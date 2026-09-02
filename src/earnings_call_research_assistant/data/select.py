"""Diversity selection and versioned train/val/test splits.

Takes filtered grounded pairs and:
  1. Caps per-source / per-task so one corpus cannot dominate.
  2. Greedy max-min diversity on token Jaccard of instruction+output.
  3. Assigns a stable split from hash(pair_id, dataset_version, seed).

No GPU. Target band is 3k–6k for the full public corpora; fixture runs
keep everything that survives the caps.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from earnings_call_research_assistant.data.filter import load_pairs_jsonl
from earnings_call_research_assistant.data.generate import InstructionPair, write_pairs_jsonl

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9\s%]+")

DATASET_VERSION = "ecra-sft-v0.1.0"
DEFAULT_TARGET_MIN = 3000
DEFAULT_TARGET_MAX = 6000


@dataclass(frozen=True)
class SelectConfig:
    """Caps, diversity knobs, and split fractions."""

    target_min: int = DEFAULT_TARGET_MIN
    target_max: int = DEFAULT_TARGET_MAX
    max_per_source: int = 2500
    max_per_task: int = 4000
    max_per_source_task: int = 2000
    diversity_jaccard_cap: float = 0.72
    train_frac: float = 0.80
    val_frac: float = 0.10
    test_frac: float = 0.10
    seed: int = 94
    dataset_version: str = DATASET_VERSION

    def __post_init__(self) -> None:
        if self.target_min < 0 or self.target_max < self.target_min:
            raise ValueError("target band must satisfy 0 <= min <= max")
        if min(self.max_per_source, self.max_per_task, self.max_per_source_task) < 1:
            raise ValueError("per-group caps must be >= 1")
        if not 0.0 <= self.diversity_jaccard_cap <= 1.0:
            raise ValueError("diversity_jaccard_cap must be in [0, 1]")
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError("train/val/test fractions must sum to 1")
        if any(f < 0 for f in (self.train_frac, self.val_frac, self.test_frac)):
            raise ValueError("split fractions must be >= 0")


@dataclass
class SplitAssignment:
    pair_id: str
    split: str
    source_id: str
    task: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SelectReport:
    dataset_version: str
    n_in: int
    n_selected: int
    n_train: int
    n_val: int
    n_test: int
    by_source: dict[str, int] = field(default_factory=dict)
    by_task: dict[str, int] = field(default_factory=dict)
    by_split: dict[str, int] = field(default_factory=dict)
    dropped_caps: int = 0
    dropped_diversity: int = 0
    seed: int = 94
    created_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def _tokens(text: str) -> set[str]:
    return {t for t in _normalize(text).split(" ") if t}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _pair_tokens(pair: InstructionPair) -> set[str]:
    return _tokens(f"{pair.task} {pair.instruction} {pair.output}")


def apply_group_caps(
    pairs: Sequence[InstructionPair],
    *,
    config: SelectConfig | None = None,
) -> tuple[list[InstructionPair], int]:
    """Keep first-seen pairs up to per-source / per-task / joint caps."""
    cfg = config or SelectConfig()
    kept: list[InstructionPair] = []
    src_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    joint_counts: Counter[tuple[str, str]] = Counter()
    dropped = 0
    for pair in pairs:
        src = pair.source_id or "unknown"
        task = pair.task or "unknown"
        if src_counts[src] >= cfg.max_per_source:
            dropped += 1
            continue
        if task_counts[task] >= cfg.max_per_task:
            dropped += 1
            continue
        if joint_counts[(src, task)] >= cfg.max_per_source_task:
            dropped += 1
            continue
        kept.append(pair)
        src_counts[src] += 1
        task_counts[task] += 1
        joint_counts[(src, task)] += 1
    return kept, dropped


def greedy_diversity(
    pairs: Sequence[InstructionPair],
    *,
    config: SelectConfig | None = None,
) -> tuple[list[InstructionPair], int]:
    """Greedy max-min: accept a pair only if max Jaccard vs kept is below cap.

    First pair in each (source, task) bucket is always kept so rare sources
    survive. Order is source then task then pair_id for determinism.
    """
    cfg = config or SelectConfig()
    ordered = sorted(
        pairs,
        key=lambda p: (p.source_id, p.task, p.pair_id),
    )
    kept: list[InstructionPair] = []
    kept_tokens: list[set[str]] = []
    seen_bucket: set[tuple[str, str]] = set()
    dropped = 0
    for pair in ordered:
        if len(kept) >= cfg.target_max:
            dropped += 1
            continue
        toks = _pair_tokens(pair)
        bucket = (pair.source_id or "unknown", pair.task or "unknown")
        if bucket not in seen_bucket:
            kept.append(pair)
            kept_tokens.append(toks)
            seen_bucket.add(bucket)
            continue
        best = 0.0
        for prior in kept_tokens:
            jac = _jaccard(toks, prior)
            if jac > best:
                best = jac
            if jac >= cfg.diversity_jaccard_cap:
                break
        if best >= cfg.diversity_jaccard_cap:
            dropped += 1
            continue
        kept.append(pair)
        kept_tokens.append(toks)
    return kept, dropped


def assign_split(pair_id: str, config: SelectConfig) -> str:
    """Stable split from SHA1(pair_id|version|seed). Independent of list order."""
    key = f"{pair_id}|{config.dataset_version}|{config.seed}".encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()
    # 8 hex chars → 32-bit space; map onto [0, 1).
    unit = int(digest[:8], 16) / 0xFFFFFFFF
    if unit < config.train_frac:
        return "train"
    if unit < config.train_frac + config.val_frac:
        return "val"
    return "test"


def split_pairs(
    pairs: Sequence[InstructionPair],
    *,
    config: SelectConfig | None = None,
) -> dict[str, list[InstructionPair]]:
    cfg = config or SelectConfig()
    buckets: dict[str, list[InstructionPair]] = {"train": [], "val": [], "test": []}
    for pair in pairs:
        buckets[assign_split(pair.pair_id, cfg)].append(pair)
    for name in buckets:
        buckets[name].sort(key=lambda p: p.pair_id)
    return buckets


def select_and_split(
    pairs: Iterable[InstructionPair],
    *,
    config: SelectConfig | None = None,
) -> tuple[dict[str, list[InstructionPair]], SelectReport]:
    """Caps → diversity → versioned splits. Returns split map + report."""
    cfg = config or SelectConfig()
    incoming = list(pairs)
    capped, n_cap = apply_group_caps(incoming, config=cfg)
    selected, n_div = greedy_diversity(capped, config=cfg)
    splits = split_pairs(selected, config=cfg)
    by_source: dict[str, int] = dict(Counter(p.source_id or "unknown" for p in selected))
    by_task: dict[str, int] = dict(Counter(p.task or "unknown" for p in selected))
    by_split = {k: len(v) for k, v in splits.items()}
    report = SelectReport(
        dataset_version=cfg.dataset_version,
        n_in=len(incoming),
        n_selected=len(selected),
        n_train=by_split.get("train", 0),
        n_val=by_split.get("val", 0),
        n_test=by_split.get("test", 0),
        by_source=by_source,
        by_task=by_task,
        by_split=by_split,
        dropped_caps=n_cap,
        dropped_diversity=n_div,
        seed=cfg.seed,
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return splits, report


def write_splits(
    splits: dict[str, list[InstructionPair]],
    out_dir: str | Path,
    *,
    report: SelectReport | None = None,
    config: SelectConfig | None = None,
) -> dict[str, Path]:
    """Write train/val/test JSONL plus manifest.json under ``out_dir``."""
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    cfg = config or SelectConfig()
    paths: dict[str, Path] = {}
    for name in ("train", "val", "test"):
        paths[name] = write_pairs_jsonl(splits.get(name, []), dest / f"{name}.jsonl")
    manifest = {
        "dataset_version": cfg.dataset_version,
        "seed": cfg.seed,
        "target_min": cfg.target_min,
        "target_max": cfg.target_max,
        "split_fractions": {
            "train": cfg.train_frac,
            "val": cfg.val_frac,
            "test": cfg.test_frac,
        },
        "files": {name: paths[name].name for name in ("train", "val", "test")},
        "report": report.to_dict() if report is not None else {},
    }
    man_path = dest / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["manifest"] = man_path
    if report is not None:
        report_path = dest / "select_report.json"
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        paths["report"] = report_path
    return paths


def load_split_jsonl(path: str | Path) -> list[InstructionPair]:
    return load_pairs_jsonl(path)
