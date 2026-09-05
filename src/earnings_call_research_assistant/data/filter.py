"""Multi-stage quality filter for grounded instruction pairs.

Pipeline (offline by default):
  1. Heuristic length / emptiness / citation checks.
  2. Exact then near-duplicate drop (hash + bucketed token Jaccard).
  3. Optional LLM-as-judge stub for Kaggle.

Near-dup is **bucketed** (not full O(n²)) so 10k–60k pair runs finish on CPU.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from earnings_call_research_assistant.data.generate import InstructionPair

LLMJudge = Callable[[InstructionPair], dict[str, Any]]

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9\s%]+")
_CITE_HINT = re.compile(r"\b(citation|citations|chunk_|source)\b", re.I)


@dataclass(frozen=True)
class FilterConfig:
    min_instruction_chars: int = 12
    max_instruction_chars: int = 600
    min_output_chars: int = 40
    max_output_chars: int = 4000
    min_context_chars: int = 40
    require_citation: bool = True
    require_citation_in_output: bool = True
    near_dup_jaccard: float = 0.88
    use_llm_judge: bool = False
    min_judge_score: float = 0.6
    # Cap pairwise compares inside each near-dup bucket (keeps CPU bounded).
    near_dup_bucket_compare_limit: int = 64

    def __post_init__(self) -> None:
        if self.min_instruction_chars < 0 or self.min_output_chars < 0:
            raise ValueError("length floors must be >= 0")
        if not 0.0 <= self.near_dup_jaccard <= 1.0:
            raise ValueError("near_dup_jaccard must be in [0, 1]")
        if not 0.0 <= self.min_judge_score <= 1.0:
            raise ValueError("min_judge_score must be in [0, 1]")


@dataclass
class FilterDecision:
    pair_id: str
    kept: bool
    stage: str
    reasons: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FilterReport:
    n_in: int
    n_kept: int
    dropped_by_stage: dict[str, int] = field(default_factory=dict)
    decisions: list[FilterDecision] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_in": self.n_in,
            "n_kept": self.n_kept,
            "n_dropped": self.n_in - self.n_kept,
            "dropped_by_stage": dict(self.dropped_by_stage),
            # Keep report light for large runs
            "decisions": [d.to_dict() for d in self.decisions[:5000]],
            "decisions_truncated": len(self.decisions) > 5000,
        }


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


def _fingerprint(pair: InstructionPair) -> str:
    key = _normalize(f"{pair.task}|{pair.instruction}|{pair.output}")
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _near_dup_bucket_key(pair: InstructionPair, toks: set[str]) -> str:
    """Coarse bucket so near-dup only compares similar pairs."""
    head = " ".join(sorted(toks)[:4]) if toks else ""
    length_band = len(pair.output or "") // 80
    return f"{pair.source_id}|{pair.task}|{length_band}|{head}"


def heuristic_check(pair: InstructionPair, config: FilterConfig) -> FilterDecision:
    reasons: list[str] = []
    inst = (pair.instruction or "").strip()
    out = (pair.output or "").strip()
    ctx = (pair.context or "").strip()

    if len(inst) < config.min_instruction_chars:
        reasons.append("instruction_too_short")
    if len(inst) > config.max_instruction_chars:
        reasons.append("instruction_too_long")
    if len(out) < config.min_output_chars:
        reasons.append("output_too_short")
    if len(out) > config.max_output_chars:
        reasons.append("output_too_long")
    if len(ctx) < config.min_context_chars:
        reasons.append("context_too_short")
    if not inst or not out:
        reasons.append("empty_fields")

    if config.require_citation and not pair.citations:
        reasons.append("missing_citation_list")
    if config.require_citation_in_output:
        cite_ok = bool(_CITE_HINT.search(out))
        if pair.chunk_id and pair.chunk_id in out:
            cite_ok = True
        if pair.citations and any(c.split("]", 1)[0] in out for c in pair.citations):
            cite_ok = True
        if not cite_ok:
            reasons.append("output_missing_citation_marker")

    kept = not reasons
    return FilterDecision(
        pair_id=pair.pair_id,
        kept=kept,
        stage="heuristic",
        reasons=reasons,
        scores={
            "instruction_chars": float(len(inst)),
            "output_chars": float(len(out)),
            "n_citations": float(len(pair.citations)),
        },
    )


def drop_duplicates(
    pairs: Sequence[InstructionPair],
    *,
    config: FilterConfig | None = None,
) -> tuple[list[InstructionPair], list[FilterDecision]]:
    """Exact hash drop, then bucketed near-dup (not full pairwise O(n²))."""
    cfg = config or FilterConfig()
    kept: list[InstructionPair] = []
    decisions: list[FilterDecision] = []
    seen_fp: set[str] = set()
    # bucket_key -> list of token sets for kept items in that bucket
    bucket_tokens: dict[str, list[set[str]]] = defaultdict(list)

    n = len(pairs)
    report_every = max(5000, n // 10) if n else 5000

    for i, pair in enumerate(pairs):
        if report_every and i and i % report_every == 0:
            print(f"[filter:dedup] {i}/{n} scanned, kept={len(kept)}")

        fp = _fingerprint(pair)
        if fp in seen_fp:
            decisions.append(
                FilterDecision(
                    pair_id=pair.pair_id,
                    kept=False,
                    stage="exact_dup",
                    reasons=["exact_duplicate"],
                    scores={"jaccard": 1.0},
                )
            )
            continue

        toks = _tokens(f"{pair.instruction} {pair.output}")
        bkey = _near_dup_bucket_key(pair, toks)
        prior_list = bucket_tokens[bkey]
        near = False
        best = 0.0
        # Only compare against a limited number of priors in the same bucket
        for prior in prior_list[-cfg.near_dup_bucket_compare_limit :]:
            jac = _jaccard(toks, prior)
            if jac > best:
                best = jac
            if jac >= cfg.near_dup_jaccard:
                near = True
                break
        if near:
            decisions.append(
                FilterDecision(
                    pair_id=pair.pair_id,
                    kept=False,
                    stage="near_dup",
                    reasons=["near_duplicate"],
                    scores={"jaccard": best},
                )
            )
            continue

        seen_fp.add(fp)
        bucket_tokens[bkey].append(toks)
        kept.append(pair)
        decisions.append(
            FilterDecision(
                pair_id=pair.pair_id,
                kept=True,
                stage="dedup",
                reasons=[],
                scores={"jaccard": best},
            )
        )

    print(f"[filter:dedup] done scanned={n} kept={len(kept)}")
    return kept, decisions


def default_llm_judge(pair: InstructionPair) -> dict[str, Any]:
    out = (pair.output or "").lower()
    hits = 0
    for cite in pair.citations:
        snippet = cite.split("]", 1)[-1].strip().lower()[:48]
        if snippet and snippet in out:
            hits += 1
        elif pair.chunk_id and pair.chunk_id.lower() in out:
            hits += 1
    denom = max(len(pair.citations), 1)
    grounded = min(1.0, hits / denom)
    length_ok = 1.0 if 40 <= len(pair.output or "") <= 2000 else 0.4
    score = 0.7 * grounded + 0.3 * length_ok
    return {
        "score": round(score, 4),
        "grounded": round(grounded, 4),
        "reason": "heuristic-proxy-judge",
    }


def apply_llm_judge(
    pairs: Sequence[InstructionPair],
    *,
    config: FilterConfig | None = None,
    judge: LLMJudge | None = None,
) -> tuple[list[InstructionPair], list[FilterDecision]]:
    cfg = config or FilterConfig()
    fn = judge or default_llm_judge
    kept: list[InstructionPair] = []
    decisions: list[FilterDecision] = []
    for pair in pairs:
        raw = fn(pair) or {}
        score = float(raw.get("score", 0.0))
        reason = str(raw.get("reason") or "llm_judge")
        accept = score >= cfg.min_judge_score
        decisions.append(
            FilterDecision(
                pair_id=pair.pair_id,
                kept=accept,
                stage="llm_judge",
                reasons=[] if accept else [f"judge_score<{cfg.min_judge_score}", reason],
                scores={"judge": score},
            )
        )
        if accept:
            meta = dict(pair.metadata)
            meta["judge_score"] = score
            meta["judge_reason"] = reason
            pair.metadata = meta
            kept.append(pair)
    return kept, decisions


def filter_pairs(
    pairs: Iterable[InstructionPair],
    *,
    config: FilterConfig | None = None,
    judge: LLMJudge | None = None,
) -> tuple[list[InstructionPair], FilterReport]:
    cfg = config or FilterConfig()
    incoming = list(pairs)
    print(f"[filter] start n_in={len(incoming)}")
    dropped: dict[str, int] = {"heuristic": 0, "exact_dup": 0, "near_dup": 0, "llm_judge": 0}
    decisions: list[FilterDecision] = []

    after_h: list[InstructionPair] = []
    for pair in incoming:
        dec = heuristic_check(pair, cfg)
        decisions.append(dec)
        if dec.kept:
            after_h.append(pair)
        else:
            dropped["heuristic"] += 1
    print(f"[filter] after heuristic kept={len(after_h)}")

    after_d, d_decs = drop_duplicates(after_h, config=cfg)
    for dec in d_decs:
        if not dec.kept:
            dropped[dec.stage] = dropped.get(dec.stage, 0) + 1
            decisions.append(dec)
        elif dec.stage == "dedup":
            decisions.append(dec)

    kept = after_d
    if cfg.use_llm_judge:
        kept, j_decs = apply_llm_judge(after_d, config=cfg, judge=judge)
        for dec in j_decs:
            decisions.append(dec)
            if not dec.kept:
                dropped["llm_judge"] += 1

    report = FilterReport(
        n_in=len(incoming),
        n_kept=len(kept),
        dropped_by_stage=dropped,
        decisions=decisions,
    )
    print(f"[filter] done n_kept={report.n_kept} dropped={report.dropped_by_stage}")
    return kept, report


def pair_from_dict(row: dict[str, Any]) -> InstructionPair:
    return InstructionPair(
        pair_id=str(row.get("pair_id") or "unknown"),
        chunk_id=str(row.get("chunk_id") or ""),
        source_id=str(row.get("source_id") or ""),
        task=str(row.get("task") or "qa"),
        instruction=str(row.get("instruction") or ""),
        context=str(row.get("context") or ""),
        output=str(row.get("output") or ""),
        citations=list(row.get("citations") or []),
        metadata=dict(row.get("metadata") or {}),
    )


def load_pairs_jsonl(path: str | Path) -> list[InstructionPair]:
    dest = Path(path)
    rows: list[InstructionPair] = []
    with dest.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(pair_from_dict(json.loads(line)))
    return rows


def write_filter_report(report: FilterReport, path: str | Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return dest
