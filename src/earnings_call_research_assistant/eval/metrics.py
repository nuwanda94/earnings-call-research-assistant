"""Quantitative metrics stub for research-panel side-by-side JSON.

Scores base vs adapter columns with CPU-only token overlap and citation-hit
rates. Does not load weights or start a train. Feed the JSON written by
``eval.panel.write_comparison`` (dry-run placeholders or Kaggle generations).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COMPARISON = ROOT / "evals" / "reports" / "research_panel_comparison.json"
DEFAULT_METRICS_OUT = ROOT / "evals" / "reports" / "research_panel_metrics.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?%?|\$[0-9]+(?:\.[0-9]+)?[a-z]*", re.I)
_REFUSAL_CUES = (
    "not in the context",
    "not stated",
    "insufficient",
    "cannot determine",
    "does not say",
    "not provided",
    "no go-live",
    "will update",
    "refuse",
    "missing from the excerpt",
)


def tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")}


def token_overlap(hypothesis: str, reference: str) -> dict[str, float]:
    """Unigram precision / recall / F1 / Jaccard of hypothesis vs reference."""
    hyp = tokenize(hypothesis)
    ref = tokenize(reference)
    if not hyp and not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0}
    inter = hyp & ref
    precision = len(inter) / len(hyp) if hyp else 0.0
    recall = len(inter) / len(ref) if ref else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    union = hyp | ref
    jaccard = len(inter) / len(union) if union else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "jaccard": round(jaccard, 4),
    }


def citation_hit_rate(text: str, needles: Sequence[str]) -> dict[str, Any]:
    hits = [n for n in needles if n and n.lower() in (text or "").lower()]
    n = len(needles)
    rate = (len(hits) / n) if n else None
    return {
        "n_needles": n,
        "n_hits": len(hits),
        "hits": hits,
        "rate": None if rate is None else round(rate, 4),
    }


def looks_like_refusal(text: str) -> bool:
    lower = (text or "").lower()
    return any(cue in lower for cue in _REFUSAL_CUES)


@dataclass
class SideScores:
    token_vs_context: dict[str, float]
    citation: dict[str, Any]
    refusal_flag: bool
    refusal_correct: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ItemScores:
    id: str
    task: str
    theme: str
    refuse_if_missing: bool
    base: SideScores
    adapter: SideScores
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "theme": self.theme,
            "refuse_if_missing": self.refuse_if_missing,
            "base": self.base.to_dict(),
            "adapter": self.adapter.to_dict(),
            "notes": self.notes,
        }


@dataclass
class MetricsReport:
    n_items: int
    n_citation_items: int
    n_refusal_items: int
    dry_run: bool
    aggregate: dict[str, Any] = field(default_factory=dict)
    items: list[ItemScores] = field(default_factory=list)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_items": self.n_items,
            "n_citation_items": self.n_citation_items,
            "n_refusal_items": self.n_refusal_items,
            "dry_run": self.dry_run,
            "source": self.source,
            "aggregate": self.aggregate,
            "items": [i.to_dict() for i in self.items],
        }


def _score_side(
    text: str,
    context: str,
    must_cite: Sequence[str],
    refuse_if_missing: bool,
) -> SideScores:
    citation = citation_hit_rate(text, must_cite)
    refused = looks_like_refusal(text)
    refusal_correct: bool | None
    if refuse_if_missing:
        refusal_correct = refused
    else:
        refusal_correct = None
    return SideScores(
        token_vs_context=token_overlap(text, context),
        citation=citation,
        refusal_flag=refused,
        refusal_correct=refusal_correct,
    )


def _mean(values: Iterable[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 4)


def score_comparison(
    payload: dict[str, Any] | None = None,
    *,
    comparison_path: Path | str | None = None,
) -> MetricsReport:
    if payload is None:
        path = Path(comparison_path) if comparison_path else DEFAULT_COMPARISON
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = str(path)
    else:
        source = comparison_path or "memory"

    rows = payload.get("rows") or []
    items: list[ItemScores] = []
    for raw in rows:
        context = str(raw.get("context", ""))
        must_cite = [str(x) for x in raw.get("must_cite", [])]
        refuse = bool(raw.get("refuse_if_missing", False))
        items.append(
            ItemScores(
                id=str(raw.get("id", "")),
                task=str(raw.get("task", "")),
                theme=str(raw.get("theme", "")),
                refuse_if_missing=refuse,
                base=_score_side(
                    str(raw.get("base_output", "")),
                    context,
                    must_cite,
                    refuse,
                ),
                adapter=_score_side(
                    str(raw.get("adapter_output", "")),
                    context,
                    must_cite,
                    refuse,
                ),
                notes=str(raw.get("notes", "")),
            )
        )

    cite_items = [i for i in items if i.base.citation["n_needles"]]
    refuse_items = [i for i in items if i.refuse_if_missing]

    def agg(side: str) -> dict[str, Any]:
        sides = [getattr(i, side) for i in items]
        cite_sides = [getattr(i, side) for i in cite_items]
        refuse_sides = [getattr(i, side) for i in refuse_items]
        return {
            "token_f1_vs_context": _mean(s.token_vs_context["f1"] for s in sides),
            "token_jaccard_vs_context": _mean(
                s.token_vs_context["jaccard"] for s in sides
            ),
            "citation_hit_rate": _mean(s.citation["rate"] for s in cite_sides),
            "refusal_accuracy": _mean(
                (1.0 if s.refusal_correct else 0.0) for s in refuse_sides
            ),
        }

    dry_run = bool(payload.get("dry_run", False))
    return MetricsReport(
        n_items=len(items),
        n_citation_items=len(cite_items),
        n_refusal_items=len(refuse_items),
        dry_run=dry_run,
        source=str(source),
        aggregate={"base": agg("base"), "adapter": agg("adapter")},
        items=items,
    )


def write_metrics(
    report: MetricsReport,
    out_path: Path | str | None = None,
) -> Path:
    path = Path(out_path) if out_path else DEFAULT_METRICS_OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def score_research_panel(
    comparison_path: Path | str | None = None,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    report = score_comparison(comparison_path=comparison_path)
    written = write_metrics(report, out_path)
    return {
        "n_items": report.n_items,
        "dry_run": report.dry_run,
        "out_path": str(written),
        "aggregate": report.aggregate,
    }
