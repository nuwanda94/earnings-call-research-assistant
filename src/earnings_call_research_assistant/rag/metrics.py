"""Retrieval metrics: Recall@k and nDCG@k (binary gold chunk IDs)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from earnings_call_research_assistant.rag.bm25 import DEFAULT_INDEX_DIR, build_bm25_from_corpus, load_bm25_index
from earnings_call_research_assistant.rag.corpus import DEFAULT_CORPUS_DIR
from earnings_call_research_assistant.rag.dense import DEFAULT_DENSE_DIR, build_dense_from_corpus, load_dense_index
from earnings_call_research_assistant.rag.eval_set import (
    DEFAULT_EVAL_PATH,
    RagEvalRow,
    build_rag_eval_set,
    load_eval_set,
    write_eval_set,
)
from earnings_call_research_assistant.rag.hybrid import DEFAULT_RRF_K, HybridRetriever

DEFAULT_K_VALUES = (1, 3, 5, 10)
DEFAULT_METRICS_OUT = Path("evals") / "reports" / "rag_metrics.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def recall_at_k(gold_ids: Sequence[str], ranked_ids: Sequence[str], k: int) -> float:
    """Fraction of gold chunk IDs appearing in the top-k ranked list."""
    gold = {g for g in gold_ids if g}
    if not gold or k <= 0:
        return 0.0
    top = set(ranked_ids[:k])
    return len(gold & top) / len(gold)


def ndcg_at_k(gold_ids: Sequence[str], ranked_ids: Sequence[str], k: int) -> float:
    """nDCG@k with binary relevance on gold chunk IDs."""
    gold = {g for g in gold_ids if g}
    if not gold or k <= 0:
        return 0.0

    def _gain(rank_1: int) -> float:
        return 1.0 / math.log2(rank_1 + 1)

    dcg = 0.0
    for i, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold:
            dcg += _gain(i)
    ideal_hits = min(len(gold), k)
    idcg = sum(_gain(i) for i in range(1, ideal_hits + 1))
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def _mean(values: Iterable[float]) -> float:
    nums = list(values)
    if not nums:
        return 0.0
    return sum(nums) / len(nums)


@dataclass
class QueryMetrics:
    query_id: str
    n_gold: int
    ranked_ids: list[str]
    recall: dict[str, float]
    ndcg: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BackendReport:
    backend: str
    n_queries: int
    k_values: list[int]
    mean_recall: dict[str, float]
    mean_ndcg: dict[str, float]
    queries: list[QueryMetrics] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "n_queries": self.n_queries,
            "k_values": self.k_values,
            "mean_recall": self.mean_recall,
            "mean_ndcg": self.mean_ndcg,
            "queries": [q.to_dict() for q in self.queries],
        }


def score_ranked(
    gold_ids: Sequence[str],
    ranked_ids: Sequence[str],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> tuple[dict[str, float], dict[str, float]]:
    recall = {f"@{k}": round(recall_at_k(gold_ids, ranked_ids, k), 6) for k in k_values}
    ndcg = {f"@{k}": round(ndcg_at_k(gold_ids, ranked_ids, k), 6) for k in k_values}
    return recall, ndcg


def evaluate_retriever(
    rows: Sequence[RagEvalRow],
    retriever: HybridRetriever,
    *,
    backend: str,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> BackendReport:
    max_k = max(k_values) if k_values else 10
    per_q: list[QueryMetrics] = []
    for row in rows:
        hits = retriever.retrieve(row.query, k=max_k)
        ranked = [h.chunk_id for h in hits]
        rec, nd = score_ranked(row.gold_chunk_ids, ranked, k_values)
        per_q.append(
            QueryMetrics(
                query_id=row.query_id,
                n_gold=len([g for g in row.gold_chunk_ids if g]),
                ranked_ids=ranked,
                recall=rec,
                ndcg=nd,
            )
        )
    mean_recall = {
        f"@{k}": round(_mean(q.recall[f"@{k}"] for q in per_q), 6) for k in k_values
    }
    mean_ndcg = {
        f"@{k}": round(_mean(q.ndcg[f"@{k}"] for q in per_q), 6) for k in k_values
    }
    return BackendReport(
        backend=backend,
        n_queries=len(per_q),
        k_values=list(k_values),
        mean_recall=mean_recall,
        mean_ndcg=mean_ndcg,
        queries=per_q,
    )


def _ensure_eval_rows(eval_path: Path) -> list[RagEvalRow]:
    if eval_path.is_file():
        rows = load_eval_set(eval_path)
        if rows:
            return rows
    rows, manifest = build_rag_eval_set()
    write_eval_set(rows, eval_path, manifest=manifest)
    return rows


def evaluate_retrieval(
    *,
    eval_path: str | Path | None = None,
    corpus_dir: str | Path | None = None,
    bm25_dir: str | Path | None = None,
    dense_dir: str | Path | None = None,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    rrf_k: int = DEFAULT_RRF_K,
    use_st_model: bool = False,
    backends: Sequence[str] = ("bm25", "dense", "hybrid"),
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    eval_file = Path(eval_path) if eval_path is not None else DEFAULT_EVAL_PATH
    rows = _ensure_eval_rows(eval_file)
    corpus = Path(corpus_dir) if corpus_dir is not None else DEFAULT_CORPUS_DIR
    bdir = Path(bm25_dir) if bm25_dir is not None else DEFAULT_INDEX_DIR
    ddir = Path(dense_dir) if dense_dir is not None else DEFAULT_DENSE_DIR

    if not (bdir / "index.json").is_file():
        build_bm25_from_corpus(corpus, bdir)
    bm25 = load_bm25_index(bdir)

    want_dense = any(b in backends for b in ("dense", "hybrid"))
    dense = None
    if want_dense:
        if not (ddir / "ids.json").is_file():
            build_dense_from_corpus(corpus, ddir, use_model=use_st_model)
        dense = load_dense_index(ddir)

    reports: dict[str, BackendReport] = {}
    if "bm25" in backends:
        reports["bm25"] = evaluate_retriever(
            rows,
            HybridRetriever(bm25=bm25, dense=None, rrf_k=rrf_k),
            backend="bm25",
            k_values=k_values,
        )
    if "dense" in backends and dense is not None:
        reports["dense"] = evaluate_retriever(
            rows,
            HybridRetriever(bm25=None, dense=dense, rrf_k=rrf_k),
            backend=f"dense:{dense.backend}",
            k_values=k_values,
        )
    if "hybrid" in backends:
        reports["hybrid"] = evaluate_retriever(
            rows,
            HybridRetriever(bm25=bm25, dense=dense, rrf_k=rrf_k),
            backend="hybrid-rrf" if dense is not None else "bm25",
            k_values=k_values,
        )

    payload: dict[str, Any] = {
        "created_utc": _utc_now(),
        "eval_set": str(eval_file),
        "n_queries": len(rows),
        "k_values": list(k_values),
        "rrf_k": rrf_k,
        "corpus_dir": str(corpus),
        "bm25_dir": str(bdir),
        "dense_dir": str(ddir),
        "dense_backend": None if dense is None else dense.backend,
        "dense_embed_model": None if dense is None else dense.embed_model,
        "n_index_docs": bm25.n_docs,
        "corpus_version": bm25.corpus_version,
        "note": (
            "Means are computed on this eval set only. Do not put percentages "
            "on a resume until a full-corpus --run (ST embeddings) is logged."
        ),
        "backends": {name: report.to_dict() for name, report in reports.items()},
    }
    dest = Path(out_path) if out_path is not None else DEFAULT_METRICS_OUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    payload["out_path"] = str(dest)
    return payload
