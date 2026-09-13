"""Reciprocal Rank Fusion of BM25 and dense ranks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from earnings_call_research_assistant.rag.bm25 import Bm25Index, Hit, load_bm25_index
from earnings_call_research_assistant.rag.dense import DenseIndex, load_dense_index

DEFAULT_RRF_K = 60


def rrf_score(rank: int, k: int = DEFAULT_RRF_K) -> float:
    return 1.0 / (k + rank)


def fuse_hits(
    *lists: Sequence[Hit],
    k: int = 5,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[Hit]:
    """Fuse one or more ranked lists with Reciprocal Rank Fusion."""
    by_id: dict[str, Hit] = {}
    scores: dict[str, float] = {}
    for ranked in lists:
        for hit in ranked:
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + rrf_score(hit.rank, rrf_k)
            existing = by_id.get(hit.chunk_id)
            if existing is None:
                by_id[hit.chunk_id] = Hit(
                    chunk_id=hit.chunk_id,
                    score=0.0,
                    rank=0,
                    source_id=hit.source_id,
                    section=hit.section,
                    text=hit.text,
                    bm25=hit.bm25,
                    dense=hit.dense,
                )
            else:
                if hit.bm25 and not existing.bm25:
                    existing.bm25 = hit.bm25
                if hit.dense is not None and existing.dense is None:
                    existing.dense = hit.dense
                if not existing.text and hit.text:
                    existing.text = hit.text
                    existing.source_id = hit.source_id
                    existing.section = hit.section
    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))
    fused: list[Hit] = []
    for rank, cid in enumerate(ordered[: max(0, k)], start=1):
        hit = by_id[cid]
        hit.score = float(scores[cid])
        hit.rank = rank
        fused.append(hit)
    return fused


@dataclass
class HybridRetriever:
    bm25: Bm25Index | None = None
    dense: DenseIndex | None = None
    rrf_k: int = DEFAULT_RRF_K
    candidate_k: int = 50

    def retrieve(self, query: str, k: int = 5) -> list[Hit]:
        lists: list[list[Hit]] = []
        pool = max(self.candidate_k, k)
        if self.bm25 is not None:
            lists.append(self.bm25.retrieve(query, k=pool))
        if self.dense is not None:
            lists.append(self.dense.retrieve(query, k=pool))
        if not lists:
            return []
        if len(lists) == 1:
            hits = lists[0][:k]
            for i, hit in enumerate(hits, start=1):
                hit.rank = i
            return hits
        return fuse_hits(*lists, k=k, rrf_k=self.rrf_k)


def load_hybrid(
    bm25_dir: str | Path | None = None,
    dense_dir: str | Path | None = None,
    *,
    rrf_k: int = DEFAULT_RRF_K,
) -> HybridRetriever:
    bm25 = load_bm25_index(bm25_dir) if bm25_dir is not None else None
    dense = None
    if dense_dir is not None and Path(dense_dir).exists():
        ids = Path(dense_dir) / "ids.json"
        if ids.is_file():
            dense = load_dense_index(dense_dir)
    return HybridRetriever(bm25=bm25, dense=dense, rrf_k=rrf_k)
