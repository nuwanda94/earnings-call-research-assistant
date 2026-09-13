"""Hybrid RAG stack (Phase 5): corpus inventory, BM25 + dense + RRF, eval set."""

from earnings_call_research_assistant.rag.bm25 import (
    Bm25Index,
    Hit,
    build_bm25_from_corpus,
    build_bm25_index,
    load_bm25_index,
)
from earnings_call_research_assistant.rag.corpus import (
    CORPUS_VERSION,
    CorpusManifest,
    RagChunk,
    build_corpus,
    load_corpus_chunks,
    write_corpus,
)
from earnings_call_research_assistant.rag.dense import (
    DenseIndex,
    build_dense_from_corpus,
    build_dense_index,
    load_dense_index,
)
from earnings_call_research_assistant.rag.eval_set import (
    EVAL_SET_VERSION,
    EvalSetManifest,
    RagEvalRow,
    build_rag_eval_set,
    load_eval_set,
    write_eval_set,
)
from earnings_call_research_assistant.rag.hybrid import HybridRetriever, fuse_hits, load_hybrid

__all__ = [
    "CORPUS_VERSION",
    "EVAL_SET_VERSION",
    "CorpusManifest",
    "EvalSetManifest",
    "RagChunk",
    "RagEvalRow",
    "Bm25Index",
    "DenseIndex",
    "Hit",
    "HybridRetriever",
    "build_corpus",
    "load_corpus_chunks",
    "write_corpus",
    "build_bm25_index",
    "build_bm25_from_corpus",
    "load_bm25_index",
    "build_dense_index",
    "build_dense_from_corpus",
    "load_dense_index",
    "fuse_hits",
    "load_hybrid",
    "build_rag_eval_set",
    "load_eval_set",
    "write_eval_set",
]
