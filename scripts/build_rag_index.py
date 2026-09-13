#!/usr/bin/env python3
"""Build BM25 (always) and dense (hash dry-run or --run ST) RAG indices.

CPU-safe by default: dense uses a hashed bag-of-tokens embedding so retrieve
works offline. Pass ``--run`` on Kaggle to embed with sentence-transformers
and persist ``data/rag/indices/dense/``.

Examples
--------
    python scripts/build_rag_corpus.py
    python scripts/build_rag_index.py
    python scripts/build_rag_index.py --query "operating margin guidance" --k 3
    python scripts/build_rag_index.py --run --query "operating margin" --k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from earnings_call_research_assistant.rag.bm25 import (  # noqa: E402
    DEFAULT_INDEX_DIR,
    build_bm25_from_corpus,
    load_bm25_index,
)
from earnings_call_research_assistant.rag.corpus import (  # noqa: E402
    DEFAULT_CORPUS_DIR,
    build_corpus,
    write_corpus,
)
from earnings_call_research_assistant.rag.dense import (  # noqa: E402
    DEFAULT_DENSE_DIR,
    DEFAULT_EMBED_MODEL,
    build_dense_from_corpus,
    load_dense_index,
)
from earnings_call_research_assistant.rag.hybrid import (  # noqa: E402
    DEFAULT_RRF_K,
    HybridRetriever,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=ROOT / DEFAULT_CORPUS_DIR,
        help="Directory with chunks.jsonl + manifest.json.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=ROOT / DEFAULT_INDEX_DIR,
        help="Where to write BM25 index.json + meta.json.",
    )
    parser.add_argument(
        "--dense-dir",
        type=Path,
        default=ROOT / DEFAULT_DENSE_DIR,
        help="Where to write dense ids.json + embeddings.",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default=DEFAULT_EMBED_MODEL,
        help="sentence-transformers model used only with --run.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="If set, retrieve top-k hybrid (or BM25-only) hits after build.",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k for --query.")
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K, help="RRF constant.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Download / load sentence-transformers and persist real dense vectors.",
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Skip dense index; retrieve with BM25 only.",
    )
    return parser.parse_args()


def _ensure_corpus(corpus_dir: Path) -> Path:
    chunks_path = corpus_dir / "chunks.jsonl"
    if chunks_path.is_file():
        return corpus_dir
    rag_chunks, manifest = build_corpus()
    write_corpus(rag_chunks, manifest, corpus_dir)
    print(f"built missing corpus at {corpus_dir} (N={manifest.n_chunks})")
    return corpus_dir


def main() -> int:
    args = _parse_args()
    corpus_dir = _ensure_corpus(args.corpus_dir)
    index, dest = build_bm25_from_corpus(corpus_dir, args.index_dir)
    print("backend: bm25")
    print(f"n_docs: {index.n_docs}")
    print(f"avgdl: {index.avgdl:.2f}")
    print(f"corpus_version: {index.corpus_version}")
    print(f"Wrote {dest / 'index.json'}")
    print(f"Wrote {dest / 'meta.json'}")

    dense = None
    if not args.bm25_only:
        use_model = bool(args.run)
        dense, ddest = build_dense_from_corpus(
            corpus_dir,
            args.dense_dir,
            embed_model=args.embed_model,
            use_model=use_model,
        )
        print(f"dense_backend: {dense.backend}")
        print(f"dense_model: {dense.embed_model}")
        print(f"dense_dim: {dense.dim}")
        print(f"Wrote {ddest / 'ids.json'}")
        print(f"Wrote {ddest / 'meta.json'}")
        if not use_model:
            print("note: dense used hash embeddings (offline). Pass --run for ST model.")

    if args.query.strip():
        bm25_loaded = load_bm25_index(dest)
        dense_loaded = None
        if dense is not None:
            dense_loaded = load_dense_index(args.dense_dir)
        retriever = HybridRetriever(
            bm25=bm25_loaded,
            dense=None if args.bm25_only else dense_loaded,
            rrf_k=args.rrf_k,
        )
        hits = retriever.retrieve(args.query, k=args.k)
        mode = "bm25" if args.bm25_only or dense_loaded is None else "hybrid-rrf"
        print(f"retrieve_mode: {mode}")
        print(f"query: {args.query}")
        print(json.dumps([h.to_dict() for h in hits], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
