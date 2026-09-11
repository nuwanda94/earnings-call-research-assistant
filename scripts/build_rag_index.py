#!/usr/bin/env python3
"""Build the BM25 RAG index and optionally retrieve top-k hits.

CPU / offline. Dense embeddings are a later Phase 5.3 step (``--run`` will
then download an embed model). This script always builds BM25.

Examples
--------
    python scripts/build_rag_corpus.py
    python scripts/build_rag_index.py
    python scripts/build_rag_index.py --query "operating margin guidance" --k 3
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
        help="Where to write index.json + meta.json.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="If set, retrieve top-k BM25 hits after (re)building the index.",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k for --query.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Reserved for Phase 5.3 dense embed. Ignored for BM25.",
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
    print(f"backend: bm25")
    print(f"n_docs: {index.n_docs}")
    print(f"avgdl: {index.avgdl:.2f}")
    print(f"corpus_version: {index.corpus_version}")
    print(f"Wrote {dest / 'index.json'}")
    print(f"Wrote {dest / 'meta.json'}")
    if args.run:
        print("note: --run reserved for dense embeddings (Phase 5.3); BM25 built on CPU.")
    if args.query.strip():
        # reload to prove persist path works offline
        loaded = load_bm25_index(dest)
        hits = loaded.retrieve(args.query, k=args.k)
        print(f"query: {args.query}")
        print(json.dumps([h.to_dict() for h in hits], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
