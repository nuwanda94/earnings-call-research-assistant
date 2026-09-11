#!/usr/bin/env python3
"""Build a versioned RAG corpus and print measured N.

Reads Phase-1 ``data/processed/chunks.jsonl`` when present; otherwise re-chunks
offline public fixtures. Writes ``chunks.jsonl`` + ``manifest.json``.

Examples
--------
    python scripts/build_rag_corpus.py
    python scripts/build_rag_corpus.py --chunks data/processed/chunks.jsonl
    python scripts/build_rag_corpus.py --out-dir data/rag/corpus_v0.1.0
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

from earnings_call_research_assistant.rag.corpus import (  # noqa: E402
    CORPUS_VERSION,
    DEFAULT_CORPUS_DIR,
    build_corpus,
    write_corpus,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=ROOT / "data" / "processed" / "chunks.jsonl",
        help="Phase-1 chunks JSONL (used when the file exists).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / DEFAULT_CORPUS_DIR,
        help="Directory for chunks.jsonl + manifest.json.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help="Fixture cap per source when Phase-1 JSONL is missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    chunks_path = args.chunks if args.chunks.is_file() else None
    rag_chunks, manifest = build_corpus(
        chunks_path=chunks_path,
        max_samples=args.max_samples,
    )
    dest = write_corpus(rag_chunks, manifest, args.out_dir)
    print(f"corpus_version: {CORPUS_VERSION}")
    print(f"N (n_chunks): {manifest.n_chunks}")
    print(f"n_documents: {manifest.n_documents}")
    print(f"sources: {json.dumps(manifest.sources)}")
    print(f"Wrote {dest / 'chunks.jsonl'}")
    print(f"Wrote {dest / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
