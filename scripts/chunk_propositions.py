#!/usr/bin/env python3
"""Chunk ingested public samples and extract factual propositions.

Uses offline fixtures by default (same as ingest). Writes JSONL of section-aware
windows plus heuristic propositions — no model calls.

Examples
--------
    python scripts/chunk_propositions.py
    python scripts/chunk_propositions.py --out data/processed/chunks.jsonl
    python scripts/chunk_propositions.py --source earnings_transcripts --window 3 --stride 2
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

from earnings_call_research_assistant.data.chunk import (  # noqa: E402
    ChunkConfig,
    chunk_records,
    write_chunks_jsonl,
)
from earnings_call_research_assistant.data.ingest import ingest_catalog  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        default=None,
        help="Source id to include (repeatable). Default: all catalog entries.",
    )
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Stream a tiny HF sample instead of offline fixtures.",
    )
    parser.add_argument("--window", type=int, default=4, help="Sentences per chunk.")
    parser.add_argument("--stride", type=int, default=2, help="Sentence stride.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "processed" / "chunks.jsonl",
        help="JSONL output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    records = ingest_catalog(
        source_ids=args.sources,
        max_samples=args.max_samples,
        download=args.download,
    )
    cfg = ChunkConfig(window_sentences=args.window, stride_sentences=args.stride)
    chunks = chunk_records(records, config=cfg)
    dest = write_chunks_jsonl(chunks, args.out)
    n_props = sum(len(c.propositions) for c in chunks)
    print(f"Records: {len(records)}  chunks: {len(chunks)}  propositions: {n_props}")
    print(f"Wrote {dest}")
    preview = [c.to_dict() for c in chunks[:2]]
    print(json.dumps(preview, indent=2)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
