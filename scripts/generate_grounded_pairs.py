#!/usr/bin/env python3
"""Build citation-grounded instruction pairs from chunk JSONL or live ingest.

Default generator is offline templates (no model). Pass --use-llm only on Kaggle
once you wire a callable; this CLI still refuses to invent an LLM backend.

Examples
--------
    python scripts/generate_grounded_pairs.py
    python scripts/generate_grounded_pairs.py --chunks data/processed/chunks.jsonl
    python scripts/generate_grounded_pairs.py --max-qa 1 --no-summary
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
)
from earnings_call_research_assistant.data.generate import (  # noqa: E402
    GenerateConfig,
    generate_pairs,
    load_chunks_jsonl,
    write_pairs_jsonl,
)
from earnings_call_research_assistant.data.ingest import ingest_catalog  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=None,
        help="Existing chunks JSONL. If omitted, ingest + chunk offline fixtures.",
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        default=None,
        help="Source id when building chunks from ingest (repeatable).",
    )
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-qa", type=int, default=2, help="QA templates per chunk.")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Reserved: requires a caller-supplied LLM. The CLI exits unless implemented.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "processed" / "grounded_pairs.jsonl",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.use_llm:
        print(
            "--use-llm is a Kaggle hook: pass an LLMGenerator into generate_pairs() "
            "from a notebook. The CLI stays template-only so automation never bills GPU.",
            file=sys.stderr,
        )
        return 2

    if args.chunks is not None:
        chunks = load_chunks_jsonl(args.chunks)
    else:
        records = ingest_catalog(
            source_ids=args.sources,
            max_samples=args.max_samples,
            download=args.download,
        )
        cfg = ChunkConfig(window_sentences=args.window, stride_sentences=args.stride)
        chunks = chunk_records(records, config=cfg)

    gen_cfg = GenerateConfig(
        max_qa_per_chunk=args.max_qa,
        include_summary=not args.no_summary,
        use_llm=False,
    )
    pairs = generate_pairs(chunks, config=gen_cfg)
    dest = write_pairs_jsonl(pairs, args.out)
    n_qa = sum(1 for p in pairs if p.task == "qa")
    n_sum = sum(1 for p in pairs if p.task == "summary")
    print(f"Chunks: {len(chunks)}  pairs: {len(pairs)}  qa: {n_qa}  summary: {n_sum}")
    print(f"Wrote {dest}")
    preview = [p.to_dict() for p in pairs[:2]]
    print(json.dumps(preview, indent=2)[:1600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
