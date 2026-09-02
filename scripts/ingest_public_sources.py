#!/usr/bin/env python3
"""Document and optionally sample public Phase-1 sources.

Default: print the catalog and write a tiny offline JSONL sample.
Never downloads large corpora unless ``--download`` is passed.

Examples
--------
    python scripts/ingest_public_sources.py
    python scripts/ingest_public_sources.py --max-samples 2 --out data/raw/public_sample.jsonl
    python scripts/ingest_public_sources.py --source fiqa --download --max-samples 8
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

from earnings_call_research_assistant.data.ingest import (  # noqa: E402
    ingest_catalog,
    list_sources,
    write_jsonl,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        default=None,
        help="Source id to include (repeatable). Default: all catalog entries.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help="Cap per source (default 3). Keep small; this is a stub.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Stream a tiny sample from Hugging Face. Off by default.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "raw" / "public_sample.jsonl",
        help="JSONL output path for the sampled records.",
    )
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="Print the source catalog and exit without writing samples.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    sources = list_sources()
    print("Public sources (Phase 1 catalog)\n")
    for spec in sources:
        print(f"- {spec.source_id}: {spec.display_name}")
        print(f"    role:     {spec.role}")
        print(f"    hf:       {spec.hf_id or '(none)'}")
        print(f"    home:     {spec.homepage}")
        print(f"    license:  {spec.license_note}")
        print(f"    notes:    {spec.notes}\n")

    if args.catalog_only:
        return 0

    records = ingest_catalog(
        source_ids=args.sources,
        max_samples=args.max_samples,
        download=args.download,
    )
    dest = write_jsonl(records, args.out)
    print(f"Wrote {len(records)} records -> {dest}")
    print(json.dumps([r.to_dict() for r in records[:2]], indent=2)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
