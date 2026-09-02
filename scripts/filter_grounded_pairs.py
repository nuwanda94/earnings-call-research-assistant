#!/usr/bin/env python3
"""Filter grounded instruction pairs: heuristic → dedup → optional judge stub.

Default path is fully offline. --use-llm-judge runs the deterministic proxy
judge (no GPU). Wire a real model from a Kaggle notebook via filter_pairs().

Examples
--------
    python scripts/filter_grounded_pairs.py
    python scripts/filter_grounded_pairs.py --pairs data/processed/grounded_pairs.jsonl
    python scripts/filter_grounded_pairs.py --use-llm-judge --min-judge-score 0.5
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
from earnings_call_research_assistant.data.filter import (  # noqa: E402
    FilterConfig,
    filter_pairs,
    load_pairs_jsonl,
    write_filter_report,
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
        "--pairs",
        type=Path,
        default=None,
        help="Existing grounded pairs JSONL. If omitted, generate from fixtures.",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=None,
        help="Chunks JSONL used when --pairs is omitted.",
    )
    parser.add_argument("--source", action="append", dest="sources", default=None)
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--min-output-chars", type=int, default=40)
    parser.add_argument("--near-dup-jaccard", type=float, default=0.88)
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Enable judge stage (proxy by default; no GPU billed).",
    )
    parser.add_argument("--min-judge-score", type=float, default=0.6)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "processed" / "filtered_pairs.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data" / "processed" / "filter_report.json",
    )
    return parser.parse_args()


def _load_or_build(args: argparse.Namespace):
    if args.pairs is not None:
        return load_pairs_jsonl(args.pairs)
    if args.chunks is not None:
        chunks = load_chunks_jsonl(args.chunks)
    else:
        records = ingest_catalog(
            source_ids=args.sources,
            max_samples=args.max_samples,
            download=args.download,
        )
        chunks = chunk_records(records, config=ChunkConfig())
    return generate_pairs(chunks, config=GenerateConfig())


def main() -> int:
    args = _parse_args()
    pairs = _load_or_build(args)
    cfg = FilterConfig(
        min_output_chars=args.min_output_chars,
        near_dup_jaccard=args.near_dup_jaccard,
        use_llm_judge=args.use_llm_judge,
        min_judge_score=args.min_judge_score,
    )
    kept, report = filter_pairs(pairs, config=cfg)
    dest = write_pairs_jsonl(kept, args.out)
    report_path = write_filter_report(report, args.report)
    print(
        f"in={report.n_in} kept={report.n_kept} dropped={report.n_in - report.n_kept} "
        f"by_stage={report.dropped_by_stage}"
    )
    print(f"Wrote pairs {dest}")
    print(f"Wrote report {report_path}")
    preview = [p.to_dict() for p in kept[:2]]
    print(json.dumps(preview, indent=2)[:1600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
