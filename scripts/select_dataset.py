#!/usr/bin/env python3
"""Diversity-select filtered pairs and write versioned train/val/test splits.

Default path is fully offline: if --pairs is omitted the script rebuilds the
fixture ingest → chunk → generate → filter chain.

Examples
--------
    python scripts/select_dataset.py
    python scripts/select_dataset.py --pairs data/processed/filtered_pairs.jsonl
    python scripts/select_dataset.py --target-max 12 --out-dir data/processed/ecra-sft-v0.1.0
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
)
from earnings_call_research_assistant.data.generate import (  # noqa: E402
    GenerateConfig,
    generate_pairs,
    load_chunks_jsonl,
)
from earnings_call_research_assistant.data.ingest import ingest_catalog  # noqa: E402
from earnings_call_research_assistant.data.select import (  # noqa: E402
    DATASET_VERSION,
    SelectConfig,
    select_and_split,
    write_splits,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=None,
        help="Filtered grounded pairs JSONL. If omitted, rebuild from fixtures.",
    )
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--source", action="append", dest="sources", default=None)
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--target-min", type=int, default=1)
    parser.add_argument("--target-max", type=int, default=6000)
    parser.add_argument("--max-per-source", type=int, default=2500)
    parser.add_argument("--diversity-jaccard-cap", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=94)
    parser.add_argument("--dataset-version", type=str, default=DATASET_VERSION)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for train/val/test JSONL + manifest (default data/processed/<version>).",
    )
    return parser.parse_args()


def _load_or_build(args: argparse.Namespace):
    if args.pairs is not None:
        return load_pairs_jsonl(args.pairs)
    if args.chunks is not None:
        chunks = load_chunks_jsonl(args.chunks)
        pairs = generate_pairs(chunks, config=GenerateConfig())
    else:
        records = ingest_catalog(
            source_ids=args.sources,
            max_samples=args.max_samples,
            download=args.download,
        )
        chunks = chunk_records(records, config=ChunkConfig())
        pairs = generate_pairs(chunks, config=GenerateConfig())
    kept, _report = filter_pairs(pairs, config=FilterConfig())
    return kept


def main() -> int:
    args = _parse_args()
    pairs = _load_or_build(args)
    cfg = SelectConfig(
        target_min=args.target_min,
        target_max=args.target_max,
        max_per_source=args.max_per_source,
        diversity_jaccard_cap=args.diversity_jaccard_cap,
        seed=args.seed,
        dataset_version=args.dataset_version,
    )
    splits, report = select_and_split(pairs, config=cfg)
    out_dir = args.out_dir or (ROOT / "data" / "processed" / cfg.dataset_version)
    paths = write_splits(splits, out_dir, report=report, config=cfg)
    print(
        f"version={report.dataset_version} in={report.n_in} selected={report.n_selected} "
        f"train={report.n_train} val={report.n_val} test={report.n_test} "
        f"dropped_caps={report.dropped_caps} dropped_diversity={report.dropped_diversity}"
    )
    print(f"by_source={report.by_source} by_task={report.by_task}")
    print(f"Wrote {paths['train']}")
    print(f"Wrote {paths['val']}")
    print(f"Wrote {paths['test']}")
    print(f"Wrote {paths['manifest']}")
    preview = {
        split: [p.to_dict() for p in rows[:1]] for split, rows in splits.items() if rows
    }
    print(json.dumps(preview, indent=2)[:1600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
