#!/usr/bin/env python3
"""Build the fixed RAG retrieval eval set (gold chunk IDs).

Deterministic sample, seed 3407. Uses the versioned corpus when present;
otherwise builds it from Phase-1 chunks or offline fixtures. Drops pair_ids
that appear in SFT train.jsonl when that split exists.

Examples
--------
    python scripts/build_rag_eval_set.py
    python scripts/build_rag_eval_set.py --out evals/rag_eval_set.jsonl
    python scripts/build_rag_eval_set.py --dataset-dir data/processed/ecra-sft-v0.1.0
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

from earnings_call_research_assistant.rag.eval_set import (  # noqa: E402
    DEFAULT_EVAL_PATH,
    DEFAULT_SEED,
    DEFAULT_TARGET_N,
    build_rag_eval_set,
    write_eval_set,
)
from earnings_call_research_assistant.rag.corpus import DEFAULT_CORPUS_DIR  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=ROOT / DEFAULT_CORPUS_DIR,
        help="RAG corpus directory (chunks.jsonl + manifest.json).",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=ROOT / "data" / "processed" / "chunks.jsonl",
        help="Phase-1 chunks JSONL used if the corpus must be rebuilt.",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=ROOT / "data" / "processed" / "grounded_pairs.jsonl",
        help="Grounded pairs JSONL when present.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "ecra-sft-v0.1.0",
        help="SFT split dir; train.jsonl pair_ids are excluded.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / DEFAULT_EVAL_PATH,
        help="Output JSONL path.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-n", type=int, default=DEFAULT_TARGET_N)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rows, manifest = build_rag_eval_set(
        corpus_dir=args.corpus_dir,
        chunks_path=args.chunks if args.chunks.is_file() else None,
        pairs_path=args.pairs if args.pairs.is_file() else None,
        dataset_dir=args.dataset_dir if args.dataset_dir.is_dir() else None,
        seed=args.seed,
        max_n=args.max_n,
    )
    dest = write_eval_set(rows, args.out, manifest=manifest)
    print(f"eval_set_version: {manifest.version}")
    print(f"seed: {manifest.seed}")
    print(f"n_queries: {manifest.n_queries}")
    print(f"corpus_n_chunks: {manifest.corpus_n_chunks}")
    print(f"n_unique_gold_chunks: {manifest.n_unique_gold_chunks}")
    print(f"excluded_train_pairs: {manifest.excluded_train_pairs}")
    print(f"Wrote {dest}")
    print(f"Wrote {dest.with_suffix('.manifest.json')}")
    print(json.dumps({"notes": manifest.notes}, ensure_ascii=False))
    if manifest.n_queries < 1:
        print("warning: empty eval set — build the corpus / grounded pairs first", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
