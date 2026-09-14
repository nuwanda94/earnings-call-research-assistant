#!/usr/bin/env python3
"""Write a filled HF model-card README into the adapter directory.

Reads optional SFT plan + RAG metric JSON; leaves TBD when files are missing.
Does not invent percentages.

Examples
--------
    python scripts/write_model_card.py
    python scripts/write_model_card.py --adapter-dir outputs/adapters/llama32-3b-ecra-sft
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

from earnings_call_research_assistant.model_card import (  # noqa: E402
    DEFAULT_ADAPTER_DIR,
    write_model_card,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter-dir", type=Path, default=ROOT / DEFAULT_ADAPTER_DIR)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="README path (default: <adapter-dir>/README.md).",
    )
    p.add_argument("--sft-plan", type=Path, default=ROOT / "outputs" / "sft_plan.json")
    p.add_argument(
        "--dataset-manifest",
        type=Path,
        default=ROOT / "data" / "processed" / "ecra-sft-v0.1.0" / "manifest.json",
    )
    p.add_argument(
        "--corpus-manifest",
        type=Path,
        default=ROOT / "data" / "rag" / "corpus_v0.1.0" / "manifest.json",
    )
    p.add_argument(
        "--rag-metrics",
        type=Path,
        default=ROOT / "evals" / "reports" / "rag_metrics.json",
    )
    p.add_argument(
        "--gen-metrics",
        type=Path,
        default=ROOT / "evals" / "reports" / "rag_generation_metrics.json",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out = args.out or (args.adapter_dir / "README.md")
    path, inp = write_model_card(
        out,
        sft_plan_path=args.sft_plan if args.sft_plan.is_file() else None,
        dataset_manifest=args.dataset_manifest if args.dataset_manifest.is_file() else None,
        corpus_manifest=args.corpus_manifest if args.corpus_manifest.is_file() else None,
        rag_metrics=args.rag_metrics if args.rag_metrics.is_file() else None,
        gen_metrics=args.gen_metrics if args.gen_metrics.is_file() else None,
        adapter_dir=args.adapter_dir,
    )
    print(f"Wrote {path}")
    print(f"inputs_meta: {path.with_suffix('.inputs.json')}")
    print(json.dumps(inp.to_dict(), indent=2)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
