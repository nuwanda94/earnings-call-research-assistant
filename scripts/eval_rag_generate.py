#!/usr/bin/env python3
"""Grounded RAG generation eval: retrieve top-k → pack context → base vs adapter.

CPU dry-run by default (no GPU, no weights). Writes
``evals/reports/rag_generation_metrics.json`` with citation-hit and optional
token F1. On Kaggle after an adapter exists::

    python scripts/eval_rag_generate.py --run
    python scripts/eval_rag_generate.py --run --adapter-dir outputs/adapters/llama32-3b-ecra-sft
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from earnings_call_research_assistant.rag.bm25 import DEFAULT_INDEX_DIR  # noqa: E402
from earnings_call_research_assistant.rag.corpus import DEFAULT_CORPUS_DIR  # noqa: E402
from earnings_call_research_assistant.rag.dense import DEFAULT_DENSE_DIR  # noqa: E402
from earnings_call_research_assistant.rag.eval_set import DEFAULT_EVAL_PATH  # noqa: E402
from earnings_call_research_assistant.rag.generate_eval import (  # noqa: E402
    DEFAULT_GEN_OUT,
    DEFAULT_TOP_K,
    evaluate_rag_generate,
)
from earnings_call_research_assistant.rag.hybrid import DEFAULT_RRF_K  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=ROOT / DEFAULT_EVAL_PATH)
    parser.add_argument("--corpus-dir", type=Path, default=ROOT / DEFAULT_CORPUS_DIR)
    parser.add_argument("--index-dir", type=Path, default=ROOT / DEFAULT_INDEX_DIR)
    parser.add_argument("--dense-dir", type=Path, default=ROOT / DEFAULT_DENSE_DIR)
    parser.add_argument("--out", type=Path, default=ROOT / DEFAULT_GEN_OUT)
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Optional LoRA adapter directory for the fine-tuned column.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Load Unsloth InferenceHarness and generate. Off by default.",
    )
    return parser.parse_args()


def _make_generators(args: argparse.Namespace):
    if not args.run:
        return None, None

    from earnings_call_research_assistant.config import load_config
    from earnings_call_research_assistant.inference import InferenceConfig, InferenceHarness

    app = load_config(args.config)
    cfg = InferenceConfig.from_mapping(app.to_dict())
    base = InferenceHarness.from_pretrained(cfg)

    adapter = None
    if args.adapter_dir is not None:
        adapter_cfg = InferenceConfig.from_mapping(app.to_dict())
        adapter_cfg.model_name = str(args.adapter_dir)
        try:
            adapter = InferenceHarness.from_pretrained(adapter_cfg)
        except Exception as exc:  # pragma: no cover
            logging.warning("Could not load adapter %s: %s", args.adapter_dir, exc)

    def base_generate(text: str) -> str:
        return base.generate(text)

    def adapter_generate(text: str) -> str:
        if adapter is None:
            return (
                "[adapter missing] train with scripts/train_sft.py --run "
                "then pass --adapter-dir"
            )
        return adapter.generate(text)

    return base_generate, adapter_generate


def main() -> int:
    args = _parse_args()
    base_fn, adapter_fn = _make_generators(args)
    payload = evaluate_rag_generate(
        eval_path=args.eval_set,
        corpus_dir=args.corpus_dir,
        bm25_dir=args.index_dir,
        dense_dir=args.dense_dir,
        out_path=args.out,
        top_k=args.k,
        rrf_k=args.rrf_k,
        use_st_model=bool(args.run),
        dry_run=not args.run,
        adapter_dir=str(args.adapter_dir) if args.adapter_dir else None,
        base_generate=base_fn,
        adapter_generate=adapter_fn,
    )
    summary = {
        "n_queries": payload.get("n_queries"),
        "dry_run": payload.get("dry_run"),
        "top_k": payload.get("top_k"),
        "corpus_version": payload.get("corpus_version"),
        "n_index_docs": payload.get("n_index_docs"),
        "adapter_dir": payload.get("adapter_dir"),
        "aggregate": payload.get("aggregate"),
        "out_path": payload.get("out_path"),
        "note": payload.get("note"),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
