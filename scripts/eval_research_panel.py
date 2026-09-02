#!/usr/bin/env python3
"""Run the qualitative research panel (base vs adapter).

Default is a CPU dry-run: no model download. Writes
``evals/reports/research_panel_comparison.json`` with placeholders.

On Kaggle after a QLoRA adapter exists::

    python scripts/eval_research_panel.py --run
    python scripts/eval_research_panel.py --run --adapter-dir outputs/adapters/llama32-3b-ecra-sft
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

from earnings_call_research_assistant.eval.panel import (  # noqa: E402
    DEFAULT_OUT,
    DEFAULT_PANEL,
    run_research_panel,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=DEFAULT_PANEL,
        help="JSONL research panel (20 grounded prompts).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Side-by-side JSON report path.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
        help="YAML used when --run loads the base harness.",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Optional LoRA adapter directory for the fine-tuned column.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Load Unsloth and generate. Off by default (no GPU).",
    )
    return parser.parse_args()


def _make_generators(args: argparse.Namespace):
    if not args.run:
        return None, None

    from earnings_call_research_assistant.config import load_config
    from earnings_call_research_assistant.inference import (
        InferenceConfig,
        InferenceHarness,
    )

    app = load_config(args.config)
    cfg = InferenceConfig.from_mapping(app.to_dict())
    base = InferenceHarness.from_pretrained(cfg)

    adapter = None
    if args.adapter_dir is not None:
        adapter_cfg = InferenceConfig.from_mapping(app.to_dict())
        adapter_cfg.model_name = str(args.adapter_dir)
        try:
            adapter = InferenceHarness.from_pretrained(adapter_cfg)
        except Exception as exc:  # pragma: no cover - environment-specific
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
    summary = run_research_panel(
        args.panel,
        args.out,
        dry_run=not args.run,
        base_generate=base_fn,
        adapter_generate=adapter_fn,
        extra_meta={
            "config": str(args.config),
            "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        },
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
