#!/usr/bin/env python3
"""Launch the CPU-safe Gradio research-panel demo.

Default is a dry-run UI: no model download, no train.

On a GPU box after adapters exist::

    pip install -e ".[demo]"
    python scripts/demo_gradio.py --run
    python scripts/demo_gradio.py --run --adapter-dir outputs/adapters/llama32-3b-ecra-sft
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from earnings_call_research_assistant.demo import launch_demo  # noqa: E402
from earnings_call_research_assistant.eval.panel import DEFAULT_PANEL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
        help="YAML used when --run loads InferenceHarness.",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Optional LoRA / merged snapshot directory (used only with --run).",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=DEFAULT_PANEL,
        help="JSONL panel used for example dropdown prompts.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Load InferenceHarness and generate. Off by default (CPU stub).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Ask Gradio for a temporary public link.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    launch_demo(
        load_model=args.run,
        config_path=args.config,
        adapter_dir=args.adapter_dir,
        panel_path=args.panel,
        share=args.share,
        server_name=args.host,
        server_port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
