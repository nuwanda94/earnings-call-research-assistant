#!/usr/bin/env python3
"""Launch Unsloth QLoRA SFT from versioned splits + configs/default.yaml.

Default is a CPU dry-run: no model download, no trainer.train().
Pass --run only on a Kaggle T4 (or local CUDA box) after the dataset exists.

Examples
--------
    python scripts/train_sft.py
    python scripts/train_sft.py --dataset-dir data/processed/ecra-sft-v0.1.0
    python scripts/train_sft.py --run --max-steps 20
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

from earnings_call_research_assistant.data.select import DATASET_VERSION  # noqa: E402
from earnings_call_research_assistant.training.sft import run_sft  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
        help="YAML/JSON AppConfig path.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "data" / "processed" / DATASET_VERSION,
        help="Directory with train.jsonl / val.jsonl from select_dataset.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Trainer logs/checkpoints (default: training.output_dir in YAML).",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Where to save the LoRA adapter after --run.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually load Unsloth and call trainer.train(). Off by default.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap for a Kaggle smoke train (e.g. 20).",
    )
    parser.add_argument("--save-steps", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    plan = run_sft(
        config_path=args.config,
        dataset_dir=args.dataset_dir,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        dry_run=not args.run,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        require_train=args.run,
    )
    print(
        f"dry_run={plan.dry_run} model={plan.model_name} "
        f"train={plan.n_train} val={plan.n_val} "
        f"dataset={plan.dataset_dir} adapter={plan.adapter_dir}"
    )
    print(json.dumps(plan.to_dict(), indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
