#!/usr/bin/env python3
"""Push a local QLoRA adapter folder to the Hugging Face Hub.

Default is a **dry-run**: inspects the adapter path, checks whether a token is
present in the environment, and writes ``outputs/publish_plan.json``. No upload.

Auth (token never committed; never printed):

    huggingface-cli login
    # or
    export HF_TOKEN=hf_xxx          # write-scoped token
    # HUGGING_FACE_HUB_TOKEN is also accepted

Examples
--------
    python scripts/publish_adapter.py
    python scripts/publish_adapter.py --adapter-dir outputs/adapters/llama32-3b-ecra-sft
    python scripts/publish_adapter.py --repo-id nuwanda94/llama32-3b-ecra-sft --run
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

from earnings_call_research_assistant.publish import (  # noqa: E402
    DEFAULT_ADAPTER_DIR,
    DEFAULT_REPO_ID,
    publish_adapter,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=ROOT / DEFAULT_ADAPTER_DIR,
        help="Local PEFT / Unsloth adapter directory (gitignored weights).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Hugging Face model repo id (namespace/name).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create / keep the Hub repo private.",
    )
    parser.add_argument(
        "--commit-message",
        default="feat: upload ECRA QLoRA adapter",
        help="Hub commit message used only with --run.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually upload. Off by default (dry-run plan only).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    plan = publish_adapter(
        adapter_dir=args.adapter_dir,
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message,
        dry_run=not args.run,
    )
    print(
        f"dry_run={plan.dry_run} repo={plan.repo_id} "
        f"adapter={plan.adapter_dir} exists={plan.adapter_exists} "
        f"token_present={plan.token_present} uploaded={plan.uploaded}"
    )
    print(json.dumps(plan.to_dict(), indent=2)[:2000])
    if args.run and not plan.uploaded:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
