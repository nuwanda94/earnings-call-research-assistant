#!/usr/bin/env python3
"""Push ``spaces/ecra-demo`` to a Hugging Face **Space** (Gradio).

Default is a **dry-run** that writes ``outputs/space_publish_plan.json``.

Auth (never commit the token):

    huggingface-cli login
    # or
    export HF_TOKEN=hf_xxx

Examples
--------
    python scripts/publish_space.py
    python scripts/publish_space.py --repo-id nuwanda94/earnings-call-research-assistant --run

After upload, open the Space → Settings → Hardware → choose **T4** for live
base vs adapter generation. Set Space variable ``ADAPTER_REPO`` if the adapter
lives under a different model id.
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

from earnings_call_research_assistant.space_publish import (  # noqa: E402
    DEFAULT_SPACE_DIR,
    DEFAULT_SPACE_REPO_ID,
    publish_space,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--space-dir",
        type=Path,
        default=ROOT / DEFAULT_SPACE_DIR,
        help="Local Gradio Space folder to upload.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_SPACE_REPO_ID,
        help="Hugging Face Space id (namespace/name).",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--commit-message",
        default="feat: deploy ECRA Gradio Space",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually create/upload the Space. Off by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    plan = publish_space(
        space_dir=args.space_dir,
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message,
        dry_run=not args.run,
    )
    print(
        f"dry_run={plan.dry_run} repo={plan.repo_id} "
        f"dir={plan.space_dir} ok_files={plan.has_required_files} "
        f"token_present={plan.token_present} uploaded={plan.uploaded}"
    )
    if plan.hub_url:
        print("Open:", plan.hub_url)
    print(json.dumps(plan.to_dict(), indent=2)[:2500])
    if args.run and not plan.uploaded:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
