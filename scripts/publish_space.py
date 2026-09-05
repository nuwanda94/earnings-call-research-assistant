#!/usr/bin/env python3
"""Push a Space folder to the Hugging Face Hub.

Default is a **static** Space (``spaces/ecra-static``) — free, no PRO.
Gradio Spaces on free cpu-basic may return HTTP 402.

    export HF_TOKEN=hf_xxx
    python scripts/publish_space.py --run
    python scripts/publish_space.py --sdk static --run
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
    DEFAULT_SPACE_SDK,
    publish_space,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-dir", type=Path, default=ROOT / DEFAULT_SPACE_DIR)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_SPACE_REPO_ID)
    parser.add_argument(
        "--sdk",
        choices=("static", "gradio"),
        default=DEFAULT_SPACE_SDK,
        help="static = free HTML Space; gradio may require HF PRO on free CPU.",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--commit-message",
        default="feat: deploy ECRA Space",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually create/upload. Off by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    plan = publish_space(
        space_dir=args.space_dir,
        repo_id=args.repo_id,
        private=args.private,
        space_sdk=args.sdk,
        commit_message=args.commit_message,
        dry_run=not args.run,
    )
    print(
        f"dry_run={plan.dry_run} sdk={plan.space_sdk} repo={plan.repo_id} "
        f"ok_files={plan.has_required_files} token={plan.token_present} "
        f"uploaded={plan.uploaded}"
    )
    if plan.hub_url:
        print("Open:", plan.hub_url)
    print(json.dumps(plan.to_dict(), indent=2)[:2500])
    if args.run and not plan.uploaded:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
