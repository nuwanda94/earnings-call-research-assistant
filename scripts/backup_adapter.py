#!/usr/bin/env python3
"""Backup a QLoRA adapter from local disk or the Hugging Face Hub.

Use after a Kaggle train/publish so weights are not trapped in an ephemeral session.

Examples
--------
    # Copy local adapter folder
    python scripts/backup_adapter.py --src outputs/adapters/llama32-3b-ecra-sft \
        --dest backups/llama32-3b-ecra-sft

    # Download from Hub (needs HF_TOKEN only if private)
    python scripts/backup_adapter.py --repo-id nuwanda94/llama32-3b-ecra-sft \
        --dest backups/llama32-3b-ecra-sft
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "outputs" / "adapters" / "llama32-3b-ecra-sft"
DEFAULT_DEST = ROOT / "backups" / "llama32-3b-ecra-sft"
DEFAULT_REPO = "nuwanda94/llama32-3b-ecra-sft"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=None, help="Local adapter directory.")
    p.add_argument("--repo-id", type=str, default=None, help="Hub model id to download.")
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Backup destination dir.")
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite dest if it already exists.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    dest = args.dest
    if dest.exists() and any(dest.iterdir()) and not args.force:
        print(f"Destination not empty: {dest} (pass --force to overwrite)")
        return 1

    if args.src is not None or (args.repo_id is None and DEFAULT_SRC.is_dir()):
        src = args.src or DEFAULT_SRC
        if not src.is_dir():
            print(f"Local adapter missing: {src}")
            return 1
        if dest.exists() and args.force:
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest)
        print(f"Copied local adapter {src} -> {dest}")
        return 0

    repo_id = args.repo_id or DEFAULT_REPO
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("pip install huggingface_hub") from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and args.force:
        shutil.rmtree(dest)
    path = snapshot_download(repo_id=repo_id, local_dir=str(dest))
    print(f"Downloaded {repo_id} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
