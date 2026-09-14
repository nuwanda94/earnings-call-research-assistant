#!/usr/bin/env python3
"""Commit and push small post-train metadata to GitHub (not adapter weights).

Default dry-run lists files. Pass --run with GITHUB_TOKEN (env or Kaggle secret).

Typical files:
  - outputs/sft_plan.json
  - data/processed/ecra-sft-v0.1.0/manifest.json (and select_report if present)
  - evals/reports/* metrics + RAG_EVAL_REPORT.md
  - PROGRESS.md / README.md when present

Never pushes large JSONL, indices, or adapter weights.

Examples
--------
    python scripts/push_run_metadata.py
    python scripts/push_run_metadata.py --run
    python scripts/push_run_metadata.py --run --message "chore: post-SFT metadata from Kaggle"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IST = timezone(timedelta(hours=5, minutes=30))

DEFAULT_FILES = [
    "outputs/sft_plan.json",
    "data/processed/ecra-sft-v0.1.0/manifest.json",
    "data/processed/ecra-sft-v0.1.0/select_report.json",
    "data/processed/filter_report.json",
    "evals/reports/corpus_manifest_snapshot.json",
    "evals/reports/rag_metrics.json",
    "evals/reports/rag_generation_metrics.json",
    "evals/reports/RAG_EVAL_REPORT.md",
    "evals/rag_eval_set.manifest.json",
    "README.md",
    "PROGRESS.md",
]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(c if "token" not in c.lower() else "***" for c in cmd))
    subprocess.check_call(cmd)


def _load_github_token(in_kaggle: bool) -> str | None:
    tok = os.environ.get("GITHUB_TOKEN", "").strip()
    if tok:
        return tok
    if not in_kaggle:
        return None
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        val = UserSecretsClient().get_secret("GITHUB_TOKEN")
        if val:
            os.environ["GITHUB_TOKEN"] = val
            return val
    except Exception as exc:
        print(f"GITHUB_TOKEN secret: {type(exc).__name__}")
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--owner", default="nuwanda94")
    p.add_argument("--repo", default="earnings-call-research-assistant")
    p.add_argument("--branch", default="main")
    p.add_argument(
        "--message",
        default=None,
        help="Commit message (default includes IST timestamp).",
    )
    p.add_argument(
        "--file",
        action="append",
        dest="files",
        default=None,
        help="Extra relative path to include (repeatable).",
    )
    p.add_argument(
        "--run",
        action="store_true",
        help="Actually git commit + push. Off by default.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    os.chdir(ROOT)
    in_kaggle = Path("/kaggle").exists()
    now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")

    candidates = list(DEFAULT_FILES)
    if args.files:
        candidates.extend(args.files)
    # de-dupe preserve order
    seen: set[str] = set()
    existing: list[str] = []
    for f in candidates:
        if f in seen:
            continue
        seen.add(f)
        if Path(f).is_file():
            existing.append(f)

    print(f"candidates_on_disk: {len(existing)}")
    for f in existing:
        print(f"  - {f} ({Path(f).stat().st_size} bytes)")

    if not args.run:
        print("Dry-run only. Pass --run with GITHUB_TOKEN to push.")
        return 0

    token = _load_github_token(in_kaggle)
    if not token:
        raise RuntimeError("GITHUB_TOKEN required for --run (env or Kaggle secret).")

    if not existing:
        print("Nothing to commit (no metadata files found).")
        return 0

    msg = args.message or f"chore: post-SFT run metadata from Kaggle ({now_ist})"
    remote = f"https://x-access-token:{token}@github.com/{args.owner}/{args.repo}.git"
    public = f"https://github.com/{args.owner}/{args.repo}.git"

    _run(["git", "config", "user.email", "kaggle-bot@users.noreply.github.com"])
    _run(["git", "config", "user.name", "kaggle-ecra"])
    _run(["git", "remote", "set-url", "origin", remote])
    try:
        for f in existing:
            _run(["git", "add", "-f", f])
        st = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        if not st:
            print("Nothing to commit (already up to date).")
        else:
            _run(["git", "commit", "-m", msg])
            _run(["git", "push", "origin", f"HEAD:{args.branch}"])
            print(f"Pushed to {args.owner}/{args.repo}@{args.branch}")
    finally:
        _run(["git", "remote", "set-url", "origin", public])
        print("remote URL scrubbed (no token)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
