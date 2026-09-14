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

GIT_NAME = "kaggle-ecra"
GIT_EMAIL = "kaggle-bot@users.noreply.github.com"

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


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(c if "token" not in c.lower() else "***" for c in cmd))
    subprocess.check_call(cmd, env=env)


def _run_capture(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(c if "token" not in c.lower() else "***" for c in cmd))
    return subprocess.run(cmd, env=env, text=True, capture_output=True)


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = GIT_NAME
    env["GIT_AUTHOR_EMAIL"] = GIT_EMAIL
    env["GIT_COMMITTER_NAME"] = GIT_NAME
    env["GIT_COMMITTER_EMAIL"] = GIT_EMAIL
    # Avoid interactive prompts / dubious ownership on Kaggle clones.
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    return env


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
    env = _git_env()

    candidates = list(DEFAULT_FILES)
    if args.files:
        candidates.extend(args.files)
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

    # Mark the working tree safe (Kaggle clones often trip "dubious ownership").
    _run(["git", "config", "--global", "--add", "safe.directory", str(ROOT)], env=env)
    _run(["git", "config", "user.email", GIT_EMAIL], env=env)
    _run(["git", "config", "user.name", GIT_NAME], env=env)
    _run(["git", "remote", "set-url", "origin", remote], env=env)

    try:
        for f in existing:
            _run(["git", "add", "-f", "--", f], env=env)

        st = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, env=env
        ).strip()
        print("git status --porcelain:")
        print(st or "(empty)")

        if not st:
            print("Nothing to commit (already up to date).")
            return 0

        # Pass identity via -c so commit never depends on missing local config.
        commit_cmd = [
            "git",
            "-c", f"user.name={GIT_NAME}",
            "-c", f"user.email={GIT_EMAIL}",
            "commit",
            "--allow-empty-message",
            "-m",
            msg,
        ]
        proc = _run_capture(commit_cmd, env=env)
        if proc.returncode != 0:
            err = (proc.stderr or "") + (proc.stdout or "")
            # Treat "nothing to commit" as success (race / already staged elsewhere).
            if "nothing to commit" in err.lower() or "no changes added" in err.lower():
                print("Commit skipped: nothing to commit after staging.")
            else:
                print("git commit FAILED")
                print(err)
                raise subprocess.CalledProcessError(proc.returncode, commit_cmd, output=proc.stdout, stderr=proc.stderr)
        else:
            print((proc.stdout or "").strip() or "commit ok")

        push = _run_capture(["git", "push", "origin", f"HEAD:{args.branch}"], env=env)
        if push.returncode != 0:
            print("git push FAILED")
            print((push.stderr or "") + (push.stdout or ""))
            raise subprocess.CalledProcessError(push.returncode, push.args, output=push.stdout, stderr=push.stderr)
        print((push.stdout or "").strip() or "push ok")
        print(f"Pushed to {args.owner}/{args.repo}@{args.branch}")
    finally:
        _run(["git", "remote", "set-url", "origin", public], env=env)
        print("remote URL scrubbed (no token)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
