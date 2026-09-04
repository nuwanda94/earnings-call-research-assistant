"""Publish the Gradio Space folder under ``spaces/ecra-demo`` to the Hugging Face Hub.

Token from env only (``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``). Dry-run by default.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from earnings_call_research_assistant.publish import resolve_hub_token, token_source

logger = logging.getLogger(__name__)

DEFAULT_SPACE_DIR = Path("spaces/ecra-demo")
DEFAULT_SPACE_REPO_ID = "nuwanda94/earnings-call-research-assistant"
DEFAULT_PLAN_PATH = Path("outputs/space_publish_plan.json")
REQUIRED_FILES = ("app.py", "requirements.txt", "README.md")


@dataclass
class SpacePublishPlan:
    dry_run: bool
    space_dir: str
    repo_id: str
    private: bool
    commit_message: str
    space_dir_exists: bool
    has_required_files: bool
    token_present: bool
    token_env: str | None
    files: list[str] = field(default_factory=list)
    uploaded: bool = False
    hub_url: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_space_publish(
    space_dir: Path | str = DEFAULT_SPACE_DIR,
    repo_id: str = DEFAULT_SPACE_REPO_ID,
    *,
    private: bool = False,
    commit_message: str = "feat: deploy ECRA Gradio Space",
    dry_run: bool = True,
) -> SpacePublishPlan:
    root = Path(space_dir)
    exists = root.is_dir()
    files = sorted(p.name for p in root.iterdir()) if exists else []
    has_req = exists and all((root / name).is_file() for name in REQUIRED_FILES)
    token = resolve_hub_token()
    notes: list[str] = []
    if not exists:
        notes.append(f"Space directory missing: {root}")
    elif not has_req:
        notes.append(f"Expected files {REQUIRED_FILES} under {root}")
    if token is None:
        notes.append(
            "No Hub token. Use `huggingface-cli login` or export HF_TOKEN (write scope)."
        )
    notes.append(
        "After upload: Space Settings → Hardware → GPU (T4) recommended for live weights."
    )
    notes.append(
        "Optional Space variables: ADAPTER_REPO, BASE_MODEL, SIDE_BY_SIDE, MAX_NEW_TOKENS."
    )
    return SpacePublishPlan(
        dry_run=dry_run,
        space_dir=str(root),
        repo_id=repo_id,
        private=private,
        commit_message=commit_message,
        space_dir_exists=exists,
        has_required_files=has_req,
        token_present=token is not None,
        token_env=token_source(),
        files=files,
        notes=notes,
    )


def write_space_plan(
    plan: SpacePublishPlan, path: Path | str = DEFAULT_PLAN_PATH
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote space publish plan to %s", out)
    return out


def publish_space(
    space_dir: Path | str = DEFAULT_SPACE_DIR,
    repo_id: str = DEFAULT_SPACE_REPO_ID,
    *,
    private: bool = False,
    commit_message: str = "feat: deploy ECRA Gradio Space",
    dry_run: bool = True,
    plan_path: Path | str = DEFAULT_PLAN_PATH,
) -> SpacePublishPlan:
    plan = plan_space_publish(
        space_dir,
        repo_id,
        private=private,
        commit_message=commit_message,
        dry_run=dry_run,
    )
    if dry_run:
        plan.notes.append("Dry-run only; no Hub upload. Pass --run after login.")
        write_space_plan(plan, plan_path)
        return plan

    if not plan.space_dir_exists:
        raise FileNotFoundError(plan.space_dir)
    if not plan.has_required_files:
        raise ValueError(f"Incomplete Space folder: {plan.space_dir}")
    token = resolve_hub_token()
    if not token:
        raise RuntimeError(
            "Missing HF token. Run `huggingface-cli login` or set HF_TOKEN."
        )

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required for --run. pip install huggingface_hub"
        ) from exc

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        exist_ok=True,
        private=private,
        repo_type="space",
        space_sdk="gradio",
    )
    api.upload_folder(
        folder_path=str(space_dir),
        repo_id=repo_id,
        repo_type="space",
        commit_message=commit_message,
        token=token,
    )
    plan.uploaded = True
    plan.hub_url = f"https://huggingface.co/spaces/{repo_id}"
    plan.notes.append(f"Uploaded Space to {plan.hub_url}")
    write_space_plan(plan, plan_path)
    return plan
