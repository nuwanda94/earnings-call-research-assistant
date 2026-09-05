"""Publish a local LoRA adapter directory to the Hugging Face Hub.

Token is read from the environment only (``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN``).
Never write tokens to disk or logs. Default path is a dry-run that prints a plan
and writes ``outputs/publish_plan.json`` without calling the Hub.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ADAPTER_DIR = Path("outputs/adapters/llama32-3b-ecra-sft")
# HF username (Hub), not the GitHub org/user
DEFAULT_REPO_ID = "skaran786/llama32-3b-ecra-sft"
DEFAULT_PLAN_PATH = Path("outputs/publish_plan.json")
TOKEN_ENV_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

ADAPTER_HINT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "README.md",
)


def resolve_hub_token() -> str | None:
    """Return a Hub token from env vars. Does not print the value."""
    for key in TOKEN_ENV_KEYS:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return None


def token_source() -> str | None:
    for key in TOKEN_ENV_KEYS:
        if os.environ.get(key, "").strip():
            return key
    return None


def _looks_like_adapter(path: Path) -> bool:
    if not path.is_dir():
        return False
    names = {p.name for p in path.iterdir()}
    return any(name in names for name in ADAPTER_HINT_FILES)


@dataclass
class PublishPlan:
    dry_run: bool
    adapter_dir: str
    repo_id: str
    private: bool
    commit_message: str
    adapter_exists: bool
    looks_like_adapter: bool
    token_present: bool
    token_env: str | None
    files: list[str] = field(default_factory=list)
    uploaded: bool = False
    hub_url: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_publish(
    adapter_dir: Path | str = DEFAULT_ADAPTER_DIR,
    repo_id: str = DEFAULT_REPO_ID,
    *,
    private: bool = False,
    commit_message: str = "feat: upload ECRA QLoRA adapter",
    dry_run: bool = True,
) -> PublishPlan:
    adapter = Path(adapter_dir)
    exists = adapter.is_dir()
    files = sorted(p.name for p in adapter.iterdir()) if exists else []
    token = resolve_hub_token()
    notes: list[str] = []
    if not exists:
        notes.append(
            f"Adapter directory missing: {adapter}. Train on Kaggle first "
            "(`python scripts/train_sft.py --run`) and copy the folder locally."
        )
    elif not _looks_like_adapter(adapter):
        notes.append(
            "Directory exists but does not look like a PEFT adapter "
            "(expected adapter_config.json or adapter_model.*)."
        )
    if token is None:
        notes.append(
            "No Hub token in the environment. Use `huggingface-cli login` "
            "or export HF_TOKEN (write scope). Never commit the token."
        )
    notes.append(
        "Auth: huggingface-cli login  OR  export HF_TOKEN=...  "
        "(HUGGING_FACE_HUB_TOKEN is also accepted)."
    )
    return PublishPlan(
        dry_run=dry_run,
        adapter_dir=str(adapter),
        repo_id=repo_id,
        private=private,
        commit_message=commit_message,
        adapter_exists=exists,
        looks_like_adapter=_looks_like_adapter(adapter) if exists else False,
        token_present=token is not None,
        token_env=token_source(),
        files=files,
        notes=notes,
    )


def write_plan(plan: PublishPlan, path: Path | str = DEFAULT_PLAN_PATH) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote publish plan to %s", out)
    return out


def publish_adapter(
    adapter_dir: Path | str = DEFAULT_ADAPTER_DIR,
    repo_id: str = DEFAULT_REPO_ID,
    *,
    private: bool = False,
    commit_message: str = "feat: upload ECRA QLoRA adapter",
    dry_run: bool = True,
    plan_path: Path | str = DEFAULT_PLAN_PATH,
) -> PublishPlan:
    """Dry-run by default. ``dry_run=False`` calls HfApi.upload_folder."""
    plan = plan_publish(
        adapter_dir,
        repo_id,
        private=private,
        commit_message=commit_message,
        dry_run=dry_run,
    )
    if dry_run:
        plan.notes.append("Dry-run only; no Hub upload. Pass --run after login.")
        write_plan(plan, plan_path)
        return plan

    if not plan.adapter_exists:
        raise FileNotFoundError(plan.adapter_dir)
    if not plan.looks_like_adapter:
        raise ValueError(f"Not an adapter directory: {plan.adapter_dir}")
    token = resolve_hub_token()
    if not token:
        raise RuntimeError(
            "Missing HF token. Run `huggingface-cli login` or set HF_TOKEN."
        )

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required for --run. "
            "`pip install huggingface_hub` (or `transformers` extra)."
        ) from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private, repo_type="model")
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
        token=token,
    )
    plan.uploaded = True
    plan.hub_url = f"https://huggingface.co/{repo_id}"
    plan.notes.append(f"Uploaded adapter to {plan.hub_url}")
    write_plan(plan, plan_path)
    return plan
