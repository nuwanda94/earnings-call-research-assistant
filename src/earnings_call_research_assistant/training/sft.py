"""Unsloth QLoRA SFT trainer for versioned grounded instruction splits.

Default path is a CPU-safe dry run: load ``configs/default.yaml``, read the
versioned JSONL splits, format chat examples, and write a training plan.
The GPU path (Unsloth + TRL ``SFTTrainer``) is intended for Kaggle T4 and
is never launched unless ``--run`` is passed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from earnings_call_research_assistant.config import (
    AppConfig,
    DEFAULT_SYSTEM_PROMPT,
    load_config,
)
from earnings_call_research_assistant.data.filter import load_pairs_jsonl
from earnings_call_research_assistant.data.generate import InstructionPair
from earnings_call_research_assistant.data.select import DATASET_VERSION

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = _REPO_ROOT / "data" / "processed" / DATASET_VERSION


@dataclass(frozen=True)
class SFTRunConfig:
    """Resolved SFT knobs (YAML training block + CLI overrides)."""

    model_name: str
    max_seq_length: int
    load_in_4bit: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: float
    warmup_steps: int
    logging_steps: int
    save_steps: int = 50
    max_steps: int | None = None
    optim: str = "adamw_8bit"
    seed: int = 3407
    output_dir: str = "outputs"
    adapter_dir: str = "outputs/adapters/llama32-3b-ecra-sft"
    dataset_dir: str = str(DEFAULT_DATASET_DIR)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    packing: bool = False

    @classmethod
    def from_app(
        cls,
        app: AppConfig,
        *,
        dataset_dir: str | Path | None = None,
        adapter_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        max_steps: int | None = None,
        save_steps: int | None = None,
    ) -> "SFTRunConfig":
        extras = app.extras or {}
        train_extra = extras.get("training") if isinstance(extras.get("training"), dict) else {}
        # Fields that may live in YAML extras or the training mapping via to_dict.
        train_dict = app.to_dict().get("training", {})
        resolved_dataset = (
            dataset_dir
            or train_dict.get("dataset_dir")
            or extras.get("dataset_dir")
            or DEFAULT_DATASET_DIR
        )
        resolved_adapter = (
            adapter_dir
            or train_dict.get("adapter_dir")
            or extras.get("adapter_dir")
            or Path(str(train_dict.get("output_dir", "outputs")))
            / "adapters"
            / "llama32-3b-ecra-sft"
        )
        resolved_output = output_dir or train_dict.get("output_dir") or "outputs"
        raw_max = max_steps if max_steps is not None else train_dict.get("max_steps")
        raw_save = save_steps if save_steps is not None else train_dict.get("save_steps", 50)
        return cls(
            model_name=app.model.name,
            max_seq_length=app.model.max_seq_length,
            load_in_4bit=app.model.load_in_4bit,
            lora_r=app.lora.r,
            lora_alpha=app.lora.lora_alpha,
            lora_dropout=app.lora.lora_dropout,
            target_modules=app.lora.target_modules,
            per_device_train_batch_size=app.training.per_device_train_batch_size,
            gradient_accumulation_steps=app.training.gradient_accumulation_steps,
            learning_rate=app.training.learning_rate,
            num_train_epochs=app.training.num_train_epochs,
            warmup_steps=app.training.warmup_steps,
            logging_steps=app.training.logging_steps,
            save_steps=int(raw_save) if raw_save is not None else 50,
            max_steps=int(raw_max) if raw_max not in (None, "") else None,
            optim=app.training.optim,
            seed=app.training.seed,
            output_dir=str(resolved_output),
            adapter_dir=str(resolved_adapter),
            dataset_dir=str(resolved_dataset),
            system_prompt=app.inference.system_prompt,
            packing=bool(train_dict.get("packing", False) or train_extra.get("packing", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_modules"] = list(self.target_modules)
        return payload


@dataclass
class SFTPlan:
    """What a run would do — written even when ``dry_run=True``."""

    dry_run: bool
    n_train: int
    n_val: int
    dataset_dir: str
    model_name: str
    adapter_dir: str
    output_dir: str
    seed: int
    preview_texts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pair_to_messages(pair: InstructionPair, system_prompt: str) -> list[dict[str, str]]:
    user = pair.instruction.strip()
    if pair.context and pair.context.strip():
        user = (
            f"{user}\n\nContext from the source excerpt:\n{pair.context.strip()}"
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
        {"role": "assistant", "content": pair.output.strip()},
    ]


def format_pair_text(
    pair: InstructionPair,
    *,
    system_prompt: str,
    tokenizer: Any | None = None,
) -> str:
    messages = pair_to_messages(pair, system_prompt)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    parts = []
    for msg in messages:
        parts.append(f"<|{msg['role']}|>\n{msg['content']}")
    return "\n".join(parts) + "\n"


def resolve_split_paths(dataset_dir: str | Path) -> dict[str, Path]:
    root = Path(dataset_dir)
    return {
        "train": root / "train.jsonl",
        "val": root / "val.jsonl",
        "test": root / "test.jsonl",
        "manifest": root / "manifest.json",
    }


def load_sft_splits(
    dataset_dir: str | Path,
    *,
    require_train: bool = True,
) -> dict[str, list[InstructionPair]]:
    paths = resolve_split_paths(dataset_dir)
    splits: dict[str, list[InstructionPair]] = {}
    for name in ("train", "val", "test"):
        path = paths[name]
        if path.exists():
            splits[name] = load_pairs_jsonl(path)
        else:
            splits[name] = []
    if require_train and not splits["train"]:
        raise FileNotFoundError(
            f"No training rows under {dataset_dir}. "
            "Run `python scripts/select_dataset.py` first "
            f"(expected {paths['train']})."
        )
    return splits


def _write_plan(plan: SFTPlan, output_dir: str | Path) -> Path:
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "sft_plan.json"
    path.write_text(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def build_plan(
    run: SFTRunConfig,
    splits: dict[str, list[InstructionPair]],
    *,
    dry_run: bool,
    tokenizer: Any | None = None,
) -> SFTPlan:
    preview = [
        format_pair_text(p, system_prompt=run.system_prompt, tokenizer=tokenizer)
        for p in splits.get("train", [])[:2]
    ]
    notes = [
        "Public grounded pairs only; do not invent figures at train time.",
        f"Versioned splits expected at {run.dataset_dir}.",
        "Kaggle: enable T4 GPU, pip install unsloth + trl, then pass --run.",
        "Dry-run never loads weights or starts SFTTrainer.",
    ]
    if dry_run:
        notes.append("dry_run=True — skipped Unsloth FastLanguageModel and trainer.train().")
    return SFTPlan(
        dry_run=dry_run,
        n_train=len(splits.get("train", [])),
        n_val=len(splits.get("val", [])),
        dataset_dir=run.dataset_dir,
        model_name=run.model_name,
        adapter_dir=run.adapter_dir,
        output_dir=run.output_dir,
        seed=run.seed,
        preview_texts=preview,
        notes=notes,
    )


def _load_unsloth_model(run: SFTRunConfig):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=run.model_name,
        max_seq_length=run.max_seq_length,
        load_in_4bit=run.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=run.lora_r,
        lora_alpha=run.lora_alpha,
        lora_dropout=run.lora_dropout,
        target_modules=list(run.target_modules),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=run.seed,
    )
    return model, tokenizer


def _pairs_to_hf_dataset(pairs: Sequence[InstructionPair], run: SFTRunConfig, tokenizer: Any):
    from datasets import Dataset

    texts = [
        format_pair_text(p, system_prompt=run.system_prompt, tokenizer=tokenizer)
        for p in pairs
    ]
    return Dataset.from_dict({"text": texts})


def _make_trainer(model: Any, tokenizer: Any, run: SFTRunConfig, train_ds: Any, val_ds: Any | None):
    from trl import SFTTrainer
    from transformers import TrainingArguments

    out = Path(run.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    args_kwargs: dict[str, Any] = {
        "output_dir": str(out),
        "per_device_train_batch_size": run.per_device_train_batch_size,
        "gradient_accumulation_steps": run.gradient_accumulation_steps,
        "warmup_steps": run.warmup_steps,
        "logging_steps": run.logging_steps,
        "save_steps": run.save_steps,
        "learning_rate": run.learning_rate,
        "num_train_epochs": run.num_train_epochs,
        "optim": run.optim,
        "seed": run.seed,
        "fp16": True,
        "bf16": False,
        "report_to": "none",
        "save_total_limit": 2,
    }
    if run.max_steps is not None:
        args_kwargs["max_steps"] = run.max_steps
    if val_ds is not None and len(val_ds) > 0:
        args_kwargs["eval_strategy"] = "steps"
        args_kwargs["eval_steps"] = max(run.save_steps, run.logging_steps)
    try:
        args = TrainingArguments(**args_kwargs)
    except TypeError:
        # Older transformers used evaluation_strategy.
        args_kwargs.pop("eval_strategy", None)
        if val_ds is not None and len(val_ds) > 0:
            args_kwargs["evaluation_strategy"] = "steps"
            args_kwargs["eval_steps"] = max(run.save_steps, run.logging_steps)
        args = TrainingArguments(**args_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": train_ds,
        "args": args,
        "dataset_text_field": "text",
        "max_seq_length": run.max_seq_length,
        "packing": run.packing,
    }
    if val_ds is not None and len(val_ds) > 0:
        trainer_kwargs["eval_dataset"] = val_ds
    try:
        return SFTTrainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs.pop("tokenizer", None)
        trainer_kwargs["processing_class"] = tokenizer
        return SFTTrainer(**trainer_kwargs)


def run_sft(
    *,
    config_path: str | Path | None = None,
    dataset_dir: str | Path | None = None,
    adapter_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    dry_run: bool = True,
    max_steps: int | None = None,
    save_steps: int | None = None,
    require_train: bool = True,
) -> SFTPlan:
    """Load config + splits and either write a plan or run QLoRA SFT."""
    app = load_config(config_path)
    run = SFTRunConfig.from_app(
        app,
        dataset_dir=dataset_dir,
        adapter_dir=adapter_dir,
        output_dir=output_dir,
        max_steps=max_steps,
        save_steps=save_steps,
    )
    splits = load_sft_splits(run.dataset_dir, require_train=require_train and not dry_run)
    if dry_run:
        plan = build_plan(run, splits, dry_run=True)
        path = _write_plan(plan, run.output_dir)
        logger.info("Wrote dry-run plan to %s", path)
        return plan

    if not splits.get("train"):
        raise FileNotFoundError(f"Empty train split in {run.dataset_dir}")

    model, tokenizer = _load_unsloth_model(run)
    train_ds = _pairs_to_hf_dataset(splits["train"], run, tokenizer)
    val_ds = (
        _pairs_to_hf_dataset(splits["val"], run, tokenizer) if splits.get("val") else None
    )
    trainer = _make_trainer(model, tokenizer, run, train_ds, val_ds)
    trainer.train()
    adapter = Path(run.adapter_dir)
    adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    plan = build_plan(run, splits, dry_run=False, tokenizer=tokenizer)
    plan.notes.append(f"Saved LoRA adapter to {adapter}")
    _write_plan(plan, run.output_dir)
    return plan
