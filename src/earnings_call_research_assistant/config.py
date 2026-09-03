"""Typed project settings loaded from YAML or JSON configs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CONFIG_NAME = "default.yaml"
DEFAULT_MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
DEFAULT_SYSTEM_PROMPT = (
    "You are a financial research assistant. Answer clearly and conservatively. "
    "If the question cannot be answered from the provided context, say so. "
    "Do not invent numbers."
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / DEFAULT_CONFIG_NAME
DEFAULT_DATASET_DIR = "data/processed/ecra-sft-v0.1.0"
DEFAULT_ADAPTER_DIR = "outputs/adapters/llama32-3b-ecra-sft"


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Expected mapping, got {type(value).__name__}")


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


@dataclass(frozen=True)
class ModelSettings:
    name: str = DEFAULT_MODEL_NAME
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "ModelSettings":
        data = _as_dict(mapping)
        return cls(
            name=str(data.get("name", DEFAULT_MODEL_NAME)),
            max_seq_length=int(data.get("max_seq_length", 2048)),
            load_in_4bit=bool(data.get("load_in_4bit", True)),
        )


@dataclass(frozen=True)
class LoRASettings:
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "LoRASettings":
        data = _as_dict(mapping)
        modules = data.get("target_modules")
        if modules is None:
            target = cls.target_modules
        else:
            target = tuple(str(m) for m in modules)
        return cls(
            r=int(data.get("r", 16)),
            lora_alpha=int(data.get("lora_alpha", 16)),
            lora_dropout=float(data.get("lora_dropout", 0.0)),
            target_modules=target,
        )


@dataclass(frozen=True)
class TrainingSettings:
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2.0e-4
    num_train_epochs: float = 1.0
    warmup_steps: int = 10
    logging_steps: int = 10
    save_steps: int = 50
    max_steps: int | None = None
    max_grad_norm: float = 1.0
    optim: str = "adamw_8bit"
    seed: int = 3407
    output_dir: str = "outputs"
    adapter_dir: str = DEFAULT_ADAPTER_DIR
    dataset_dir: str = DEFAULT_DATASET_DIR
    packing: bool = False

    @property
    def effective_batch_size(self) -> int:
        """Micro-batch × grad-accum (single process / single GPU)."""
        return int(self.per_device_train_batch_size) * int(self.gradient_accumulation_steps)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "TrainingSettings":
        data = _as_dict(mapping)
        return cls(
            per_device_train_batch_size=int(data.get("per_device_train_batch_size", 2)),
            gradient_accumulation_steps=int(data.get("gradient_accumulation_steps", 8)),
            learning_rate=float(data.get("learning_rate", 2.0e-4)),
            num_train_epochs=float(data.get("num_train_epochs", 1)),
            warmup_steps=int(data.get("warmup_steps", 10)),
            logging_steps=int(data.get("logging_steps", 10)),
            save_steps=int(data.get("save_steps", 50)),
            max_steps=_optional_int(data.get("max_steps")),
            max_grad_norm=float(data.get("max_grad_norm", 1.0)),
            optim=str(data.get("optim", "adamw_8bit")),
            seed=int(data.get("seed", 3407)),
            output_dir=str(data.get("output_dir", "outputs")),
            adapter_dir=str(data.get("adapter_dir", DEFAULT_ADAPTER_DIR)),
            dataset_dir=str(data.get("dataset_dir", DEFAULT_DATASET_DIR)),
            packing=bool(data.get("packing", False)),
        )


@dataclass(frozen=True)
class InferenceSettings:
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "InferenceSettings":
        data = _as_dict(mapping)
        return cls(
            max_new_tokens=int(data.get("max_new_tokens", 256)),
            do_sample=bool(data.get("do_sample", False)),
            temperature=float(data.get("temperature", 0.0)),
            system_prompt=str(data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)),
        )


@dataclass(frozen=True)
class AppConfig:
    model: ModelSettings = field(default_factory=ModelSettings)
    lora: LoRASettings = field(default_factory=LoRASettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    inference: InferenceSettings = field(default_factory=InferenceSettings)
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "AppConfig":
        data = _as_dict(mapping)
        known = {"model", "lora", "training", "inference"}
        extras = {k: v for k, v in data.items() if k not in known}
        return cls(
            model=ModelSettings.from_mapping(data.get("model")),
            lora=LoRASettings.from_mapping(data.get("lora")),
            training=TrainingSettings.from_mapping(data.get("training")),
            inference=InferenceSettings.from_mapping(data.get("inference")),
            extras=extras,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "model": asdict(self.model),
            "lora": {
                **asdict(self.lora),
                "target_modules": list(self.lora.target_modules),
            },
            "training": asdict(self.training),
            "inference": asdict(self.inference),
        }
        payload.update(self.extras)
        return payload

    def replace(self, **changes: Any) -> "AppConfig":
        return replace(self, **changes)


def _parse_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        import yaml

        loaded = yaml.safe_load(text) or {}
    elif suffix == ".json":
        loaded = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix} ({path})")
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Config root must be a mapping, got {type(loaded).__name__}")
    return dict(loaded)


def resolve_config_path(path: str | Path | None = None) -> Path:
    if path is None:
        return DEFAULT_CONFIG_PATH
    candidate = Path(path)
    if candidate.exists():
        return candidate
    bundled = _REPO_ROOT / "configs" / candidate.name
    if bundled.exists():
        return bundled
    raise FileNotFoundError(f"Config not found: {path}")


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load typed settings from YAML or JSON (defaults to configs/default.yaml)."""
    resolved = resolve_config_path(path)
    return AppConfig.from_mapping(_parse_file(resolved))


def load_config_mapping(path: str | Path | None = None) -> dict[str, Any]:
    return load_config(path).to_dict()
