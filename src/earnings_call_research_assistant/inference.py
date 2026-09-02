"""Reusable base / adapter inference harness (Unsloth + chat template)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

DEFAULT_SYSTEM_PROMPT = (
    "You are a financial research assistant. Answer clearly and conservatively. "
    "If the question cannot be answered from the provided context, say so. "
    "Do not invent numbers."
)

DEFAULT_MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"


@dataclass
class InferenceConfig:
    model_name: str = DEFAULT_MODEL_NAME
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    extra_from_pretrained: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "InferenceConfig":
        """Build from a YAML/JSON mapping (``model`` block + optional ``inference``)."""
        mapping = mapping or {}
        model = mapping.get("model", mapping)
        infer = mapping.get("inference", {})
        return cls(
            model_name=model.get("name", DEFAULT_MODEL_NAME),
            max_seq_length=int(model.get("max_seq_length", 2048)),
            load_in_4bit=bool(model.get("load_in_4bit", True)),
            max_new_tokens=int(infer.get("max_new_tokens", 256)),
            do_sample=bool(infer.get("do_sample", False)),
            temperature=float(infer.get("temperature", 0.0)),
            system_prompt=str(infer.get("system_prompt", DEFAULT_SYSTEM_PROMPT)),
        )


class InferenceHarness:
    """Load a model once and run chat-templated greedy (or sampled) generation."""

    def __init__(self, model: Any, tokenizer: Any, config: InferenceConfig | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or InferenceConfig()

    @classmethod
    def from_pretrained(
        cls,
        config: InferenceConfig | None = None,
        *,
        model_name: str | None = None,
        max_seq_length: int | None = None,
        load_in_4bit: bool | None = None,
    ) -> "InferenceHarness":
        from unsloth import FastLanguageModel

        cfg = config or InferenceConfig()
        if model_name is not None:
            cfg.model_name = model_name
        if max_seq_length is not None:
            cfg.max_seq_length = max_seq_length
        if load_in_4bit is not None:
            cfg.load_in_4bit = load_in_4bit

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model_name,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit,
            **cfg.extra_from_pretrained,
        )
        FastLanguageModel.for_inference(model)
        return cls(model, tokenizer, cfg)

    def build_messages(
        self,
        user_text: str,
        *,
        system_prompt: str | None = None,
        history: Sequence[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt or self.config.system_prompt},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_text})
        return messages

    def encode(self, messages: list[dict[str, str]]):
        device = getattr(self.model, "device", None)
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if device is not None:
            encoded = encoded.to(device)
        return encoded

    def generate(
        self,
        user_text: str,
        *,
        system_prompt: str | None = None,
        history: Sequence[dict[str, str]] | None = None,
        max_new_tokens: int | None = None,
        do_sample: bool | None = None,
        temperature: float | None = None,
    ) -> str:
        import torch

        messages = self.build_messages(
            user_text, system_prompt=system_prompt, history=history
        )
        inputs = self.encode(messages)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "do_sample": self.config.do_sample if do_sample is None else do_sample,
        }
        temp = self.config.temperature if temperature is None else temperature
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = temp

        with torch.inference_mode():
            out = self.model.generate(inputs, **gen_kwargs)
        new_tokens = out[0, inputs.shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_many(self, prompts: Sequence[str], **kwargs) -> list[str]:
        return [self.generate(p, **kwargs) for p in prompts]


def load_harness(
    config: InferenceConfig | dict[str, Any] | None = None,
    **overrides: Any,
) -> InferenceHarness:
    """Convenience loader used by notebooks and evals."""
    if isinstance(config, dict):
        cfg = InferenceConfig.from_mapping(config)
    else:
        cfg = config or InferenceConfig()
    return InferenceHarness.from_pretrained(cfg, **overrides)
