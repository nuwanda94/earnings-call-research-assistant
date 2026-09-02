"""Earnings Call Research Assistant — data, training, and eval package."""

from earnings_call_research_assistant.config import (
    AppConfig,
    InferenceSettings,
    LoRASettings,
    ModelSettings,
    TrainingSettings,
    load_config,
)
from earnings_call_research_assistant.inference import (
    DEFAULT_SYSTEM_PROMPT,
    InferenceConfig,
    InferenceHarness,
    load_harness,
)

__version__ = "0.1.0"

__all__ = [
    "AppConfig",
    "DEFAULT_SYSTEM_PROMPT",
    "InferenceConfig",
    "InferenceHarness",
    "InferenceSettings",
    "LoRASettings",
    "ModelSettings",
    "TrainingSettings",
    "load_config",
    "load_harness",
    "__version__",
]
