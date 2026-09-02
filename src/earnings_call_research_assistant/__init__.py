"""Earnings Call Research Assistant — data, training, and eval package."""

from earnings_call_research_assistant.inference import (
    DEFAULT_SYSTEM_PROMPT,
    InferenceConfig,
    InferenceHarness,
    load_harness,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "InferenceConfig",
    "InferenceHarness",
    "load_harness",
    "__version__",
]
