# Source package

Installable package root (`pip install -e .`). Layout:

```
earnings_call_research_assistant/
  inference.py  # base / adapter chat-template generate harness
  data/         # Phase 1 loaders and filters
  training/     # Phase 2 SFT / QLoRA
  eval/         # Phase 3 metrics and panels
```

Keep modules small and importable from Kaggle notebooks.

```python
from earnings_call_research_assistant import InferenceHarness, InferenceConfig

harness = InferenceHarness.from_pretrained(InferenceConfig())
print(harness.generate("What is a beat vs miss on an earnings call?"))
```
