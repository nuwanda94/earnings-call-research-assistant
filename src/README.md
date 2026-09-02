# Source package

Installable package root (`pip install -e .`). Layout:

```
earnings_call_research_assistant/
  config.py     # YAML/JSON loader → typed AppConfig
  inference.py  # base / adapter chat-template generate harness
  data/         # Phase 1 loaders and filters
  training/     # Phase 2 SFT / QLoRA
  eval/         # Phase 3 metrics and panels
```

Keep modules small and importable from Kaggle notebooks.

```python
from earnings_call_research_assistant import InferenceHarness, load_config

cfg = load_config()  # configs/default.yaml
harness = InferenceHarness.from_pretrained(
    model_name=cfg.model.name,
    max_seq_length=cfg.model.max_seq_length,
    load_in_4bit=cfg.model.load_in_4bit,
)
print(harness.generate("What is a beat vs miss on an earnings call?"))
```
