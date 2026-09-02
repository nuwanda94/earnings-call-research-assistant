# Source package

Installable package root (`pip install -e .`). Layout:

```
earnings_call_research_assistant/
  data/       # Phase 1 loaders and filters
  training/   # Phase 2 SFT / QLoRA
  eval/       # Phase 3 metrics and panels
```

Keep modules small and importable from Kaggle notebooks.
