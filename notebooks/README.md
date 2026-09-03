# Notebooks

Kaggle-first workflows. Prefer short, restartable cells and public data only.

- `00_baseline_inference.ipynb` — base-model smoke test (Unsloth 4-bit, Llama-3.2-3B-Instruct)
- `01_data_and_sft.ipynb` — Phase 1 data pipeline (ingest → chunk → generate → filter → select) + Phase 2 QLoRA dry-run / optional `--run` train
- `02_train_qlora.ipynb` — (optional alias; use `01_data_and_sft.ipynb`)
- `03_eval_compare.ipynb` — base vs adapter (planned)

## Quick start on Kaggle

1. New notebook, **GPU (T4)** enabled.
2. First cell:

```bash
%cd /kaggle/working
!git clone --depth 1 https://github.com/nuwanda94/earnings-call-research-assistant.git
%cd earnings-call-research-assistant
```

3. Open `notebooks/00_baseline_inference.ipynb` for base smoke **or** `notebooks/01_data_and_sft.ipynb` for data + train.

In `01_data_and_sft.ipynb`:

- Leave `RUN_TRAIN = False` for a CPU-safe full data pass + `outputs/sft_plan.json`.
- Set `RUN_TRAIN = True` and `MAX_STEPS = 20` for a short QLoRA smoke on T4.
- Copy `outputs/adapters/llama32-3b-ecra-sft/` off the session before it expires.

Package imports must use `earnings_call_research_assistant` after `src/` is on `sys.path` (not `from src....`).
