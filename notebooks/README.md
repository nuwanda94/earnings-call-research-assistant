# Notebooks

Kaggle-first workflows. Prefer short, restartable cells and public data only.

- `00_baseline_inference.ipynb` — base-model smoke test (Unsloth 4-bit, Llama-3.2-3B-Instruct)
- `01_data_pipeline.ipynb` — Phase 1 generation / filter (planned)
- `02_train_qlora.ipynb` — Unsloth SFT on T4 / 2×T4 (planned)
- `03_eval_compare.ipynb` — base vs adapter (planned)

Copy `00_baseline_inference.ipynb` into a Kaggle notebook, enable a T4 GPU, and run top-to-bottom. No training.
