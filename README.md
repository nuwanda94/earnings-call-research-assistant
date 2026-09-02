# Earnings Call Research Assistant

Domain-adapted LLM for financial research Q&A and summarization from public earnings call transcripts.

**Goal**: Fine-tune a small open model (Llama-3.2-3B / Qwen2.5-3B) with QLoRA on Kaggle so it outperforms the base model on research-style financial questions — suitable as an interview portfolio piece for roles at firms like Morningstar / PitchBook.

## Status

Progress is tracked in [`PROGRESS.md`](PROGRESS.md). An hourly automation advances one action item per run and commits the result.

| Phase | Name | Status |
|-------|------|--------|
| 0 | Foundation | Done |
| 1 | Robust Data Foundation | Done |
| 2 | Training Pipeline | Done |
| 3 | Evaluation & Iteration | Next |
| 4 | Packaging & Portfolio Polish | Pending |

Training reproducibility (seed `3407`, adapter dirs, Kaggle dry-run vs `--run`) is in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Key Principles

- Public data only
- Grounded synthetic generation + multi-stage quality filtering
- Quality over quantity (target 3k–6k high-signal examples)
- Reproducible on free Kaggle GPUs (T4 / 2×T4)
- Clear before/after qualitative evidence for interviews

## Clone and run baseline (< 30 min)

This path loads `unsloth/Llama-3.2-3B-Instruct` in 4-bit and runs three research-style smoke prompts. It does **not** train.

You need:

- Python 3.10+
- A CUDA GPU with ~8 GB+ VRAM (Kaggle T4 is the intended target)
- Hugging Face access to the base model (public Unsloth snapshot)

### Option A — Kaggle (recommended)

1. Create a new Kaggle notebook with **GPU (T4)** enabled.
2. Add this repo as a dataset *or* clone it in the first cell:

```bash
!git clone https://github.com/nuwanda94/earnings-call-research-assistant.git
%cd earnings-call-research-assistant
```

3. Open [`notebooks/00_baseline_inference.ipynb`](notebooks/00_baseline_inference.ipynb) (or copy its cells into the Kaggle notebook).
4. Run all cells top-to-bottom.

What the notebook does:

- Detects Kaggle and `pip install`s Unsloth / Transformers / bitsandbytes / PyYAML.
- Puts `../src` on `sys.path` so `earnings_call_research_assistant` imports without a full install.
- Loads [`configs/default.yaml`](configs/default.yaml) into `InferenceConfig` when the file is present.
- Instantiates `InferenceHarness.from_pretrained(...)` and generates answers for three earnings-research prompts.

If all three prompts return coherent text, the Phase 0 harness works. Keep the notebook output as the qualitative *base* snapshot for later base-vs-adapter comparison.

### Option B — local GPU

```bash
git clone https://github.com/nuwanda94/earnings-call-research-assistant.git
cd earnings-call-research-assistant
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Optional extras:

```bash
pip install -e ".[eval,demo]"   # metrics + Gradio later
pip install -e ".[all]"         # eval + demo + dev
```

Then either open the same notebook:

```bash
jupyter notebook notebooks/00_baseline_inference.ipynb
```

or run a one-shot smoke generate from Python:

```python
from earnings_call_research_assistant import load_config
from earnings_call_research_assistant.inference import InferenceConfig, InferenceHarness

app = load_config()                       # configs/default.yaml
cfg = InferenceConfig.from_mapping(app.to_dict())
harness = InferenceHarness.from_pretrained(cfg)
print(harness.generate("What is the difference between prepared remarks and Q&A on an earnings call?"))
```

`load_config("configs/default.yaml")` also accepts a path (YAML or JSON). Missing keys fall back to the typed defaults in `src/earnings_call_research_assistant/config.py`.

### Config that the baseline uses

[`configs/default.yaml`](configs/default.yaml) is the single source of truth for Phase 0/2:

- `model` — name, max sequence length, 4-bit load
- `lora` — QLoRA ranks / targets (unused until Phase 2)
- `training` — SFT hyperparameters (unused until Phase 2)
- `inference` — `max_new_tokens`, sampling, conservative system prompt

The notebook maps that YAML through `InferenceConfig.from_mapping(...)`. The package-level `load_config()` returns a typed `AppConfig` with the same blocks.

## Train on Kaggle (Phase 2)

Do **not** start a full epoch from CI. Dry-run first, then `--run` on a T4:

```bash
python scripts/train_sft.py
python scripts/train_sft.py --run --max-steps 20
```

- Seed: `3407`
- 3B adapter: `outputs/adapters/llama32-3b-ecra-sft`
- 8B adapter: `outputs/adapters/llama31-8b-ecra-sft` (`--config configs/llama32-8b.yaml`)

Details: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Structure

```
src/earnings_call_research_assistant/   # config, inference, data, train, eval
notebooks/     # Kaggle notebooks; start with 00_baseline_inference.ipynb
data/          # raw / processed (gitignored large files)
configs/       # default.yaml — model, LoRA, training, inference
evals/         # evaluation scripts & reports
scripts/       # utilities
docs/          # plan, data card, reproducibility, design notes
```

See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the full plan and acceptance criteria.
