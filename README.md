# Earnings Call Research Assistant

Domain-adapted LLM for financial research Q&A and summarization from public earnings call transcripts.

**Goal**: Fine-tune a small open model (Llama-3.2-3B / optional Llama-3.1-8B) with QLoRA on Kaggle so it outperforms the base model on research-style financial questions — a portfolio piece for research-tooling roles (Morningstar / PitchBook style).

## Status

Progress: [`PROGRESS.md`](PROGRESS.md). Hourly automation advanced one action item per run; **Phases 0–4 are complete**.

| Phase | Name | Status |
|-------|------|--------|
| 0 | Foundation | Done |
| 1 | Robust Data Foundation | Done |
| 2 | Training Pipeline | Done |
| 3 | Evaluation & Iteration | Done |
| 4 | Packaging & Portfolio Polish | Done |

- Reproducibility (seed `3407`, adapter dirs, dry-run vs `--run`): [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
- Data card (`ecra-sft-v0.1.0`): [`docs/DATA_CARD.md`](docs/DATA_CARD.md)
- Eval write-up: [`evals/reports/EVALUATION_REPORT.md`](evals/reports/EVALUATION_REPORT.md) · [`evals/reports/ITERATION_NOTE_v0.1.md`](evals/reports/ITERATION_NOTE_v0.1.md)
- Demo video script (record off-repo): [`docs/DEMO_VIDEO.md`](docs/DEMO_VIDEO.md)

## Key principles

- Public data only
- Grounded synthetic generation + multi-stage quality filtering
- Quality over quantity (target 3k–6k high-signal examples)
- Reproducible on free Kaggle GPUs (T4 / 2×T4)
- Clear before/after qualitative evidence for interviews
- Every GPU script is **dry-run by default**; pass `--run` only on a real GPU box

---

## Clone and run baseline (< 30 min)

This path loads `unsloth/Llama-3.2-3B-Instruct` in 4-bit and runs three research-style smoke prompts. It does **not** train.

You need:

- Python 3.10+
- A CUDA GPU with ~8 GB+ VRAM (Kaggle T4 is the intended target)
- Hugging Face access to the public Unsloth snapshot

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
pip install -e ".[eval,demo]"   # metrics + Gradio
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

### Config the baseline uses

[`configs/default.yaml`](configs/default.yaml) is the single source of truth:

- `model` — name, max sequence length, 4-bit load
- `lora` — QLoRA ranks / targets
- `training` — SFT hyperparameters
- `inference` — `max_new_tokens`, sampling, conservative system prompt

The notebook maps that YAML through `InferenceConfig.from_mapping(...)`. The package-level `load_config()` returns a typed `AppConfig` with the same blocks.

---

## Full walkthrough (CPU dry-run → Kaggle GPU)

Commands below are **CPU-safe** unless you add `--run`. After `pip install -e ".[eval,demo]"` you can walk the pipeline on a laptop in a few minutes; GPU steps stay on Kaggle.

### 1. Baseline (Phase 0)

Use the <30 min path above. Success = three coherent generations from `InferenceHarness`.

### 2. Data foundation (Phase 1)

Offline fixtures only; no large downloads unless you pass `--download`.

```bash
python scripts/ingest_public_sources.py --catalog-only
python scripts/ingest_public_sources.py --out data/raw/public_sample.jsonl
python scripts/chunk_propositions.py --out data/processed/chunks.jsonl
python scripts/generate_grounded_pairs.py --out data/processed/grounded_pairs.jsonl
python scripts/filter_grounded_pairs.py --out data/processed/filtered_pairs.jsonl
python scripts/select_dataset.py --out-dir data/processed/ecra-sft-v0.1.0
```

Writes hash-stable `train.jsonl` / `val.jsonl` / `test.jsonl` plus `manifest.json`. Schema and lineage: [`docs/DATA_CARD.md`](docs/DATA_CARD.md). CLI details: [`scripts/README.md`](scripts/README.md).

### 3. Train dry-run (Phase 2)

```bash
python scripts/train_sft.py
python scripts/train_sft.py --dataset-dir data/processed/ecra-sft-v0.1.0
```

Dry-run formats chat examples and writes `outputs/sft_plan.json`. No weights load.

On a Kaggle T4 after Unsloth is installed (smoke, not a full epoch):

```bash
python scripts/train_sft.py --run --max-steps 20
```

- Seed: `3407`
- 3B adapter: `outputs/adapters/llama32-3b-ecra-sft`
- Optional 8B: `--config configs/llama32-8b.yaml` → `outputs/adapters/llama31-8b-ecra-sft`

### 4. Eval (Phase 3)

```bash
python scripts/eval_research_panel.py
python scripts/score_research_panel.py
```

Dry-run writes placeholder base/adapter columns and near-zero metrics to `evals/reports/`.
After an adapter exists on Kaggle:

```bash
python scripts/eval_research_panel.py --run --adapter-dir outputs/adapters/llama32-3b-ecra-sft
python scripts/score_research_panel.py
```

How to read the numbers: [`evals/reports/EVALUATION_REPORT.md`](evals/reports/EVALUATION_REPORT.md).

### 5. Gradio demo (Phase 4)

```bash
pip install -e ".[demo]"
python scripts/demo_gradio.py
```

CPU stub: dropdown of four research-panel prompts (guidance, missing-FCF refusal, margin summary, unanswerable date). No weights.

On a GPU box:

```bash
python scripts/demo_gradio.py --run --adapter-dir outputs/adapters/llama32-3b-ecra-sft
```

### 6. Publish adapter to Hugging Face Hub (Phase 4)

```bash
python scripts/publish_adapter.py
```

Dry-run inspects `outputs/adapters/llama32-3b-ecra-sft`, notes whether `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` is set (value never printed), and writes `outputs/publish_plan.json`. No Hub call.

Login on the machine that will upload:

```bash
huggingface-cli login
# or: export HF_TOKEN=hf_xxx     # write-scoped; do not commit
python scripts/publish_adapter.py --repo-id nuwanda94/llama32-3b-ecra-sft --run
```

### 7. Record the portfolio clip (optional, off-repo)

Follow [`docs/DEMO_VIDEO.md`](docs/DEMO_VIDEO.md) (3–4.5 min script: baseline → grounded data → adapter demo). Do not commit the video file.

---

## Structure

```
src/earnings_call_research_assistant/   # config, inference, data, train, eval, demo, publish
notebooks/     # Kaggle notebooks; start with 00_baseline_inference.ipynb
data/          # raw / processed (gitignored large files)
configs/       # default.yaml + llama32-8b.yaml
evals/         # research panel + reports
scripts/       # thin CLIs around the package
docs/          # plan, data card, reproducibility, demo video script
```

See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the full plan.
