# Earnings Call Research Assistant

Domain-adapted LLM for financial research Q&A and summarization from public earnings call transcripts.

**Goal**: Fine-tune a small open model (Llama-3.2-3B / optional Llama-3.1-8B) with QLoRA on Kaggle so it outperforms the base model on research-style financial questions — a portfolio piece for research-tooling roles (Morningstar / PitchBook style).

## Status

Progress: [`PROGRESS.md`](PROGRESS.md). **Phases 0–5 complete** (full-scale T4 SFT + HF publish + GitHub metadata on 2026-09-14). Retrieval R@k on the scaled index needs **query realignment** before citing IR metrics on a resume.

| Phase | Name | Status |
|-------|------|--------|
| 0 | Foundation | Done |
| 1 | Robust Data Foundation | Done |
| 2 | Training Pipeline | Done |
| 3 | Evaluation & Iteration | Done |
| 4 | Packaging & Portfolio Polish | Done |
| 5 | Hybrid RAG + measurable metrics | Done (scaled N; see Results caveat on R@k) |

Docs: [`PROGRESS.md`](PROGRESS.md) · [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) · [`docs/DATA_CARD.md`](docs/DATA_CARD.md) · [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)

## Artifacts

| Artifact | Link |
|----------|------|
| **QLoRA adapter (Hugging Face)** | [nuwanda94/llama32-3b-ecra-sft](https://huggingface.co/nuwanda94/llama32-3b-ecra-sft) |
| Full eval report | [`evals/reports/RAG_EVAL_REPORT.md`](evals/reports/RAG_EVAL_REPORT.md) |
| SFT plan (train/val, seed, batch) | [`outputs/sft_plan.json`](outputs/sft_plan.json) |
| Retrieval metrics JSON | [`evals/reports/rag_metrics.json`](evals/reports/rag_metrics.json) |
| Generation metrics JSON | [`evals/reports/rag_generation_metrics.json`](evals/reports/rag_generation_metrics.json) |
| HF model card template | [`docs/MODEL_CARD_RAG.md`](docs/MODEL_CARD_RAG.md) |
| Scale notebook (Kaggle T4) | [`notebooks/05_full_scale_sft_t4.ipynb`](notebooks/05_full_scale_sft_t4.ipynb) |

Load the published adapter:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "unsloth/Llama-3.2-3B-Instruct"
adapter = "nuwanda94/llama32-3b-ecra-sft"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
```

## Results

Numbers from committed JSON / SFT plan after the **full-scale** Kaggle run (2026-09-14):

| Claim | Source | Value |
|-------|--------|-------|
| SFT train / val | `outputs/sft_plan.json` | **2127 / 240** (seed **3407**, effective batch **16**) |
| Selected pairs | dataset `manifest.json` | **2627** |
| N (chunk count) | corpus snapshot `n_chunks` | **19990** |
| Sources | `sources` | transcripts 18474 · fiqa 860 · finance_alpaca 656 |
| Recall@k (hybrid) | `backends.hybrid.mean_recall` | **0.0** at k=1,3,5,10 (generic template queries; see report) |
| nDCG@k (hybrid) | `backends.hybrid.mean_ndcg` | **0.0** at k=1,3,5,10 |
| Grounded answer accuracy | `aggregate.*.grounded_answer_accuracy` | **base 0.58 / adapter 1.0** (50 queries, `dry_run=false`) |
| Citation-hit rate | `aggregate.*.citation_hit_rate` | **base 0.593 / adapter 0.876** |
| Dense backend | `rag_metrics.json` | `sentence-transformers` / `all-MiniLM-L6-v2` |
| Hub adapter | publish log | https://huggingface.co/nuwanda94/llama32-3b-ecra-sft |

**Caveats:** Do not cite R@k as product quality until `scripts/realign_rag_eval.py` + re-eval. Generator lift is the stronger current claim. Not SEC-wide coverage.

```bash
# Realign gold IDs to current corpus and re-score
python scripts/realign_rag_eval.py --max-n 50 --eval-retrieval
python scripts/realign_rag_eval.py --max-n 50 --eval-retrieval --eval-generate \
  --adapter-dir outputs/adapters/llama32-3b-ecra-sft

# Backup adapter from Hub
python scripts/backup_adapter.py --repo-id nuwanda94/llama32-3b-ecra-sft
```

Scale notebook: [`notebooks/05_full_scale_sft_t4.ipynb`](notebooks/05_full_scale_sft_t4.ipynb).

## Key principles

- Public data only
- Grounded synthetic generation + multi-stage quality filtering
- Quality over quantity (target 3k–6k high-signal examples)
- Reproducible on free Kaggle GPUs (T4 / 2×T4)
- Every GPU script is **dry-run by default**; pass `--run` only on a real GPU box

---

## Clone and run baseline (< 30 min)

```bash
git clone https://github.com/nuwanda94/earnings-call-research-assistant.git
cd earnings-call-research-assistant
# Kaggle: open notebooks/00_baseline_inference.ipynb with T4 GPU
```

See full walkthrough in git history / prior README sections for Phase 0–4 CLIs. Primary scale path: `notebooks/05_full_scale_sft_t4.ipynb`.

## Structure

```
src/earnings_call_research_assistant/
notebooks/     # 00 baseline … 05 full-scale SFT + publish
data/          # raw / processed (gitignored large files)
configs/       # default.yaml + llama32-8b.yaml + rag.yaml
evals/         # research panel + RAG eval set + reports
scripts/       # CLIs including backup_adapter, realign_rag_eval, publish_adapter
docs/          # plan, data card, reproducibility, model card
```

See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md).
