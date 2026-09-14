# Progress Tracker

Last updated: 2026-09-14 14:25 IST

## Current Phase
Phase 5 complete (full-scale SFT + HF publish + GitHub metadata). Optional follow-up: realign RAG eval queries for non-zero Recall@k.

## Next Action Item
**Optional (human, corpus machine):** `python scripts/realign_rag_eval.py --max-n 50 --eval-retrieval --eval-generate --adapter-dir outputs/adapters/llama32-3b-ecra-sft` then refresh report/Hub card from new JSON only. **Backup:** `python scripts/backup_adapter.py --repo-id nuwanda94/llama32-3b-ecra-sft`. Automation must not invent R@k or start long GPU jobs.

## Completed Items

- [x] Create GitHub repository `earnings-call-research-assistant`
- [x] Initial README and progress tracker
- [x] Replace requirements.txt with pyproject.toml for project dependencies
- [x] Directory structure: `src/`, `notebooks/`, `data/`, `configs/`, `evals/`, `scripts/`, `docs/`
- [x] Kaggle notebook template `notebooks/00_baseline_inference.ipynb`
- [x] Reusable base-model inference harness (`src/earnings_call_research_assistant/inference.py`)
- [x] Typed config loader (`src/earnings_call_research_assistant/config.py`) reading YAML/JSON
- [x] Root README with <30 min clone-and-run baseline (Kaggle + local, config + notebook)
- [x] Public-source ingestion stubs (transcripts, FiQA, Finance-Alpaca)
- [x] Chunking + proposition extraction
- [x] Grounded synthetic Q&A / summary generation
- [x] Multi-stage filtering (heuristic → dedup → LLM-as-judge stub)
- [x] Diversity selection + versioned splits + data card
- [x] Unsloth QLoRA SFT training script (3B) + externalized config / logging / checkpoints
- [x] Optional 8B config path (`configs/llama32-8b.yaml` + `--model-name` override)
- [x] Reproducibility notes (seed 3407, adapter output dirs, Kaggle dry-run vs `--run`)
- [x] Qualitative research panel (20 grounded prompts) + base vs adapter eval stub
- [x] Quantitative metrics stub (token-overlap / citation-hit rates on panel JSON)
- [x] Evaluation report + first iteration note (refusal-and-citation data mix)
- [x] CPU-safe Gradio demo stub (`demo.py` + `scripts/demo_gradio.py`)
- [x] Hugging Face Hub adapter publish script (env token only; dry-run default)
- [x] Polished README + clone-to-demo walkthrough
- [x] Portfolio demo video script + recording notes (`docs/DEMO_VIDEO.md`)
- [x] Docs: Phase 5 hybrid RAG plan
- [x] Phase 5.1–5.8 RAG stack + reports + notebooks
- [x] Notebook `05_full_scale_sft_t4.ipynb`
- [x] Full-scale SFT: train **2127** / val **240** / selected **2627** (seed 3407)
- [x] Scaled RAG corpus N=**19990**; metrics JSON + generate eval (`dry_run=false`)
- [x] HF adapter publish `nuwanda94/llama32-3b-ecra-sft` (`uploaded=true`)
- [x] GitHub metadata push (sft_plan, manifests, reports)
- [x] Phase 5.9 closed in docs (honest R@k=0 caveat + realign script)
- [x] `scripts/backup_adapter.py` + `scripts/realign_rag_eval.py`

## Phase 0–4 Checklists

All items complete (see git history / prior PROGRESS revisions).

## Phase 5 Checklist — Hybrid RAG + metrics

- [x] 5.1 Corpus builder + `manifest.json` with measured **N**
- [x] 5.2 BM25 index + retrieve CLI
- [x] 5.3 Dense embeddings + hybrid (RRF) fusion
- [x] 5.4 Fixed eval set with gold chunk IDs
- [x] 5.5 `eval_retrieval.py` → Recall@k + nDCG@k
- [x] 5.6 `eval_rag_generate.py` → grounded answer accuracy (base vs adapter)
- [x] 5.7 Kaggle notebook `03_rag_eval_and_publish.ipynb`
- [x] 5.8 `RAG_EVAL_REPORT.md` + README results + HF model card template
- [x] 5.9 Human Kaggle full run + publish adapter/metrics to HF + GitHub (scaled N=19990; train 2127; Hub uploaded; R@k realign optional)

## Notes for Automation

1. Phase 5.9 is **closed**.
2. Optional next work is human realign of RAG eval queries on a machine that has the corpus.
3. Do not invent R@k; do not start long GPU jobs in automation.

## Log

- 2026-09-14 14:25 IST — docs: closed Phase 5.9 with scaled metrics (N=19990, train 2127, HF uploaded). Honest R@k=0 documented; added `scripts/backup_adapter.py` + `scripts/realign_rag_eval.py`. Next = optional realign re-eval.
- 2026-09-14 14:00 IST — chore: Phase 5.9 waiting log (superseded by full-scale run + close-out).
- 2026-09-14 12:35 IST — feat: `notebooks/05_full_scale_sft_t4.ipynb`.
- Earlier Phase 0–5.8 log entries retained in git history.
