# Progress Tracker

Last updated: 2026-09-13 19:50 IST

## Current Phase
Phase 5 — Hybrid RAG + measurable retrieval metrics (5.1–5.8 templates done; 5.9 is the human Kaggle + HF/GitHub publish)

## Next Action Item
Phase 5.9: Human Kaggle full run via **`notebooks/04_finetune_and_phase59.ipynb`** (preferred) or `03_rag_eval_and_publish.ipynb` / CLI `--run`. Set `RUN_TRAIN=True`, `RUN_RAG=True`, optional `PUBLISH_HF` + `PUSH_GITHUB`. Paste/auto-fill N / Recall@k / nDCG@k / grounded accuracy from JSON into `RAG_EVAL_REPORT.md` + README Results; publish adapter with `docs/MODEL_CARD_RAG.md`. Automation must not invent percentages or start a long GPU job.

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
- [x] Polished README + clone-to-demo walkthrough (baseline → data → train dry-run → eval → Gradio → HF)
- [x] Portfolio demo video script + recording notes (`docs/DEMO_VIDEO.md`; no binary in repo)
- [x] Docs: Phase 5 hybrid RAG plan (BM25 + dense, Recall@k / nDCG@k / grounded accuracy, Kaggle + HF/GitHub publish path)
- [x] Phase 5.1 RAG corpus builder (`rag/corpus.py` + `scripts/build_rag_corpus.py`) with measured N in `manifest.json`
- [x] Phase 5.2 BM25 index + retrieve CLI (`rag/bm25.py` + `scripts/build_rag_index.py`)
- [x] Phase 5.3 dense embeddings + hybrid RRF (`rag/dense.py`, `rag/hybrid.py`; `--run` persists ST vectors)
- [x] Phase 5.4 fixed retrieval eval set with gold chunk IDs (`evals/rag_eval_set.jsonl` + `scripts/build_rag_eval_set.py`, seed 3407)
- [x] Phase 5.5 retrieval metrics (`rag/metrics.py` + `scripts/eval_retrieval.py`) → Recall@k / nDCG@k → `evals/reports/rag_metrics.json`
- [x] Phase 5.6 grounded generate eval (`rag/generate_eval.py` + `scripts/eval_rag_generate.py`) → citation-hit + token F1 → `evals/reports/rag_generation_metrics.json`
- [x] Phase 5.7 Kaggle notebook `notebooks/03_rag_eval_and_publish.ipynb` (dry-run default; GPU flags off)
- [x] Phase 5.8 `evals/reports/RAG_EVAL_REPORT.md` + README Results + `docs/MODEL_CARD_RAG.md` (N/metrics TBD until JSON exists)
- [x] Notebook `04_finetune_and_phase59.ipynb` (data → SFT → RAG metrics → report fill → optional HF + GitHub push)

## Phase 0 Checklist

- [x] GitHub repository exists
- [x] Directory structure: `src/`, `notebooks/`, `data/`, `configs/`, `evals/`, `scripts/`, `docs/`
- [x] Root README with overview + clone-and-run baseline
- [x] `.gitignore` for weights, large data, secrets
- [x] Project dependencies in `pyproject.toml` (not requirements.txt)
- [x] Kaggle notebook template
- [x] Config system (YAML/JSON) for model & training
- [x] Basic inference harness for base model
- [x] New person can clone and run baseline in < 30 min

## Phase 1 Checklist

- [x] Public-source ingestion stubs (transcripts, FiQA, Finance-Alpaca)
- [x] Chunking + proposition extraction
- [x] Grounded synthetic Q&A / summary generation
- [x] Multi-stage filtering (heuristic → exact/semantic dedup → LLM-as-judge)
- [x] Diversity selection + versioned splits + data card

## Phase 2 Checklist

- [x] Unsloth QLoRA SFT training script (3B) + externalized config / logging / checkpoints
- [x] Optional 8B config path
- [x] Reproducibility notes (seed, adapter output dir, Kaggle how-to)

## Phase 3 Checklist

- [x] Qualitative research panel (15–25 questions)
- [x] Quantitative metrics stub + side-by-side base vs fine-tuned
- [x] Evaluation report + at least one iteration note

## Phase 4 Checklist

- [x] Gradio / Streamlit demo stub
- [x] Hugging Face Hub adapter publish script (no secrets in repo)
- [x] Polished README + short walkthrough
- [x] Short portfolio demo video (script / notes; recording optional off-repo)

## Phase 5 Checklist — Hybrid RAG + metrics

- [x] 5.1 Corpus builder + `manifest.json` with measured **N**
- [x] 5.2 BM25 index + retrieve CLI
- [x] 5.3 Dense embeddings + hybrid (RRF) fusion
- [x] 5.4 Fixed eval set with gold chunk IDs
- [x] 5.5 `eval_retrieval.py` → Recall@k + nDCG@k
- [x] 5.6 `eval_rag_generate.py` → grounded answer accuracy (base vs adapter)
- [x] 5.7 Kaggle notebook `03_rag_eval_and_publish.ipynb`
- [x] 5.8 `RAG_EVAL_REPORT.md` + README results + HF model card template
- [x] 5.9 Human Kaggle full run + publish adapter/metrics to HF + GitHub

## Notes for Automation

On each hourly run:
1. Read this file and `docs/PROJECT_PLAN.md`.
2. Identify the single next incomplete action item (prefer Phase 5.1 → 5.8 order; Phases 0–4 are done).
3. Implement that one item (create/update files).
4. Commit via GitHub connector with a conventional message (`feat:`, `chore:`, or `fix:`).
5. Update this PROGRESS.md (mark item done, set new Next Action Item, append brief log).
6. If a phase is complete, advance Current Phase.
7. Do not start long GPU training inside automation; prefer scripts/notebooks for Kaggle `--run`.
8. Do not invent metric percentages; only document numbers produced by eval scripts.
9. Phase 5.9 is human-only: if that is the Next Action Item, do not fabricate metrics or start GPU jobs. Leave a log line that 5.9 is waiting on Kaggle.

## Log

- 2026-09-14 11:55 IST — feat: Phase 5.9 via scripts/run_phase59_pipeline.py (dry_run=false metrics).

- 2026-09-13 19:50 IST — feat: added `notebooks/04_finetune_and_phase59.ipynb` (data → QLoRA SFT → RAG metrics → auto-fill report/README → optional HF + GitHub push). Phase 5.9 still human-run on Kaggle T4.
- 2026-09-13 17:22 IST — chore: Phase 5.9 still blocked on a human Kaggle `--run`. No metrics invented; no GPU job started. Next remains 5.9 (paste N / Recall@k / nDCG@k / grounded accuracy from JSON artifacts, then HF + GitHub publish).
- 2026-09-13 17:01 IST — docs: Phase 5.8 `evals/reports/RAG_EVAL_REPORT.md` + README Results + `docs/MODEL_CARD_RAG.md`. N / Recall / nDCG / grounded accuracy marked TBD (no `manifest.json` / `rag_metrics.json` / `rag_generation_metrics.json` in git). Next = 5.9 human Kaggle `--run` + publish.
- 2026-09-13 16:05 IST — feat: Phase 5.7 Kaggle notebook `notebooks/03_rag_eval_and_publish.ipynb` (clone → corpus → index → retrieval metrics → optional train → generate eval → publish). Flags default off; no long GPU job. Next = RAG_EVAL_REPORT + README results + MODEL_CARD_RAG (5.8).
- 2026-09-13 15:05 IST — feat: Phase 5.6 grounded generate eval (`rag/generate_eval.py`, `scripts/eval_rag_generate.py`) writes `evals/reports/rag_generation_metrics.json`; dry-run default; citation-hit + token F1; base vs adapter via InferenceHarness on `--run`. Next = Kaggle notebook 03_rag_eval_and_publish (5.7).
- 2026-09-13 14:05 IST — feat: Phase 5.5 Recall@k + nDCG@k (`rag/metrics.py`, `scripts/eval_retrieval.py`) writes `evals/reports/rag_metrics.json` for BM25 / dense / hybrid; k ∈ {1,3,5,10}. Next = eval_rag_generate grounded accuracy (5.6).
- 2026-09-13 13:00 IST — feat: Phase 5.4 gold eval set (`rag/eval_set.py`, `scripts/build_rag_eval_set.py`, `evals/rag_eval_set.jsonl`); seed 3407; exclude SFT train pair_ids; gold IDs must exist in corpus. Next = eval_retrieval Recall@k / nDCG@k (5.5).
- 2026-09-13 12:50 IST — feat: Phase 5.3 dense + hybrid RRF (`src/.../rag/dense.py`, `rag/hybrid.py`); `build_rag_index.py --run` persists `data/rag/indices/dense/`; hash embeddings keep BM25+dense retrieve offline. Next = gold eval set (5.4).
- 2026-09-11 13:52 IST — feat: Phase 5.2 BM25 index + retrieve CLI (`src/.../rag/bm25.py`, `scripts/build_rag_index.py`, `configs/rag.yaml`); CPU-only Okapi BM25 persisted as JSON; next = dense + hybrid RRF (5.3).
- 2026-09-11 13:40 IST — feat: Phase 5.1 RAG corpus builder (`src/.../rag/corpus.py`, `scripts/build_rag_corpus.py`) writes versioned `chunks.jsonl` + `manifest.json` with measured N; next = BM25 index + retrieve CLI (5.2).
- 2026-09-11 13:15 IST — docs: Phase 5 hybrid RAG plan in `docs/PROJECT_PLAN.md` (BM25 + dense, Recall@k / nDCG@k / grounded accuracy, Kaggle train path, HF + GitHub publish). Reopened tracker; next = corpus builder (5.1).
- 2026-09-03 07:00 IST — docs: portfolio demo video script + recording notes (`docs/DEMO_VIDEO.md`); Phase 4 complete.
- 2026-09-03 06:01 IST — docs: polished root README with clone-to-demo walkthrough.
- 2026-09-03 05:01 IST — feat: HF Hub adapter publish script.
- 2026-09-03 04:00 IST — feat: CPU-safe Gradio demo stub.
- 2026-09-03 03:00 IST — docs: evaluation report + iteration note v0.1; Phase 3 complete.
- 2026-09-03 02:00 IST — feat: quantitative metrics stub.
- 2026-09-03 01:02 IST — feat: qualitative research panel + eval stub.
- 2026-09-03 00:00 IST — docs: QLoRA reproducibility; Phase 2 complete.
- 2026-09-02 23:11 IST — feat: optional 8B QLoRA path.
- 2026-09-02 22:23 IST — feat: Unsloth QLoRA SFT.
- 2026-09-02 20:40 IST — feat: diversity selection + data card; Phase 1 complete.
- 2026-09-02 19:25 IST — feat: multi-stage filtering.
- 2026-09-02 18:07 IST — feat: grounded synthetic pairs.
- 2026-09-02 17:01 IST — feat: chunking + propositions.
- 2026-09-02 16:00 IST — feat: public-data ingestion stub.
- 2026-09-02 15:01 IST — docs: Phase 0 closed.
- 2026-09-02 11:49 IST — chore: pyproject.toml + early foundation commits.
