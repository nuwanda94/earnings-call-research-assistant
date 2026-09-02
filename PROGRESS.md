# Progress Tracker

Last updated: 2026-09-03 03:00 IST

## Current Phase
Phase 4 – Packaging & Portfolio Polish

## Next Action Item
Add a CPU-safe Gradio demo stub that loads `InferenceHarness` (base or `--adapter-dir`) for a few research-panel prompts without starting a train.

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

- [ ] Gradio / Streamlit demo stub
- [ ] Hugging Face Hub adapter publish script (no secrets in repo)
- [ ] Polished README + short walkthrough

## Notes for Automation

On each hourly run:
1. Read this file and `docs/PROJECT_PLAN.md`.
2. Identify the single next incomplete action item (prefer Phase 0 → 1 → 2 → 3 → 4 order).
3. Implement that one item (create/update files).
4. Commit via GitHub connector with a conventional message (`feat:`, `chore:`, or `fix:`).
5. Update this PROGRESS.md (mark item done, set new Next Action Item, append brief log).
6. If a phase is complete, advance Current Phase.

## Log

- 2026-09-03 03:00 IST — docs: evaluation report + iteration note v0.1 (`evals/reports/`); dry-run metrics interpreted as harness-only; next change is an insufficient-context SFT slice. Phase 3 complete, advanced to Phase 4.
- 2026-09-03 02:00 IST — feat: quantitative metrics stub (`eval/metrics.py`, `scripts/score_research_panel.py`) scoring panel comparison JSON with token-overlap F1/Jaccard vs context, citation-hit rate, and refusal accuracy; CPU-only, no GPU train.
- 2026-09-03 01:02 IST — feat: qualitative research panel (`evals/research_panel.jsonl`, 20 grounded QA/summarization items) plus CPU dry-run eval stub (`eval/panel.py`, `scripts/eval_research_panel.py`); `--run` is Kaggle inference only.
- 2026-09-03 00:00 IST — docs: QLoRA reproducibility (`docs/REPRODUCIBILITY.md`) — seed `3407`, `outputs/adapters/*`, dry-run vs `--run` Kaggle how-to; Phase 2 complete, advanced to Phase 3.
- 2026-09-02 23:11 IST — feat: optional 8B QLoRA path (`configs/llama32-8b.yaml` = Llama-3.1-8B-Instruct, batch 1 / accum 8 for T4) plus `--model-name` override on `scripts/train_sft.py`.
- 2026-09-02 22:23 IST — feat: Unsloth QLoRA SFT (`training/sft.py` + `scripts/train_sft.py`); dry-run default writes `outputs/sft_plan.json`; `--run` is the Kaggle GPU path. Wired `training.dataset_dir` / `adapter_dir` / `save_steps` / `max_steps` in `configs/default.yaml`.
- 2026-09-02 20:40 IST — feat: diversity selection + versioned splits (`data/select.py` + `scripts/select_dataset.py`) and `docs/DATA_CARD.md` for `ecra-sft-v0.1.0`; Phase 1 complete, advanced to Phase 2.
- 2026-09-02 19:25 IST — feat: multi-stage filtering (`data/filter.py` + `scripts/filter_grounded_pairs.py`); heuristic length/citation checks, exact/near-dup Jaccard drop, optional LLM-as-judge stub (proxy default, Kaggle hook).
- 2026-09-02 18:07 IST — feat: grounded synthetic Q&A / summary pairs (`data/generate.py` + `scripts/generate_grounded_pairs.py`); template path cites chunk+proposition; optional LLM rewrite hook for Kaggle.
- 2026-09-02 17:01 IST — feat: section-aware chunking + heuristic proposition extraction (`data/chunk.py` + `scripts/chunk_propositions.py`); offline-safe windows for grounded generation.
- 2026-09-02 16:00 IST — feat: public-data ingestion stub (`data/ingest.py` + `scripts/ingest_public_sources.py`); catalog for transcripts / FiQA / Finance-Alpaca with offline fixtures and opt-in HF streaming.
- 2026-09-02 15:01 IST — docs: polished root README with Kaggle + local <30 min baseline path (`load_config` / `configs/default.yaml` / `notebooks/00_baseline_inference.ipynb`); closed Phase 0 and advanced to Phase 1.
- 2026-09-02 14:02 IST — feat: added `src/earnings_call_research_assistant/config.py` (`AppConfig` + model/LoRA/training/inference settings) loading `configs/default.yaml` (or JSON); extended default config with an `inference` block.
- 2026-09-02 13:01 IST — feat: extracted Unsloth chat-template generate harness into `src/earnings_call_research_assistant/inference.py`; notebook now imports `InferenceHarness` / `InferenceConfig`.
- 2026-09-02 12:04 IST — feat: added Kaggle baseline inference notebook (`notebooks/00_baseline_inference.ipynb`) for Unsloth 4-bit smoke generation on Llama-3.2-3B-Instruct.
- 2026-09-02 11:52 IST — chore: added src package layout (`earnings_call_research_assistant`) plus notebooks/, evals/, scripts/ placeholders; marked directory structure and .gitignore done.
- 2026-09-02 11:49 IST — chore: replaced requirements.txt with pyproject.toml (PEP 621 dependencies + optional eval/demo/dev extras).
