# Progress Tracker

Last updated: 2026-09-02 20:40 IST

## Current Phase
Phase 2 – Training Pipeline

## Next Action Item
Add Unsloth QLoRA SFT training script + config wiring (`src/earnings_call_research_assistant/training/sft.py`, `scripts/train_sft.py`) that reads versioned splits and `configs/default.yaml`. Do not launch a long GPU run in automation — keep it a Kaggle-ready script with a CPU/dry-run path.

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
- [x] Multi-stage filtering (heuristic → dedup → LLM-as-judge)
- [x] Diversity selection + versioned splits + data card

## Phase 2 Checklist

- [ ] Unsloth QLoRA SFT training script (3B) + externalized config / logging / checkpoints
- [ ] Optional 8B config path
- [ ] Reproducibility notes (seed, adapter output dir, Kaggle how-to)

## Notes for Automation

On each hourly run:
1. Read this file and `docs/PROJECT_PLAN.md`.
2. Identify the single next incomplete action item (prefer Phase 0 → 1 → 2 → 3 → 4 order).
3. Implement that one item (create/update files).
4. Commit via GitHub connector with a conventional message (`feat:`, `chore:`, or `fix:`).
5. Update this PROGRESS.md (mark item done, set new Next Action Item, append brief log).
6. If a phase is complete, advance Current Phase.

## Log

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
