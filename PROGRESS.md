# Progress Tracker

Last updated: 2026-09-02 13:01 IST

## Current Phase
Phase 0 – Foundation

## Next Action Item
Add a small config loader (`src/earnings_call_research_assistant/config.py`) that reads YAML/JSON (starting with `configs/default.yaml`) and exposes typed settings for model, LoRA, training, and inference.

## Completed Items

- [x] Create GitHub repository `earnings-call-research-assistant`
- [x] Initial README and progress tracker
- [x] Replace requirements.txt with pyproject.toml for project dependencies
- [x] Directory structure: `src/`, `notebooks/`, `data/`, `configs/`, `evals/`, `scripts/`, `docs/`
- [x] Kaggle notebook template `notebooks/00_baseline_inference.ipynb`
- [x] Reusable base-model inference harness (`src/earnings_call_research_assistant/inference.py`)

## Phase 0 Checklist

- [x] GitHub repository exists
- [x] Directory structure: `src/`, `notebooks/`, `data/`, `configs/`, `evals/`, `scripts/`, `docs/`
- [ ] Root README with overview (done at high level)
- [x] `.gitignore` for weights, large data, secrets
- [x] Project dependencies in `pyproject.toml` (not requirements.txt)
- [x] Kaggle notebook template
- [ ] Config system (YAML/JSON) for model & training
- [x] Basic inference harness for base model
- [ ] New person can clone and run baseline in < 30 min

## Notes for Automation

On each hourly run:
1. Read this file and `docs/PROJECT_PLAN.md`.
2. Identify the single next incomplete action item (prefer Phase 0 → 1 → 2 → 3 → 4 order).
3. Implement that one item (create/update files).
4. Commit via GitHub connector with a conventional message (`feat:`, `chore:`, or `fix:`).
5. Update this PROGRESS.md (mark item done, set new Next Action Item, append brief log).
6. If a phase is complete, advance Current Phase.

## Log

- 2026-09-02 13:01 IST — feat: extracted Unsloth chat-template generate harness into `src/earnings_call_research_assistant/inference.py`; notebook now imports `InferenceHarness` / `InferenceConfig`.
- 2026-09-02 12:04 IST — feat: added Kaggle baseline inference notebook (`notebooks/00_baseline_inference.ipynb`) for Unsloth 4-bit smoke generation on Llama-3.2-3B-Instruct.
- 2026-09-02 11:52 IST — chore: added src package layout (`earnings_call_research_assistant`) plus notebooks/, evals/, scripts/ placeholders; marked directory structure and .gitignore done.
- 2026-09-02 11:49 IST — chore: replaced requirements.txt with pyproject.toml (PEP 621 dependencies + optional eval/demo/dev extras).
