# Earnings Call Research Assistant

Domain-adapted LLM for financial research Q&A and summarization from public earnings call transcripts.

**Goal**: Fine-tune a small open model (Llama-3.2-3B / Qwen2.5-3B) with QLoRA on Kaggle so it outperforms the base model on research-style financial questions — suitable as an interview portfolio piece for roles at firms like Morningstar / PitchBook.

## Status

Progress is tracked in [`PROGRESS.md`](PROGRESS.md). An hourly automation advances one action item per run and commits the result.

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 0 | Foundation | In progress |
| 1 | Robust Data Foundation | Pending |
| 2 | Training Pipeline | Pending |
| 3 | Evaluation & Iteration | Pending |
| 4 | Packaging & Portfolio Polish | Pending |

## Key Principles

- Public data only
- Grounded synthetic generation + multi-stage quality filtering
- Quality over quantity (target 3k–6k high-signal examples)
- Reproducible on free Kaggle GPUs (T4 / 2×T4)
- Clear before/after qualitative evidence for interviews

## Structure

```
src/           # training, data, eval scripts
notebooks/     # Kaggle notebooks
data/          # raw / processed (gitignored large files)
configs/       # model & training configs
evals/         # evaluation scripts & reports
scripts/       # utilities
docs/          # plan, data card, design notes
```

See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the full plan and acceptance criteria.
