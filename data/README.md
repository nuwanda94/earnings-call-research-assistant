# Data

Public sources only. Large raw/processed files are gitignored.

## Planned sources
- S&P 500 / large-cap earnings call transcripts (Hugging Face)
- FiQA / financial QA sets
- Finance-Alpaca or Finance-Instruct sample

## Pipeline (Phase 1)
1. Ingestion & normalization
2. Chunking & proposition extraction
3. Grounded synthetic generation
4. Multi-stage filtering (heuristic → dedup → LLM-as-judge)
5. Diversity selection → versioned train/val/test + data card

See `docs/PROJECT_PLAN.md` and root `PROGRESS.md`.
