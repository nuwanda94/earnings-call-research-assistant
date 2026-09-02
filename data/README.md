# Data

Public sources only. Large raw/processed files are gitignored.

## Planned sources
- S&P 500 / large-cap earnings call transcripts (`kurry/sp500_earnings_transcripts`)
- FiQA / financial QA sets (`LLukas22/fiqa`)
- Finance-Alpaca sample (`gbharti/finance-alpaca`)

See `src/earnings_call_research_assistant/data/ingest.py` for the catalog and
`scripts/ingest_public_sources.py` to print it or write a tiny JSONL sample.

```bash
python scripts/ingest_public_sources.py --catalog-only
```

## Pipeline (Phase 1)
1. Ingestion & normalization  ← stub in place (offline fixtures; optional HF stream)
2. Chunking & proposition extraction
3. Grounded synthetic generation  ← `data/generate.py` + `scripts/generate_grounded_pairs.py`
4. Multi-stage filtering (heuristic → dedup → LLM-as-judge)
5. Diversity selection → versioned train/val/test + data card

See `docs/PROJECT_PLAN.md` and root `PROGRESS.md`.
