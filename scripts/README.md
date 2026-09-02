# Scripts

CLI utilities for ingestion, filtering, training launch, and report generation.
Prefer thin wrappers around `earnings_call_research_assistant` so notebooks and Kaggle stay in sync.

## Public-source catalog (Phase 1)

```bash
python scripts/ingest_public_sources.py --catalog-only
python scripts/ingest_public_sources.py --out data/raw/public_sample.jsonl
```

Default mode uses tiny offline fixtures and does **not** download corpora.
Pass `--download` only when you explicitly want a streamed Hugging Face sample.
