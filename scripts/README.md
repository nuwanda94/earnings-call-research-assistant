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

## Chunking + propositions

```bash
python scripts/chunk_propositions.py --out data/processed/chunks.jsonl
```

Splits ingested text into section-aware sentence windows and extracts short
heuristic propositions (numbers, metrics, guidance). No LLM required.

## Grounded synthetic pairs

```bash
python scripts/generate_grounded_pairs.py --out data/processed/grounded_pairs.jsonl
```

Turns chunks + propositions into citation-grounded Q&A and summary instruction
pairs via offline templates. Optional LLM rewrite is a notebook hook only
(`generate_pairs(..., llm=..., config=GenerateConfig(use_llm=True))`).

## Multi-stage filtering

```bash
python scripts/filter_grounded_pairs.py --out data/processed/filtered_pairs.jsonl
```

Applies heuristic length/citation checks, exact + near-duplicate drop, and an
optional LLM-as-judge stage. `--use-llm-judge` runs a deterministic proxy so
automation never bills GPU; pass a real `judge` callable from a Kaggle notebook.
