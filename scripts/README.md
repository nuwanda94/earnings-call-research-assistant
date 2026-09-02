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

## Diversity selection + versioned splits

```bash
python scripts/select_dataset.py --out-dir data/processed/ecra-sft-v0.1.0
```

Applies per-source/task caps and greedy Jaccard diversity, then writes
hash-stable `train.jsonl` / `val.jsonl` / `test.jsonl` plus `manifest.json`.
See [`docs/DATA_CARD.md`](../docs/DATA_CARD.md).

## QLoRA SFT (Phase 2)

```bash
python scripts/train_sft.py
python scripts/train_sft.py --dataset-dir data/processed/ecra-sft-v0.1.0
```

Dry-run by default: reads `configs/default.yaml`, formats chat examples from
the versioned splits, and writes `outputs/sft_plan.json`. No weights load.

On Kaggle T4, after Unsloth is installed:

```bash
python scripts/train_sft.py --run --max-steps 20
```

Full epoch uses YAML `training.*` (batch, LR, seed `3407`). The LoRA adapter
lands in `outputs/adapters/llama32-3b-ecra-sft` unless `--adapter-dir` is set.
