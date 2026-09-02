# Data Card — `ecra-sft-v0.1.0`

Earnings Call Research Assistant supervised fine-tuning set.
Public sources only. Grounded generation + multi-stage quality controls.

## Snapshot

| Field | Value |
|-------|--------|
| Version | `ecra-sft-v0.1.0` |
| Intended size | 3,000–6,000 instruction pairs |
| Splits | train 80% / val 10% / test 10% (hash-stable) |
| Seed | 94 |
| Layout | `data/processed/ecra-sft-v0.1.0/{train,val,test}.jsonl` + `manifest.json` |
| License posture | Public research corpora only; do not redistribute paywalled transcripts |

The versioned files themselves are gitignored when large. Rebuild with:

```bash
python scripts/select_dataset.py --out-dir data/processed/ecra-sft-v0.1.0
```

Fixture mode (no downloads) writes a tiny deterministic snapshot suitable for CI.

## Motivation

Train a 3B research assistant that answers and summarizes from earnings-call
context without inventing figures. Every kept example must cite the chunk it
was built from.

## Sources (catalog)

Defined in `src/earnings_call_research_assistant/data/ingest.py`:

- Earnings call transcripts — `kurry/sp500_earnings_transcripts` (public HF mirror).
- FiQA financial QA — `LLukas22/fiqa`.
- Finance-Alpaca sample — `gbharti/finance-alpaca`.

Offline fixtures ship inside the ingest module so the pipeline runs without
network. `--download` on ingest scripts is opt-in Hugging Face streaming.

## Lineage

```
ingest (public catalog / fixtures)
  → chunk + heuristic propositions
  → grounded Q&A / summary templates (optional LLM rewrite on Kaggle)
  → filter: length/citation heuristic → exact + Jaccard near-dup → LLM-as-judge stub
  → select: per-source/task caps → greedy max-min diversity → hash splits
```

Record fields (`InstructionPair`):

- `pair_id`, `chunk_id`, `source_id`, `task` (`qa` or `summary`)
- `instruction`, `context`, `output`, `citations[]`, `metadata`

## Selection rules

- Cap pairs per `source_id`, per `task`, and per `(source, task)` so one
  corpus cannot drown the mix.
- Greedy diversity: a candidate is dropped if token-Jaccard vs any already
  kept pair is ≥ `diversity_jaccard_cap` (default 0.72). The first pair in
  each `(source, task)` bucket is always kept.
- Hard ceiling `target_max` (default 6,000). `target_min` (3,000) is a
  planning floor for the full corpora, not a fixture requirement.

## Splits

`assign_split(pair_id)` hashes `pair_id|dataset_version|seed` with SHA-1 and
maps the unit interval onto train/val/test. Assignment does not depend on
row order, so regenerating the same filtered set yields the same split.

Do **not** tune on `test`. Val is for early stopping and qualitative review.

## Filtering that precedes this card

See `src/earnings_call_research_assistant/data/filter.py`:

- Instruction / output / context length bounds.
- Required citation list and a citation marker in the output.
- Exact SHA-1 duplicate drop, then near-dup Jaccard 0.88.
- Optional LLM-as-judge (deterministic proxy offline; real model on Kaggle).

## Risks and limitations

- Template-generated answers can be formulaic until the Kaggle rewrite hook
  is used.
- Public transcript mirrors may lag or miss speakers / corrections.
- FiQA and Finance-Alpaca are adjacent finance text, not earnings calls;
  they add instruction variety, not grounded call facts.
- Jaccard diversity is lexical, not semantic. Embeddings can replace it later.
- No material non-public information. Not investment advice.

## Maintenance

Bump `DATASET_VERSION` in `select.py` when sources, filters, or split seed
change. Write a new directory (`ecra-sft-v0.1.1`, …) and append a row here.
Keep `manifest.json` next to the JSONL files.
