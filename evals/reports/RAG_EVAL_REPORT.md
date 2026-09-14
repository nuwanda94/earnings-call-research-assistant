# RAG Evaluation Report (Phase 5)

Date: 2026-09-14 14:22 IST  
Artifacts:
- `evals/reports/corpus_manifest_snapshot.json`
- `evals/reports/rag_metrics.json`
- `evals/reports/rag_generation_metrics.json`
- `evals/rag_eval_set.jsonl` (seed `3407`)
- SFT plan: `outputs/sft_plan.json`

**Honesty rule:** every N and every percentage is copied from those JSON files (or SFT plan). No invented numbers.

## Run summary (full-scale Kaggle T4)

| Stage | Result |
|-------|--------|
| SFT | Full-epoch QLoRA, seed **3407**, train **2127** / val **240** / selected **2627** pairs |
| Adapter Hub | https://huggingface.co/nuwanda94/llama32-3b-ecra-sft (`uploaded=true`) |
| RAG corpus N | **19990** chunks |
| Eval queries | **50** |
| Dense backend | `sentence-transformers` / `all-MiniLM-L6-v2` |

## Corpus (N)

| Field | Source key | Value |
|-------|------------|-------|
| N (chunks) | `n_chunks` | **19990** |
| Sources | `sources` | earnings_transcripts: 18474, finance_alpaca: 656, fiqa: 860 |
| Version | `version` | **v0.1.0** |
| Created (UTC) | `created_utc` | 2026-09-14T08:05:52Z |
| Input | `input_path` | `data/processed/chunks.jsonl` |

## SFT (generator)

| Field | Value |
|-------|-------|
| Model | unsloth/Llama-3.2-3B-Instruct (4-bit QLoRA) |
| Seed | 3407 |
| Effective batch | 16 (micro 2 × accum 8) |
| Train / val | **2127 / 240** |
| Dataset version | ecra-sft-v0.1.0 |
| Adapter | `outputs/adapters/llama32-3b-ecra-sft` → Hub `nuwanda94/llama32-3b-ecra-sft` |

## Retrieval (Recall@k, nDCG@k)

Means over **50** queries · `n_index_docs=19990` · `rrf_k=60` · `dense_backend=sentence-transformers`

**From `rag_metrics.json` (this commit):** hybrid / BM25 / dense mean_recall and mean_ndcg are **0.0 at k∈{1,3,5,10}.**

| Backend | R@1 | R@3 | R@5 | R@10 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 |
|---------|-----|-----|-----|------|--------|--------|--------|---------|
| BM25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Dense (ST) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Hybrid RRF | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### Why retrieval is 0.0 (known issue)

Gold `chunk_id`s in `evals/rag_eval_set.jsonl` come from grounded pairs, but many **queries are template-generic** (e.g. “What did management report in this prepared remarks excerpt?”). On a ~20k-chunk index, rankers return other prepared-remarks spans; gold IDs never appear in top-10 → Recall@k = 0.

Fix path (regenerate candidates **from the current corpus**, then re-score):

```bash
python scripts/realign_rag_eval.py --max-n 50 --eval-retrieval
python scripts/realign_rag_eval.py --max-n 50 --eval-retrieval --eval-generate \
  --adapter-dir outputs/adapters/llama32-3b-ecra-sft
```

Do **not** invent higher R@k until a realigned JSON exists. Do not put R@k=0 on a resume without the query-genericity caveat.

## Grounded generation (retrieve → generate)

From `rag_generation_metrics.json` (`dry_run: false`, **50** queries, hybrid top-k=5, adapter present):

| Side | citation-hit rate | token F1 vs gold | grounded_answer_accuracy |
|------|------------------:|-----------------:|-------------------------:|
| base | **0.593** | **0.147** | **0.58** |
| adapter | **0.876** | **0.321** | **1.0** |

With retrieval often missing the gold span, generations are scored against **retrieved** context; treat **generator lift** (base → adapter) as the primary claim, not perfect IR.

## Limitations

- Retrieval metrics are **not portfolio-ready** until realign + re-eval.
- Template queries are under-specified for a large index.
- Citation-hit ≠ numerical consistency with filings.
- Public data only; not SEC-wide coverage.

## Resume-safe wording (current evidence)

> QLoRA fine-tuned Llama-3.2-3B on **2,127** grounded public earnings-style pairs (seed 3407; val 240). Adapter: `nuwanda94/llama32-3b-ecra-sft`. Hybrid BM25+dense index over **19,990** public chunks; retrieval eval (50 queries) currently shows **R@k = 0** due to generic template queries — realign path documented. On the same 50-query generate eval (`dry_run=false`), grounded_answer_accuracy **base 0.58 → adapter 1.0**, citation-hit **0.59 → 0.88**.

## Next

1. `python scripts/backup_adapter.py --repo-id nuwanda94/llama32-3b-ecra-sft`
2. On a machine with the corpus: realign + re-eval (commands above)
3. Re-fill this report + Hub model card from the new JSON only
