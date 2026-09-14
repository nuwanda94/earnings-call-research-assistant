# RAG Evaluation Report (Phase 5)

Date: 2026-09-14 11:55 IST  
Corpus snapshot: `evals/reports/corpus_manifest_snapshot.json`  
Retrieval: `evals/reports/rag_metrics.json`  
Generation: `evals/reports/rag_generation_metrics.json`  
Eval set: `evals/rag_eval_set.jsonl` (seed `3407`)

**Honesty rule:** every N and every percentage on this page is copied from those JSON files.  
**Caveat:** this is a **smoke** corpus (**N=4** chunks, **6** eval queries). High R@5 is expected on a tiny index — not a coverage claim for SEC / all issuers.

## Status of artifacts in this repo

| Artifact | Present in git? | Notes |
|----------|-----------------|-------|
| `evals/reports/corpus_manifest_snapshot.json` | Yes | N=4, v0.1.0 |
| `evals/reports/rag_metrics.json` | Yes | BM25 / dense / hybrid; ST backend |
| `evals/reports/rag_generation_metrics.json` | Yes | `dry_run: false` |
| `evals/rag_eval_set.jsonl` | Yes | seed 3407 |

## Corpus (N)

| Field | Source key | Value |
|-------|------------|-------|
| N (chunks) | `n_chunks` | **4** |
| Documents | `n_documents` | **3** |
| Version | `version` | **v0.1.0** |
| Sources | `sources` | earnings_transcripts: 2, finance_alpaca: 1, fiqa: 1 |
| Created (UTC) | `created_utc` | 2026-09-14T06:29:55Z |

## Retrieval (Recall@k, nDCG@k)

k ∈ {1, 3, 5, 10}. Binary relevance on gold `chunk_id`s. Means over **6** queries.  
`n_index_docs=4` · `dense_backend=sentence-transformers` · `dense_embed_model=sentence-transformers/all-MiniLM-L6-v2` · `rrf_k=60`

| Backend | R@1 | R@3 | R@5 | R@10 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 |
|---------|-----|-----|-----|------|--------|--------|--------|---------|
| BM25 | 0.500 | 0.667 | 1.000 | 1.000 | 0.500 | 0.583 | 0.727 | 0.727 |
| Dense (ST) | 0.333 | 0.833 | 1.000 | 1.000 | 0.333 | 0.605 | 0.677 | 0.677 |
| Hybrid RRF | 0.333 | 0.833 | 1.000 | 1.000 | 0.333 | 0.605 | 0.677 | 0.677 |

## Grounded generation (retrieve → generate)

Pipeline: hybrid top-k=5 → pack context → `InferenceHarness` base vs QLoRA adapter (`outputs/adapters/llama32-3b-ecra-sft`).  
`dry_run: false` · `n_queries: 6` · `corpus_version: v0.1.0`

| Side | citation-hit rate | token F1 vs gold | grounded_answer_accuracy | dry_run |
|------|------------------:|-----------------:|-------------------------:|---------|
| base | 0.486 | 0.321 | **0.667** | false |
| adapter | 0.569 | 0.485 | **1.000** | false |

Also logged in JSON: mean token F1 vs retrieved context — base 0.357 / adapter 0.250.

## Limitations

- Smoke corpus N=4 is not a coverage claim for SEC / all issuers.
- With only 4 index docs, R@5/R@10 saturating at 1.0 is expected; treat ranking quality as exploratory.
- Citation-hit is substring support against retrieved text, not numerical consistency.
- Eval set size (6 queries, seed 3407) is a lock for reproducibility, not a large IR benchmark.
- Adapter train was a short Kaggle run (60 steps); not a full-epoch portfolio claim by itself.

## Resume template (fixture-scale only)

> Hybrid retrieval (BM25 + dense MiniLM) over **4** chunked public earnings-style passages (smoke corpus); hybrid **R@5 = 1.0**, **nDCG@5 = 0.677** on 6 held-out queries; grounded answer accuracy **base 0.667 / adapter 1.0** (`dry_run=false`); generator fine-tuned with QLoRA (Llama-3.2-3B). Scale N before using these as primary resume metrics.

## Next

Optional: expand public corpus to hundreds–thousands of chunks, rebuild index + eval set, re-run metrics, then update this report and the README Results table with the new literals only.
