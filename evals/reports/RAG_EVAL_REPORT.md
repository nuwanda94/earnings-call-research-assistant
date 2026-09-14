# RAG Evaluation Report (Phase 5)

Date: 2026-09-14 11:55 IST
Corpus snapshot: `evals/reports/corpus_manifest_snapshot.json`
Retrieval: `evals/reports/rag_metrics.json`
Generation: `evals/reports/rag_generation_metrics.json`
Eval set: `evals/rag_eval_set.jsonl` (seed 3407)

## Corpus (N)

| Field | Value |
|-------|-------|
| N (chunks) | **4** |
| Documents | **3** |
| Version | **v0.1.0** |
| Sources | `{'earnings_transcripts': 2, 'finance_alpaca': 1, 'fiqa': 1}` |

## Retrieval

| Backend | R@1 | R@3 | R@5 | R@10 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 |
|---------|-----|-----|-----|------|--------|--------|--------|---------|
| BM25 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Dense | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Hybrid RRF | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Grounded generation

| Side | citation-hit | token F1 | dry_run |
|------|-------------:|---------:|---------|
| base | TBD | TBD | False |
| adapter | TBD | TBD | False |

Numbers copied from JSON only (`dry_run=false`).
