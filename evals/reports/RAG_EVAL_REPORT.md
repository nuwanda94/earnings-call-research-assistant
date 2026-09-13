# RAG Evaluation Report (Phase 5)

Date: 2026-09-13 17:01 IST  
Corpus / N source: `data/rag/corpus_v0.1.0/manifest.json` (write via `scripts/build_rag_corpus.py`)  
Retrieval metrics: `evals/reports/rag_metrics.json` (write via `scripts/eval_retrieval.py`)  
Generation metrics: `evals/reports/rag_generation_metrics.json` (write via `scripts/eval_rag_generate.py`)  
Eval set: `evals/rag_eval_set.jsonl` (seed `3407`)

**Honesty rule:** every N and every percentage on this page is copied from those JSON files or marked **TBD**. Do not invent resume numbers.

## Status of artifacts in this repo

| Artifact | Present in git? | Value used here |
|----------|-----------------|-----------------|
| `data/rag/corpus_v0.1.0/manifest.json` | No (generated locally / Kaggle; corpus dir is not committed) | **TBD** |
| `evals/reports/rag_metrics.json` | No | **TBD** |
| `evals/reports/rag_generation_metrics.json` | No | **TBD** |
| `evals/rag_eval_set.jsonl` | Yes (fixture / builder output) | committed; query count still TBD until metrics JSON exists |

Treat any local dry-run JSON as a **fixture**, not a published result. Dry-run dense embeddings may be hash-based, not sentence-transformers.

## How to fill this report (human / Kaggle)

```bash
python scripts/build_rag_corpus.py
python scripts/build_rag_eval_set.py
python scripts/build_rag_index.py --run          # ST embeddings; GPU optional
python scripts/eval_retrieval.py --run
python scripts/eval_rag_generate.py --run --adapter-dir outputs/adapters/llama32-3b-ecra-sft
```

Then copy fields:

- **N** ← `manifest.json` → `n_chunks` (also log `n_documents`, `version`, `sources`)
- Recall@k / nDCG@k ← `rag_metrics.json` → `backends.hybrid.mean_recall` / `mean_ndcg` (also BM25 and dense)
- Grounded accuracy ← `rag_generation_metrics.json` → `aggregate` (citation-hit, token F1, `dry_run` flag)

If `dry_run` is `true` in generation metrics, leave grounded accuracy as **TBD** on the resume.

## Corpus (N)

| Field | Source key | Value |
|-------|------------|-------|
| N (chunks) | `n_chunks` | **TBD** |
| Documents | `n_documents` | **TBD** |
| Version | `version` | **TBD** (expected `v0.1.0` after first write) |
| Sources | `sources` | **TBD** (public catalog only) |

Portfolio target after a full public ingest is hundreds–thousands of chunks, not the smoke fixture.

## Retrieval (Recall@k, nDCG@k)

k ∈ {1, 3, 5, 10}. Binary relevance on gold `chunk_id`s. Means are over the locked eval set.

| Backend | R@1 | R@3 | R@5 | R@10 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 |
|---------|-----|-----|-----|------|--------|--------|--------|---------|
| BM25 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Dense | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Hybrid RRF | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Metadata to copy from `rag_metrics.json` when it exists:

- `n_queries`, `n_index_docs`, `corpus_version`
- `dense_backend`, `dense_embed_model`, `rrf_k`

Default embed model (from `configs/rag.yaml` / dense builder): sentence-transformers family (e.g. MiniLM or BGE-small). Record the exact string from JSON, not this sentence.

## Grounded generation (retrieve → generate)

Pipeline: hybrid top-k → pack retrieved text → `InferenceHarness` base vs QLoRA adapter.

| Side | citation-hit | token F1 vs gold_answer | n_queries | dry_run |
|------|-------------:|------------------------:|----------:|---------|
| base | TBD | TBD | TBD | TBD |
| adapter | TBD | TBD | TBD | TBD |

Copy from `rag_generation_metrics.json` → `aggregate` (and top-level `dry_run`, `top_k`, `adapter_dir`, `corpus_version`).

Until `--run` generations exist, placeholders score near zero and must not be cited as model quality.

## Limitations

- Smoke corpus N is not a coverage claim for SEC / all issuers.
- Hash embeddings in dry-run are for wiring tests; hybrid numbers from that path are not sentence-transformer retrieval.
- Citation-hit is substring support against retrieved text, not numerical consistency.
- Eval set size in v0.1 is a lock for reproducibility (seed 3407), not a large IR benchmark.
- Train-pair IDs are excluded from the eval builder; do not mix SFT train queries back into gold.

## Resume template (fill only from JSON)

> Hybrid retrieval (BM25 + dense) over **[N from manifest]** chunked public earnings/SEC-style passages; **Recall@k = [hybrid mean_recall from rag_metrics.json]**, **nDCG@k = [hybrid mean_ndcg]**, grounded answer accuracy **[aggregate from rag_generation_metrics.json with dry_run=false]** on a fixed held-out eval set; generator fine-tuned with QLoRA (Llama-3.2-3B).

Leave brackets as TBD until Phase 5.9 Kaggle `--run` writes the three JSON files and this report is edited with those literals.

## Next

Phase 5.9: human Kaggle full run (`notebooks/03_rag_eval_and_publish.ipynb` or the CLI above), paste numbers here and into the README Results section, publish adapter + this report to Hugging Face using `docs/MODEL_CARD_RAG.md`.
