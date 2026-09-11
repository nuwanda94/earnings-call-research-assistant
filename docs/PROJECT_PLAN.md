# Project Plan – Earnings Call Research Assistant

Principal Staff ML Engineer plan. **Public data only.** Optimized for Kaggle free GPUs and interview readiness.

Phases **0–4** deliver grounded **QLoRA SFT** (domain-adapted generator) plus packaging.
Phase **5** adds a real **hybrid RAG** stack so portfolio claims can cite measured **Recall@k / nDCG@k / grounded answer accuracy** on a fixed eval set—not invented percentages.

---

## Success Criteria

### Phases 0–4 (delivered)

- Fine-tuned 3B (primary) model path for research-style financial QA/summarization.
- Reproducible on Kaggle with public data only; dry-run default, `--run` on GPU.
- Grounded synthetic dataset pipeline (design target 3k–6k pairs) with lineage + multi-stage filters.
- GitHub repo + HF adapter publish script + Gradio stub + demo notes.
- Qualitative research panel (~20) + citation / refusal–oriented metric stubs.

### Phase 5 (target – hybrid RAG + publishable metrics)

- Hybrid retrieval (**BM25 + dense**) over **N** chunked public filings/transcripts (N measured and logged).
- Fixed evaluation set with gold chunk IDs (and optional gold answers).
- Reported metrics on that set:
  - **Recall@k** (k ∈ {1, 3, 5, 10})
  - **nDCG@k**
  - **Grounded answer accuracy** (citation support + optional exact/soft match or LLM-as-judge on Kaggle only)
- End-to-end **retrieve → generate** path using the QLoRA adapter (base vs adapter comparison).
- Findings published: **GitHub** (report + numbers in `evals/reports/`) and **Hugging Face** (model card + optional dataset card for the eval index).

**Resume-ready template (fill only after Phase 5 metrics exist):**

> Hybrid retrieval (BM25 + dense) over **[N]** chunked public earnings/SEC-style passages; **Recall@k = [R%]**, **nDCG@k = [D]**, grounded answer accuracy **[A%]** on a fixed held-out eval set; generator fine-tuned with QLoRA (Llama-3.2-3B).

Do **not** put R/D/A on a resume until `evals/reports/RAG_EVAL_REPORT.md` records them from a real run.

---

## Phases 0–4 (complete)

### Phase 0 – Foundation
Repo structure, configs, Kaggle baseline inference harness.

### Phase 1 – Robust Data Foundation
Ingest public sources → chunk + propositions → grounded pairs → filter → diversity select → data card.

### Phase 2 – Training Pipeline
Unsloth QLoRA SFT (3B + optional 8B), seed `3407`, effective batch via grad accum, dry-run vs `--run`.

### Phase 3 – Evaluation & Iteration (generator)
Research panel, token-overlap / citation-hit stubs, evaluation report + iteration note.

### Phase 4 – Packaging
Gradio stub, HF publish script, polished README, demo video notes.

---

## Phase 5 – Hybrid RAG + measurable retrieval metrics

**Goal:** Turn the existing **chunk store** into a **retrieval corpus**, add hybrid search, lock an eval set, measure retrieval + grounded generation, then publish findings.

### 5.0 Design constraints

| Constraint | Choice |
|------------|--------|
| Data | Public only (existing catalog + optional EDGAR *public* HTML/txt samples; no paid terminals) |
| Compute | Build index on CPU; embed on Kaggle GPU if needed; metric scripts CPU-safe |
| Defaults | Dry-run / tiny fixtures; `--run` for full embed + optional LLM judge |
| Honesty | N and all % come from `manifest.json` + `rag_metrics.json` written by scripts |
| Secrets | `HF_TOKEN` via env / Kaggle secrets only |

### 5.1 Corpus & chunk inventory (define **N**)

**Implement**

- `src/earnings_call_research_assistant/rag/corpus.py`
  - Load Phase-1 chunks (`data/processed/chunks.jsonl`) and/or re-chunk ingested docs.
  - Assign stable `chunk_id` (hash of source_id + span).
  - Write `data/rag/corpus_v0.1.0/chunks.jsonl` + `manifest.json` with **N = chunk count**, source breakdown, license notes.
- `scripts/build_rag_corpus.py` (dry-run prints N; writes corpus dir).

**Acceptance**

- [ ] `manifest.json` contains `n_chunks`, `n_documents`, `sources`, `version`.
- [ ] N is reproducible from the same input JSONL (hash-stable IDs).

**Portfolio note:** Prefer N in the **hundreds–thousands of chunks** (e.g. 500–5 000) once HF/download paths are used—not the 6-row smoke set.

### 5.2 Hybrid index (BM25 + dense)

**Implement**

- `rag/bm25.py` — lexical index (e.g. `rank_bm25` or whoosh-lite); persist under `data/rag/indices/bm25/`.
- `rag/dense.py` — sentence-transformer embeddings (default: small public model, e.g. `BAAI/bge-small-en-v1.5` or `sentence-transformers/all-MiniLM-L6-v2`); FAISS or numpy top-k; persist `data/rag/indices/dense/`.
- `rag/hybrid.py` — fuse ranks (RRF or weighted score normalize); `retrieve(query, k) -> list[Hit]`.
- `configs/rag.yaml` — embed model, k, rrf_k, paths, seed.
- `scripts/build_rag_index.py` — build BM25 always; dense with `--run` (downloads embed model).

**Acceptance**

- [ ] CPU path: BM25-only retrieve works offline on fixtures.
- [ ] Hybrid path returns fused top-k with both score components logged.
- [ ] Index build is idempotent given the same corpus version.

### 5.3 Fixed retrieval eval set (gold chunk IDs)

**Implement**

- `evals/rag_eval_set.jsonl` — each row: `query_id`, `query`, `gold_chunk_ids[]`, optional `gold_answer`, `source`.
- Bootstrap: derive from grounded pairs where the cited chunk is known (template pairs already carry context lineage); hold out **≥30–50** queries for v0.1, grow to **100+** when corpus scales.
- `scripts/build_rag_eval_set.py` — deterministic sample (seed `3407`).

**Acceptance**

- [ ] No train leakage: eval queries excluded from any retrieval-tuning labels if we add them later.
- [ ] Every `gold_chunk_id` exists in the corpus manifest.

### 5.4 Metrics: Recall@k, nDCG@k, grounded answer accuracy

**Implement**

- `rag/metrics.py`
  - `recall_at_k(gold_ids, ranked_ids, k)`
  - `ndcg_at_k(...)` (binary relevance on gold chunk IDs)
  - Aggregate mean over the eval set; write `evals/reports/rag_metrics.json`.
- `scripts/eval_retrieval.py` — run BM25 / dense / hybrid; print table for k=1,3,5,10.
- **Grounded answer accuracy** (`scripts/eval_rag_generate.py`):
  1. Retrieve top-k with hybrid.
  2. Pack context → `InferenceHarness` (base and/or adapter).
  3. Score: **citation-hit** (answer spans supported by retrieved text) + optional token F1 vs gold_answer; optional LLM-judge only under `--run` on Kaggle.
  4. Write `evals/reports/rag_generation_metrics.json`.

**Acceptance**

- [ ] One command produces machine-readable metrics + a short markdown report.
- [ ] Report states **N**, index version, embed model, k, and whether adapter was used.

### 5.5 RAG + QLoRA generator path (Kaggle)

**User-run on T4 (not automation long jobs)**

1. Build corpus + eval set (CPU).
2. `build_rag_index.py --run` (embed).
3. `eval_retrieval.py` → baseline hybrid Recall/nDCG.
4. Existing SFT path: `train_sft.py --run` (or notebooks `01` / `02`) on grounded pairs.
5. `eval_rag_generate.py --run --adapter-dir outputs/adapters/llama32-3b-ecra-sft`.
6. Compare base vs adapter grounded accuracy in the report.

**Notebook:** `notebooks/03_rag_eval_and_publish.ipynb` (clone → corpus → index → metrics → optional train → generate eval → publish cells).

### 5.6 Publish findings (GitHub + Hugging Face)

**GitHub**

- [ ] `evals/reports/RAG_EVAL_REPORT.md` — N, tables for Recall@k / nDCG@k, grounded accuracy, limitations.
- [ ] Update root README “Results” section with **numbers from the report only**.
- [ ] Tag release e.g. `v0.2.0-rag` when metrics stabilize.

**Hugging Face**

- [ ] Adapter repo model card: link to GitHub report; document base model, seed, effective batch, RAG metrics summary.
- [ ] Optional: upload **eval corpus manifest + eval set JSONL** as a dataset repo (`nuwanda94/ecra-rag-eval-v0.1`) — public, no secrets.
- [ ] `scripts/publish_adapter.py --run` remains the upload path; extend model card template under `docs/MODEL_CARD_RAG.md`.

**Profile narrative**

- GitHub README pin + HF model card should tell the same story: hybrid RAG numbers + QLoRA generator, public data, reproducible Kaggle commands.

### 5.7 Implementation order (one PR / automation item at a time)

1. Corpus builder + manifest (**defines N**).
2. BM25 index + retrieve CLI.
3. Dense embed + FAISS/numpy + hybrid RRF.
4. Gold eval set builder from grounded pairs.
5. `eval_retrieval.py` → Recall@k + nDCG@k JSON + markdown stub.
6. `eval_rag_generate.py` → grounded answer accuracy.
7. Kaggle notebook `03_rag_eval_and_publish.ipynb`.
8. RAG_EVAL_REPORT + README results + HF model card template.
9. Human: full Kaggle `--run`, fill numbers, publish adapter + report commit.

### 5.8 Out of scope (explicit)

- Paid filings terminals, non-public transcripts.
- Claiming SEC coverage without public EDGAR (or similar) in the data card.
- Tuning dense retriever with private labels before a locked eval set exists.
- Inventing X% before `rag_metrics.json` exists.

---

## Detailed acceptance criteria (Phase 5 checklist)

- [ ] `data/rag/corpus_v*/manifest.json` with measured **N**
- [ ] Hybrid retrieve API + BM25-only fallback
- [ ] `evals/rag_eval_set.jsonl` with gold chunk IDs
- [ ] `evals/reports/rag_metrics.json` with Recall@k and nDCG@k
- [ ] `evals/reports/rag_generation_metrics.json` with grounded answer accuracy
- [ ] `evals/reports/RAG_EVAL_REPORT.md` suitable for GitHub + HF card
- [ ] Kaggle notebook path documented; HF publish + GitHub report instructions

---

## Automation notes

Phases 0–4 remain complete. Automation should advance **Phase 5** one checklist item at a time (prefer 5.1 → 5.7 order). Do not start long GPU trains inside automation; prefer scripts/notebooks the user runs on Kaggle. When Phase 5 metrics exist, human publishes findings to HF + GitHub profile.
