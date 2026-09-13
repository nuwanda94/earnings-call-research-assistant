# Model card template — ECRA QLoRA + hybrid RAG

Copy this file into the Hugging Face adapter repo `README.md` when publishing
(`scripts/publish_adapter.py --run`). Replace every **TBD** with values from
`data/rag/corpus_v0.1.0/manifest.json`, `evals/reports/rag_metrics.json`, and
`evals/reports/rag_generation_metrics.json`. If those files are missing or
`dry_run: true`, keep TBD.

---

```markdown
---
language: en
license: apache-2.0
base_model: unsloth/Llama-3.2-3B-Instruct
library_name: peft
tags:
  - qlora
  - finance
  - earnings-calls
  - rag
  - retrieval
---

# llama32-3b-ecra-sft (Earnings Call Research Assistant)

QLoRA adapter on Llama-3.2-3B-Instruct for grounded financial research Q&A.
Paired with a **hybrid BM25 + dense** retriever over public earnings / SEC-style
passages. All metrics below are measured on a locked eval set (seed 3407).

GitHub report: https://github.com/nuwanda94/earnings-call-research-assistant/blob/main/evals/reports/RAG_EVAL_REPORT.md

## Intended use

Research-style questions over *retrieved public excerpts* (guidance, margins,
segment color, named risks). The model should refuse figures that are not in
context. Not investment advice.

## Training

| Item | Value |
|------|-------|
| Base | unsloth/Llama-3.2-3B-Instruct (4-bit QLoRA) |
| Seed | 3407 |
| Dataset | ecra-sft-v0.1.0 (public sources; see repo `docs/DATA_CARD.md`) |
| Effective batch | see `configs/default.yaml` / `outputs/sft_plan.json` |
| Adapter dir | `outputs/adapters/llama32-3b-ecra-sft` |

## Retrieval corpus

| Item | Source | Value |
|------|--------|-------|
| N chunks | manifest.json `n_chunks` | TBD |
| Documents | `n_documents` | TBD |
| Corpus version | `version` | TBD |
| Embed model | rag_metrics.json `dense_embed_model` | TBD |
| Fusion | Reciprocal Rank Fusion | rrf_k from JSON (TBD) |

## Measured metrics

Do not edit these cells by hand except to paste JSON literals.

### Recall@k / nDCG@k (`rag_metrics.json`, hybrid backend)

| k | Recall | nDCG |
|--:|-------:|-----:|
| 1 | TBD | TBD |
| 3 | TBD | TBD |
| 5 | TBD | TBD |
| 10 | TBD | TBD |

`n_queries`: TBD · `n_index_docs`: TBD · `dense_backend`: TBD

### Grounded generation (`rag_generation_metrics.json`)

| Side | citation-hit | token F1 | dry_run |
|------|-------------:|---------:|---------|
| base | TBD | TBD | TBD |
| adapter | TBD | TBD | TBD |

`top_k`: TBD · `adapter_dir`: TBD

## How to reproduce

See the GitHub notebook `notebooks/03_rag_eval_and_publish.ipynb` or:

```bash
python scripts/eval_retrieval.py --run
python scripts/eval_rag_generate.py --run --adapter-dir outputs/adapters/llama32-3b-ecra-sft
```

Token: set `HF_TOKEN` or `huggingface-cli login`. Never commit secrets.

## Limitations

Public-data smoke or mid-size corpus only until N is logged. Dry-run hash
embeddings are not ST retrieval. Citation-hit ≠ numerical correctness.
```
