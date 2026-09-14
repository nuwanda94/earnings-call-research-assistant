# Model card — ECRA QLoRA + hybrid RAG

Copy into the Hugging Face adapter repo `README.md` when publishing
(`scripts/publish_adapter.py --run` also auto-writes from JSON).
Values below match the **2026-09-14 full-scale** artifacts in git.

---

```markdown
---
language:
- en
license: apache-2.0
base_model: unsloth/Llama-3.2-3B-Instruct
library_name: peft
tags:
  - qlora
  - finance
  - earnings-calls
  - rag
  - retrieval
pipeline_tag: text-generation
---

# llama32-3b-ecra-sft (Earnings Call Research Assistant)

QLoRA adapter on Llama-3.2-3B-Instruct for grounded financial research Q&A.
Paired with a hybrid BM25 + dense retriever over public earnings-style passages.

GitHub report: https://github.com/nuwanda94/earnings-call-research-assistant/blob/main/evals/reports/RAG_EVAL_REPORT.md

## Intended use

Research-style questions over *retrieved public excerpts*. Prefer refusing figures
not in context. **Not investment advice.**

## Training

| Item | Value |
|------|-------|
| Base | unsloth/Llama-3.2-3B-Instruct (4-bit QLoRA) |
| Seed | 3407 |
| Dataset | ecra-sft-v0.1.0 |
| Train / val | **2127 / 240** |
| Effective batch | 16 |
| Adapter | outputs/adapters/llama32-3b-ecra-sft |

## Retrieval corpus

| Item | Value |
|------|-------|
| N chunks | **19990** |
| Sources | transcripts 18474 · fiqa 860 · finance_alpaca 656 |
| Embed model | sentence-transformers/all-MiniLM-L6-v2 |
| Fusion | RRF (rrf_k=60) |

## Measured metrics

### Recall@k / nDCG@k (`rag_metrics.json`, hybrid)

| k | Recall | nDCG |
|--:|-------:|-----:|
| 1 | **0.0** | **0.0** |
| 3 | **0.0** | **0.0** |
| 5 | **0.0** | **0.0** |
| 10 | **0.0** | **0.0** |

`n_queries`: **50** · `n_index_docs`: **19990**  
**Note:** R@k is 0 with generic template queries on the large index. Re-run
`scripts/realign_rag_eval.py` before citing IR quality.

### Grounded generation (`rag_generation_metrics.json`, dry_run=false)

| Side | citation-hit | token F1 vs gold | grounded accuracy |
|------|-------------:|-----------------:|------------------:|
| base | **0.593** | **0.147** | **0.58** |
| adapter | **0.876** | **0.321** | **1.0** |

## How to load

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "unsloth/Llama-3.2-3B-Instruct"
adapter = "nuwanda94/llama32-3b-ecra-sft"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
```

## Limitations

Public data only. Report measured N. Retrieval R@k not portfolio-ready until
query realignment. Citation-hit ≠ numerical correctness.
```
