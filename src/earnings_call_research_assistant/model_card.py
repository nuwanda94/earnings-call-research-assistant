"""Fill Hugging Face model card README from measured SFT / RAG artifacts.

Never invents percentages. Missing JSON leaves TBD placeholders.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"
DEFAULT_DATASET = "ecra-sft-v0.1.0"
DEFAULT_ADAPTER_DIR = Path("outputs/adapters/llama32-3b-ecra-sft")
DEFAULT_OUT_README = DEFAULT_ADAPTER_DIR / "README.md"
GITHUB_REPORT = (
    "https://github.com/nuwanda94/earnings-call-research-assistant/"
    "blob/main/evals/reports/RAG_EVAL_REPORT.md"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "TBD"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    if isinstance(x, dict):
        # compact mean_recall style {"1": 0.3, ...}
        parts = []
        for k in sorted(x.keys(), key=lambda z: (len(str(z)), str(z))):
            parts.append(f"{k}={_fmt(x[k], digits)}")
        return ", ".join(parts) if parts else "TBD"
    return str(x)


@dataclass
class ModelCardInputs:
    base_model: str = DEFAULT_BASE_MODEL
    dataset_version: str = DEFAULT_DATASET
    seed: int = 3407
    adapter_dir: str = str(DEFAULT_ADAPTER_DIR)
    n_train: str = "TBD"
    n_val: str = "TBD"
    effective_batch: str = "TBD"
    n_chunks: str = "TBD"
    n_documents: str = "TBD"
    corpus_version: str = "TBD"
    embed_model: str = "TBD"
    dense_backend: str = "TBD"
    n_queries: str = "TBD"
    recall: dict[str, str] = field(default_factory=dict)
    ndcg: dict[str, str] = field(default_factory=dict)
    base_citation: str = "TBD"
    base_f1: str = "TBD"
    adapter_citation: str = "TBD"
    adapter_f1: str = "TBD"
    grounded_accuracy_base: str = "TBD"
    grounded_accuracy_adapter: str = "TBD"
    gen_dry_run: str = "TBD"
    created_utc: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_card_inputs(
    *,
    sft_plan_path: Path | str | None = Path("outputs/sft_plan.json"),
    dataset_manifest: Path | str | None = Path("data/processed/ecra-sft-v0.1.0/manifest.json"),
    corpus_manifest: Path | str | None = Path("data/rag/corpus_v0.1.0/manifest.json"),
    rag_metrics: Path | str | None = Path("evals/reports/rag_metrics.json"),
    gen_metrics: Path | str | None = Path("evals/reports/rag_generation_metrics.json"),
    base_model: str = DEFAULT_BASE_MODEL,
    adapter_dir: Path | str = DEFAULT_ADAPTER_DIR,
) -> ModelCardInputs:
    plan = _load_json(Path(sft_plan_path) if sft_plan_path else None)
    ds_man = _load_json(Path(dataset_manifest) if dataset_manifest else None)
    corp = _load_json(Path(corpus_manifest) if corpus_manifest else None)
    rag = _load_json(Path(rag_metrics) if rag_metrics else None)
    gen = _load_json(Path(gen_metrics) if gen_metrics else None)

    notes: list[str] = []
    report = (ds_man.get("report") or {}) if ds_man else {}
    n_train = report.get("n_train", plan.get("n_train"))
    n_val = report.get("n_val", plan.get("n_val"))

    hybrid = ((rag.get("backends") or {}).get("hybrid") or {}) if rag else {}
    mean_r = hybrid.get("mean_recall") or {}
    mean_n = hybrid.get("mean_ndcg") or {}
    recall = {str(k): _fmt(mean_r.get(str(k), mean_r.get(k))) for k in (1, 3, 5, 10)}
    ndcg = {str(k): _fmt(mean_n.get(str(k), mean_n.get(k))) for k in (1, 3, 5, 10)}

    agg = (gen.get("aggregate") or {}) if gen else {}
    base = agg.get("base") or {}
    adap = agg.get("adapter") or {}

    if not corp:
        notes.append("No corpus manifest; N left TBD.")
    if not rag:
        notes.append("No rag_metrics.json; retrieval cells TBD.")
    if not gen or gen.get("dry_run") is not False:
        notes.append("Generation metrics missing or dry_run!=false; gen cells may be TBD.")

    return ModelCardInputs(
        base_model=str(plan.get("model_name") or base_model),
        dataset_version=str(
            ds_man.get("dataset_version")
            or plan.get("dataset_version")
            or DEFAULT_DATASET
        ),
        seed=int(plan.get("seed") or 3407),
        adapter_dir=str(adapter_dir),
        n_train=_fmt(n_train) if n_train is not None else "TBD",
        n_val=_fmt(n_val) if n_val is not None else "TBD",
        effective_batch=_fmt(
            plan.get("effective_batch_size")
            or plan.get("effective_batch")
        ),
        n_chunks=_fmt(corp.get("n_chunks")) if corp else "TBD",
        n_documents=_fmt(corp.get("n_documents")) if corp else "TBD",
        corpus_version=str(corp.get("version") or "TBD") if corp else "TBD",
        embed_model=str(
            rag.get("dense_embed_model")
            or rag.get("embed_model")
            or "TBD"
        ),
        dense_backend=str(rag.get("dense_backend") or "TBD"),
        n_queries=_fmt(rag.get("n_queries")),
        recall=recall,
        ndcg=ndcg,
        base_citation=_fmt(base.get("citation_hit_rate", base.get("citation_hit"))),
        base_f1=_fmt(base.get("mean_token_f1", base.get("token_f1"))),
        adapter_citation=_fmt(adap.get("citation_hit_rate", adap.get("citation_hit"))),
        adapter_f1=_fmt(adap.get("mean_token_f1", adap.get("token_f1"))),
        grounded_accuracy_base=_fmt(
            base.get("grounded_answer_accuracy", base.get("grounded_accuracy"))
        ),
        grounded_accuracy_adapter=_fmt(
            adap.get("grounded_answer_accuracy", adap.get("grounded_accuracy"))
        ),
        gen_dry_run=_fmt(gen.get("dry_run")) if gen else "TBD",
        created_utc=_utc_now(),
        notes=notes,
    )


def render_model_card(inp: ModelCardInputs) -> str:
    r = inp.recall
    n = inp.ndcg
    return f"""---
language:
- en
license: apache-2.0
base_model: {inp.base_model}
library_name: peft
tags:
- qlora
- peft
- finance
- earnings-calls
- rag
- retrieval
pipeline_tag: text-generation
---

# llama32-3b-ecra-sft (Earnings Call Research Assistant)

QLoRA adapter on **{inp.base_model}** for grounded financial research Q&A / summarization
over public earnings-call style text. Paired in the GitHub repo with hybrid **BM25 +
dense** retrieval. Metrics below are copied from machine-written JSON only — never hand-invented.

- GitHub: https://github.com/nuwanda94/earnings-call-research-assistant
- RAG report: {GITHUB_REPORT}

## Intended use

Research-style questions over *retrieved public excerpts* (guidance, margins, segment
color, named risks). Prefer refusing figures that are not in context. **Not investment advice.**

## Training

| Item | Value |
|------|-------|
| Base | {inp.base_model} (4-bit QLoRA) |
| Seed | {inp.seed} |
| Dataset | {inp.dataset_version} |
| Train / val rows | {inp.n_train} / {inp.n_val} |
| Effective batch | {inp.effective_batch} |
| Adapter path | `{inp.adapter_dir}` |
| Card generated (UTC) | {inp.created_utc} |

## Retrieval corpus (when RAG was run)

| Item | Value |
|------|-------|
| N chunks | **{inp.n_chunks}** |
| Documents | **{inp.n_documents}** |
| Corpus version | {inp.corpus_version} |
| Embed model | {inp.embed_model} |
| Dense backend | {inp.dense_backend} |
| Eval queries | {inp.n_queries} |

## Measured metrics (hybrid retrieval)

| k | Recall | nDCG |
|--:|-------:|-----:|
| 1 | {r.get("1", "TBD")} | {n.get("1", "TBD")} |
| 3 | {r.get("3", "TBD")} | {n.get("3", "TBD")} |
| 5 | {r.get("5", "TBD")} | {n.get("5", "TBD")} |
| 10 | {r.get("10", "TBD")} | {n.get("10", "TBD")} |

### Grounded generation

| Side | citation-hit | token F1 | grounded accuracy | dry_run |
|------|-------------:|---------:|------------------:|--------|
| base | {inp.base_citation} | {inp.base_f1} | {inp.grounded_accuracy_base} | {inp.gen_dry_run} |
| adapter | {inp.adapter_citation} | {inp.adapter_f1} | {inp.grounded_accuracy_adapter} | {inp.gen_dry_run} |

## How to load

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "{inp.base_model}"
adapter = "<this-repo-id>"  # e.g. nuwanda94/llama32-3b-ecra-sft
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
```

Reproduce metrics from the GitHub repo:

```bash
python scripts/eval_retrieval.py --run
python scripts/eval_rag_generate.py --run --adapter-dir outputs/adapters/llama32-3b-ecra-sft
```

## Limitations

Public data only. Report measured **N** and query counts; fixture-scale runs are not
SEC coverage. Citation-hit ≠ numerical correctness. Token: `HF_TOKEN` / `huggingface-cli login` only — never commit secrets.

## Notes from card builder

{chr(10).join("- " + n for n in inp.notes) if inp.notes else "- (none)"}
"""


def write_model_card(
    out_path: Path | str = DEFAULT_OUT_README,
    *,
    inputs: ModelCardInputs | None = None,
    **collect_kwargs: Any,
) -> tuple[Path, ModelCardInputs]:
    inp = inputs or collect_card_inputs(**collect_kwargs)
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render_model_card(inp), encoding="utf-8")
    meta = dest.with_suffix(".inputs.json")
    meta.write_text(json.dumps(inp.to_dict(), indent=2) + "\n", encoding="utf-8")
    return dest, inp
