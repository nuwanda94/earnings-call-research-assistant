# Evaluations

Hold quantitative scripts, qualitative research-panel questions, and generated reports.
Large model outputs stay gitignored; commit rubrics and summaries only.

## Research panel (Phase 3)

[`research_panel.jsonl`](research_panel.jsonl) is 20 grounded prompts across
QA and summarization. Each row has a short public-style transcript excerpt,
citation needles, and a flag for questions that should be refused when the
number is not in context.

CPU dry-run (no weights):

```bash
python scripts/eval_research_panel.py
```

Writes [`reports/research_panel_comparison.json`](reports/research_panel_comparison.json)
with placeholder base/adapter columns. On Kaggle after an adapter exists:

```bash
python scripts/eval_research_panel.py --run --adapter-dir outputs/adapters/llama32-3b-ecra-sft
```

Library entry point: `earnings_call_research_assistant.eval.run_research_panel`.

## Quantitative metrics stub

Token-overlap (unigram F1 / Jaccard vs context) and citation-hit rate on the
side-by-side JSON. CPU only; dry-run placeholders score near zero, which is
expected until `--run` fills real generations.

```bash
python scripts/score_research_panel.py
```

Writes [`reports/research_panel_metrics.json`](reports/research_panel_metrics.json).
Library entry point: `earnings_call_research_assistant.eval.score_research_panel`.

## Report and first iteration

- [`reports/EVALUATION_REPORT.md`](reports/EVALUATION_REPORT.md) — how to read dry-run vs Kaggle scores.
- [`reports/ITERATION_NOTE_v0.1.md`](reports/ITERATION_NOTE_v0.1.md) — single next change: add an insufficient-context slice to SFT data before a longer train.

## RAG retrieval eval set (Phase 5.4)

[`rag_eval_set.jsonl`](rag_eval_set.jsonl) is the locked query set for Recall@k /
nDCG@k. Rebuild (CPU, seed `3407`):

```bash
python scripts/build_rag_eval_set.py
```

Rows: `query_id`, `query`, `gold_chunk_ids[]`, optional `gold_answer`, `source`.
Every gold ID must exist in `data/rag/corpus_v0.1.0`. Train-split pair IDs from
the SFT dataset are excluded. Fixture corpora produce a small N; grow the set
when the public chunk inventory scales to hundreds–thousands.

## Grounded generation metrics (Phase 5.6)

```bash
python scripts/eval_rag_generate.py
```

Writes [`reports/rag_generation_metrics.json`](reports/rag_generation_metrics.json):
per-query retrieved IDs, packed context size, base vs adapter answers, citation-hit
and token F1. Dry-run does not load weights. Quote `grounded_answer_accuracy` only
after a Kaggle `--run`.
