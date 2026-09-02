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
