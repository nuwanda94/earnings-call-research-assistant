# Evaluation Report — Research Panel (Phase 3)

Date: 2026-09-03 03:00 IST  
Dataset version referenced: `ecra-sft-v0.1.0`  
Models compared: Llama-3.2-3B-Instruct base vs planned QLoRA adapter `outputs/adapters/llama32-3b-ecra-sft`  
Scoring: `scripts/score_research_panel.py` → `evals/reports/research_panel_metrics.json`

## What was measured

The qualitative panel is 20 grounded items in [`evals/research_panel.jsonl`](../research_panel.jsonl):

| Slice | Count | Intent |
|-------|------:|--------|
| QA | 16 | Extract a cited figure or contrast from the excerpt |
| Summarization | 4 | Compress only named drivers / risks / outlook |
| Refusal (`refuse_if_missing`) | 2 (`p03` FCF, `p16` EU AI Act date) | Must say the number/date is not in context |
| Citation needles present | 18 | Unigram / phrase hits such as `$4.12B`, `70 basis points` |

Metrics (CPU-only, no weights):

- Token overlap of the model text vs the *context* (precision / recall / F1 / Jaccard). This is a grounding proxy, not a gold-answer BLEU.
- Citation-hit rate: fraction of `must_cite` needles that appear in the generation.
- Refusal accuracy: on the two missing-number items, does the text contain a refusal cue (`not in the context`, `not stated`, `insufficient`, `no go-live`, …).

## Dry-run numbers (CI / laptop, no GPU)

`scripts/eval_research_panel.py` without `--run` writes placeholder columns:

- base: `[dry-run] base model not loaded; run on Kaggle with --run`
- adapter: `[dry-run] adapter not loaded; pass --adapter-dir after QLoRA --run`

Those strings share almost no tokens with the transcript excerpts and contain none of the citation needles or refusal cues. Expected aggregate from `score_comparison`:

| Side | token F1 vs context | citation-hit rate | refusal accuracy |
|------|--------------------:|------------------:|-----------------:|
| base (placeholder) | ~0.00 | 0.00 (18 items) | 0.00 (2 items) |
| adapter (placeholder) | ~0.00 | 0.00 | 0.00 |

`dry_run: true` in the comparison JSON is the authoritative flag. **Do not treat placeholder equality as “base ties adapter.”** It only proves the eval harness is wired.

## Optional Kaggle `--run` (not executed in this automation)

On a T4, after `scripts/train_sft.py --run` has written an adapter:

```bash
python scripts/eval_research_panel.py --run --adapter-dir outputs/adapters/llama32-3b-ecra-sft
python scripts/score_research_panel.py
```

Interpret the *real* JSON as follows:

1. **Citation-hit rate is the primary portfolio metric.** A successful 3B SFT should lift adapter hits on guidance, margin, and segment items (`p02`, `p04`, `p05`, `p06`, `p11`, `p18`) without inventing figures that are absent from context.
2. **Token F1 vs context will rise for both models** once generations quote the excerpt. Prefer a *gap* (adapter F1 − base F1) over the absolute level; a high F1 with zero citations can still be a paraphrase that dropped the numbers.
3. **Refusal accuracy must not collapse.** If the adapter answers `p03` with a made-up FCF or invents an EU AI Act quarter on `p16`, that is a data-quality failure, not a reason to train longer.
4. **Qualitative spot-check** `p20` (grounding-rule) and `p08` (named risks only). These catch generic “strong demand / macro risk” language that the base Instruct model likes and that grounded SFT should suppress.

Until that Kaggle JSON exists, the only honest claim is: the panel, comparison writer, and scorer are reproducible; model ranking is pending GPU generations.

## Limitations

- Unigram overlap rewards copying the excerpt; it does not score numerical consistency or unit errors.
- Citation hits are substring matches (`$4.20` hits `$4.20 billion`).
- Two refusal items is too small for a confidence interval; they are a canary, not a benchmark.
- Adapter path is not loaded unless `--adapter-dir` points at a trained folder. Missing adapters write an explicit `[adapter missing]` string and will still score near zero.

## Decision

Phase 3 instrumentation is complete. The next change should not be “train more steps on the same mix.” Dry-run scores cannot justify a longer run. See [`ITERATION_NOTE_v0.1.md`](ITERATION_NOTE_v0.1.md) for the single data change to make before the first scored Kaggle comparison.
