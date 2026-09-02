# Iteration Note v0.1 — Refusal-and-citation mix before first scored train

Date: 2026-09-03 03:00 IST  
Trigger: Phase 3 dry-run metrics (placeholders only; no Kaggle adapter scores yet)

## Observation

Dry-run base and adapter both score ~0 token F1, 0 citation-hit rate, and 0 refusal accuracy. That is expected. What the *panel design* already flags, independent of GPU output:

- 18 / 20 items reward quoting a number that is in the excerpt.
- Only 2 / 20 items (`p03`, `p16`) reward refusing a missing figure.
- SFT v0.1.0 is built from template-grounded pairs that almost always have a citeable proposition. The filter drops pairs that fail citation checks, so the train split is biased toward “always answer with a number.”

If we QLoRA on that mix and then score the panel, the likely failure mode is **over-answering**: higher citation hits on easy items, invented FCF / go-live dates on refusal items. That would look like a metric win and a research-assistant loss.

## Single next change (data, not more steps)

**Add an explicit insufficient-context slice to `ecra-sft-v0.1.0` (or cut `v0.1.1`) before any multi-epoch `--run`.**

Concretely, in the grounded generator / selector:

1. For a subset of chunks, emit a question whose answer is *not* in the attached propositions (e.g. ask for FCF when only OCF and capex appear; ask for a go-live date when counsel only says “will update”).
2. Target answer template: state what *is* in the excerpt, then refuse the missing field with a cue the scorer already recognizes (`not in the context`, `not stated`, `insufficient`, `no go-live`).
3. Keep these pairs after filtering: citation needles may be empty when `refuse_if_missing` is true; do not drop them in the heuristic “must cite” stage.
4. Cap the slice at roughly 8–12% of train (enough that 3B QLoRA sees the pattern; not enough to drown guidance QA).
5. Re-run `scripts/select_dataset.py` to a new versioned dir + data-card bump. Do not change seed `3407` or LoRA rank yet.

Only after that mix exists should Kaggle do `train_sft.py --run --max-steps 20` and `eval_research_panel.py --run` so base vs adapter metrics are interpretable on both citation and refusal.

## Explicitly not this iteration

- Jumping to the 8B YAML
- Raising `max_steps` / epochs on v0.1.0
- Swapping the overlap metric for an LLM judge
- Publishing an adapter to the Hub

Those wait on a scored comparison that includes the refusal canaries.
