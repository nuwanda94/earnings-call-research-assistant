# Scripts

CLI utilities for ingestion, filtering, training launch, and report generation.
Prefer thin wrappers around `earnings_call_research_assistant` so notebooks and Kaggle stay in sync.

## Public-source catalog (Phase 1)

```bash
python scripts/ingest_public_sources.py --catalog-only
python scripts/ingest_public_sources.py --out data/raw/public_sample.jsonl
```

Default mode uses tiny offline fixtures and does **not** download corpora.
Pass `--download` only when you explicitly want a streamed Hugging Face sample.

## Chunking + propositions

```bash
python scripts/chunk_propositions.py --out data/processed/chunks.jsonl
```

Splits ingested text into section-aware sentence windows and extracts short
heuristic propositions (numbers, metrics, guidance). No LLM required.

## Grounded synthetic pairs

```bash
python scripts/generate_grounded_pairs.py --out data/processed/grounded_pairs.jsonl
```

Turns chunks + propositions into citation-grounded Q&A and summary instruction
pairs via offline templates. Optional LLM rewrite is a notebook hook only
(`generate_pairs(..., llm=..., config=GenerateConfig(use_llm=True))`).

## Multi-stage filtering

```bash
python scripts/filter_grounded_pairs.py --out data/processed/filtered_pairs.jsonl
```

Applies heuristic length/citation checks, exact + near-duplicate drop, and an
optional LLM-as-judge stage. `--use-llm-judge` runs a deterministic proxy so
automation never bills GPU; pass a real `judge` callable from a Kaggle notebook.

## Diversity selection + versioned splits

```bash
python scripts/select_dataset.py --out-dir data/processed/ecra-sft-v0.1.0
```

Applies per-source/task caps and greedy Jaccard diversity, then writes
hash-stable `train.jsonl` / `val.jsonl` / `test.jsonl` plus `manifest.json`.
See [`docs/DATA_CARD.md`](../docs/DATA_CARD.md).

## QLoRA SFT (Phase 2)

Seed is always `3407`. Adapters land under `outputs/adapters/`.
Full dry-run vs `--run` Kaggle steps: [`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md).

```bash
python scripts/train_sft.py
python scripts/train_sft.py --dataset-dir data/processed/ecra-sft-v0.1.0
```

Dry-run by default: reads `configs/default.yaml` (Llama-3.2-3B-Instruct),
formats chat examples from the versioned splits, and writes `outputs/sft_plan.json`.
No weights load. Adapter path: `outputs/adapters/llama32-3b-ecra-sft`.

On Kaggle T4, after Unsloth is installed:

```bash
python scripts/train_sft.py --run --max-steps 20
```

### Optional 8B path (same script)

Llama 3.2 has no 8B checkpoint. Use Llama-3.1-8B-Instruct with a smaller
micro-batch so a single T4 stays in 4-bit VRAM (effective batch still 8):

```bash
python scripts/train_sft.py --config configs/llama32-8b.yaml
python scripts/train_sft.py --config configs/llama32-8b.yaml --run --max-steps 20
```

`--model-name` overrides only `model.name` and keeps the rest of the YAML:

```bash
python scripts/train_sft.py --model-name unsloth/Meta-Llama-3.1-8B-Instruct
```

Prefer the dedicated YAML for 8B so batch size is 1 and `adapter_dir` is
`outputs/adapters/llama31-8b-ecra-sft`. Full-epoch knobs live under `training.*`
(seed `3407`).

## Qualitative research panel (Phase 3)

```bash
python scripts/eval_research_panel.py
python scripts/eval_research_panel.py --run --adapter-dir outputs/adapters/llama32-3b-ecra-sft
```

Dry-run loads [`evals/research_panel.jsonl`](../evals/research_panel.jsonl)
(20 grounded QA/summarization prompts) and writes placeholder base vs adapter
rows to `evals/reports/research_panel_comparison.json`. `--run` is the Kaggle
inference path and does not train.

## Quantitative metrics stub (Phase 3)

```bash
python scripts/score_research_panel.py
```

Reads the side-by-side JSON (writes a dry-run comparison first if missing) and
prints token-overlap F1/Jaccard vs context plus citation-hit and refusal rates
for base vs adapter. Output: `evals/reports/research_panel_metrics.json`.

Interpretation and the first training/data iteration:
[`evals/reports/EVALUATION_REPORT.md`](../evals/reports/EVALUATION_REPORT.md),
[`evals/reports/ITERATION_NOTE_v0.1.md`](../evals/reports/ITERATION_NOTE_v0.1.md).

## Gradio demo stub (Phase 4)

```bash
pip install -e ".[demo]"
python scripts/demo_gradio.py
```

CPU-safe by default: dropdown of four research-panel prompts (guidance, missing
FCF refusal, margin summary, unanswerable date) plus optional question/context
overrides. Replies are a dry-run placeholder — no weights, no train.

On a GPU box:

```bash
python scripts/demo_gradio.py --run
python scripts/demo_gradio.py --run --adapter-dir outputs/adapters/llama32-3b-ecra-sft
```

`--run` loads `InferenceHarness` from `configs/default.yaml` (or the adapter
directory). Implementation: `src/earnings_call_research_assistant/demo.py`.
