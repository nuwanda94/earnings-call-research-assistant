# Reproducibility — Unsloth QLoRA SFT

Phase 2 trains a LoRA adapter on the versioned grounded splits. This note is the single source of truth for seed, output paths, and the Kaggle dry-run vs `--run` path. It does **not** launch a long GPU train; that stays a user/Kaggle action.

## Fixed knobs

| Knob | Value | Where |
|------|--------|--------|
| Seed | `3407` | `training.seed` in YAML; `SFTRunConfig.seed`; Unsloth `random_state`; `TrainingArguments.seed` |
| Trainer logs / checkpoints | `outputs/` | `training.output_dir` |
| Dry-run plan | `outputs/sft_plan.json` | Written by every `train_sft.py` invocation |
| 3B adapter | `outputs/adapters/llama32-3b-ecra-sft` | `configs/default.yaml` |
| 8B adapter | `outputs/adapters/llama31-8b-ecra-sft` | `configs/llama32-8b.yaml` |
| Dataset | `data/processed/ecra-sft-v0.1.0` | `train.jsonl` / `val.jsonl` / `test.jsonl` + `manifest.json` |
| Primary model | `unsloth/Llama-3.2-3B-Instruct` | 4-bit, `max_seq_length` 2048 |
| Optional 8B | `unsloth/Meta-Llama-3.1-8B-Instruct` | batch 1 / grad-accum 8 on a T4 |

`outputs/` and adapter weights are gitignored. Keep the adapter directory name stable so Phase 3 eval and HF upload can hard-code the same path.

## What the seed covers

`3407` is Unsloth's conventional demo seed. It is applied in three places so a T4 rerun is as close as the stack allows:

1. `FastLanguageModel.get_peft_model(..., random_state=3407)` — LoRA init.
2. `TrainingArguments(seed=3407)` — data order / dropout (dropout is 0.0 by default).
3. YAML `training.seed` so CLI dry-runs print the same number in `sft_plan.json`.

Non-determinism that still remains: CUDA kernels, Unsloth packing (off), and any change to the JSONL row order. Do not reshuffle splits; `select_dataset.py` writes hash-stable files.

## Dry-run vs `--run`

`scripts/train_sft.py` defaults to **dry-run**. That path:

- Loads YAML (`configs/default.yaml` or `--config`).
- Resolves dataset / adapter / output dirs (CLI flags win over YAML).
- Optionally reads existing splits if present; missing train JSONL is allowed in dry-run.
- Formats two preview chat examples.
- Writes `outputs/sft_plan.json` with `dry_run=true`, `seed`, `adapter_dir`, row counts.
- Does **not** download weights, import Unsloth, or call `trainer.train()`.

`--run` is the only GPU path. It loads Unsloth, trains, then `save_pretrained`s the adapter to `adapter_dir`.

```bash
# CPU / CI / hourly automation — safe, no GPU
python scripts/train_sft.py
python scripts/train_sft.py --config configs/llama32-8b.yaml

# Inspect the plan
cat outputs/sft_plan.json
```

Expect `seed` 3407 and `adapter_dir` under `outputs/adapters/` in that JSON.

## Kaggle how-to (user-run, not automation)

Use a **GPU T4** notebook. Do not rely on the hourly bot for `--run`.

1. Clone the repo in the first cell and `cd` into it.
2. Build (or attach) the versioned splits if they are not already on disk:

   ```bash
   python scripts/ingest_public_sources.py --out data/raw/public_sample.jsonl
   python scripts/chunk_propositions.py --out data/processed/chunks.jsonl
   python scripts/generate_grounded_pairs.py --out data/processed/grounded_pairs.jsonl
   python scripts/filter_grounded_pairs.py --out data/processed/filtered_pairs.jsonl
   python scripts/select_dataset.py --out-dir data/processed/ecra-sft-v0.1.0
   ```

3. Install Unsloth + TRL in the notebook (same extras as `notebooks/00_baseline_inference.ipynb`).
4. Confirm the plan on CPU first:

   ```bash
   python scripts/train_sft.py --dataset-dir data/processed/ecra-sft-v0.1.0
   ```

5. Smoke train (short, optional) then a full epoch when you have time:

   ```bash
   python scripts/train_sft.py --run --max-steps 20
   python scripts/train_sft.py --run
   ```

6. 8B T4 path (optional; smaller micro-batch):

   ```bash
   python scripts/train_sft.py --config configs/llama32-8b.yaml --run --max-steps 20
   ```

`--model-name` overrides only `model.name`. Prefer the 8B YAML so batch / `adapter_dir` stay T4-safe.

After `--run`, the adapter lives at:

- 3B: `outputs/adapters/llama32-3b-ecra-sft/`
- 8B: `outputs/adapters/llama31-8b-ecra-sft/`

Copy that folder off the Kaggle working directory before the session dies. Phase 3 eval loads it next to the base model from `notebooks/00_baseline_inference.ipynb`.

## Checklist before claiming a run is reproducible

- [ ] `sft_plan.json` shows `seed: 3407`
- [ ] Dataset dir is `ecra-sft-v0.1.0` (or a new versioned name + updated data card)
- [ ] Adapter path matches the YAML for that model size
- [ ] `--run` was used only on GPU; dry-run was used to verify the plan first
- [ ] No private filings or non-public transcripts entered the JSONL
