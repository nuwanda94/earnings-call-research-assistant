# Portfolio demo video — script and recording notes

Target length: **3:00–4:30**. No binary recording lives in this repo (keep `.mp4` / `.mov` off Git; host on YouTube, Loom, or a private Drive link and paste the URL in the README when you have it).

This is a talking-head + screen-share walkthrough of **baseline → grounded data → adapter demo**. Every GPU step shown on screen can be a **CPU dry-run** plus a still of a previously captured Kaggle cell if you do not have a live T4.

## What interviewers should take away

1. Public data only; no paid transcript firehose.
2. Grounded generation + multi-stage filtering, not raw web scrape → SFT.
3. Dry-run-by-default CLIs so a laptop clone is safe.
4. Base vs adapter is judged on a fixed research panel (citation + refusal), not vibes.

## Shot list (clock)

| Time | Shot | On screen |
|------|------|-----------|
| 0:00–0:25 | Title + one-liner | Repo README header + this table of phases |
| 0:25–1:05 | Baseline | `notebooks/00_baseline_inference.ipynb` or `InferenceHarness` smoke prompt |
| 1:05–2:10 | Grounded data | Terminal: ingest → chunk → generate → filter → select; flash `docs/DATA_CARD.md` |
| 2:10–2:50 | Train / eval (dry) | `outputs/sft_plan.json` + `evals/reports/EVALUATION_REPORT.md` |
| 2:50–4:00 | Adapter demo | Gradio stub (`scripts/demo_gradio.py`) — guidance Q + missing-context refusal |
| 4:00–4:20 | Close | Link to HF adapter plan + seed `3407` |

## Spoken script (tight)

### 0:00 Hook

> This is the Earnings Call Research Assistant — a 3B QLoRA portfolio piece for research-tooling roles. The claim is simple: a small instruct model, fine-tuned on *grounded* public earnings text, should cite the chunk it was given and refuse when free cash flow is not in context.

### 0:25 Baseline

> Phase 0 is a clone-and-run baseline. On Kaggle T4 I load `unsloth/Llama-3.2-3B-Instruct` in 4-bit through `InferenceHarness`. Three research-style prompts, no training. That output is the *before* snapshot for the panel.
>
> Locally the same path is `load_config()` plus `InferenceHarness.from_pretrained`. Every later GPU script is dry-run unless you pass `--run`.

Show one prompt, e.g. prepared remarks vs Q&A. Do not scroll a wall of tokens.

### 1:05 Grounded data

> Phase 1 is the actual product risk. I do not dump Finance-Alpaca into SFT and hope. The pipeline is:
>
> 1. Ingest public catalogs — sample transcripts, FiQA, Finance-Alpaca — fixtures offline, `--download` only when I opt in.
> 2. Section-aware chunks and heuristic propositions so a later answer can point at a span.
> 3. Template-grounded Q&A and summaries that *must* cite chunk + proposition.
> 4. Filter: length and citation heuristics, exact and Jaccard near-dup drop, optional LLM-as-judge hook.
> 5. Diversity select into versioned `ecra-sft-v0.1.0` splits and a data card.

Run (or paste pre-captured output of):

```bash
python scripts/ingest_public_sources.py --catalog-only
python scripts/chunk_propositions.py --out data/processed/chunks.jsonl
python scripts/generate_grounded_pairs.py --out data/processed/grounded_pairs.jsonl
python scripts/filter_grounded_pairs.py --out data/processed/filtered_pairs.jsonl
python scripts/select_dataset.py --out-dir data/processed/ecra-sft-v0.1.0
```

Hold `docs/DATA_CARD.md` for three seconds. Say the target band is 3k–6k high-signal rows, not a million noisy ones.

### 2:10 Train plan and eval

> Training is Unsloth QLoRA, seed **3407**, adapter dir `outputs/adapters/llama32-3b-ecra-sft`. On this laptop I only run the dry-run so you see the formatted chat examples and `outputs/sft_plan.json`. The `--run --max-steps 20` smoke lives on Kaggle.
>
> Eval is a 20-item research panel: guidance, margin summary, citation, and deliberate unanswerable items. Metrics are token-overlap against the provided context plus citation-hit and refusal accuracy — not open-web fact-checking.

Flash `evals/reports/EVALUATION_REPORT.md` and `ITERATION_NOTE_v0.1.md` (insufficient-context mix).

### 2:50 Gradio adapter demo

> Same four prompts as the demo dropdown. Watch two:
>
> - *Guidance* — answer should stay inside the chunk and name the source span.
> - *Missing FCF* — the model should refuse instead of inventing a cash-flow number.

```bash
pip install -e ".[demo]"
python scripts/demo_gradio.py          # CPU stub, no weights
# GPU box only:
# python scripts/demo_gradio.py --run --adapter-dir outputs/adapters/llama32-3b-ecra-sft
```

If you only have the CPU stub, narrate: *this screen is the interface; the adapter generations are in the panel JSON from the Kaggle `--run`.* Do not fake live tokens.

### 4:00 Close

> Adapter publish is `scripts/publish_adapter.py` — dry-run writes `outputs/publish_plan.json`, token only from `HF_TOKEN` or `huggingface-cli login`, never from the repo.
>
> Repo: `github.com/nuwanda94/earnings-call-research-assistant`. Repro notes in `docs/REPRODUCIBILITY.md`. That is the loop: baseline, grounded set, small adapter, panel evidence.

## Recording notes

- Resolution 1920×1080, 16:9. Terminal font ≥16 pt. Dark theme is fine; avoid low-contrast grey on grey.
- Do not record API keys, `HF_TOKEN`, or a logged-in Hugging Face settings page.
- Do not commit the video. Add `*.mp4` / `*.mov` / `*.webm` in `.gitignore` if they are not already covered.
- Prefer a single take with jump cuts between phases over a slide deck.
- If Kaggle is slow, screenshot the completed baseline cell and cut to it; say it is a captured T4 run.
- Captions: burn-in or YouTube auto-captions reviewed once (finance jargon: FCF, Q&A, QLoRA).
- Alternate 90-second cut: hook + one data-card sentence + Gradio refusal only. Use that for a resume link; keep the 4-minute cut for interviews.

## Checklist before you hit record

- [ ] `python scripts/ingest_public_sources.py --catalog-only` prints the catalog
- [ ] `python scripts/train_sft.py` writes `outputs/sft_plan.json`
- [ ] `python scripts/demo_gradio.py` serves locally
- [ ] Browser zoom 125% on README and data card
- [ ] No secrets in the shell history visible on screen

## After recording

Paste the public URL under **Status** in the root README (one line, no embed). Leave this file as the canonical script so a re-record stays consistent.
