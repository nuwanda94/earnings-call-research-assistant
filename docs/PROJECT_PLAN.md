# Project Plan – Earnings Call Research Assistant

Principal Staff ML Engineer plan. Public data only. Optimized for Kaggle free GPUs and interview readiness.

## Success Criteria

- Fine-tuned 3B (primary) model outperforms base on research-style financial QA/summarization.
- Fully reproducible on Kaggle with public data only.
- High-quality grounded dataset (3k–6k examples) with lineage and multi-stage quality controls.
- Clean GitHub + HF adapter + short demo.
- Documented quantitative + qualitative evaluation.

## Phases

### Phase 0 – Foundation
- Repo structure, requirements, Kaggle template, configs, basic inference harness.

### Phase 1 – Robust Data Foundation
- Ingestion of public sources (earnings transcripts, FiQA, Finance-Alpaca sample).
- Chunking + proposition extraction.
- Grounded synthetic Q&A / summary generation.
- Multi-stage filtering: heuristic → exact/semantic dedup → LLM-as-judge.
- Diversity selection → final 3k–6k set + versioned splits + data card.

### Phase 2 – Training Pipeline
- Unsloth QLoRA SFT on 3B (then optional 8B).
- Externalized config, logging, checkpoints, reproducibility.

### Phase 3 – Evaluation & Iteration
- Quantitative metrics + qualitative research panel (15–25 questions).
- Side-by-side base vs fine-tuned; evaluation report; at least one iteration.

### Phase 4 – Packaging & Portfolio Polish
- Push adapter to HF Hub, Gradio/Streamlit demo, polished README, short video.

## Detailed Acceptance Criteria

See conversation history / keep this file updated as items are completed. Full criteria live in the original plan; automation advances one checklist item at a time.
