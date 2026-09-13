# Notebooks

Kaggle-first workflows. Prefer short, restartable cells and public data only.

| Notebook | Purpose |
|----------|---------|
| `00_baseline_inference.ipynb` | Base-model smoke test |
| `01_data_and_sft.ipynb` | Data pipeline + optional short train |
| `02_full_train_and_hf_demo.ipynb` | Full train + side-by-side compare + **HF adapter** + **HF Space Gradio deploy** |
| `03_publish_static_space.ipynb` | Static Space helper |
| `03_rag_eval_and_publish.ipynb` | Phase 5 hybrid RAG: corpus → index → Recall@k/nDCG → optional SFT → generate eval → publish |
| `04_finetune_and_phase59.ipynb` | **End-to-end Phase 5.9**: data → QLoRA SFT → RAG metrics → fill report/README → optional HF + **GitHub push** |

`03_rag_eval_and_publish.ipynb` is **dry-run by default** (`RUN_GPU = False`). Flip flags only on a Kaggle T4. Do not treat fixture metric JSON as resume numbers.

`04_finetune_and_phase59.ipynb` is the recommended human path to close Phase 5.9. Defaults keep train/RAG/push **off**. Set:

- `RUN_TRAIN=True` for Unsloth QLoRA (seed 3407)
- `RUN_RAG=True` for sentence-transformer embeddings + real generate eval
- `PUBLISH_HF=True` + Kaggle secret `HF_TOKEN`
- `PUSH_GITHUB=True` + Kaggle secret `GITHUB_TOKEN` (classic PAT, `repo` scope)

It commits only small artifacts (`rag_metrics.json`, `rag_generation_metrics.json`, report, README, PROGRESS) — not adapter weights or full corpus.

## Hugging Face Space (permanent Gradio app)

Source folder: [`spaces/ecra-demo`](../spaces/ecra-demo) (`app.py`, `requirements.txt`, Space README).

```bash
export HF_TOKEN=hf_xxx   # write token; never commit
python scripts/publish_space.py              # dry-run plan
python scripts/publish_space.py --run        # create/upload Space
```

Default Space id: `nuwanda94/earnings-call-research-assistant`  
URL: https://huggingface.co/spaces/nuwanda94/earnings-call-research-assistant

After upload: **Space Settings → Hardware → T4** for live base vs adapter.  
Set Space variable `ADAPTER_REPO` to your adapter model id (default `nuwanda94/llama32-3b-ecra-sft`).

Notebook 02 does the same when `PUBLISH_SPACE=True`.

### Hugging Face token (notebook 02 / 03 RAG / 04 publish cell)

1. Write token: https://huggingface.co/settings/tokens  
2. Kaggle → **Add-ons → Secrets** → `HF_TOKEN`  
3. Never paste the token into cells or git  

### GitHub token (notebook 04 push cell)

1. Classic PAT with `repo` scope: https://github.com/settings/tokens  
2. Kaggle → **Add-ons → Secrets** → `GITHUB_TOKEN`  
3. Never print or commit the token  
