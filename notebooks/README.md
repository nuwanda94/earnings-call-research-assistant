# Notebooks

Kaggle-first workflows. Prefer short, restartable cells and public data only.

| Notebook | Purpose |
|----------|---------|
| `00_baseline_inference.ipynb` | Base-model smoke test |
| `01_data_and_sft.ipynb` | Data pipeline + optional short train |
| `02_full_train_and_hf_demo.ipynb` | Full train + side-by-side compare + **HF adapter** + **HF Space Gradio deploy** |

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

### Hugging Face token (notebook 02)

1. Write token: https://huggingface.co/settings/tokens  
2. Kaggle → **Add-ons → Secrets** → `HF_TOKEN`  
3. Never paste the token into cells or git  
