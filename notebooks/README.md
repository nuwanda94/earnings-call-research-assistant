# Notebooks

Kaggle-first workflows. Prefer short, restartable cells and public data only.

| Notebook | Purpose |
|----------|---------|
| `00_baseline_inference.ipynb` | Base-model smoke test (Unsloth 4-bit, Llama-3.2-3B-Instruct) |
| `01_data_and_sft.ipynb` | Phase 1 data pipeline + Phase 2 dry-run / optional short train |
| `02_full_train_and_hf_demo.ipynb` | **Full epoch train** + **Hugging Face Hub publish** + optional Gradio share |

## Quick start on Kaggle

1. New notebook, **GPU (T4)** enabled.
2. First cell (or let notebook 02 clone itself):

```bash
%cd /kaggle/working
!git clone --depth 1 https://github.com/nuwanda94/earnings-call-research-assistant.git
%cd earnings-call-research-assistant
```

3. Choose a notebook:

- **Baseline only:** `00_baseline_inference.ipynb`
- **Data + smoke train:** `01_data_and_sft.ipynb` (`RUN_TRAIN=True`, `MAX_STEPS=20`)
- **Full train + HF upload:** `02_full_train_and_hf_demo.ipynb`

### Hugging Face token (notebook 02)

1. Create a write token at https://huggingface.co/settings/tokens  
2. Kaggle → **Add-ons → Secrets** → add `HF_TOKEN`  
3. Never paste the token into notebook cells or git  

Default upload target: `nuwanda94/llama32-3b-ecra-sft` (override `HF_REPO_ID`).

Package imports must use `earnings_call_research_assistant` after `src/` is on `sys.path` (not `from src....`).
