---
title: Earnings Call Research Assistant
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
suggested_hardware: t4-small
tags:
  - finance
  - earnings
  - qlora
  - llm
---

# Earnings Call Research Assistant — demo Space

Side-by-side **base** vs **fine-tuned (QLoRA adapter)** answers on grounded research-panel prompts.

| Setting | Value |
|---------|--------|
| Base model | `unsloth/Llama-3.2-3B-Instruct` (or `BASE_MODEL` env) |
| Adapter repo | `nuwanda94/llama32-3b-ecra-sft` (or `ADAPTER_REPO` env) |
| Mode | Side-by-side when a GPU is available; CPU shows guidance text |

## Space secrets / variables

- **Secret** `HF_TOKEN` — only needed if the adapter or base model is gated/private
- **Variable** `ADAPTER_REPO` — Hub model id of the LoRA adapter (default above)
- **Variable** `BASE_MODEL` — base instruct model id
- **Variable** `SIDE_BY_SIDE` — `1` (default) or `0` for adapter-only UI

## Source

Built from [nuwanda94/earnings-call-research-assistant](https://github.com/nuwanda94/earnings-call-research-assistant).
Publish this folder with:

```bash
python scripts/publish_space.py --run
```
