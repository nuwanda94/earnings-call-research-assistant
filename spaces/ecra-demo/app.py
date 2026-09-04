"""Hugging Face Space entrypoint: side-by-side base vs QLoRA adapter Gradio demo.

Designed for Spaces with a GPU (e.g. T4). On CPU-only hardware the UI still
loads and explains how to attach hardware / secrets; it does not train.
"""

from __future__ import annotations

import gc
import logging
import os
from functools import lru_cache
from typing import Any

import gradio as gr

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("ecra-space")

BASE_MODEL = os.environ.get("BASE_MODEL", "unsloth/Llama-3.2-3B-Instruct")
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "nuwanda94/llama32-3b-ecra-sft")
SIDE_BY_SIDE = os.environ.get("SIDE_BY_SIDE", "1").strip() not in {"0", "false", "False"}
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
SYSTEM_PROMPT = (
    "You are a financial research assistant. Answer clearly and conservatively. "
    "If the question cannot be answered from the provided context, say so. "
    "Do not invent numbers."
)

# Lightweight built-in panel (avoids shipping the full repo into the Space).
PANEL: list[dict[str, str]] = [
    {
        "id": "p02",
        "label": "p02 · guidance tone",
        "user": (
            "Ticker: EXAMPLE\nTheme: guidance\n\n"
            "Context:\nManagement reiterated full-year revenue guidance and noted "
            "conservative assumptions on enterprise demand in the second half.\n\n"
            "Question:\nWhat did management say about full-year guidance and demand?"
        ),
    },
    {
        "id": "p03",
        "label": "p03 · missing FCF refusal",
        "user": (
            "Ticker: EXAMPLE\nTheme: free cash flow\n\n"
            "Context:\nThe prepared remarks discussed operating margin expansion and "
            "headcount plans. Free cash flow was not quantified on this call.\n\n"
            "Question:\nWhat was free cash flow for the quarter?"
        ),
    },
    {
        "id": "p04",
        "label": "p04 · margin summary",
        "user": (
            "Ticker: EXAMPLE\nTheme: margins\n\n"
            "Context:\nGross margin improved 80 bps year over year on mix; operating "
            "margin was flat as the company reinvested in go-to-market.\n\n"
            "Question:\nSummarize the margin commentary."
        ),
    },
    {
        "id": "p16",
        "label": "p16 · unanswerable date",
        "user": (
            "Ticker: EXAMPLE\nTheme: calendar\n\n"
            "Context:\nThe call covered Q2 results and reiterated the product roadmap. "
            "No date was given for the next investor day.\n\n"
            "Question:\nWhen is the next investor day?"
        ),
    },
]


def _cuda_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _release() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _build_messages(user_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def _generate_with(model: Any, tokenizer: Any, user_text: str) -> str:
    import torch

    messages = _build_messages(user_text)
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    encoded = encoded.to(device)
    with torch.inference_mode():
        out = model.generate(
            encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    new_tokens = out[0, encoded.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _load_base():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tok


def _load_adapter():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_REPO)
    model.eval()
    return model, tok


def compare(user_text: str) -> tuple[str, str]:
    if not user_text.strip():
        return "", ""
    if not _cuda_ok():
        msg = (
            "[Space has no GPU] Attach **T4** (or similar) hardware in Space settings, "
            f"set ADAPTER_REPO={ADAPTER_REPO}, and restart. "
            "CPU builds stay UI-only so the Space does not OOM."
        )
        return msg, msg

    base_reply = ""
    adapter_reply = ""
    try:
        logger.info("Loading base %s", BASE_MODEL)
        model, tok = _load_base()
        base_reply = _generate_with(model, tok, user_text)
    except Exception as exc:
        base_reply = f"[base load/generate failed] {type(exc).__name__}: {exc}"
    finally:
        try:
            del model, tok
        except Exception:
            pass
        _release()

    try:
        logger.info("Loading adapter %s on %s", ADAPTER_REPO, BASE_MODEL)
        model, tok = _load_adapter()
        adapter_reply = _generate_with(model, tok, user_text)
    except Exception as exc:
        adapter_reply = (
            f"[adapter load/generate failed] {type(exc).__name__}: {exc}\n"
            f"Publish the adapter first: python scripts/publish_adapter.py --run"
        )
    finally:
        try:
            del model, tok
        except Exception:
            pass
        _release()

    return base_reply, adapter_reply


def pack_from_dropdown(label: str, custom_q: str, custom_ctx: str) -> str:
    if custom_q.strip():
        ctx = custom_ctx.strip() or "(no extra context)"
        return f"Context:\n{ctx}\n\nQuestion:\n{custom_q.strip()}"
    for row in PANEL:
        if row["label"] == label:
            return row["user"]
    return PANEL[0]["user"]


def on_submit(label: str, custom_q: str, custom_ctx: str) -> tuple[str, str, str]:
    user = pack_from_dropdown(label, custom_q, custom_ctx)
    if SIDE_BY_SIDE:
        base_r, ad_r = compare(user)
        return user, base_r, ad_r
    # Adapter-only path still fills both boxes for a stable UI schema.
    base_r, ad_r = compare(user)
    return user, base_r, ad_r


def build() -> gr.Blocks:
    labels = [p["label"] for p in PANEL]
    hw = "GPU" if _cuda_ok() else "CPU (no live weights)"
    with gr.Blocks(title="Earnings Call Research Assistant") as demo:
        gr.Markdown(
            f"# Earnings Call Research Assistant\n"
            f"**Hardware:** {hw} · **Base:** `{BASE_MODEL}` · **Adapter:** `{ADAPTER_REPO}`\n\n"
            "Side-by-side **before (base)** vs **after (fine-tuned adapter)**. "
            "Does not train. Public demo prompts only."
        )
        example = gr.Dropdown(choices=labels, value=labels[0], label="Panel example")
        question = gr.Textbox(label="Override question (optional)", lines=2)
        context = gr.Textbox(label="Override context (optional)", lines=4)
        btn = gr.Button("Compare base vs fine-tuned", variant="primary")
        user_box = gr.Textbox(label="Packed user message", lines=8)
        with gr.Row():
            base_box = gr.Textbox(label="Base model (before)", lines=12)
            adapter_box = gr.Textbox(label="Fine-tuned adapter (after)", lines=12)
        btn.click(on_submit, [example, question, context], [user_box, base_box, adapter_box])
        example.change(on_submit, [example, question, context], [user_box, base_box, adapter_box])
    return demo


demo = build()

if __name__ == "__main__":
    demo.queue().launch()
