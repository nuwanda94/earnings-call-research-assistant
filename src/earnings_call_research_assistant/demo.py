"""CPU-safe Gradio demo stub for a few research-panel prompts.

Default path never loads weights and never trains. Pass ``load_model=True``
(or ``scripts/demo_gradio.py --run``) on a GPU box to wire ``InferenceHarness``.
Optional ``adapter_dir`` swaps the loaded snapshot for a local LoRA folder.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from earnings_call_research_assistant.eval.panel import DEFAULT_PANEL, PanelItem, load_panel

logger = logging.getLogger(__name__)

DRY_RUN_REPLY = (
    "[demo dry-run] InferenceHarness not loaded. "
    "Install extras with `pip install -e '.[demo]'` and relaunch with "
    "`python scripts/demo_gradio.py --run` (optional `--adapter-dir`). "
    "This stub never starts a training run."
)

EXAMPLE_IDS = ("p02", "p03", "p04", "p16")


def example_items(panel_path: Path | str | None = None) -> list[PanelItem]:
    items = load_panel(panel_path or DEFAULT_PANEL)
    wanted = {i: None for i in EXAMPLE_IDS}
    for item in items:
        if item.id in wanted:
            wanted[item.id] = item
    chosen = [item for item in wanted.values() if item is not None]
    return chosen or items[:4]


def example_choices(items: list[PanelItem] | None = None) -> list[str]:
    rows = items if items is not None else example_items()
    return [f"{item.id} · {item.ticker} · {item.theme}" for item in rows]


def _make_generate(
    *,
    load_model: bool,
    config_path: Path | str | None,
    adapter_dir: Path | str | None,
) -> Callable[[str], str]:
    if not load_model:
        return lambda _text: DRY_RUN_REPLY

    from earnings_call_research_assistant.config import load_config
    from earnings_call_research_assistant.inference import InferenceConfig, InferenceHarness

    app = load_config(config_path) if config_path else load_config()
    cfg = InferenceConfig.from_mapping(app.to_dict())
    if adapter_dir:
        cfg.model_name = str(adapter_dir)
        logger.info("Demo loading adapter snapshot from %s", adapter_dir)
    else:
        logger.info("Demo loading base model %s", cfg.model_name)
    harness = InferenceHarness.from_pretrained(cfg)
    return lambda text: harness.generate(text)


def answer_turn(
    example_label: str,
    custom_question: str,
    custom_context: str,
    *,
    items: list[PanelItem] | None = None,
    generate_fn: Callable[[str], str] | None = None,
) -> tuple[str, str]:
    """Return (user_text, model_or_stub_reply) for the Gradio callback."""
    catalog = items if items is not None else example_items()
    by_label = {f"{item.id} · {item.ticker} · {item.theme}": item for item in catalog}
    item = by_label.get(example_label, catalog[0] if catalog else None)

    if custom_question.strip():
        context = custom_context.strip() or (item.context if item else "")
        ticker = item.ticker if item else "CUSTOM"
        theme = item.theme if item else "custom"
        user_text = (
            f"Ticker: {ticker}\nTheme: {theme}\n\n"
            f"Context:\n{context}\n\nQuestion:\n{custom_question.strip()}"
        )
    elif item is not None:
        user_text = item.user_text()
    else:
        user_text = "No panel examples found."

    fn = generate_fn or (lambda _t: DRY_RUN_REPLY)
    return user_text, fn(user_text)


def build_demo(
    *,
    load_model: bool = False,
    config_path: Path | str | None = None,
    adapter_dir: Path | str | None = None,
    panel_path: Path | str | None = None,
    share: bool = False,
) -> Any:
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - optional extra
        raise SystemExit(
            "gradio is required for the demo. Install with: pip install -e '.[demo]'"
        ) from exc

    items = example_items(panel_path)
    labels = example_choices(items)
    generate_fn = _make_generate(
        load_model=load_model,
        config_path=config_path,
        adapter_dir=adapter_dir,
    )
    source = str(adapter_dir) if adapter_dir else "base model"
    mode = f"live InferenceHarness ({source})" if load_model else "CPU dry-run stub"

    def _on_submit(example_label: str, question: str, context: str) -> tuple[str, str]:
        return answer_turn(
            example_label,
            question,
            context,
            items=items,
            generate_fn=generate_fn,
        )

    with gr.Blocks(title="Earnings Call Research Assistant") as demo:
        gr.Markdown(
            "# Earnings Call Research Assistant\n"
            f"Mode: **{mode}**. Grounded prompts from "
            "`evals/research_panel.jsonl`. Does not train."
        )
        example = gr.Dropdown(choices=labels, value=labels[0] if labels else None, label="Panel example")
        question = gr.Textbox(label="Override question (optional)", lines=2)
        context = gr.Textbox(label="Override context (optional)", lines=4)
        submit = gr.Button("Generate", variant="primary")
        user_box = gr.Textbox(label="Packed user message", lines=8)
        reply_box = gr.Textbox(label="Assistant", lines=8)
        submit.click(_on_submit, [example, question, context], [user_box, reply_box])
        example.change(_on_submit, [example, question, context], [user_box, reply_box])
    demo.queue()
    return demo


def launch_demo(
    *,
    load_model: bool = False,
    config_path: Path | str | None = None,
    adapter_dir: Path | str | None = None,
    panel_path: Path | str | None = None,
    share: bool = False,
    server_name: str = "127.0.0.1",
    server_port: int | None = None,
) -> Any:
    demo = build_demo(
        load_model=load_model,
        config_path=config_path,
        adapter_dir=adapter_dir,
        panel_path=panel_path,
        share=share,
    )
    demo.launch(share=share, server_name=server_name, server_port=server_port)
    return demo
