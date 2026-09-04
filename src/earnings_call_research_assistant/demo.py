"""Gradio demo for research-panel prompts (single model or base vs adapter).

Default path never loads weights and never trains. Pass ``load_model=True``
(or ``scripts/demo_gradio.py --run``) on a GPU box to wire ``InferenceHarness``.
Optional ``adapter_dir`` swaps the loaded snapshot for a local LoRA folder.
Pass ``side_by_side=True`` with an adapter to show base and fine-tuned answers.
"""

from __future__ import annotations

import gc
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

DRY_RUN_SIDE_BY_SIDE = (
    "[demo dry-run] Base vs adapter comparison requires GPU + `--run`. "
    "Example: `python scripts/demo_gradio.py --run --side-by-side "
    "--adapter-dir outputs/adapters/llama32-3b-ecra-sft`"
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


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


def _load_harness(
    *,
    config_path: Path | str | None,
    model_name: str | None = None,
) -> Any:
    from earnings_call_research_assistant.config import load_config
    from earnings_call_research_assistant.inference import InferenceConfig, InferenceHarness

    app = load_config(config_path) if config_path else load_config()
    cfg = InferenceConfig.from_mapping(app.to_dict())
    if model_name:
        return InferenceHarness.from_pretrained(cfg, model_name=model_name)
    return InferenceHarness.from_pretrained(cfg)


def _make_generate(
    *,
    load_model: bool,
    config_path: Path | str | None,
    adapter_dir: Path | str | None,
) -> Callable[[str], str]:
    if not load_model:
        return lambda _text: DRY_RUN_REPLY

    adapter = str(adapter_dir) if adapter_dir else None
    if adapter:
        logger.info("Demo loading adapter snapshot from %s", adapter)
        harness = _load_harness(config_path=config_path, model_name=adapter)
    else:
        logger.info("Demo loading base model")
        harness = _load_harness(config_path=config_path)
    return lambda text: harness.generate(text)


def _make_side_by_side_generate(
    *,
    load_model: bool,
    config_path: Path | str | None,
    adapter_dir: Path | str | None,
) -> Callable[[str], tuple[str, str]]:
    """Return (base_reply, adapter_reply). Loads models sequentially to fit T4."""

    if not load_model:
        return lambda _text: (DRY_RUN_SIDE_BY_SIDE, DRY_RUN_SIDE_BY_SIDE)

    adapter_path = Path(adapter_dir) if adapter_dir else None
    has_adapter = bool(adapter_path and adapter_path.exists())

    def _both(text: str) -> tuple[str, str]:
        base_reply = ""
        adapter_reply = ""

        logger.info("Side-by-side: loading base model")
        base = _load_harness(config_path=config_path)
        try:
            base_reply = base.generate(text)
        finally:
            del base
            _release_cuda()

        if not has_adapter:
            adapter_reply = (
                "[no adapter] Train first or pass a valid --adapter-dir. "
                f"Looked for: {adapter_path}"
            )
            return base_reply, adapter_reply

        logger.info("Side-by-side: loading adapter from %s", adapter_path)
        try:
            tuned = _load_harness(config_path=config_path, model_name=str(adapter_path))
            try:
                adapter_reply = tuned.generate(text)
            finally:
                del tuned
                _release_cuda()
        except Exception as exc:
            # Fallback: base + PEFT load_adapter when folder is pure LoRA weights
            logger.warning("Direct adapter load failed (%s); trying base+load_adapter", exc)
            tuned = _load_harness(config_path=config_path)
            try:
                tuned.model.load_adapter(str(adapter_path))
                adapter_reply = tuned.generate(text)
            except Exception as exc2:
                adapter_reply = f"[adapter load failed] {type(exc2).__name__}: {exc2}"
            finally:
                del tuned
                _release_cuda()

        return base_reply, adapter_reply

    return _both


def pack_user_text(
    example_label: str,
    custom_question: str,
    custom_context: str,
    *,
    items: list[PanelItem] | None = None,
) -> str:
    catalog = items if items is not None else example_items()
    by_label = {f"{item.id} · {item.ticker} · {item.theme}": item for item in catalog}
    item = by_label.get(example_label, catalog[0] if catalog else None)

    if custom_question.strip():
        context = custom_context.strip() or (item.context if item else "")
        ticker = item.ticker if item else "CUSTOM"
        theme = item.theme if item else "custom"
        return (
            f"Ticker: {ticker}\nTheme: {theme}\n\n"
            f"Context:\n{context}\n\nQuestion:\n{custom_question.strip()}"
        )
    if item is not None:
        return item.user_text()
    return "No panel examples found."


def answer_turn(
    example_label: str,
    custom_question: str,
    custom_context: str,
    *,
    items: list[PanelItem] | None = None,
    generate_fn: Callable[[str], str] | None = None,
) -> tuple[str, str]:
    """Return (user_text, model_or_stub_reply) for the single-model Gradio callback."""
    user_text = pack_user_text(
        example_label, custom_question, custom_context, items=items
    )
    fn = generate_fn or (lambda _t: DRY_RUN_REPLY)
    return user_text, fn(user_text)


def answer_turn_side_by_side(
    example_label: str,
    custom_question: str,
    custom_context: str,
    *,
    items: list[PanelItem] | None = None,
    generate_fn: Callable[[str], tuple[str, str]] | None = None,
) -> tuple[str, str, str]:
    """Return (user_text, base_reply, adapter_reply)."""
    user_text = pack_user_text(
        example_label, custom_question, custom_context, items=items
    )
    fn = generate_fn or (lambda _t: (DRY_RUN_SIDE_BY_SIDE, DRY_RUN_SIDE_BY_SIDE))
    base_reply, adapter_reply = fn(user_text)
    return user_text, base_reply, adapter_reply


def build_demo(
    *,
    load_model: bool = False,
    config_path: Path | str | None = None,
    adapter_dir: Path | str | None = None,
    panel_path: Path | str | None = None,
    share: bool = False,
    side_by_side: bool = False,
) -> Any:
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - optional extra
        raise SystemExit(
            "gradio is required for the demo. Install with: pip install -e '.[demo]'"
        ) from exc

    items = example_items(panel_path)
    labels = example_choices(items)

    if side_by_side:
        generate_fn = _make_side_by_side_generate(
            load_model=load_model,
            config_path=config_path,
            adapter_dir=adapter_dir,
        )
        mode = (
            f"live base vs adapter ({adapter_dir or 'no adapter dir'})"
            if load_model
            else "CPU dry-run stub (side-by-side)"
        )

        def _on_submit(example_label: str, question: str, context: str) -> tuple[str, str, str]:
            return answer_turn_side_by_side(
                example_label,
                question,
                context,
                items=items,
                generate_fn=generate_fn,
            )

        with gr.Blocks(title="Earnings Call Research Assistant — base vs fine-tuned") as demo:
            gr.Markdown(
                "# Earnings Call Research Assistant\n"
                f"Mode: **{mode}**. Side-by-side **base** vs **fine-tuned** answers. "
                "Grounded prompts from `evals/research_panel.jsonl`. Does not train."
            )
            example = gr.Dropdown(
                choices=labels, value=labels[0] if labels else None, label="Panel example"
            )
            question = gr.Textbox(label="Override question (optional)", lines=2)
            context = gr.Textbox(label="Override context (optional)", lines=4)
            submit = gr.Button("Compare base vs fine-tuned", variant="primary")
            user_box = gr.Textbox(label="Packed user message", lines=8)
            with gr.Row():
                base_box = gr.Textbox(label="Base model (before)", lines=10)
                adapter_box = gr.Textbox(label="Fine-tuned adapter (after)", lines=10)
            submit.click(
                _on_submit, [example, question, context], [user_box, base_box, adapter_box]
            )
            example.change(
                _on_submit, [example, question, context], [user_box, base_box, adapter_box]
            )
        demo.queue()
        return demo

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
        example = gr.Dropdown(
            choices=labels, value=labels[0] if labels else None, label="Panel example"
        )
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
    side_by_side: bool = False,
) -> Any:
    demo = build_demo(
        load_model=load_model,
        config_path=config_path,
        adapter_dir=adapter_dir,
        panel_path=panel_path,
        share=share,
        side_by_side=side_by_side,
    )
    demo.launch(share=share, server_name=server_name, server_port=server_port)
    return demo
