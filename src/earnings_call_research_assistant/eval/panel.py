"""Qualitative research panel + base vs adapter comparison stub.

Default path is CPU-only: load the JSONL panel, optionally attach placeholder
outputs, and write a side-by-side workbook. Pass ``run=True`` plus an
``InferenceHarness`` on Kaggle to fill real generations. Long GPU trains are
never started from this module.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PANEL = ROOT / "evals" / "research_panel.jsonl"
DEFAULT_OUT = ROOT / "evals" / "reports" / "research_panel_comparison.json"

PLACEHOLDER_BASE = "[dry-run] base model not loaded; run on Kaggle with --run"
PLACEHOLDER_ADAPTER = (
    "[dry-run] adapter not loaded; pass --adapter-dir after QLoRA --run"
)


@dataclass
class PanelItem:
    id: str
    task: str
    ticker: str
    theme: str
    prompt: str
    context: str
    must_cite: list[str] = field(default_factory=list)
    refuse_if_missing: bool = False

    def user_text(self) -> str:
        return (
            f"Ticker: {self.ticker}\nTheme: {self.theme}\n\n"
            f"Context:\n{self.context}\n\nQuestion:\n{self.prompt}"
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PanelItem":
        return cls(
            id=str(raw["id"]),
            task=str(raw.get("task", "qa")),
            ticker=str(raw.get("ticker", "")),
            theme=str(raw.get("theme", "")),
            prompt=str(raw["prompt"]),
            context=str(raw.get("context", "")),
            must_cite=[str(x) for x in raw.get("must_cite", [])],
            refuse_if_missing=bool(raw.get("refuse_if_missing", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonRow:
    id: str
    task: str
    ticker: str
    theme: str
    prompt: str
    context: str
    must_cite: list[str]
    refuse_if_missing: bool
    base_output: str
    adapter_output: str
    base_citation_hits: list[str]
    adapter_citation_hits: list[str]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


GenerateFn = Callable[[str], str]


def load_panel(path: Path | str | None = None) -> list[PanelItem]:
    panel_path = Path(path) if path else DEFAULT_PANEL
    items: list[PanelItem] = []
    with panel_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(PanelItem.from_dict(json.loads(line)))
    return items


def _hits(text: str, needles: Sequence[str]) -> list[str]:
    lower = text.lower()
    return [n for n in needles if n.lower() in lower]


def _call_generate(fn: GenerateFn | None, user_text: str, placeholder: str) -> str:
    if fn is None:
        return placeholder
    return fn(user_text)


def compare_panel(
    items: Iterable[PanelItem] | None = None,
    *,
    base_generate: GenerateFn | None = None,
    adapter_generate: GenerateFn | None = None,
    dry_run: bool = True,
) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    for item in items if items is not None else load_panel():
        user = item.user_text()
        if dry_run:
            base_out = PLACEHOLDER_BASE
            adapter_out = PLACEHOLDER_ADAPTER
        else:
            base_out = _call_generate(base_generate, user, PLACEHOLDER_BASE)
            adapter_out = _call_generate(adapter_generate, user, PLACEHOLDER_ADAPTER)
        rows.append(
            ComparisonRow(
                id=item.id,
                task=item.task,
                ticker=item.ticker,
                theme=item.theme,
                prompt=item.prompt,
                context=item.context,
                must_cite=list(item.must_cite),
                refuse_if_missing=item.refuse_if_missing,
                base_output=base_out,
                adapter_output=adapter_out,
                base_citation_hits=_hits(base_out, item.must_cite),
                adapter_citation_hits=_hits(adapter_out, item.must_cite),
                notes=(
                    "expect refusal / insufficient-context"
                    if item.refuse_if_missing
                    else "expect grounded citations"
                ),
            )
        )
    return rows


def write_comparison(
    rows: Sequence[ComparisonRow],
    out_path: Path | str | None = None,
    *,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    path = Path(out_path) if out_path else DEFAULT_OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_items": len(rows),
        "dry_run": all(
            r.base_output.startswith("[dry-run]") for r in rows
        ),
        "meta": extra_meta or {},
        "rows": [r.to_dict() for r in rows],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def run_research_panel(
    panel_path: Path | str | None = None,
    out_path: Path | str | None = None,
    *,
    dry_run: bool = True,
    base_generate: GenerateFn | None = None,
    adapter_generate: GenerateFn | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    items = load_panel(panel_path)
    rows = compare_panel(
        items,
        base_generate=base_generate,
        adapter_generate=adapter_generate,
        dry_run=dry_run,
    )
    written = write_comparison(rows, out_path, extra_meta=extra_meta)
    return {
        "n_items": len(rows),
        "out_path": str(written),
        "dry_run": dry_run,
        "themes": sorted({r.theme for r in rows}),
    }
