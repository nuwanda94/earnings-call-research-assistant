"""Public-source ingestion stubs for Phase 1.

Default path is offline: catalog + tiny baked-in samples so clone-and-run
never pulls multi-GB transcript corpora. Optional Hugging Face sampling is
opt-in via ``load_public_source(..., download=True)`` or the CLI ``--download``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

# ---------------------------------------------------------------------------
# Catalog (public data only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    """One public corpus we may sample from later in Phase 1."""

    source_id: str
    display_name: str
    role: str
    hf_id: str | None
    hf_config: str | None
    homepage: str
    license_note: str
    text_fields: tuple[str, ...]
    notes: str
    default_split: str = "train"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["text_fields"] = list(self.text_fields)
        return d


PUBLIC_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        source_id="earnings_transcripts",
        display_name="S&P 500 earnings call transcripts",
        role="primary grounding corpus (prepared remarks + Q&A)",
        hf_id="kurry/sp500_earnings_transcripts",
        hf_config=None,
        homepage="https://huggingface.co/datasets/kurry/sp500_earnings_transcripts",
        license_note="MIT on the HF card; still public-company disclosures only.",
        text_fields=("transcript", "content", "text"),
        notes=(
            "Large (tens of thousands of calls). Never stream the full set by "
            "default. Use streaming + ticker/year filters when download=True."
        ),
    ),
    SourceSpec(
        source_id="fiqa",
        display_name="FiQA financial QA",
        role="seed research-style question distribution",
        hf_id="LLukas22/fiqa",
        hf_config=None,
        homepage="https://sites.google.com/view/fiqa/",
        license_note="Academic / task release; check the HF card before redistribution.",
        text_fields=("question", "answer", "text"),
        notes=(
            "Opinion-oriented financial QA. Useful for question style, not as a "
            "grounded transcript source. Pair later with extracted propositions."
        ),
    ),
    SourceSpec(
        source_id="finance_alpaca",
        display_name="Finance-Alpaca / Wealth-Alpaca",
        role="instruction-style finance pairs (style prior only)",
        hf_id="gbharti/finance-alpaca",
        hf_config=None,
        homepage="https://huggingface.co/datasets/gbharti/finance-alpaca",
        license_note="Derived Alpaca + FiQA mix; treat as public research sample.",
        text_fields=("instruction", "input", "output", "text"),
        notes=(
            "Not grounded in a specific call. Sample a few hundred rows max for "
            "prompt-style diversity; do not dump the full 60k+ set into SFT."
        ),
    ),
)

SOURCE_BY_ID: dict[str, SourceSpec] = {s.source_id: s for s in PUBLIC_SOURCES}


# Tiny offline fixtures so the stub is runnable without `datasets` or network.
OFFLINE_SAMPLES: dict[str, list[dict[str, Any]]] = {
    "earnings_transcripts": [
        {
            "source_id": "earnings_transcripts",
            "ticker": "AAPL",
            "period": "FY2024-Q1",
            "section": "prepared_remarks",
            "text": (
                "Thank you for joining Apple's first quarter fiscal 2024 results. "
                "We set an all-time revenue record of $119.6 billion, up 2 percent "
                "year over year, driven by iPhone and Services."
            ),
        },
        {
            "source_id": "earnings_transcripts",
            "ticker": "MSFT",
            "period": "FY2024-Q2",
            "section": "qa",
            "text": (
                "Analyst: Can you talk about Azure growth excluding the impact of "
                "the extra calendar day? CEO: Azure grew 30 percent, and 28 percent "
                "constant currency, with AI services contributing several points."
            ),
        },
    ],
    "fiqa": [
        {
            "source_id": "fiqa",
            "question": "What does a beat versus a miss mean on an earnings call?",
            "answer": (
                "A beat means reported EPS or revenue exceeded the consensus estimate; "
                "a miss means results came in below that estimate."
            ),
        }
    ],
    "finance_alpaca": [
        {
            "source_id": "finance_alpaca",
            "instruction": "Explain prepared remarks versus the Q&A portion of an earnings call.",
            "input": "",
            "output": (
                "Prepared remarks are a scripted management overview of results and outlook. "
                "Q&A is unscripted analyst questioning and is usually richer for research."
            ),
        }
    ],
}


@dataclass
class IngestRecord:
    """Normalized row used by later chunking / generation stages."""

    source_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "text": self.text, "metadata": self.metadata}


def list_sources() -> list[SourceSpec]:
    return list(PUBLIC_SOURCES)


def get_source(source_id: str) -> SourceSpec:
    try:
        return SOURCE_BY_ID[source_id]
    except KeyError as exc:
        known = ", ".join(SOURCE_BY_ID)
        raise KeyError(f"Unknown source_id={source_id!r}. Known: {known}") from exc


def _text_from_row(row: Mapping[str, Any], fields: Iterable[str]) -> str:
    parts: list[str] = []
    for name in fields:
        value = row.get(name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    if parts:
        return "\n\n".join(parts)
    # last-ditch: first reasonably long string field
    for value in row.values():
        if isinstance(value, str) and len(value.strip()) >= 40:
            return value.strip()
    return ""


def _offline_records(source_id: str, max_samples: int) -> list[IngestRecord]:
    spec = get_source(source_id)
    rows = OFFLINE_SAMPLES.get(source_id, [])[:max_samples]
    records: list[IngestRecord] = []
    for row in rows:
        text = _text_from_row(row, spec.text_fields)
        if not text:
            continue
        meta = {k: v for k, v in row.items() if k != "text"}
        records.append(IngestRecord(source_id=source_id, text=text, metadata=meta))
    return records


def _hf_records(spec: SourceSpec, max_samples: int) -> list[IngestRecord]:
    if not spec.hf_id:
        raise ValueError(f"{spec.source_id} has no Hugging Face id configured")
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Optional download requires the `datasets` package (already in pyproject)."
        ) from exc

    kwargs: dict[str, Any] = {"split": spec.default_split, "streaming": True}
    if spec.hf_config:
        ds = load_dataset(spec.hf_id, spec.hf_config, **kwargs)
    else:
        ds = load_dataset(spec.hf_id, **kwargs)

    records: list[IngestRecord] = []
    for row in ds:
        text = _text_from_row(row, spec.text_fields)
        if not text:
            continue
        meta = {
            k: v
            for k, v in dict(row).items()
            if k not in spec.text_fields and not isinstance(v, (list, dict))
        }
        records.append(IngestRecord(source_id=spec.source_id, text=text, metadata=meta))
        if len(records) >= max_samples:
            break
    return records


def load_public_source(
    source_id: str,
    *,
    max_samples: int = 3,
    download: bool = False,
) -> list[IngestRecord]:
    """Return a small sample from ``source_id``.

    ``download=False`` (default) uses the baked-in fixtures and never hits the network.
    """
    if max_samples < 1:
        raise ValueError("max_samples must be >= 1")
    spec = get_source(source_id)
    if download:
        return _hf_records(spec, max_samples)
    return _offline_records(source_id, max_samples)


def ingest_catalog(
    *,
    source_ids: Iterable[str] | None = None,
    max_samples: int = 3,
    download: bool = False,
) -> list[IngestRecord]:
    ids = list(source_ids) if source_ids is not None else [s.source_id for s in PUBLIC_SOURCES]
    out: list[IngestRecord] = []
    for sid in ids:
        out.extend(load_public_source(sid, max_samples=max_samples, download=download))
    return out


def write_jsonl(records: Iterable[IngestRecord], path: str | Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    return dest
