"""Section-aware chunking and heuristic proposition extraction.

Splits ingested transcript text into overlapping sentence windows and extracts
short factual propositions for later grounded Q&A / summary generation.
No LLM calls: this stage is deterministic and offline-safe.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from earnings_call_research_assistant.data.ingest import IngestRecord

# Headings that typically separate an earnings call.
_SECTION_HEADINGS = (
    ("prepared_remarks", re.compile(r"^(prepared\s+remarks|operator|management\s+remarks|opening\s+remarks)\b", re.I)),
    ("qa", re.compile(r"^(question[-\s]?and[-\s]?answer|q\s*&\s*a|qa session|questions?\s+and\s+answers?)\b", re.I)),
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_WHITESPACE = re.compile(r"\s+")

# Signals that a sentence is likely a grounded fact rather than filler.
_FACT_CUES = re.compile(
    r"(\d|%|basis\s+points|bps|revenue|eps|margin|guidance|grew|declined|"
    r"increased|decreased|billion|million|percent|year[- ]over[- ]year|"
    r"quarter|outlook|customers?|subscribers?|arr|gmav)",
    re.I,
)
_FILLER = re.compile(
    r"^(thank you|thanks|good (morning|afternoon)|please go ahead|"
    r"next question|operator|turn it over)\b",
    re.I,
)
_SPEAKER_PREFIX = re.compile(
    r"^(analyst|operator|ceo|cfo|cto|coo|host|speaker)\s*[:\-]\s*", re.I
)


@dataclass(frozen=True)
class ChunkConfig:
    """Window sizes in sentences. Keep small so later prompts stay grounded."""

    window_sentences: int = 4
    stride_sentences: int = 2
    min_chars: int = 40
    max_chars: int = 1800
    max_propositions_per_chunk: int = 8

    def __post_init__(self) -> None:
        if self.window_sentences < 1:
            raise ValueError("window_sentences must be >= 1")
        if self.stride_sentences < 1:
            raise ValueError("stride_sentences must be >= 1")
        if self.min_chars < 1:
            raise ValueError("min_chars must be >= 1")


@dataclass
class Proposition:
    text: str
    source_sentence_index: int
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TextChunk:
    chunk_id: str
    source_id: str
    section: str
    text: str
    start_sentence: int
    end_sentence: int
    propositions: list[Proposition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "section": self.section,
            "text": self.text,
            "start_sentence": self.start_sentence,
            "end_sentence": self.end_sentence,
            "propositions": [p.to_dict() for p in self.propositions],
            "metadata": self.metadata,
        }


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _split_sentences(text: str) -> list[str]:
    cleaned = _normalize(text)
    if not cleaned:
        return []
    parts = _SENTENCE_SPLIT.split(cleaned)
    return [p.strip() for p in parts if p.strip()]


def detect_section(text: str, metadata: dict[str, Any] | None = None) -> str:
    """Prefer explicit metadata, then heading cues, else ``unknown``."""
    meta = metadata or {}
    raw = str(meta.get("section") or "").strip().lower().replace(" ", "_")
    if raw in {"prepared_remarks", "qa", "unknown"}:
        return raw
    if raw in {"q&a", "q_and_a", "questions"}:
        return "qa"
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    for name, pattern in _SECTION_HEADINGS:
        if pattern.search(first_line):
            return name
    lowered = text.lower()
    if "analyst:" in lowered or "q&a" in lowered or "question-and-answer" in lowered:
        return "qa"
    return "unknown"


def split_sections(text: str, metadata: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    """Split a document into (section_name, body) pairs.

    If no in-text headings exist, return a single section from metadata/heuristics.
    """
    lines = text.splitlines()
    blocks: list[tuple[str, list[str]]] = []
    current_name = detect_section(text, metadata)
    current: list[str] = []
    found_heading = False
    for line in lines:
        stripped = line.strip()
        matched = None
        for name, pattern in _SECTION_HEADINGS:
            if pattern.search(stripped) and len(stripped) < 80:
                matched = name
                break
        if matched is not None:
            found_heading = True
            if current:
                blocks.append((current_name, current))
            current_name = matched
            current = []
            continue
        current.append(line)
    if current:
        blocks.append((current_name, current))
    if not found_heading:
        return [(detect_section(text, metadata), text)]
    return [(name, "\n".join(body).strip()) for name, body in blocks if "\n".join(body).strip()]


def _chunk_id(source_id: str, section: str, start: int, end: int, text: str) -> str:
    digest = hashlib.sha1(f"{source_id}|{section}|{start}|{end}|{text}".encode("utf-8")).hexdigest()[:12]
    return f"{source_id}:{section}:{start}-{end}:{digest}"


def _is_filler(sentence: str) -> bool:
    cleaned = _SPEAKER_PREFIX.sub("", sentence).strip()
    return bool(_FILLER.match(cleaned)) or len(cleaned) < 20


def _proposition_score(sentence: str) -> float:
    if _is_filler(sentence):
        return 0.0
    cleaned = _SPEAKER_PREFIX.sub("", sentence).strip()
    score = 0.15
    if _FACT_CUES.search(cleaned):
        score += 0.55
    if re.search(r"\d", cleaned):
        score += 0.2
    if 40 <= len(cleaned) <= 280:
        score += 0.15
    elif len(cleaned) > 400:
        score -= 0.2
    return min(score, 1.0)


def extract_propositions(
    sentences: Sequence[str],
    *,
    max_items: int = 8,
    min_score: float = 0.45,
) -> list[Proposition]:
    """Pull short factual statements from sentence windows."""
    scored: list[Proposition] = []
    for idx, sent in enumerate(sentences):
        cleaned = _SPEAKER_PREFIX.sub("", sent).strip()
        if not cleaned.endswith((".", "!", "?")):
            cleaned = cleaned.rstrip(";,") + "."
        score = _proposition_score(sent)
        if score < min_score:
            continue
        scored.append(Proposition(text=cleaned, source_sentence_index=idx, score=round(score, 3)))
    scored.sort(key=lambda p: (-p.score, p.source_sentence_index))
    return scored[:max_items]


def chunk_text(
    text: str,
    *,
    source_id: str = "unknown",
    metadata: dict[str, Any] | None = None,
    config: ChunkConfig | None = None,
) -> list[TextChunk]:
    cfg = config or ChunkConfig()
    meta = dict(metadata or {})
    chunks: list[TextChunk] = []
    for section, body in split_sections(text, meta):
        sentences = _split_sentences(body)
        if not sentences:
            continue
        i = 0
        while i < len(sentences):
            window = sentences[i : i + cfg.window_sentences]
            joined = _normalize(" ".join(window))
            if len(joined) < cfg.min_chars:
                i += cfg.stride_sentences
                continue
            if len(joined) > cfg.max_chars:
                joined = joined[: cfg.max_chars].rsplit(" ", 1)[0]
            end = i + len(window) - 1
            props = extract_propositions(
                window, max_items=cfg.max_propositions_per_chunk
            )
            chunk_meta = {k: v for k, v in meta.items() if k != "text"}
            chunk_meta["section"] = section
            chunks.append(
                TextChunk(
                    chunk_id=_chunk_id(source_id, section, i, end, joined),
                    source_id=source_id,
                    section=section,
                    text=joined,
                    start_sentence=i,
                    end_sentence=end,
                    propositions=props,
                    metadata=chunk_meta,
                )
            )
            if i + cfg.window_sentences >= len(sentences):
                break
            i += cfg.stride_sentences
    return chunks


def chunk_records(
    records: Iterable[IngestRecord],
    *,
    config: ChunkConfig | None = None,
) -> list[TextChunk]:
    out: list[TextChunk] = []
    for rec in records:
        out.extend(chunk_text(rec.text, source_id=rec.source_id, metadata=rec.metadata, config=config))
    return out


def write_chunks_jsonl(chunks: Iterable[TextChunk], path: str | Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    return dest
