"""Grounded synthetic Q&A and summary pairs from chunk + proposition records.

Default path is offline templates: every answer is composed from extracted
propositions and quotes a citation span from the source chunk. An optional
LLM callback can be passed for Kaggle-side rewriting without changing the
grounding contract (answers still must cite chunk_id + proposition text).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from earnings_call_research_assistant.data.chunk import Proposition, TextChunk

LLMGenerator = Callable[[str, dict[str, Any]], str]

_QA_TEMPLATES: tuple[str, ...] = (
    "What did management report in this {section} excerpt?",
    "Which quantitative results are stated in the {section} passage?",
    "Summarize the factual claims supported by this {section} excerpt.",
    "According to this {section} excerpt, what happened to the key metrics?",
)

_SUMMARY_TEMPLATES: tuple[str, ...] = (
    "Write a short research note grounded only in this {section} excerpt.",
    "Produce a one-paragraph briefing of the facts in this {section} passage.",
)

_SECTION_LABEL = {
    "prepared_remarks": "prepared remarks",
    "qa": "Q&A",
    "unknown": "earnings-call",
}


@dataclass(frozen=True)
class GenerateConfig:
    """Controls how many pairs to emit per chunk and whether to call an LLM."""

    max_qa_per_chunk: int = 2
    include_summary: bool = True
    min_propositions: int = 1
    require_citation: bool = True
    use_llm: bool = False

    def __post_init__(self) -> None:
        if self.max_qa_per_chunk < 0:
            raise ValueError("max_qa_per_chunk must be >= 0")
        if self.min_propositions < 0:
            raise ValueError("min_propositions must be >= 0")


@dataclass
class InstructionPair:
    pair_id: str
    chunk_id: str
    source_id: str
    task: str
    instruction: str
    context: str
    output: str
    citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section_label(section: str) -> str:
    return _SECTION_LABEL.get(section, section.replace("_", " ") or "earnings-call")


def _pair_id(chunk_id: str, task: str, instruction: str) -> str:
    digest = hashlib.sha1(f"{chunk_id}|{task}|{instruction}".encode("utf-8")).hexdigest()[:12]
    return f"{chunk_id}:{task}:{digest}"


def _props_for_chunk(chunk: TextChunk) -> list[Proposition]:
    return [p for p in chunk.propositions if p.text.strip()]


def _citation_block(chunk: TextChunk, props: Sequence[Proposition]) -> list[str]:
    cites: list[str] = []
    for prop in props:
        cites.append(f"[{chunk.chunk_id}#s{prop.source_sentence_index}] {prop.text}")
    if not cites:
        snippet = chunk.text.strip()[:240]
        cites.append(f"[{chunk.chunk_id}] {snippet}")
    return cites


def _template_answer(task: str, chunk: TextChunk, props: Sequence[Proposition]) -> str:
    bullets = "\n".join(f"- {p.text}" for p in props) or f"- {chunk.text.strip()}"
    cites = "; ".join(f"{chunk.chunk_id}#s{p.source_sentence_index}" for p in props) or chunk.chunk_id
    section = _section_label(chunk.section)
    if task == "summary":
        lead = f"Research note from the {section} excerpt (grounded):"
    else:
        lead = f"From the {section} excerpt, the supported facts are:"
    return (
        f"{lead}\n{bullets}\n\n"
        f"Citations: {cites}. No figures beyond these source sentences."
    )


def _maybe_rewrite(
    draft: str,
    *,
    task: str,
    chunk: TextChunk,
    props: Sequence[Proposition],
    llm: LLMGenerator | None,
    use_llm: bool,
) -> str:
    if not use_llm or llm is None:
        return draft
    prompt = (
        "Rewrite the draft answer so it reads like a concise research assistant. "
        "Keep every number and proper noun exactly as written. Do not add facts. "
        "Keep the Citations line intact.\n\n"
        f"TASK: {task}\nCHUNK:\n{chunk.text}\n\nDRAFT:\n{draft}\n"
    )
    payload = {
        "task": task,
        "chunk_id": chunk.chunk_id,
        "propositions": [p.to_dict() for p in props],
    }
    rewritten = llm(prompt, payload).strip()
    return rewritten or draft


def pairs_from_chunk(
    chunk: TextChunk,
    *,
    config: GenerateConfig | None = None,
    llm: LLMGenerator | None = None,
) -> list[InstructionPair]:
    """Turn one chunk into citation-grounded instruction pairs."""
    cfg = config or GenerateConfig()
    props = _props_for_chunk(chunk)
    if len(props) < cfg.min_propositions and not chunk.text.strip():
        return []
    if cfg.require_citation and not props and not chunk.text.strip():
        return []

    section = _section_label(chunk.section)
    cites = _citation_block(chunk, props)
    used_props = props or [
        Proposition(text=chunk.text.strip()[:280], source_sentence_index=chunk.start_sentence, score=0.0)
    ]
    pairs: list[InstructionPair] = []

    qa_templates = _QA_TEMPLATES[: cfg.max_qa_per_chunk]
    for tmpl in qa_templates:
        instruction = tmpl.format(section=section)
        draft = _template_answer("qa", chunk, used_props)
        output = _maybe_rewrite(
            draft, task="qa", chunk=chunk, props=used_props, llm=llm, use_llm=cfg.use_llm
        )
        pairs.append(
            InstructionPair(
                pair_id=_pair_id(chunk.chunk_id, "qa", instruction),
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                task="qa",
                instruction=instruction,
                context=chunk.text,
                output=output,
                citations=cites,
                metadata={
                    "section": chunk.section,
                    "generator": "llm" if cfg.use_llm and llm is not None else "template",
                    "n_propositions": len(props),
                },
            )
        )

    if cfg.include_summary:
        instruction = _SUMMARY_TEMPLATES[0].format(section=section)
        draft = _template_answer("summary", chunk, used_props)
        output = _maybe_rewrite(
            draft, task="summary", chunk=chunk, props=used_props, llm=llm, use_llm=cfg.use_llm
        )
        pairs.append(
            InstructionPair(
                pair_id=_pair_id(chunk.chunk_id, "summary", instruction),
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                task="summary",
                instruction=instruction,
                context=chunk.text,
                output=output,
                citations=cites,
                metadata={
                    "section": chunk.section,
                    "generator": "llm" if cfg.use_llm and llm is not None else "template",
                    "n_propositions": len(props),
                },
            )
        )
    return pairs


def generate_pairs(
    chunks: Iterable[TextChunk],
    *,
    config: GenerateConfig | None = None,
    llm: LLMGenerator | None = None,
) -> list[InstructionPair]:
    out: list[InstructionPair] = []
    for chunk in chunks:
        out.extend(pairs_from_chunk(chunk, config=config, llm=llm))
    return out


def chunk_from_dict(row: dict[str, Any]) -> TextChunk:
    """Rebuild a TextChunk from JSONL written by ``write_chunks_jsonl``."""
    props = [
        Proposition(
            text=str(p.get("text", "")),
            source_sentence_index=int(p.get("source_sentence_index", 0)),
            score=float(p.get("score", 0.0)),
        )
        for p in row.get("propositions") or []
    ]
    return TextChunk(
        chunk_id=str(row.get("chunk_id") or "unknown"),
        source_id=str(row.get("source_id") or "unknown"),
        section=str(row.get("section") or "unknown"),
        text=str(row.get("text") or ""),
        start_sentence=int(row.get("start_sentence", 0)),
        end_sentence=int(row.get("end_sentence", 0)),
        propositions=props,
        metadata=dict(row.get("metadata") or {}),
    )


def load_chunks_jsonl(path: str | Path) -> list[TextChunk]:
    dest = Path(path)
    chunks: list[TextChunk] = []
    with dest.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            chunks.append(chunk_from_dict(json.loads(line)))
    return chunks


def write_pairs_jsonl(pairs: Iterable[InstructionPair], path: str | Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
    return dest
