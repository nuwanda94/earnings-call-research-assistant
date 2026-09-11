"""Versioned RAG corpus from Phase-1 chunks (or offline ingest fixtures).

Defines measured **N** = chunk count. IDs are hash-stable given the same
input JSONL / fixture set. No embeddings, no GPU.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from earnings_call_research_assistant.data.chunk import (
    ChunkConfig,
    TextChunk,
    chunk_records,
)
from earnings_call_research_assistant.data.generate import chunk_from_dict, load_chunks_jsonl
from earnings_call_research_assistant.data.ingest import PUBLIC_SOURCES, ingest_catalog

CORPUS_VERSION = "v0.1.0"
DEFAULT_CORPUS_DIR = Path("data") / "rag" / f"corpus_{CORPUS_VERSION}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stable_chunk_id(
    source_id: str,
    section: str,
    start: int,
    end: int,
    text: str,
) -> str:
    """Hash of source + span + text. Matches Phase-1 chunk identity contract."""
    digest = hashlib.sha1(
        f"{source_id}|{section}|{start}|{end}|{text}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{source_id}:{section}:{start}-{end}:{digest}"


@dataclass
class RagChunk:
    """Retrieval unit written to ``chunks.jsonl``."""

    chunk_id: str
    source_id: str
    section: str
    text: str
    start_sentence: int
    end_sentence: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CorpusManifest:
    version: str
    n_chunks: int
    n_documents: int
    sources: dict[str, int]
    license_notes: list[str]
    created_utc: str
    input_path: str | None
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _license_notes() -> list[str]:
    notes: list[str] = []
    for spec in PUBLIC_SOURCES:
        notes.append(f"{spec.source_id}: {spec.license_note}")
    notes.append("Public data only. Do not mix paid terminal extracts.")
    return notes


def rag_chunk_from_text_chunk(chunk: TextChunk) -> RagChunk:
    chunk_id = chunk.chunk_id or stable_chunk_id(
        chunk.source_id, chunk.section, chunk.start_sentence, chunk.end_sentence, chunk.text
    )
    meta = dict(chunk.metadata or {})
    meta["n_propositions"] = len(chunk.propositions)
    return RagChunk(
        chunk_id=chunk_id,
        source_id=chunk.source_id,
        section=chunk.section,
        text=chunk.text,
        start_sentence=chunk.start_sentence,
        end_sentence=chunk.end_sentence,
        metadata=meta,
    )


def _dedupe(chunks: Sequence[RagChunk]) -> list[RagChunk]:
    seen: set[str] = set()
    out: list[RagChunk] = []
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        if not chunk.text.strip():
            continue
        seen.add(chunk.chunk_id)
        out.append(chunk)
    return out


def load_phase1_chunks(path: str | Path) -> list[TextChunk]:
    return load_chunks_jsonl(path)


def chunks_from_fixtures(
    *,
    source_ids: Iterable[str] | None = None,
    max_samples: int = 3,
    config: ChunkConfig | None = None,
) -> list[TextChunk]:
    records = ingest_catalog(
        source_ids=source_ids,
        max_samples=max_samples,
        download=False,
    )
    return chunk_records(records, config=config or ChunkConfig())


def build_corpus(
    *,
    chunks_path: str | Path | None = None,
    source_ids: Iterable[str] | None = None,
    max_samples: int = 3,
    config: ChunkConfig | None = None,
) -> tuple[list[RagChunk], CorpusManifest]:
    """Load Phase-1 JSONL when present; otherwise re-chunk offline fixtures."""
    input_path: str | None = None
    text_chunks: list[TextChunk]
    if chunks_path is not None and Path(chunks_path).is_file():
        input_path = str(Path(chunks_path))
        text_chunks = load_phase1_chunks(chunks_path)
        origin = f"phase1_jsonl:{input_path}"
    else:
        text_chunks = chunks_from_fixtures(
            source_ids=source_ids, max_samples=max_samples, config=config
        )
        origin = "offline_fixtures"

    rag_chunks = _dedupe([rag_chunk_from_text_chunk(c) for c in text_chunks])
    by_source = Counter(c.source_id for c in rag_chunks)
    n_docs = len({c.source_id for c in rag_chunks})
    manifest = CorpusManifest(
        version=CORPUS_VERSION,
        n_chunks=len(rag_chunks),
        n_documents=n_docs,
        sources=dict(sorted(by_source.items())),
        license_notes=_license_notes(),
        created_utc=_utc_now(),
        input_path=input_path,
        notes=(
            f"Origin={origin}. N is measured chunk count after empty/id dedupe. "
            "IDs are sha1(source|section|start|end|text)[:12]."
        ),
    )
    return rag_chunks, manifest


def write_corpus(
    chunks: Sequence[RagChunk],
    manifest: CorpusManifest,
    out_dir: str | Path | None = None,
) -> Path:
    dest = Path(out_dir) if out_dir is not None else DEFAULT_CORPUS_DIR
    dest.mkdir(parents=True, exist_ok=True)
    chunks_path = dest / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    manifest_path = dest / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return dest


def load_corpus_chunks(corpus_dir: str | Path) -> list[RagChunk]:
    path = Path(corpus_dir) / "chunks.jsonl"
    rows: list[RagChunk] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(
                RagChunk(
                    chunk_id=str(row["chunk_id"]),
                    source_id=str(row.get("source_id") or "unknown"),
                    section=str(row.get("section") or "unknown"),
                    text=str(row.get("text") or ""),
                    start_sentence=int(row.get("start_sentence", 0)),
                    end_sentence=int(row.get("end_sentence", 0)),
                    metadata=dict(row.get("metadata") or {}),
                )
            )
    return rows


def load_manifest(corpus_dir: str | Path) -> CorpusManifest:
    path = Path(corpus_dir) / "manifest.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return CorpusManifest(
        version=str(raw.get("version") or CORPUS_VERSION),
        n_chunks=int(raw["n_chunks"]),
        n_documents=int(raw["n_documents"]),
        sources=dict(raw.get("sources") or {}),
        license_notes=list(raw.get("license_notes") or []),
        created_utc=str(raw.get("created_utc") or ""),
        input_path=raw.get("input_path"),
        notes=str(raw.get("notes") or ""),
    )
