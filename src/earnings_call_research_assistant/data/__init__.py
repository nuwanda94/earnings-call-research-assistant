"""Ingestion, chunking, grounded generation, and filtering (Phase 1)."""

from earnings_call_research_assistant.data.chunk import (
    ChunkConfig,
    Proposition,
    TextChunk,
    chunk_records,
    chunk_text,
    extract_propositions,
    write_chunks_jsonl,
)
from earnings_call_research_assistant.data.ingest import (
    PUBLIC_SOURCES,
    IngestRecord,
    SourceSpec,
    get_source,
    ingest_catalog,
    list_sources,
    load_public_source,
    write_jsonl,
)

__all__ = [
    "PUBLIC_SOURCES",
    "ChunkConfig",
    "IngestRecord",
    "Proposition",
    "SourceSpec",
    "TextChunk",
    "chunk_records",
    "chunk_text",
    "extract_propositions",
    "get_source",
    "ingest_catalog",
    "list_sources",
    "load_public_source",
    "write_chunks_jsonl",
    "write_jsonl",
]
