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
from earnings_call_research_assistant.data.generate import (
    GenerateConfig,
    InstructionPair,
    generate_pairs,
    load_chunks_jsonl,
    pairs_from_chunk,
    write_pairs_jsonl,
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
    "GenerateConfig",
    "IngestRecord",
    "InstructionPair",
    "Proposition",
    "SourceSpec",
    "TextChunk",
    "chunk_records",
    "chunk_text",
    "extract_propositions",
    "generate_pairs",
    "get_source",
    "ingest_catalog",
    "list_sources",
    "load_chunks_jsonl",
    "load_public_source",
    "pairs_from_chunk",
    "write_chunks_jsonl",
    "write_jsonl",
    "write_pairs_jsonl",
]
