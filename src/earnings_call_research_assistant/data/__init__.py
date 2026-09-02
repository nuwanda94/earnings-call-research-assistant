"""Ingestion, chunking, grounded generation, and filtering (Phase 1)."""

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
    "IngestRecord",
    "SourceSpec",
    "get_source",
    "ingest_catalog",
    "list_sources",
    "load_public_source",
    "write_jsonl",
]
