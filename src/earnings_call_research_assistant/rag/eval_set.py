"""Fixed retrieval eval set with gold chunk IDs (Phase 5.4).

Derives queries from grounded instruction pairs whose cited ``chunk_id``
exists in the RAG corpus. Deterministic sample uses seed ``3407``.
Train-split pair IDs are excluded when a versioned SFT dataset is present.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from earnings_call_research_assistant.data.filter import load_pairs_jsonl
from earnings_call_research_assistant.data.generate import (
    GenerateConfig,
    InstructionPair,
    generate_pairs,
    load_chunks_jsonl,
)
from earnings_call_research_assistant.rag.corpus import (
    CORPUS_VERSION,
    DEFAULT_CORPUS_DIR,
    RagChunk,
    build_corpus,
    load_corpus_chunks,
    write_corpus,
)

EVAL_SET_VERSION = "v0.1.0"
DEFAULT_EVAL_PATH = Path("evals") / "rag_eval_set.jsonl"
DEFAULT_SEED = 3407
DEFAULT_TARGET_N = 50
DEFAULT_MIN_N = 8


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class RagEvalRow:
    query_id: str
    query: str
    gold_chunk_ids: list[str]
    gold_answer: str | None = None
    source: str = "grounded_pair"
    pair_id: str | None = None
    task: str = "qa"
    source_id: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalSetManifest:
    version: str
    seed: int
    n_queries: int
    corpus_version: str
    corpus_n_chunks: int
    n_unique_gold_chunks: int
    excluded_train_pairs: int
    notes: str
    created_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _query_id(pair_id: str, query: str) -> str:
    digest = hashlib.sha1(f"{pair_id}|{query}".encode("utf-8")).hexdigest()[:12]
    return f"q:{digest}"


def _corpus_id_set(chunks: Sequence[RagChunk]) -> set[str]:
    return {c.chunk_id for c in chunks if c.chunk_id}


def load_train_pair_ids(dataset_dir: str | Path | None) -> set[str]:
    """pair_id values from SFT train.jsonl — excluded to avoid retrieval-label leakage."""
    if dataset_dir is None:
        return set()
    path = Path(dataset_dir) / "train.jsonl"
    if not path.is_file():
        return set()
    ids: set[str] = set()
    for pair in load_pairs_jsonl(path):
        if pair.pair_id:
            ids.add(pair.pair_id)
    return ids


def load_or_build_corpus(
    corpus_dir: str | Path | None = None,
    *,
    chunks_path: str | Path | None = None,
) -> tuple[list[RagChunk], Path]:
    dest = Path(corpus_dir) if corpus_dir is not None else DEFAULT_CORPUS_DIR
    chunks_file = dest / "chunks.jsonl"
    if chunks_file.is_file():
        return load_corpus_chunks(dest), dest
    phase1 = Path(chunks_path) if chunks_path else None
    rag_chunks, manifest = build_corpus(
        chunks_path=phase1 if phase1 is not None and phase1.is_file() else None,
    )
    write_corpus(rag_chunks, manifest, dest)
    return rag_chunks, dest


def _pairs_from_corpus_or_jsonl(
    corpus: Sequence[RagChunk],
    *,
    pairs_path: str | Path | None = None,
    phase1_chunks: str | Path | None = None,
) -> list[InstructionPair]:
    if pairs_path is not None and Path(pairs_path).is_file():
        return load_pairs_jsonl(pairs_path)
    if phase1_chunks is not None and Path(phase1_chunks).is_file():
        return generate_pairs(load_chunks_jsonl(phase1_chunks), config=GenerateConfig())
    # Rebuild TextChunk-shaped rows from the RAG corpus so templates stay grounded.
    from earnings_call_research_assistant.data.chunk import Proposition, TextChunk

    text_chunks: list[TextChunk] = []
    for row in corpus:
        text_chunks.append(
            TextChunk(
                chunk_id=row.chunk_id,
                source_id=row.source_id,
                section=row.section,
                text=row.text,
                start_sentence=row.start_sentence,
                end_sentence=row.end_sentence,
                propositions=[
                    Proposition(text=row.text.strip()[:280], source_sentence_index=row.start_sentence, score=0.7)
                ]
                if row.text.strip()
                else [],
                metadata=dict(row.metadata or {}),
            )
        )
    return generate_pairs(text_chunks, config=GenerateConfig())


def candidates_from_pairs(
    pairs: Iterable[InstructionPair],
    corpus_ids: set[str],
    *,
    excluded_pair_ids: set[str] | None = None,
) -> tuple[list[RagEvalRow], int]:
    blocked = excluded_pair_ids or set()
    skipped_train = 0
    rows: list[RagEvalRow] = []
    seen_keys: set[tuple[str, str]] = set()
    for pair in pairs:
        if pair.pair_id and pair.pair_id in blocked:
            skipped_train += 1
            continue
        gold = pair.chunk_id
        if not gold or gold not in corpus_ids:
            continue
        query = (pair.instruction or "").strip()
        if not query:
            continue
        key = (query, gold)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rows.append(
            RagEvalRow(
                query_id=_query_id(pair.pair_id or gold, query),
                query=query,
                gold_chunk_ids=[gold],
                gold_answer=(pair.output or "").strip() or None,
                source="grounded_pair",
                pair_id=pair.pair_id,
                task=pair.task or "qa",
                source_id=pair.source_id or "unknown",
                metadata={
                    "section": (pair.metadata or {}).get("section"),
                    "generator": (pair.metadata or {}).get("generator"),
                },
            )
        )
    return rows, skipped_train


def sample_eval_rows(
    rows: Sequence[RagEvalRow],
    *,
    seed: int = DEFAULT_SEED,
    max_n: int = DEFAULT_TARGET_N,
) -> list[RagEvalRow]:
    """Stable shuffle then take up to max_n, preferring source/task diversity."""
    if not rows:
        return []
    rng = random.Random(seed)
    ordered = list(rows)
    rng.shuffle(ordered)
    # Round-robin by (source_id, task) so one fixture source cannot dominate.
    buckets: dict[tuple[str, str], list[RagEvalRow]] = {}
    for row in ordered:
        buckets.setdefault((row.source_id, row.task), []).append(row)
    keys = list(buckets.keys())
    rng.shuffle(keys)
    picked: list[RagEvalRow] = []
    seen_q: set[str] = set()
    while len(picked) < max_n and keys:
        next_keys: list[tuple[str, str]] = []
        for key in keys:
            bucket = buckets[key]
            if not bucket:
                continue
            cand = bucket.pop(0)
            if cand.query_id in seen_q:
                if bucket:
                    next_keys.append(key)
                continue
            picked.append(cand)
            seen_q.add(cand.query_id)
            if bucket:
                next_keys.append(key)
            if len(picked) >= max_n:
                break
        keys = next_keys
    picked.sort(key=lambda r: r.query_id)
    return picked


def build_rag_eval_set(
    *,
    corpus_dir: str | Path | None = None,
    chunks_path: str | Path | None = None,
    pairs_path: str | Path | None = None,
    dataset_dir: str | Path | None = None,
    seed: int = DEFAULT_SEED,
    max_n: int = DEFAULT_TARGET_N,
) -> tuple[list[RagEvalRow], EvalSetManifest]:
    corpus, dest = load_or_build_corpus(corpus_dir, chunks_path=chunks_path)
    corpus_ids = _corpus_id_set(corpus)
    pairs = _pairs_from_corpus_or_jsonl(
        corpus,
        pairs_path=pairs_path,
        phase1_chunks=chunks_path,
    )
    excluded = load_train_pair_ids(dataset_dir)
    candidates, n_excl = candidates_from_pairs(pairs, corpus_ids, excluded_pair_ids=excluded)
    selected = sample_eval_rows(candidates, seed=seed, max_n=max_n)
    golds = {gid for row in selected for gid in row.gold_chunk_ids}
    notes = (
        f"Seed={seed}. Sampled {len(selected)} / {len(candidates)} grounded candidates. "
        f"Every gold_chunk_id is in corpus {dest} (N={len(corpus)}). "
        f"Excluded {n_excl} train-split pair_ids. "
        "v0.1 fixture corpora yield fewer than 30–50 queries; grow when N scales."
    )
    manifest = EvalSetManifest(
        version=EVAL_SET_VERSION,
        seed=seed,
        n_queries=len(selected),
        corpus_version=CORPUS_VERSION,
        corpus_n_chunks=len(corpus),
        n_unique_gold_chunks=len(golds),
        excluded_train_pairs=n_excl,
        notes=notes,
        created_utc=_utc_now(),
    )
    return selected, manifest


def write_eval_set(
    rows: Sequence[RagEvalRow],
    path: str | Path | None = None,
    *,
    manifest: EvalSetManifest | None = None,
) -> Path:
    dest = Path(path) if path is not None else DEFAULT_EVAL_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    if manifest is not None:
        man_path = dest.with_suffix(".manifest.json")
        man_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return dest


def load_eval_set(path: str | Path | None = None) -> list[RagEvalRow]:
    dest = Path(path) if path is not None else DEFAULT_EVAL_PATH
    rows: list[RagEvalRow] = []
    with dest.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            rows.append(
                RagEvalRow(
                    query_id=str(raw["query_id"]),
                    query=str(raw["query"]),
                    gold_chunk_ids=[str(x) for x in raw.get("gold_chunk_ids") or []],
                    gold_answer=raw.get("gold_answer"),
                    source=str(raw.get("source") or "grounded_pair"),
                    pair_id=raw.get("pair_id"),
                    task=str(raw.get("task") or "qa"),
                    source_id=str(raw.get("source_id") or "unknown"),
                    metadata=dict(raw.get("metadata") or {}),
                )
            )
    return rows
