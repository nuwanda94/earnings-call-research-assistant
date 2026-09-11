"""Offline Okapi BM25 index over a versioned RAG corpus.

No third-party retriever package: tokenization + scoring use stdlib + the
already-declared numpy extra only if present. Persist JSON under
``data/rag/indices/bm25/`` so retrieve works without GPU or network.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from earnings_call_research_assistant.rag.corpus import (
    CORPUS_VERSION,
    DEFAULT_CORPUS_DIR,
    RagChunk,
    load_corpus_chunks,
    load_manifest,
)

TOKEN_RE = re.compile(r"[a-z0-9%]+", re.IGNORECASE)
DEFAULT_INDEX_DIR = Path("data") / "rag" / "indices" / "bm25"
BM25_K1 = 1.5
BM25_B = 0.75


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]


@dataclass
class Hit:
    chunk_id: str
    score: float
    rank: int
    source_id: str = ""
    section: str = ""
    text: str = ""
    bm25: float = 0.0
    dense: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Bm25Index:
    """In-memory BM25 plus the texts needed to return snippets."""

    version: str
    corpus_version: str
    n_docs: int
    avgdl: float
    k1: float
    b: float
    chunk_ids: list[str]
    source_ids: list[str]
    sections: list[str]
    texts: list[str]
    doc_lens: list[int]
    df: dict[str, int]
    tfs: list[dict[str, int]]
    created_utc: str = field(default_factory=_utc_now)

    def to_serializable(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "corpus_version": self.corpus_version,
            "n_docs": self.n_docs,
            "avgdl": self.avgdl,
            "k1": self.k1,
            "b": self.b,
            "chunk_ids": self.chunk_ids,
            "source_ids": self.source_ids,
            "sections": self.sections,
            "texts": self.texts,
            "doc_lens": self.doc_lens,
            "df": self.df,
            "tfs": self.tfs,
            "created_utc": self.created_utc,
        }

    @classmethod
    def from_serializable(cls, raw: dict[str, Any]) -> "Bm25Index":
        return cls(
            version=str(raw.get("version") or "bm25-v0.1"),
            corpus_version=str(raw.get("corpus_version") or CORPUS_VERSION),
            n_docs=int(raw["n_docs"]),
            avgdl=float(raw["avgdl"]),
            k1=float(raw.get("k1", BM25_K1)),
            b=float(raw.get("b", BM25_B)),
            chunk_ids=list(raw["chunk_ids"]),
            source_ids=list(raw.get("source_ids") or [""] * int(raw["n_docs"])),
            sections=list(raw.get("sections") or [""] * int(raw["n_docs"])),
            texts=list(raw.get("texts") or [""] * int(raw["n_docs"])),
            doc_lens=[int(x) for x in raw["doc_lens"]],
            df={str(k): int(v) for k, v in dict(raw.get("df") or {}).items()},
            tfs=[{str(k): int(v) for k, v in d.items()} for d in raw.get("tfs") or []],
            created_utc=str(raw.get("created_utc") or ""),
        )

    def score_query(self, query: str) -> list[float]:
        q_tokens = tokenize(query)
        if not q_tokens or self.n_docs == 0:
            return [0.0] * self.n_docs
        idf_cache: dict[str, float] = {}
        n = self.n_docs
        scores = [0.0] * n
        for term in q_tokens:
            df = self.df.get(term, 0)
            if df == 0:
                continue
            if term not in idf_cache:
                # Robertson–Sparck Jones IDF with +0.5 smoothing
                idf_cache[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            idf = idf_cache[term]
            for i, tf_map in enumerate(self.tfs):
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                dl = self.doc_lens[i] or 1
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                scores[i] += idf * (tf * (self.k1 + 1.0) / denom)
        return scores

    def retrieve(self, query: str, k: int = 5) -> list[Hit]:
        scores = self.score_query(query)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.chunk_ids[i]))
        hits: list[Hit] = []
        for rank, i in enumerate(order[: max(0, k)], start=1):
            hits.append(
                Hit(
                    chunk_id=self.chunk_ids[i],
                    score=float(scores[i]),
                    rank=rank,
                    source_id=self.source_ids[i],
                    section=self.sections[i],
                    text=self.texts[i],
                    bm25=float(scores[i]),
                    dense=None,
                )
            )
        return hits


def build_bm25_index(
    chunks: Sequence[RagChunk],
    *,
    corpus_version: str = CORPUS_VERSION,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> Bm25Index:
    chunk_ids: list[str] = []
    source_ids: list[str] = []
    sections: list[str] = []
    texts: list[str] = []
    doc_lens: list[int] = []
    tfs: list[dict[str, int]] = []
    df: dict[str, int] = {}
    for chunk in chunks:
        tokens = tokenize(chunk.text)
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        for tok in tf:
            df[tok] = df.get(tok, 0) + 1
        chunk_ids.append(chunk.chunk_id)
        source_ids.append(chunk.source_id)
        sections.append(chunk.section)
        texts.append(chunk.text)
        doc_lens.append(len(tokens))
        tfs.append(tf)
    n = len(chunk_ids)
    avgdl = (sum(doc_lens) / n) if n else 0.0
    return Bm25Index(
        version="bm25-v0.1",
        corpus_version=corpus_version,
        n_docs=n,
        avgdl=avgdl,
        k1=k1,
        b=b,
        chunk_ids=chunk_ids,
        source_ids=source_ids,
        sections=sections,
        texts=texts,
        doc_lens=doc_lens,
        df=df,
        tfs=tfs,
    )


def write_bm25_index(index: Bm25Index, out_dir: str | Path | None = None) -> Path:
    dest = Path(out_dir) if out_dir is not None else DEFAULT_INDEX_DIR
    dest.mkdir(parents=True, exist_ok=True)
    payload = index.to_serializable()
    (dest / "index.json").write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    meta = {
        "backend": "bm25",
        "version": index.version,
        "corpus_version": index.corpus_version,
        "n_docs": index.n_docs,
        "avgdl": index.avgdl,
        "k1": index.k1,
        "b": index.b,
        "created_utc": index.created_utc,
        "path": str(dest / "index.json"),
    }
    (dest / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return dest


def load_bm25_index(index_dir: str | Path) -> Bm25Index:
    path = Path(index_dir) / "index.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Bm25Index.from_serializable(raw)


def build_bm25_from_corpus(
    corpus_dir: str | Path | None = None,
    index_dir: str | Path | None = None,
) -> tuple[Bm25Index, Path]:
    cdir = Path(corpus_dir) if corpus_dir is not None else DEFAULT_CORPUS_DIR
    chunks = load_corpus_chunks(cdir)
    try:
        manifest = load_manifest(cdir)
        version = manifest.version
    except FileNotFoundError:
        version = CORPUS_VERSION
    index = build_bm25_index(chunks, corpus_version=version)
    dest = write_bm25_index(index, index_dir)
    return index, dest
