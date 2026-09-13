"""Dense retrieval over a versioned RAG corpus.

Default path is a deterministic hash embedding so dry-run / CI stays offline.
``--run`` on Kaggle loads ``sentence-transformers`` (or numpy cosine over a
precomputed ``embeddings.npy``) and writes ``data/rag/indices/dense/``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from earnings_call_research_assistant.rag.bm25 import Hit, tokenize
from earnings_call_research_assistant.rag.corpus import (
    CORPUS_VERSION,
    DEFAULT_CORPUS_DIR,
    RagChunk,
    load_corpus_chunks,
    load_manifest,
)

DEFAULT_DENSE_DIR = Path("data") / "rag" / "indices" / "dense"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HASH_DIM = 128


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _try_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except ImportError:  # pragma: no cover
        return None


def hash_embed_text(text: str, dim: int = HASH_DIM) -> list[float]:
    """Signed hashed bag-of-tokens embedding (no model download)."""
    vec = [0.0] * dim
    tokens = tokenize(text)
    if not tokens:
        return vec
    for tok in tokens:
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = sum(x * x for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def embed_texts(
    texts: Sequence[str],
    *,
    model_name: str = DEFAULT_EMBED_MODEL,
    use_model: bool = False,
    dim: int = HASH_DIM,
) -> list[list[float]]:
    if use_model:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for --run dense embeddings"
            ) from exc
        model = SentenceTransformer(model_name)
        matrix = model.encode(
            list(texts),
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [list(map(float, row)) for row in matrix]
    return [hash_embed_text(t, dim=dim) for t in texts]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    return float(sum(a[i] * b[i] for i in range(n)))


@dataclass
class DenseIndex:
    version: str
    corpus_version: str
    embed_model: str
    backend: str
    dim: int
    chunk_ids: list[str]
    source_ids: list[str]
    sections: list[str]
    texts: list[str]
    embeddings: list[list[float]]
    created_utc: str = field(default_factory=_utc_now)

    def encode_query(self, query: str, use_model: bool | None = None) -> list[float]:
        want_model = self.backend == "sentence-transformers" if use_model is None else use_model
        rows = embed_texts(
            [query],
            model_name=self.embed_model,
            use_model=want_model,
            dim=self.dim,
        )
        return rows[0]

    def score_query(self, query: str, use_model: bool | None = None) -> list[float]:
        q = self.encode_query(query, use_model=use_model)
        return [_dot(q, row) for row in self.embeddings]

    def retrieve(self, query: str, k: int = 5, use_model: bool | None = None) -> list[Hit]:
        scores = self.score_query(query, use_model=use_model)
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
                    bm25=0.0,
                    dense=float(scores[i]),
                )
            )
        return hits


def build_dense_index(
    chunks: Sequence[RagChunk],
    *,
    corpus_version: str = CORPUS_VERSION,
    embed_model: str = DEFAULT_EMBED_MODEL,
    use_model: bool = False,
    dim: int = HASH_DIM,
) -> DenseIndex:
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts, model_name=embed_model, use_model=use_model, dim=dim)
    actual_dim = len(embeddings[0]) if embeddings else dim
    return DenseIndex(
        version="dense-v0.1",
        corpus_version=corpus_version,
        embed_model=embed_model if use_model else "hash-bow",
        backend="sentence-transformers" if use_model else "hash",
        dim=actual_dim,
        chunk_ids=[c.chunk_id for c in chunks],
        source_ids=[c.source_id for c in chunks],
        sections=[c.section for c in chunks],
        texts=texts,
        embeddings=embeddings,
    )


def write_dense_index(index: DenseIndex, out_dir: str | Path | None = None) -> Path:
    dest = Path(out_dir) if out_dir is not None else DEFAULT_DENSE_DIR
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": index.version,
        "corpus_version": index.corpus_version,
        "embed_model": index.embed_model,
        "backend": index.backend,
        "dim": index.dim,
        "chunk_ids": index.chunk_ids,
        "source_ids": index.source_ids,
        "sections": index.sections,
        "texts": index.texts,
        "created_utc": index.created_utc,
    }
    (dest / "ids.json").write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    np = _try_numpy()
    if np is not None:
        np.save(dest / "embeddings.npy", np.asarray(index.embeddings, dtype="float32"))
    else:
        (dest / "embeddings.json").write_text(
            json.dumps(index.embeddings) + "\n", encoding="utf-8"
        )
    meta = {
        "backend": index.backend,
        "version": index.version,
        "corpus_version": index.corpus_version,
        "embed_model": index.embed_model,
        "dim": index.dim,
        "n_docs": len(index.chunk_ids),
        "created_utc": index.created_utc,
        "path": str(dest),
    }
    (dest / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return dest


def load_dense_index(index_dir: str | Path) -> DenseIndex:
    dest = Path(index_dir)
    raw = json.loads((dest / "ids.json").read_text(encoding="utf-8"))
    embeddings: list[list[float]]
    npy = dest / "embeddings.npy"
    js = dest / "embeddings.json"
    if npy.is_file():
        np = _try_numpy()
        if np is None:
            raise RuntimeError("numpy required to load embeddings.npy")
        matrix = np.load(npy)
        embeddings = [list(map(float, row)) for row in matrix]
    elif js.is_file():
        embeddings = json.loads(js.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError(f"no embeddings under {dest}")
    return DenseIndex(
        version=str(raw.get("version") or "dense-v0.1"),
        corpus_version=str(raw.get("corpus_version") or CORPUS_VERSION),
        embed_model=str(raw.get("embed_model") or "hash-bow"),
        backend=str(raw.get("backend") or "hash"),
        dim=int(raw.get("dim") or (len(embeddings[0]) if embeddings else HASH_DIM)),
        chunk_ids=list(raw["chunk_ids"]),
        source_ids=list(raw.get("source_ids") or [""] * len(raw["chunk_ids"])),
        sections=list(raw.get("sections") or [""] * len(raw["chunk_ids"])),
        texts=list(raw.get("texts") or [""] * len(raw["chunk_ids"])),
        embeddings=embeddings,
        created_utc=str(raw.get("created_utc") or ""),
    )


def build_dense_from_corpus(
    corpus_dir: str | Path | None = None,
    index_dir: str | Path | None = None,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    use_model: bool = False,
) -> tuple[DenseIndex, Path]:
    cdir = Path(corpus_dir) if corpus_dir is not None else DEFAULT_CORPUS_DIR
    chunks = load_corpus_chunks(cdir)
    try:
        manifest = load_manifest(cdir)
        version = manifest.version
    except FileNotFoundError:
        version = CORPUS_VERSION
    index = build_dense_index(
        chunks,
        corpus_version=version,
        embed_model=embed_model,
        use_model=use_model,
    )
    dest = write_dense_index(index, index_dir)
    return index, dest
