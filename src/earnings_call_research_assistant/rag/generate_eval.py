"""Grounded answer accuracy: retrieve → pack context → generate → score.

CPU dry-run by default (no weights). ``--run`` on Kaggle loads InferenceHarness
for base and optional adapter. Metrics: citation-hit against retrieved text,
optional token F1 vs gold_answer. Do not treat dry-run percentages as resume numbers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from earnings_call_research_assistant.eval.metrics import citation_hit_rate, tokenize, token_overlap
from earnings_call_research_assistant.rag.bm25 import DEFAULT_INDEX_DIR, build_bm25_from_corpus, load_bm25_index
from earnings_call_research_assistant.rag.corpus import DEFAULT_CORPUS_DIR
from earnings_call_research_assistant.rag.dense import DEFAULT_DENSE_DIR, build_dense_from_corpus, load_dense_index
from earnings_call_research_assistant.rag.eval_set import DEFAULT_EVAL_PATH, RagEvalRow, build_rag_eval_set, load_eval_set, write_eval_set
from earnings_call_research_assistant.rag.hybrid import DEFAULT_RRF_K, HybridRetriever

DEFAULT_GEN_OUT = Path("evals") / "reports" / "rag_generation_metrics.json"
DEFAULT_TOP_K = 5
DEFAULT_MAX_CONTEXT_CHARS = 3500
GenerateFn = Callable[[str], str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pack_context(hits: Sequence[Any], *, max_chars: int = DEFAULT_MAX_CONTEXT_CHARS) -> str:
    """Join retrieved passages with source tags; truncate to max_chars."""
    parts: list[str] = []
    used = 0
    for hit in hits:
        cid = getattr(hit, "chunk_id", "") or ""
        src = getattr(hit, "source_id", "") or ""
        text = (getattr(hit, "text", "") or "").strip()
        if not text:
            continue
        header = f"[{cid} | {src}]"
        block = f"{header}\n{text}"
        if used and used + 2 + len(block) > max_chars:
            remain = max_chars - used - 2
            if remain > 40:
                parts.append(block[:remain] + "…")
            break
        parts.append(block)
        used += len(block) + (2 if parts else 0)
        if used >= max_chars:
            break
    return "\n\n".join(parts)


def build_grounded_prompt(query: str, packed: str) -> str:
    return (
        "Use only the retrieved excerpts below. Cite numbers that appear in them. "
        "If the excerpts do not contain the answer, say the information is not in the context.\n\n"
        f"Excerpts:\n{packed}\n\nQuestion:\n{query}"
    )


def _needles_from_gold(gold_answer: str | None, packed: str) -> list[str]:
    """Citation needles: gold tokens that also appear in retrieved context."""
    if not gold_answer:
        toks = sorted(tokenize(packed))
        return [t for t in toks if len(t) >= 4][:8]
    gold_toks = tokenize(gold_answer)
    ctx = tokenize(packed)
    shared = sorted(gold_toks & ctx)
    if shared:
        return shared[:12]
    return sorted(gold_toks)[:8]


def score_generation(
    answer: str,
    packed: str,
    gold_answer: str | None,
) -> dict[str, Any]:
    needles = _needles_from_gold(gold_answer, packed)
    citation = citation_hit_rate(answer, needles)
    vs_ctx = token_overlap(answer, packed)
    vs_gold = token_overlap(answer, gold_answer or "")
    cite_rate = citation.get("rate")
    grounded = 0.0
    if cite_rate is not None and cite_rate >= 0.5 and vs_ctx.get("f1", 0.0) >= 0.05:
        grounded = 1.0
    elif gold_answer and vs_gold.get("f1", 0.0) >= 0.3 and vs_ctx.get("f1", 0.0) >= 0.1:
        grounded = 1.0
    return {
        "citation": citation,
        "token_f1_vs_context": vs_ctx.get("f1", 0.0),
        "token_f1_vs_gold": None if not gold_answer else vs_gold.get("f1", 0.0),
        "grounded_hit": grounded,
    }


def _placeholder(side: str, packed: str) -> str:
    n = len(packed)
    return (
        f"[{side} dry-run] No weights loaded. Packed context is {n} chars. "
        "Run on Kaggle with --run (and optional --adapter-dir) to fill generations."
    )


def _ensure_eval_rows(eval_path: Path) -> list[RagEvalRow]:
    if eval_path.is_file():
        rows = load_eval_set(eval_path)
        if rows:
            return rows
    rows, manifest = build_rag_eval_set()
    write_eval_set(rows, eval_path, manifest=manifest)
    return rows


def _mean(values: Sequence[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 6)


@dataclass
class SideResult:
    output: str
    scores: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryGenResult:
    query_id: str
    query: str
    gold_chunk_ids: list[str]
    retrieved_ids: list[str]
    packed_chars: int
    gold_answer: str | None
    base: SideResult
    adapter: SideResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "gold_chunk_ids": self.gold_chunk_ids,
            "retrieved_ids": self.retrieved_ids,
            "packed_chars": self.packed_chars,
            "gold_answer": self.gold_answer,
            "base": self.base.to_dict(),
            "adapter": self.adapter.to_dict(),
        }


def evaluate_rag_generate(
    *,
    eval_path: str | Path | None = None,
    corpus_dir: str | Path | None = None,
    bm25_dir: str | Path | None = None,
    dense_dir: str | Path | None = None,
    out_path: str | Path | None = None,
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    use_st_model: bool = False,
    dry_run: bool = True,
    adapter_dir: str | None = None,
    base_generate: GenerateFn | None = None,
    adapter_generate: GenerateFn | None = None,
) -> dict[str, Any]:
    eval_file = Path(eval_path) if eval_path is not None else DEFAULT_EVAL_PATH
    rows = _ensure_eval_rows(eval_file)
    corpus = Path(corpus_dir) if corpus_dir is not None else DEFAULT_CORPUS_DIR
    bdir = Path(bm25_dir) if bm25_dir is not None else DEFAULT_INDEX_DIR
    ddir = Path(dense_dir) if dense_dir is not None else DEFAULT_DENSE_DIR

    if not (bdir / "index.json").is_file():
        build_bm25_from_corpus(corpus, bdir)
    bm25 = load_bm25_index(bdir)
    dense = None
    if (ddir / "ids.json").is_file():
        dense = load_dense_index(ddir)
    elif use_st_model:
        build_dense_from_corpus(corpus, ddir, use_model=True)
        dense = load_dense_index(ddir)
    else:
        if not (ddir / "ids.json").is_file():
            build_dense_from_corpus(corpus, ddir, use_model=False)
            dense = load_dense_index(ddir)

    retriever = HybridRetriever(bm25=bm25, dense=dense, rrf_k=rrf_k)
    items: list[QueryGenResult] = []
    for row in rows:
        hits = retriever.retrieve(row.query, k=top_k)
        packed = pack_context(hits, max_chars=max_context_chars)
        prompt = build_grounded_prompt(row.query, packed)
        if dry_run or base_generate is None:
            base_out = _placeholder("base", packed)
        else:
            base_out = base_generate(prompt)
        if dry_run or adapter_generate is None:
            adapter_out = _placeholder("adapter", packed)
        else:
            adapter_out = adapter_generate(prompt)
        items.append(
            QueryGenResult(
                query_id=row.query_id,
                query=row.query,
                gold_chunk_ids=list(row.gold_chunk_ids),
                retrieved_ids=[h.chunk_id for h in hits],
                packed_chars=len(packed),
                gold_answer=row.gold_answer,
                base=SideResult(base_out, score_generation(base_out, packed, row.gold_answer)),
                adapter=SideResult(adapter_out, score_generation(adapter_out, packed, row.gold_answer)),
            )
        )

    def agg(side: str) -> dict[str, Any]:
        sides = [getattr(it, side).scores for it in items]
        return {
            "citation_hit_rate": _mean(
                s["citation"].get("rate") for s in sides
            ),
            "mean_token_f1_vs_context": _mean(s.get("token_f1_vs_context") for s in sides),
            "mean_token_f1_vs_gold": _mean(s.get("token_f1_vs_gold") for s in sides),
            "grounded_answer_accuracy": _mean(s.get("grounded_hit") for s in sides),
        }

    payload: dict[str, Any] = {
        "created_utc": _utc_now(),
        "eval_set": str(eval_file),
        "n_queries": len(items),
        "top_k": top_k,
        "rrf_k": rrf_k,
        "max_context_chars": max_context_chars,
        "corpus_dir": str(corpus),
        "bm25_dir": str(bdir),
        "dense_dir": str(ddir),
        "dense_backend": None if dense is None else dense.backend,
        "n_index_docs": bm25.n_docs,
        "corpus_version": bm25.corpus_version,
        "dry_run": dry_run,
        "adapter_dir": adapter_dir,
        "note": (
            "Dry-run placeholders score near zero by design. Fill generations with "
            "--run on Kaggle before quoting grounded_answer_accuracy."
        ),
        "aggregate": {"base": agg("base"), "adapter": agg("adapter")},
        "items": [it.to_dict() for it in items],
    }
    dest = Path(out_path) if out_path is not None else DEFAULT_GEN_OUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    payload["out_path"] = str(dest)
    return payload
