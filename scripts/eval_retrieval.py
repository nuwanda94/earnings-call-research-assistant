#!/usr/bin/env python3
"""Evaluate BM25 / dense / hybrid retrieval: Recall@k and nDCG@k.

CPU-safe. Builds corpus + hash-dense indices if they are missing.
Pass ``--run`` only when a sentence-transformers index already exists or you
want the eval script to rebuild dense with the ST model (downloads weights).

    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --backends bm25,hybrid
    python scripts/eval_retrieval.py --run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from earnings_call_research_assistant.rag.bm25 import DEFAULT_INDEX_DIR  # noqa: E402
from earnings_call_research_assistant.rag.corpus import DEFAULT_CORPUS_DIR  # noqa: E402
from earnings_call_research_assistant.rag.dense import DEFAULT_DENSE_DIR  # noqa: E402
from earnings_call_research_assistant.rag.eval_set import DEFAULT_EVAL_PATH  # noqa: E402
from earnings_call_research_assistant.rag.hybrid import DEFAULT_RRF_K  # noqa: E402
from earnings_call_research_assistant.rag.metrics import (  # noqa: E402
    DEFAULT_K_VALUES,
    DEFAULT_METRICS_OUT,
    evaluate_retrieval,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=ROOT / DEFAULT_EVAL_PATH)
    parser.add_argument("--corpus-dir", type=Path, default=ROOT / DEFAULT_CORPUS_DIR)
    parser.add_argument("--index-dir", type=Path, default=ROOT / DEFAULT_INDEX_DIR)
    parser.add_argument("--dense-dir", type=Path, default=ROOT / DEFAULT_DENSE_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / DEFAULT_METRICS_OUT,
        help="Machine-readable metrics JSON.",
    )
    parser.add_argument(
        "--k",
        type=str,
        default=",".join(str(k) for k in DEFAULT_K_VALUES),
        help="Comma-separated k values (default 1,3,5,10).",
    )
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument(
        "--backends",
        type=str,
        default="bm25,dense,hybrid",
        help="Comma-separated: bm25, dense, hybrid.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Rebuild dense with sentence-transformers if the index is missing.",
    )
    return parser.parse_args()


def _print_table(payload: dict) -> None:
    print(f"n_queries: {payload.get('n_queries')}")
    print(f"n_index_docs: {payload.get('n_index_docs')}")
    print(f"corpus_version: {payload.get('corpus_version')}")
    print(f"dense_backend: {payload.get('dense_backend')}")
    print(f"k_values: {payload.get('k_values')}")
    backends = payload.get("backends") or {}
    header = f"{'backend':<22} " + " ".join(f"R@{k:<4}" for k in payload.get("k_values") or []) 
    header += " " + " ".join(f"nDCG@{k:<3}" for k in payload.get("k_values") or [])
    print(header)
    for name, report in backends.items():
        rec = report.get("mean_recall") or {}
        nd = report.get("mean_ndcg") or {}
        ks = report.get("k_values") or payload.get("k_values") or []
        row = f"{report.get('backend', name):<22} "
        row += " ".join(f"{rec.get(f'@{k}', 0):<6.4f}" for k in ks)
        row += " "
        row += " ".join(f"{nd.get(f'@{k}', 0):<8.4f}" for k in ks)
        print(row)


def main() -> int:
    args = _parse_args()
    k_values = tuple(int(x.strip()) for x in args.k.split(",") if x.strip())
    backends = tuple(x.strip() for x in args.backends.split(",") if x.strip())
    payload = evaluate_retrieval(
        eval_path=args.eval_set,
        corpus_dir=args.corpus_dir,
        bm25_dir=args.index_dir,
        dense_dir=args.dense_dir,
        k_values=k_values,
        rrf_k=args.rrf_k,
        use_st_model=bool(args.run),
        backends=backends,
        out_path=args.out,
    )
    _print_table(payload)
    print(f"Wrote {payload.get('out_path')}")
    summary = {
        name: {
            "mean_recall": report.get("mean_recall"),
            "mean_ndcg": report.get("mean_ndcg"),
        }
        for name, report in (payload.get("backends") or {}).items()
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
