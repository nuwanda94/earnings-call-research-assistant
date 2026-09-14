#!/usr/bin/env python3
"""Rebuild the RAG eval set so every gold_chunk_id exists in the current corpus.

Prefers pairs regenerated from the versioned RAG corpus (not a stale pairs JSONL
from an older chunk inventory). Optionally re-runs retrieval metrics after rebuild.

Examples
--------
    python scripts/realign_rag_eval.py
    python scripts/realign_rag_eval.py --max-n 50 --eval-retrieval
    python scripts/realign_rag_eval.py --eval-retrieval --eval-generate --adapter-dir outputs/adapters/llama32-3b-ecra-sft
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from earnings_call_research_assistant.rag.corpus import (  # noqa: E402
    DEFAULT_CORPUS_DIR,
    load_corpus_chunks,
    load_manifest,
)
from earnings_call_research_assistant.rag.eval_set import (  # noqa: E402
    DEFAULT_EVAL_PATH,
    DEFAULT_SEED,
    DEFAULT_TARGET_N,
    build_rag_eval_set,
    write_eval_set,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-dir", type=Path, default=ROOT / DEFAULT_CORPUS_DIR)
    p.add_argument(
        "--pairs",
        type=Path,
        default=None,
        help="Optional pairs JSONL. Default: regenerate from corpus so IDs match.",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "ecra-sft-v0.1.0",
    )
    p.add_argument("--out", type=Path, default=ROOT / DEFAULT_EVAL_PATH)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-n", type=int, default=DEFAULT_TARGET_N)
    p.add_argument(
        "--from-pairs",
        action="store_true",
        help="Use data/processed/grounded_pairs.jsonl if present (may miss corpus IDs).",
    )
    p.add_argument("--eval-retrieval", action="store_true")
    p.add_argument("--eval-generate", action="store_true")
    p.add_argument(
        "--adapter-dir",
        type=Path,
        default=ROOT / "outputs" / "adapters" / "llama32-3b-ecra-sft",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    corpus_dir = args.corpus_dir
    if not (corpus_dir / "chunks.jsonl").is_file():
        print(f"Corpus missing: {corpus_dir / 'chunks.jsonl'}")
        print("Build it first: python scripts/build_rag_corpus.py --chunks data/processed/chunks.jsonl")
        return 1

    chunks = load_corpus_chunks(corpus_dir)
    man = load_manifest(corpus_dir)
    corpus_ids = {c.chunk_id for c in chunks}
    print(f"corpus N={man.n_chunks} unique_ids={len(corpus_ids)} version={man.version}")

    pairs_path = None
    if args.from_pairs:
        cand = ROOT / "data" / "processed" / "grounded_pairs.jsonl"
        pairs_path = cand if cand.is_file() else None
    if args.pairs is not None:
        pairs_path = args.pairs if args.pairs.is_file() else None

    rows, manifest = build_rag_eval_set(
        corpus_dir=corpus_dir,
        pairs_path=pairs_path,  # None => regenerate from corpus texts
        dataset_dir=args.dataset_dir if args.dataset_dir.is_dir() else None,
        seed=args.seed,
        max_n=args.max_n,
    )

    missing = [gid for r in rows for gid in r.gold_chunk_ids if gid not in corpus_ids]
    if missing:
        print(f"ERROR: {len(missing)} gold ids not in corpus (showing up to 5): {missing[:5]}")
        return 1
    if not rows:
        print("ERROR: empty eval set after alignment")
        return 1

    dest = write_eval_set(rows, args.out, manifest=manifest)
    print(f"Wrote {dest} n_queries={manifest.n_queries} gold_unique={manifest.n_unique_gold_chunks}")
    print(manifest.notes)
    print(json.dumps(manifest.to_dict(), indent=2)[:1200])

    if args.eval_retrieval:
        cmd = [sys.executable, str(ROOT / "scripts" / "eval_retrieval.py"), "--run"]
        print("+", " ".join(cmd))
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            return rc

    if args.eval_generate:
        cmd = [sys.executable, str(ROOT / "scripts" / "eval_rag_generate.py"), "--run"]
        if args.adapter_dir.is_dir():
            cmd += ["--adapter-dir", str(args.adapter_dir)]
        print("+", " ".join(cmd))
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
