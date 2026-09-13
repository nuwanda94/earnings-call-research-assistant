#!/usr/bin/env python3
"""End-to-end Phase 5.9 driver for Kaggle (data -> SFT -> RAG metrics -> report -> optional git push).

Prefer invoking from notebooks/04_finetune_and_phase59.ipynb, or:

    python scripts/run_phase59_pipeline.py
    python scripts/run_phase59_pipeline.py --train --rag --max-steps 60
    python scripts/run_phase59_pipeline.py --train --rag --publish-hf --push-github

Defaults are dry-run safe. Never invents metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)

IST = timezone(timedelta(hours=5, minutes=30))


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def _load_secret(name: str, in_kaggle: bool) -> bool:
    if os.environ.get(name):
        print(f"{name}: already in env")
        return True
    if not in_kaggle:
        print(f"{name}: not set")
        return False
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        val = UserSecretsClient().get_secret(name)
        if val:
            os.environ[name] = val
            print(f"{name}: loaded from Kaggle secrets")
            return True
        print(f"{name}: secret empty")
    except Exception as exc:
        print(f"{name}: {type(exc).__name__}")
    return False


def _fmt(x, digits=4):
    if x is None:
        return "TBD"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", action="store_true", help="Run Unsloth QLoRA train")
    p.add_argument("--rag", action="store_true", help="ST embeddings + generate eval")
    p.add_argument("--publish-hf", action="store_true")
    p.add_argument("--push-github", action="store_true")
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--max-samples", type=int, default=8)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--hf-repo-id", default="nuwanda94/llama32-3b-ecra-sft")
    p.add_argument("--github-owner", default="nuwanda94")
    p.add_argument("--github-repo", default="earnings-call-research-assistant")
    p.add_argument("--github-branch", default="main")
    args = p.parse_args()

    now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    in_kaggle = Path("/kaggle").exists()
    adapter_dir = Path("outputs/adapters/llama32-3b-ecra-sft")

    print("flags train", args.train, "rag", args.rag, "hf", args.publish_hf, "gh", args.push_github)
    print("NOW_IST", now_ist)

    has_hf = _load_secret("HF_TOKEN", in_kaggle) or _load_secret("HUGGING_FACE_HUB_TOKEN", in_kaggle)
    has_gh = _load_secret("GITHUB_TOKEN", in_kaggle)

    from earnings_call_research_assistant.data import (
        DATASET_VERSION,
        ChunkConfig,
        FilterConfig,
        GenerateConfig,
        SelectConfig,
        chunk_records,
        filter_pairs,
        generate_pairs,
        ingest_catalog,
        select_and_split,
        write_chunks_jsonl,
        write_filter_report,
        write_jsonl,
        write_pairs_jsonl,
        write_splits,
    )

    raw = Path("data/raw/public_sample.jsonl")
    chunks_path = Path("data/processed/chunks.jsonl")
    pairs_path = Path("data/processed/grounded_pairs.jsonl")
    filtered_path = Path("data/processed/filtered_pairs.jsonl")
    out_dir = Path("data/processed") / DATASET_VERSION

    records = ingest_catalog(max_samples=args.max_samples, download=False)
    write_jsonl(records, raw)
    print(f"ingest: {len(records)} -> {raw}")

    chunks = chunk_records(records, config=ChunkConfig(window_sentences=4, stride_sentences=2))
    write_chunks_jsonl(chunks, chunks_path)
    print(f"chunks: {len(chunks)} -> {chunks_path}")

    pairs = generate_pairs(
        chunks, config=GenerateConfig(max_qa_per_chunk=2, include_summary=True, use_llm=False)
    )
    write_pairs_jsonl(pairs, pairs_path)
    print(f"pairs: {len(pairs)} -> {pairs_path}")

    kept, report = filter_pairs(
        pairs, config=FilterConfig(min_output_chars=40, near_dup_jaccard=0.88, use_llm_judge=False)
    )
    write_pairs_jsonl(kept, filtered_path)
    write_filter_report(report, Path("data/processed/filter_report.json"))
    print(f"filtered: in={report.n_in} kept={report.n_kept}")

    sel_cfg = SelectConfig(
        target_min=1,
        target_max=6000,
        max_per_source=2500,
        diversity_jaccard_cap=0.72,
        seed=94,
        dataset_version=DATASET_VERSION,
    )
    splits, sel_report = select_and_split(kept, config=sel_cfg)
    write_splits(splits, out_dir, report=sel_report, config=sel_cfg)
    print(f"splits: train={sel_report.n_train} val={sel_report.n_val} test={sel_report.n_test}")

    from earnings_call_research_assistant.training.sft import run_sft

    plan = run_sft(
        config_path=args.config,
        dataset_dir=out_dir,
        dry_run=True,
        max_steps=args.max_steps,
        require_train=False,
    )
    print(f"dry_run plan model={plan.model_name} seed={plan.seed} adapter={plan.adapter_dir}")

    if args.train:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("--train requires CUDA (Kaggle T4)")
        train_plan = run_sft(
            config_path=args.config,
            dataset_dir=out_dir,
            dry_run=False,
            max_steps=args.max_steps,
            require_train=True,
        )
        print("Train finished:", train_plan.adapter_dir)

    _run([sys.executable, "scripts/build_rag_corpus.py"])
    _run([sys.executable, "scripts/build_rag_eval_set.py"])
    if args.rag:
        _run([
            sys.executable,
            "scripts/build_rag_index.py",
            "--run",
            "--query",
            "operating margin guidance",
            "--k",
            "5",
        ])
        _run([sys.executable, "scripts/eval_retrieval.py", "--run"])
    else:
        _run([
            sys.executable,
            "scripts/build_rag_index.py",
            "--query",
            "operating margin guidance",
            "--k",
            "3",
        ])
        _run([sys.executable, "scripts/eval_retrieval.py"])

    manifest_path = Path("data/rag/corpus_v0.1.0/manifest.json")
    metrics_path = Path("evals/reports/rag_metrics.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    rag_metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    print("N chunks:", manifest.get("n_chunks"))

    if args.rag and (args.train or adapter_dir.is_dir()):
        cmd = [sys.executable, "scripts/eval_rag_generate.py", "--run"]
        if adapter_dir.is_dir():
            cmd += ["--adapter-dir", str(adapter_dir)]
        _run(cmd)
    else:
        _run([sys.executable, "scripts/eval_rag_generate.py"])

    gen_path = Path("evals/reports/rag_generation_metrics.json")
    gen_metrics = json.loads(gen_path.read_text()) if gen_path.is_file() else {}
    print("gen dry_run:", gen_metrics.get("dry_run"))

    if manifest:
        snap = Path("evals/reports/corpus_manifest_snapshot.json")
        snap.write_text(json.dumps(manifest, indent=2) + "\n")

    can_fill = bool(manifest) and bool(rag_metrics) and gen_metrics.get("dry_run") is False
    print("can_fill_report:", can_fill)

    if can_fill:
        n_chunks = manifest.get("n_chunks", "TBD")
        n_docs = manifest.get("n_documents", "TBD")
        version = manifest.get("version", "TBD")
        sources = manifest.get("sources", "TBD")
        backends = rag_metrics.get("backends") or {}
        hybrid = backends.get("hybrid") or {}
        bm25 = backends.get("bm25") or {}
        dense = backends.get("dense") or {}

        def row(b):
            mr, mn = b.get("mean_recall"), b.get("mean_ndcg")
            if isinstance(mr, dict) and isinstance(mn, dict):
                cells = [_fmt(mr.get(str(k), mr.get(k))) for k in (1, 3, 5, 10)]
                cells += [_fmt(mn.get(str(k), mn.get(k))) for k in (1, 3, 5, 10)]
                return cells
            return [_fmt(mr)] * 4 + [_fmt(mn)] * 4

        agg = gen_metrics.get("aggregate") or {}
        base = agg.get("base") or agg
        adapter = agg.get("adapter") or agg
        report_md = f"""# RAG Evaluation Report (Phase 5)

Date: {now_ist}
Corpus snapshot: `evals/reports/corpus_manifest_snapshot.json`
Retrieval: `evals/reports/rag_metrics.json`
Generation: `evals/reports/rag_generation_metrics.json`
Eval set: `evals/rag_eval_set.jsonl` (seed 3407)

## Corpus (N)

| Field | Value |
|-------|-------|
| N (chunks) | **{n_chunks}** |
| Documents | **{n_docs}** |
| Version | **{version}** |
| Sources | `{sources}` |

## Retrieval

| Backend | R@1 | R@3 | R@5 | R@10 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 |
|---------|-----|-----|-----|------|--------|--------|--------|---------|
| BM25 | {" | ".join(row(bm25))} |
| Dense | {" | ".join(row(dense))} |
| Hybrid RRF | {" | ".join(row(hybrid))} |

## Grounded generation

| Side | citation-hit | token F1 | dry_run |
|------|-------------:|---------:|---------|
| base | {_fmt(base.get("citation_hit", base.get("mean_citation_hit")))} | {_fmt(base.get("token_f1", base.get("mean_token_f1")))} | {gen_metrics.get("dry_run")} |
| adapter | {_fmt(adapter.get("citation_hit", adapter.get("mean_citation_hit")))} | {_fmt(adapter.get("token_f1", adapter.get("mean_token_f1")))} | {gen_metrics.get("dry_run")} |

Numbers copied from JSON only (`dry_run=false`).
"""
        Path("evals/reports/RAG_EVAL_REPORT.md").write_text(report_md)
        print("Updated RAG_EVAL_REPORT.md")

        readme = Path("README.md")
        if readme.is_file():
            text = readme.read_text()
            hybrids = hybrid.get("mean_recall")
            hybridn = hybrid.get("mean_ndcg")
            ground = adapter.get("citation_hit", adapter.get("mean_citation_hit"))
            for old, new in [
                (
                    "| N (chunk count) | corpus `manifest.json` \u2192 `n_chunks` | **TBD** |",
                    f"| N (chunk count) | corpus `manifest.json` \u2192 `n_chunks` | **{n_chunks}** |",
                ),
                (
                    "| Recall@k (hybrid) | `rag_metrics.json` \u2192 `backends.hybrid.mean_recall` | **TBD** |",
                    f"| Recall@k (hybrid) | `rag_metrics.json` \u2192 `backends.hybrid.mean_recall` | **{_fmt(hybrids)}** |",
                ),
                (
                    "| nDCG@k (hybrid) | `rag_metrics.json` \u2192 `backends.hybrid.mean_ndcg` | **TBD** |",
                    f"| nDCG@k (hybrid) | `rag_metrics.json` \u2192 `backends.hybrid.mean_ndcg` | **{_fmt(hybridn)}** |",
                ),
                (
                    "| Grounded answer accuracy | `rag_generation_metrics.json` \u2192 `aggregate` (and `dry_run=false`) | **TBD** |",
                    f"| Grounded answer accuracy | `rag_generation_metrics.json` \u2192 `aggregate` (and `dry_run=false`) | **{_fmt(ground)}** |",
                ),
            ]:
                text = text.replace(old, new)
                # also try arrow character variants
                text = text.replace(old.replace("\u2192", "\u2192"), new)
            # Prefer literal arrows as in README
            text = text.replace(
                "| N (chunk count) | corpus `manifest.json` → `n_chunks` | **TBD** |",
                f"| N (chunk count) | corpus `manifest.json` → `n_chunks` | **{n_chunks}** |",
            )
            text = text.replace(
                "| Recall@k (hybrid) | `rag_metrics.json` → `backends.hybrid.mean_recall` | **TBD** |",
                f"| Recall@k (hybrid) | `rag_metrics.json` → `backends.hybrid.mean_recall` | **{_fmt(hybrids)}** |",
            )
            text = text.replace(
                "| nDCG@k (hybrid) | `rag_metrics.json` → `backends.hybrid.mean_ndcg` | **TBD** |",
                f"| nDCG@k (hybrid) | `rag_metrics.json` → `backends.hybrid.mean_ndcg` | **{_fmt(hybridn)}** |",
            )
            text = text.replace(
                "| Grounded answer accuracy | `rag_generation_metrics.json` → `aggregate` (and `dry_run=false`) | **TBD** |",
                f"| Grounded answer accuracy | `rag_generation_metrics.json` → `aggregate` (and `dry_run=false`) | **{_fmt(ground)}** |",
            )
            readme.write_text(text)
            print("Patched README Results")

    if args.publish_hf and has_hf and adapter_dir.is_dir():
        _run([
            sys.executable,
            "scripts/publish_adapter.py",
            "--repo-id",
            args.hf_repo_id,
            "--run",
        ])
    else:
        _run([sys.executable, "scripts/publish_adapter.py"])

    progress = Path("PROGRESS.md")
    if progress.is_file():
        ptext = progress.read_text()
        if can_fill:
            ptext = ptext.replace(
                "- [ ] 5.9 Human Kaggle full run + publish adapter/metrics to HF + GitHub",
                "- [x] 5.9 Human Kaggle full run + publish adapter/metrics to HF + GitHub",
            )
            log_entry = (
                f"- {now_ist} — feat: Phase 5.9 via scripts/run_phase59_pipeline.py "
                f"(dry_run=false metrics)."
            )
        else:
            log_entry = (
                f"- {now_ist} — chore: phase59 pipeline partial run; "
                f"5.9 still waiting on dry_run=false metrics."
            )
        if "## Log" in ptext and log_entry not in ptext:
            ptext = ptext.replace("## Log\n", f"## Log\n\n{log_entry}\n", 1)
        progress.write_text(ptext)

    if args.push_github:
        if not has_gh:
            raise RuntimeError("--push-github requires GITHUB_TOKEN")
        files = [
            "evals/reports/rag_metrics.json",
            "evals/reports/rag_generation_metrics.json",
            "evals/reports/corpus_manifest_snapshot.json",
            "evals/reports/RAG_EVAL_REPORT.md",
            "README.md",
            "PROGRESS.md",
        ]
        existing = [f for f in files if Path(f).is_file()]
        token = os.environ["GITHUB_TOKEN"]
        remote = f"https://x-access-token:{token}@github.com/{args.github_owner}/{args.github_repo}.git"
        _run(["git", "config", "user.email", "kaggle-bot@users.noreply.github.com"])
        _run(["git", "config", "user.name", "kaggle-phase59"])
        _run(["git", "remote", "set-url", "origin", remote])
        for f in existing:
            _run(["git", "add", "-f", f])
        st = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        if not st:
            print("Nothing to commit")
        else:
            msg = f"chore: phase 5.9 metrics + report from Kaggle ({now_ist})"
            _run(["git", "commit", "-m", msg])
            _run(["git", "push", "origin", f"HEAD:{args.github_branch}"])
        _run([
            "git",
            "remote",
            "set-url",
            "origin",
            f"https://github.com/{args.github_owner}/{args.github_repo}.git",
        ])
        print("remote scrubbed")
    else:
        print("skip git push")

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
