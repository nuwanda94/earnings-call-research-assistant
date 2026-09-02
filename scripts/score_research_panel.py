#!/usr/bin/env python3
"""Score a research-panel comparison JSON (token overlap + citation hits).

CPU-only. Does not train or load weights. If the comparison file is missing,
writes a dry-run comparison first, then scores it.

    python scripts/score_research_panel.py
    python scripts/score_research_panel.py --comparison evals/reports/research_panel_comparison.json
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

from earnings_call_research_assistant.eval.metrics import (  # noqa: E402
    DEFAULT_COMPARISON,
    DEFAULT_METRICS_OUT,
    score_research_panel,
)
from earnings_call_research_assistant.eval.panel import (  # noqa: E402
    DEFAULT_PANEL,
    run_research_panel,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        type=Path,
        default=DEFAULT_COMPARISON,
        help="Side-by-side JSON from scripts/eval_research_panel.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_METRICS_OUT,
        help="Metrics JSON path.",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=DEFAULT_PANEL,
        help="Used only when --comparison is missing (dry-run generate).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.comparison.exists():
        run_research_panel(args.panel, args.comparison, dry_run=True)
    summary = score_research_panel(args.comparison, args.out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
