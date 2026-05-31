"""Fill report/REPORT.md numeric tokens from results/*.json (deterministic).

Numeric tokens like [EM: baseline base] or [TabFact acc: finetune_traces] are
replaced in place with mean +/- std pulled straight from the result files. Prose
tokens ([FILL: ...]) are left alone -- those need a written sentence, so the
script just lists the ones still open. It exits non-zero if a numeric token is
left unmapped (that means a condition is missing or a token name drifted).

Usage:
    python scripts/finalize_report.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import analysis, report_fill

REPORT = Path("report/REPORT.md")
NUMERIC_TOKEN = re.compile(r"\[(?:EM|F1|ACC|TabFact|ERR|MCNEMAR|KAPPA|EM_GAP|TIME)[^\]]*\]")
PROSE_TOKEN = re.compile(r"\[FILL[^\]]*\]")


def main() -> None:
    results = analysis.load_results()
    cq_path = Path("results/chain_quality.json")
    chain_quality = (json.loads(cq_path.read_text()) if cq_path.exists()
                     else {"kappa": 0.0, "mean_a": 0.0, "mean_b": 0.0})
    token_map = report_fill.build_token_map(results, chain_quality)

    text = REPORT.read_text()
    for token, value in token_map.items():
        text = text.replace(token, value)
    REPORT.write_text(text)

    leftover = NUMERIC_TOKEN.findall(text)
    prose = PROSE_TOKEN.findall(text)

    print(f"Filled numeric tokens in {REPORT} ({len(token_map)} known tokens).")
    if prose:
        print("\nPROSE TOKENS STILL OPEN (write grounded sentences from results/summary_table.md):")
        for p in prose:
            print("  ", p)
    if leftover:
        print("\nERROR: numeric tokens with no mapping (missing condition or renamed token):")
        for t in sorted(set(leftover)):
            print("  ", t)
        sys.exit(1)


if __name__ == "__main__":
    main()
