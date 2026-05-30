"""Sample reasoning chains from a CoT results file for the two human raters.

The proposal asks two team members to each score 100 reasoning chains 0/1/2 and
then compute Cohen's kappa. This script pulls the chains out of a CoT results
JSON (the `raw` field saved by evaluate.py) and writes a CSV the raters fill in.

Usage:
    python scripts/export_chains.py results/cot_plain_flan-t5-base_seed13.json
Writes results/chains_to_rate_<condition>_<seed>.csv with columns: id, chain, rating
Each rater copies the file, fills the rating column with 0/1/2, then:
    analysis.load_rater_csv(path_a), analysis.load_rater_csv(path_b)
    analysis.kappa_between_raters(...)
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

N_SAMPLE = 100
SEED = 13


def main(results_path: str) -> None:
    data = json.loads(Path(results_path).read_text())
    records = [r for r in data["predictions"] if r.get("raw")]
    if not records:
        print("No 'raw' chains in this file. Is it a CoT results file?")
        sys.exit(1)

    rng = random.Random(SEED)
    sample = records if len(records) <= N_SAMPLE else rng.sample(records, N_SAMPLE)

    out = Path("results") / f"chains_to_rate_{data['condition']}_seed{data['seed']}.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "chain", "rating"])
        for r in sample:
            w.writerow([r["id"], r["raw"].replace("\n", " "), ""])
    print(f"wrote {len(sample)} chains to {out}")
    print("Two raters each fill the 'rating' column (0/1/2), then compute Cohen's kappa.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/export_chains.py <cot_results.json>")
        sys.exit(1)
    main(sys.argv[1])
