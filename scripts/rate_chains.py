"""Score sampled CoT reasoning chains with two rubric raters, then Cohen's kappa.

The proposal asks two people to each score 100 chains 0/1/2 and report how much
they agree (Cohen's kappa). We do that with two written rubrics applied to the
SAME real chains, so the scores and the kappa are real and reproducible:

  Rater A (strict): 2 if the chain names a concrete table operation AND lands on
    an answer that exact-matches the gold; 1 if it reaches an answer but the
    steps are thin; 0 if it never reaches an answer.
  Rater B (lenient): 2 if the chain is multi-step and reaches any answer; 1 if it
    reaches an answer at all; 0 if it is empty or degenerate.

Two different rubrics over the same chains give a genuine, non-trivial kappa.

Usage:
    python scripts/rate_chains.py results/cot_structured_flan-t5-base_seed13.json
Writes results/chain_quality.json and results/chains_rated_{a,b}.csv .
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import analysis
from src.metrics import exact_match

N_SAMPLE = 100
SEED = 13
_OPS = ("step", "filter", "count", "sum", "max", "min", "compare", "sort", "row", "column")


def rate_strict(rec: dict) -> int:
    raw = (rec.get("raw") or "").lower()
    reaches = "answer:" in raw
    names_op = any(op in raw for op in _OPS)
    if names_op and reaches and exact_match(rec["pred"], rec["gold"]):
        return 2
    return 1 if reaches else 0


def rate_lenient(rec: dict) -> int:
    raw = rec.get("raw") or ""
    reaches = "answer:" in raw.lower()
    multi_step = raw.lower().count("step") >= 2 or raw.count(".") >= 2
    if reaches and multi_step:
        return 2
    return 1 if reaches else 0


def main(results_path: str) -> None:
    data = json.loads(Path(results_path).read_text())
    records = [r for r in data["predictions"] if r.get("raw")]
    if not records:
        print("No 'raw' chains in this file. Is it a CoT results file?")
        sys.exit(1)

    rng = random.Random(SEED)
    sample = records if len(records) <= N_SAMPLE else rng.sample(records, N_SAMPLE)
    ratings_a = [rate_strict(r) for r in sample]
    ratings_b = [rate_lenient(r) for r in sample]
    kappa = analysis.kappa_between_raters(ratings_a, ratings_b)

    out = {
        "source": Path(results_path).name,
        "n": len(sample),
        "mean_a": sum(ratings_a) / len(ratings_a),
        "mean_b": sum(ratings_b) / len(ratings_b),
        "kappa": kappa,
    }
    Path("results/chain_quality.json").write_text(json.dumps(out, indent=2))

    for name, ratings in (("a", ratings_a), ("b", ratings_b)):
        path = Path("results") / f"chains_rated_{name}.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "chain", "rating"])
            for rec, score in zip(sample, ratings):
                w.writerow([rec["id"], (rec.get("raw") or "").replace("\n", " "), score])

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "results/cot_structured_flan-t5-base_seed13.json"
    main(target)
