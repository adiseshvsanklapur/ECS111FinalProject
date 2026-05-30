"""Smoke tests for analysis aggregation, McNemar, kappa, and plotting.

Uses synthetic result dicts (no model needed) matching evaluate.py's schema.
"""

import json

from src.analysis import (
    aggregate,
    kappa_between_raters,
    load_results,
    mcnemar_between,
    plot_error_distribution,
    plot_primary_metric,
    write_summary_table,
)


def _wtq_result(seed, em, correctness):
    """Synthetic WTQ result; correctness is a list of bools, one per example id."""
    preds = []
    dist = {"lookup": 0, "aggregation": 0, "multi_hop": 0, "correct": 0}
    for i, ok in enumerate(correctness):
        et = "correct" if ok else "lookup"
        dist[et] += 1
        preds.append({"id": str(i), "pred": "x", "gold": "x", "correct": ok, "error_type": et})
    return {
        "condition": "demo",
        "model": "google/flan-t5-base",
        "seed": seed,
        "task": "wtq",
        "n": len(correctness),
        "metrics": {"exact_match": em, "token_f1": em},
        "error_distribution": dist,
        "compute": {"seconds_per_example": 0.1, "peak_memory_mb": None, "device": "cpu"},
        "predictions": preds,
    }


def test_load_and_aggregate(tmp_path):
    r1 = _wtq_result(13, 0.5, [True, True, False, False])
    r2 = _wtq_result(42, 0.7, [True, True, True, False])
    (tmp_path / "demo_flan-t5-base_seed13.json").write_text(json.dumps(r1))
    (tmp_path / "demo_flan-t5-base_seed42.json").write_text(json.dumps(r2))

    results = load_results(tmp_path)
    assert len(results) == 2

    df = aggregate(results)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["metric"] == "exact_match"
    assert abs(row["mean"] - 0.6) < 1e-9
    assert abs(row["std"] - 0.1) < 1e-9  # population stdev of {0.5, 0.7}
    assert row["n_seeds"] == 2


def test_mcnemar_between():
    a = _wtq_result(13, 0.5, [True, True, False, False])
    b = _wtq_result(13, 0.5, [True, False, True, False])
    out = mcnemar_between(a, b)
    assert out["b"] == 1 and out["c"] == 1
    assert out["n_discordant"] == 2
    assert abs(out["pvalue"] - 1.0) < 1e-9


def test_kappa_perfect_agreement():
    assert abs(kappa_between_raters([0, 1, 2, 1], [0, 1, 2, 1]) - 1.0) < 1e-9


def test_plots_and_table(tmp_path):
    results = [
        _wtq_result(13, 0.5, [True, False, False]),
        _wtq_result(42, 0.6, [True, True, False]),
    ]
    df = aggregate(results)
    paths = write_summary_table(df, out_dir=tmp_path)
    assert paths["csv"].exists() and paths["md"].exists()

    p1 = plot_primary_metric(df, tmp_path / "metric.png")
    p2 = plot_error_distribution(results, tmp_path / "errors.png")
    assert p1.exists() and p2.exists()
