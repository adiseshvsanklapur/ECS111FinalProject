"""Aggregate per-(condition, seed) result JSONs into the reported numbers.

- mean +/- std of the primary metric across seeds, per condition
- McNemar's test between two conditions (paired per-example correctness)
- Cohen's kappa between two chain-quality raters
- summary table (CSV + Markdown) and bar plots

Pure aggregation over the JSON schema written by evaluate.py; no torch needed.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save figures, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .metrics import cohen_kappa, mcnemar_test  # noqa: E402


def primary_metric(result: dict) -> tuple[str, float]:
    """The headline metric for a result: EM (wtq) or classification accuracy (tabfact)."""
    if result["task"] == "tabfact":
        return "classification_accuracy", result["metrics"]["classification_accuracy"]
    return "exact_match", result["metrics"]["exact_match"]


def load_results(results_dir: Path | str = config.RESULTS_DIR) -> list[dict]:
    """Load every per-condition result JSON in results_dir.

    Skips JSON files that are not per-condition results (e.g. chain_quality.json)
    so helper files can sit next to the results without breaking aggregation.
    """
    results_dir = Path(results_dir)
    out = []
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "condition" in data and "predictions" in data:
            out.append(data)
    return out


def aggregate(results: list[dict]) -> pd.DataFrame:
    """Group results by (condition, model, task); mean/std of primary metric over seeds."""
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        groups.setdefault((r["condition"], r["model"], r["task"]), []).append(r)

    rows = []
    for (condition, model, task), items in groups.items():
        metric_name, _ = primary_metric(items[0])
        values = [primary_metric(r)[1] for r in items]
        spe = [r["compute"]["seconds_per_example"] for r in items]
        rows.append(
            {
                "condition": condition,
                "model": model.split("/")[-1],
                "task": task,
                "metric": metric_name,
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "n_seeds": len(values),
                "sec_per_example": statistics.fmean(spe),
            }
        )
    df = pd.DataFrame(rows).sort_values(["task", "condition", "model"]).reset_index(drop=True)
    return df


def _correct_by_id(result: dict) -> dict[str, bool]:
    return {rec["id"]: rec["correct"] for rec in result["predictions"]}


def mcnemar_between(result_a: dict, result_b: dict) -> dict:
    """Paired McNemar on per-example correctness over the ids common to both results."""
    a, b = _correct_by_id(result_a), _correct_by_id(result_b)
    ids = [i for i in a if i in b]
    if not ids:
        raise ValueError("No overlapping example ids between the two results.")
    return mcnemar_test([a[i] for i in ids], [b[i] for i in ids])


def kappa_between_raters(ratings_a: list[int], ratings_b: list[int]) -> float:
    """Cohen's kappa for two chain-quality raters (0/1/2 scores)."""
    return cohen_kappa(ratings_a, ratings_b)


def load_rater_csv(path: Path | str) -> list[int]:
    """Load a rater file: CSV with an 'id' and a 'rating' column, sorted by id."""
    df = pd.read_csv(path).sort_values("id")
    return df["rating"].astype(int).tolist()


def write_summary_table(df: pd.DataFrame, out_dir: Path | str = config.RESULTS_DIR) -> dict[str, Path]:
    """Write the summary table as CSV and Markdown. Returns the two paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary_table.csv"
    md_path = out_dir / "summary_table.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False))
    return {"csv": csv_path, "md": md_path}


def plot_primary_metric(df: pd.DataFrame, out_path: Path | str) -> Path:
    """Bar chart of mean primary metric per condition (error bars = std across seeds)."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f"{r.condition}\n({r.model})" for r in df.itertuples()]
    ax.bar(labels, df["mean"], yerr=df["std"], capsize=4)
    ax.set_ylabel("primary metric (EM / classification acc)")
    ax.set_title("Primary metric by condition (mean ± std over seeds)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_error_distribution(results: list[dict], out_path: Path | str) -> Path:
    """Stacked bar of WTQ error-type counts per condition."""
    out_path = Path(out_path)
    wtq = [r for r in results if r["task"] == "wtq" and r["error_distribution"]]
    types = [t for t in config.ERROR_TYPES if t != "correct"]
    labels = [f"{r['condition']}/{r['model'].split('/')[-1]}/s{r['seed']}" for r in wtq]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = [0] * len(wtq)
    for et in types:
        vals = [r["error_distribution"].get(et, 0) for r in wtq]
        ax.bar(labels, vals, bottom=bottom, label=et)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("error count")
    ax.set_title("WTQ error-type distribution by condition")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
