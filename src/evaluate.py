"""Run a condition end-to-end: predict -> metrics -> error labels -> results JSON.

Produces one JSON per (condition, model, seed) under results/. analysis.py
aggregates those files across seeds and runs the statistical tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config
from .metrics import (
    exact_match,
    exact_match_score,
    map_tabfact_label,
    token_f1_score,
)
from .prompts import extract_answer
from .trainer import generate, peak_memory_mb

# Keyword cues for the WTQ error-type heuristic (proposal section 6).
_AGG_CUES = (
    "how many", "how much", "number of", "count", "total", "sum", "average",
    "mean", "most", "least", "largest", "smallest", "highest", "lowest",
    "maximum", "minimum", "max", "min", "longest", "shortest",
)
_MULTIHOP_CUES = (" and ", " after ", " before ", " than ", " also ", " both ", " then ")


def classify_question(question: str) -> str:
    """Heuristic reasoning type for an INCORRECT WTQ prediction.

    Returns one of "aggregation" | "multi_hop" | "lookup". This is an
    approximate label (proposal calls for a 3-way error breakdown); it keys off
    surface cues in the question, not gold logic.
    """
    q = f" {question.lower()} "
    if any(cue in q for cue in _AGG_CUES):
        return "aggregation"
    if any(cue in q for cue in _MULTIHOP_CUES):
        return "multi_hop"
    return "lookup"


def label_error(example: dict, correct: bool) -> str:
    """Per-example label: 'correct' or one of the three WTQ error types."""
    if correct:
        return "correct"
    return classify_question(example["question"])


def _is_correct(pred: str, example: dict, task: str) -> bool:
    if task == "tabfact":
        return map_tabfact_label(pred) == example["answer"]
    return exact_match(pred, example["answer"])


def save_results(results: dict) -> Path:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_short = results["model"].split("/")[-1]
    path = config.RESULTS_DIR / f"{results['condition']}_{model_short}_seed{results['seed']}.json"
    path.write_text(json.dumps(results, indent=2))
    return path


def predict_and_evaluate(
    model,
    tokenizer,
    examples: list[dict],
    prompt_fn,
    *,
    condition: str,
    model_id: str,
    seed: int,
    task: str,
    device: str | None = None,
    batch_size: int = 16,
    save: bool = True,
) -> dict:
    """Generate predictions for `examples`, score them, label errors, and assemble results.

    prompt_fn: example -> source string (baseline or CoT builder).
    task: "wtq" (EM + token-F1) or "tabfact" (classification accuracy).
    """
    device = device or config.get_device()
    is_cot = condition.startswith("cot")
    # CoT prompts are long and put the question last -> truncate from the left.
    max_source_len = config.MAX_SOURCE_LEN_PROMPT if is_cot else config.MAX_SOURCE_LEN
    truncation_side = "left" if is_cot else "right"

    sources = [prompt_fn(ex) for ex in examples]
    raw_preds, seconds_per_example = generate(
        model, tokenizer, sources, device, batch_size,
        max_source_len=max_source_len, truncation_side=truncation_side,
    )
    preds = [extract_answer(p) for p in raw_preds]
    golds = [ex["answer"] for ex in examples]

    records = []
    for ex, pred, raw in zip(examples, preds, raw_preds):
        correct = _is_correct(pred, ex, task)
        rec = {
            "id": ex["id"],
            "pred": pred,
            "gold": ex["answer"],
            "correct": correct,
            "error_type": label_error(ex, correct),
        }
        if is_cot:
            rec["raw"] = raw  # full generation incl. reasoning, for chain-quality rating
        records.append(rec)

    if task == "tabfact":
        n_correct = sum(r["correct"] for r in records)
        metrics = {"classification_accuracy": n_correct / max(len(records), 1)}
        error_distribution = None
    else:
        metrics = {
            "exact_match": exact_match_score(preds, golds),
            "token_f1": token_f1_score(preds, golds),
        }
        error_distribution = {et: 0 for et in config.ERROR_TYPES}
        for r in records:
            error_distribution[r["error_type"]] += 1

    results = {
        "condition": condition,
        "model": model_id,
        "seed": seed,
        "task": task,
        "n": len(examples),
        "metrics": metrics,
        "error_distribution": error_distribution,
        "compute": {
            "seconds_per_example": seconds_per_example,
            "peak_memory_mb": peak_memory_mb(device),
            "device": device,
        },
        "predictions": records,
    }
    if save:
        save_results(results)
    return results
