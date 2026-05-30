"""Tests for src/report_fill.build_token_map.

Builds a synthetic but full result set (every condition, both seeds) and checks
that each report/slide token resolves to a sensible real value. No GPU, no
download -- pure dict math over the results schema written by evaluate.py.
"""
from src.report_fill import build_token_map


def _wtq(condition, model, seed, em, f1, errs):
    return {
        "condition": condition, "model": f"google/{model}", "seed": seed,
        "task": "wtq", "n": 1000,
        "metrics": {"exact_match": em, "token_f1": f1},
        "error_distribution": errs,
        "compute": {"seconds_per_example": 0.1, "peak_memory_mb": 4000.0, "device": "cuda"},
        "predictions": [{"id": str(i), "pred": "x", "gold": "y",
                         "correct": i < int(em * 10), "error_type": "lookup"} for i in range(10)],
    }


def _tabfact(condition, seed, acc):
    return {
        "condition": condition, "model": "google/flan-t5-base", "seed": seed,
        "task": "tabfact", "n": 1000,
        "metrics": {"classification_accuracy": acc}, "error_distribution": None,
        "compute": {"seconds_per_example": 0.1, "peak_memory_mb": 4000.0, "device": "cuda"},
        "predictions": [{"id": str(i), "pred": "true", "gold": "true",
                         "correct": i < int(acc * 10), "error_type": "correct"} for i in range(10)],
    }


ERRS = {"lookup": 30, "aggregation": 20, "multi_hop": 10, "correct": 40}


def _full_results():
    rs = []
    for seed in (13, 42):
        rs += [
            _wtq("baseline", "flan-t5-base", seed, 0.20, 0.25, ERRS),
            _wtq("baseline", "flan-t5-large", seed, 0.18, 0.23, ERRS),
            _wtq("cot_plain", "flan-t5-base", seed, 0.22, 0.28, ERRS),
            _wtq("cot_plain", "flan-t5-large", seed, 0.24, 0.30, ERRS),
            _wtq("cot_structured", "flan-t5-base", seed, 0.23, 0.29, ERRS),
            _wtq("cot_structured", "flan-t5-large", seed, 0.26, 0.31, ERRS),
            _wtq("finetune_answers", "flan-t5-base", seed, 0.31, 0.36, ERRS),
            _wtq("finetune_traces", "flan-t5-base", seed, 0.30, 0.35, ERRS),
            _tabfact("generalization_finetune_answers", seed, 0.58),
            _tabfact("generalization_finetune_traces", seed, 0.62),
        ]
    return rs


def test_numeric_tokens_resolve():
    tm = build_token_map(_full_results(), chain_quality={"kappa": 0.71, "mean_a": 1.3, "mean_b": 1.2})
    assert tm["[EM: baseline base]"].startswith("0.2")           # mean over seeds
    assert "±" in tm["[EM: baseline base]"]                       # mean ± std form
    assert tm["[ACC: tabfact finetune_traces base]"].startswith("0.62")
    assert tm["[TabFact acc: finetune_traces]"].startswith("0.62")
    assert tm["[KAPPA: chain quality]"].startswith("0.71")
    assert tm["[EM_GAP: best vs baseline]"].startswith("+")       # best base EM - baseline base EM
    for share in ("[ERR: lookup]", "[ERR: aggregation]", "[ERR: multihop]"):
        assert tm[share].endswith("%")
    assert "p" in tm["[MCNEMAR: cot vs finetune]"].lower()


def test_no_unresolved_known_tokens():
    tm = build_token_map(_full_results(), chain_quality={"kappa": 0.71, "mean_a": 1.3, "mean_b": 1.2})
    assert not any(v.startswith("[") for v in tm.values())        # no token maps to another token
