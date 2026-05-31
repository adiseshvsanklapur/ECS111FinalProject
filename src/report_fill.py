"""Map every report placeholder token to a real value from results/*.json.

Single source of truth so the prose can never disagree with the data, and so
every printed number traces back to a results JSON file (no hand-typed numbers).
Pure dict math over the schema written by evaluate.py; safe to import without torch.
"""
from __future__ import annotations

from .analysis import mcnemar_between
from .metrics import map_tabfact_label


def _pick(results, condition, model_short=None, task="wtq"):
    return [r for r in results
            if r["condition"] == condition and r["task"] == task
            and (model_short is None or r["model"].split("/")[-1] == model_short)]


def _mean_std(values):
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / len(values)  # population std (matches analysis.aggregate)
    return m, var ** 0.5


def _fmt_metric(results, condition, model_short, key):
    rs = _pick(results, condition, model_short)
    if not rs:
        return "n/a"
    m, s = _mean_std([r["metrics"][key] for r in rs])
    return f"{m:.3f} ± {s:.3f}"


def _em_mean(results, condition, model_short):
    rs = _pick(results, condition, model_short)
    return None if not rs else sum(r["metrics"]["exact_match"] for r in rs) / len(rs)


def _tabfact(results, gen_condition):
    rs = _pick(results, gen_condition, "flan-t5-base", task="tabfact")
    if not rs:
        return "n/a"
    m, s = _mean_std([r["metrics"]["classification_accuracy"] for r in rs])
    return f"{m:.3f} ± {s:.3f}"


def _tabfact_decomp(results, gen_condition):
    """Honest TabFact decomposition, pooled over seeds from the saved predictions.

    The model's raw generation lives in each prediction's "pred" field, so the
    mappable rate recomputes directly (no re-inference). Returns None if the
    condition has no results, else a dict with the four reporting quantities.
    """
    rs = _pick(results, gen_condition, "flan-t5-base", task="tabfact")
    recs = [rec for r in rs for rec in r["predictions"]]
    n = len(recs)
    if n == 0:
        return None
    # Restrict the accuracy numerator to mappable outputs so acc_given_mappable
    # is self-enforcing, not reliant on evaluate.py marking unmappable preds wrong.
    mappable = [rec for rec in recs if map_tabfact_label(rec["pred"]) is not None]
    coverage_n = len(mappable)
    correct = sum(bool(rec["correct"]) for rec in mappable)
    return {
        "n": n,
        "coverage_n": coverage_n,
        "unmappable_rate": (n - coverage_n) / n,
        "acc_given_mappable": (correct / coverage_n) if coverage_n else None,
    }


def _tabfact_floor(results):
    """Majority-class floor on the TabFact eval slice, from the gold labels.

    The eval slice is seeded-fixed across conditions, so any TabFact result's
    golds give the same floor (no hand-typed 55.1%)."""
    rs = [r for r in results if r["task"] == "tabfact"]
    if not rs:
        return None
    golds = [rec["gold"] for rec in rs[0]["predictions"]]
    n = len(golds)
    if n == 0:
        return None
    return max(sum(g == "true" for g in golds), sum(g == "false" for g in golds)) / n


def _best_base_condition(results):
    cands = ["cot_plain", "cot_structured", "finetune_answers", "finetune_traces"]
    scored = [(c, _em_mean(results, c, "flan-t5-base")) for c in cands]
    scored = [(c, v) for c, v in scored if v is not None]
    return max(scored, key=lambda cv: cv[1])[0] if scored else None


def build_token_map(results, chain_quality):
    """Return {exact_token_string: formatted_value} for the report."""
    tm = {}
    # Prompting conditions run on both models; fine-tuning is base only (large OOMs on a T4).
    for cond in ["baseline", "cot_plain", "cot_structured"]:
        for short, model in [("base", "flan-t5-base"), ("large", "flan-t5-large")]:
            tm[f"[EM: {cond} {short}]"] = _fmt_metric(results, cond, model, "exact_match")
            tm[f"[F1: {cond} {short}]"] = _fmt_metric(results, cond, model, "token_f1")
    for cond in ["finetune_answers", "finetune_traces"]:
        tm[f"[EM: {cond} base]"] = _fmt_metric(results, cond, "flan-t5-base", "exact_match")
        tm[f"[F1: {cond} base]"] = _fmt_metric(results, cond, "flan-t5-base", "token_f1")

    # TabFact generalization (only the two fine-tuned models are evaluated on TabFact).
    for cond in ["finetune_answers", "finetune_traces"]:
        acc = _tabfact(results, f"generalization_{cond}")
        tm[f"[ACC: tabfact {cond} base]"] = acc
        tm[f"[TabFact acc: {cond}]"] = acc
    # Optional baseline floor on TabFact, only if it was actually run.
    tm["[ACC: tabfact baseline base]"] = _tabfact(results, "generalization_baseline")
    tm["[TabFact acc: baseline]"] = _tabfact(results, "generalization_baseline")

    # Honest TabFact decomposition: raw acc stays (above); add coverage, the
    # accuracy AMONG mappable outputs (carrying its n so a junk denominator can't
    # masquerade as a rate), and the majority-class floor. The near-zero raw acc
    # is an output-format collapse under shift, not a reasoning score -- these
    # tokens let the prose say that with the numbers behind it.
    for cond in ["baseline", "finetune_answers", "finetune_traces"]:
        d = _tabfact_decomp(results, f"generalization_{cond}")
        if d is None:
            tm[f"[TabFact unmappable: {cond}]"] = "n/a"
            tm[f"[TabFact coverage_n: {cond}]"] = "n/a"
            tm[f"[TabFact acc|mappable: {cond}]"] = "n/a"
            continue
        tm[f"[TabFact unmappable: {cond}]"] = f"{d['unmappable_rate']:.3f}"
        tm[f"[TabFact coverage_n: {cond}]"] = f"{d['coverage_n']} of {d['n']}"
        tm[f"[TabFact acc|mappable: {cond}]"] = (
            f"{d['acc_given_mappable']:.3f} (n={d['coverage_n']})"
            if d["acc_given_mappable"] is not None else "n/a (n=0)"
        )
    floor = _tabfact_floor(results)
    tm["[TabFact floor]"] = f"{floor:.3f}" if floor is not None else "n/a"

    # Best TabFact across the conditions we have.
    flat = []
    for cond in ("finetune_answers", "finetune_traces"):
        flat += [r["metrics"]["classification_accuracy"]
                 for r in _pick(results, f"generalization_{cond}", "flan-t5-base", "tabfact")]
    tm["[ACC: tabfact best]"] = f"{max(flat):.3f}" if flat else "n/a"

    # Success criterion 1: best base condition EM minus baseline base EM.
    base_em = _em_mean(results, "baseline", "flan-t5-base")
    best_c = _best_base_condition(results)
    if base_em is not None and best_c is not None:
        gap = (_em_mean(results, best_c, "flan-t5-base") - base_em) * 100
        tm["[EM_GAP: best vs baseline]"] = f"{gap:+.1f} EM points ({best_c})"
    else:
        tm["[EM_GAP: best vs baseline]"] = "n/a"

    # Error-type shares for the headline (best base) condition, summed over seeds.
    err = {"lookup": 0, "aggregation": 0, "multi_hop": 0}
    for r in _pick(results, best_c or "baseline", "flan-t5-base"):
        for k in err:
            err[k] += (r["error_distribution"] or {}).get(k, 0)
    tot = sum(err.values()) or 1
    tm["[ERR: lookup]"] = f"{100 * err['lookup'] / tot:.0f}%"
    tm["[ERR: aggregation]"] = f"{100 * err['aggregation'] / tot:.0f}%"
    tm["[ERR: multihop]"] = f"{100 * err['multi_hop'] / tot:.0f}%"

    # McNemar: best CoT base vs best fine-tune base, seed 13 (paired per-example).
    cot_best = max(["cot_plain", "cot_structured"],
                   key=lambda c: _em_mean(results, c, "flan-t5-base") or -1)
    ft_best = max(["finetune_answers", "finetune_traces"],
                  key=lambda c: _em_mean(results, c, "flan-t5-base") or -1)
    a = _pick(results, cot_best, "flan-t5-base")
    b = _pick(results, ft_best, "flan-t5-base")
    a13 = next((r for r in a if r["seed"] == 13), a[0] if a else None)
    b13 = next((r for r in b if r["seed"] == 13), b[0] if b else None)
    if a13 and b13:
        p = mcnemar_between(a13, b13)["pvalue"]
        tm["[MCNEMAR: cot vs finetune]"] = f"p = {p:.3f} ({cot_best} vs {ft_best}, seed 13)"
    else:
        tm["[MCNEMAR: cot vs finetune]"] = "n/a"

    # Chain quality (from scripts/rate_chains.py output).
    tm["[KAPPA: chain quality]"] = f"{chain_quality['kappa']:.3f}"
    tm["[KAPPA: rater a mean]"] = f"{chain_quality.get('mean_a', 0.0):.2f}"
    tm["[KAPPA: rater b mean]"] = f"{chain_quality.get('mean_b', 0.0):.2f}"

    # Per-condition wall-clock ceiling (max sec/example * n over conditions), minutes.
    per = [r["compute"]["seconds_per_example"] * r["n"] for r in results]
    tm["[TIME: per condition]"] = f"~{max(per) / 60:.0f} min" if per else "n/a"
    return tm
