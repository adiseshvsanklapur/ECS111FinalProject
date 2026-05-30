"""Map every report/slide placeholder token to a real value from results/*.json.

Single source of truth so the report and the slides can never disagree, and so
every printed number traces back to a results JSON file (no hand-typed numbers).
Pure dict math over the schema written by evaluate.py; safe to import without torch.
"""
from __future__ import annotations

from .analysis import mcnemar_between


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


def _best_base_condition(results):
    cands = ["cot_plain", "cot_structured", "finetune_answers", "finetune_traces"]
    scored = [(c, _em_mean(results, c, "flan-t5-base")) for c in cands]
    scored = [(c, v) for c, v in scored if v is not None]
    return max(scored, key=lambda cv: cv[1])[0] if scored else None


def build_token_map(results, chain_quality):
    """Return {exact_token_string: formatted_value} for report + slides."""
    tm = {}
    conditions = ["baseline", "cot_plain", "cot_structured", "finetune_answers", "finetune_traces"]
    for cond in conditions:
        for short, model in [("base", "flan-t5-base"), ("large", "flan-t5-large")]:
            tm[f"[EM: {cond} {short}]"] = _fmt_metric(results, cond, model, "exact_match")
            tm[f"[F1: {cond} {short}]"] = _fmt_metric(results, cond, model, "token_f1")

    # TabFact generalization (only the two fine-tuned models are evaluated on TabFact).
    for cond in ["finetune_answers", "finetune_traces"]:
        acc = _tabfact(results, f"generalization_{cond}")
        tm[f"[ACC: tabfact {cond} base]"] = acc
        tm[f"[TabFact acc: {cond}]"] = acc
    # Optional baseline floor on TabFact, only if it was actually run.
    tm["[ACC: tabfact baseline base]"] = _tabfact(results, "generalization_baseline")

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
        tm["[EM_GAP: best vs baseline]"] = f"+{gap:.1f} EM points ({best_c})"
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

    # Per-condition wall-clock ceiling (max sec/example * n over conditions), minutes.
    per = [r["compute"]["seconds_per_example"] * r["n"] for r in results]
    tm["[TIME: per condition]"] = f"~{max(per) / 60:.0f} min" if per else "n/a"
    return tm
