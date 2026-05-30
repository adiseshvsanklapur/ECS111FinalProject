"""Real end-to-end smoke test of the whole pipeline.

Downloads small real slices of WTQ + TabFact, runs a genuine (tiny) fine-tune on
the local device (cuda / mps / cpu), generates predictions, scores them, labels
errors, and writes a results JSON. Proves every module executes for real -- it
is NOT a mock. Also serves as the first sanity step on Colab.

Run from repo root:  python scripts/smoke_local.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data import load_tabfact, load_wtq_eval, load_wtq_train, serialize_table
from src.evaluate import predict_and_evaluate
from src.prompts import build_baseline_prompt, build_cot_prompt, build_train_source, build_train_target
from src.trace_templates import generate_trace, trace_coverage
from src.trainer import generate, load_model_and_tokenizer, train


def main() -> None:
    device = config.get_device()
    print(f"[smoke] device = {device}")

    # 1) Real data ---------------------------------------------------------- #
    print("[smoke] downloading small real slices ...")
    wtq_eval = load_wtq_eval(n=config.SMOKE_EVAL_N, seed=config.EVAL_SEED)
    wtq_train = load_wtq_train(n=config.SMOKE_TRAIN_N, seed=13)
    tabfact = load_tabfact(n=config.SMOKE_EVAL_N, seed=config.EVAL_SEED)
    print(f"[smoke] wtq_eval={len(wtq_eval)} wtq_train={len(wtq_train)} tabfact={len(tabfact)}")

    ex = wtq_eval[0]
    print(f"[smoke] sample question: {ex['question']!r}  gold={ex['answer']!r}")
    print("[smoke] serialized table (first 3 lines):")
    for line in serialize_table(ex["table"]).splitlines()[:3]:
        print("        " + line)

    # 2) Trace generator on real rows -------------------------------------- #
    cov = trace_coverage(wtq_train)
    n_traces = sum(generate_trace(e) is not None for e in wtq_train)
    print(f"[smoke] trace coverage on {cov['total']} train rows: {n_traces} got a trace")

    # 3) Real (tiny) fine-tune on MPS/cuda/cpu ----------------------------- #
    sources = [build_train_source(e) for e in wtq_train]
    targets = [build_train_target(e) for e in wtq_train]
    print("[smoke] fine-tuning (smoke) ...")
    model, tok, device = train(config.FLAN_BASE, sources, targets, seed=13, smoke=True, device=device)

    # 4) Generate + evaluate (baseline prompt on the fine-tuned model) ------ #
    print("[smoke] predict + evaluate WTQ ...")
    res_wtq = predict_and_evaluate(
        model, tok, wtq_eval, build_baseline_prompt,
        condition="smoke_finetune", model_id=config.SMOKE_MODEL,
        seed=13, task="wtq", device=device, batch_size=8, save=True,
    )
    print(f"[smoke] WTQ metrics: {res_wtq['metrics']}  errors: {res_wtq['error_distribution']}")

    # 5) Zero-shot CoT generation sanity on the base small model ----------- #
    base_model, base_tok, _ = load_model_and_tokenizer(config.SMOKE_MODEL, device)
    cot_sources = [build_cot_prompt(e, style="plain", n_shots=2) for e in wtq_eval[:2]]
    cot_preds, spe = generate(base_model, base_tok, cot_sources, device, batch_size=2)
    print(f"[smoke] CoT generated {len(cot_preds)} outputs at {spe:.3f}s/ex; e.g. {cot_preds[0][:60]!r}")

    # 6) Generalization: same model on TabFact ----------------------------- #
    print("[smoke] predict + evaluate TabFact (generalization) ...")
    res_tf = predict_and_evaluate(
        model, tok, tabfact, build_baseline_prompt,
        condition="smoke_generalization", model_id=config.SMOKE_MODEL,
        seed=13, task="tabfact", device=device, batch_size=8, save=True,
    )
    print(f"[smoke] TabFact metrics: {res_tf['metrics']}  compute: {res_tf['compute']}")

    print("[smoke] OK -- full pipeline executed end-to-end on real data.")


if __name__ == "__main__":
    main()
