"""Run every condition for real on this machine and save results.

This produces REAL numbers (not smoke): baseline, CoT (plain + structured),
fine-tune answers-only, fine-tune with traces, and the TabFact generalization
test. It then aggregates everything into a summary table and plots.

Two scales:
  --scale quick   one seed, small slices. ~30-45 min. For a first real look.
  --scale full    config scale (2 seeds, 1000 eval, 8000 train). Hours. Final.

Usage:
    python scripts/run_all_local.py --scale quick
Results land in results/ . Re-running overwrites per (condition, model, seed).
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data import load_tabfact, load_wtq_eval, load_wtq_train
from src.evaluate import predict_and_evaluate
from src.prompts import (
    build_baseline_prompt,
    build_cot_prompt,
    build_train_source,
    build_train_target,
)
from src.trace_templates import generate_trace
from src.trainer import load_model_and_tokenizer, train


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def results_exist(condition: str, model_id: str, seed: int) -> bool:
    """True if this (condition, model, seed) already has a saved result (resume support)."""
    model_short = model_id.split("/")[-1]
    return (config.RESULTS_DIR / f"{condition}_{model_short}_seed{seed}.json").exists()


def get_settings(scale: str) -> dict:
    if scale == "full":
        return {
            "prompt_models": config.PROMPT_MODELS,
            "seeds": config.SEEDS,
            "eval_n": config.EVAL_N,
            "eval_n_cot": config.EVAL_N_COT,
            "train_n": config.TRAIN_N,
        }
    # quick: one seed, small slices, base + large still both prompted
    return {
        "prompt_models": config.PROMPT_MODELS,
        "seeds": [13],
        "eval_n": 120,
        "eval_n_cot": 80,
        "train_n": 1200,
    }


def run_baseline(s: dict, device: str) -> None:
    log("=== BASELINE ===")
    for model_id in s["prompt_models"]:
        todo = [seed for seed in s["seeds"] if not results_exist("baseline", model_id, seed)]
        if not todo:
            log(f"baseline {model_id.split('/')[-1]}: all seeds done, skip")
            continue
        model, tok, device = load_model_and_tokenizer(model_id, device)
        for seed in todo:
            ex = load_wtq_eval(n=s["eval_n"], seed=seed)
            res = predict_and_evaluate(
                model, tok, ex, build_baseline_prompt,
                condition="baseline", model_id=model_id, seed=seed,
                task="wtq", device=device,
            )
            log(f"baseline {model_id.split('/')[-1]} seed{seed}: {res['metrics']}")


def run_cot(s: dict, device: str) -> None:
    log("=== CHAIN OF THOUGHT ===")
    for style in ["plain", "structured"]:
        for model_id in s["prompt_models"]:
            todo = [seed for seed in s["seeds"] if not results_exist(f"cot_{style}", model_id, seed)]
            if not todo:
                log(f"cot_{style} {model_id.split('/')[-1]}: all seeds done, skip")
                continue
            model, tok, device = load_model_and_tokenizer(model_id, device)
            for seed in todo:
                ex = load_wtq_eval(n=s["eval_n_cot"], seed=seed)
                pf = functools.partial(build_cot_prompt, style=style, n_shots=config.N_SHOTS)
                res = predict_and_evaluate(
                    model, tok, ex, pf,
                    condition=f"cot_{style}", model_id=model_id, seed=seed,
                    task="wtq", device=device,
                )
                log(f"cot_{style} {model_id.split('/')[-1]} seed{seed}: {res['metrics']}")


def run_finetune(s: dict, device: str, condition: str, use_traces: bool) -> None:
    log(f"=== FINE TUNE: {condition} ===")
    Path("checkpoints").mkdir(exist_ok=True)
    for seed in s["seeds"]:
        ckpt = f"checkpoints/{condition}_seed{seed}"
        if results_exist(condition, config.FINETUNE_MODEL, seed) and Path(ckpt).exists():
            log(f"{condition} seed{seed}: results + checkpoint exist, skip")
            continue
        train_ex = load_wtq_train(n=s["train_n"], seed=seed)
        sources = [build_train_source(e) for e in train_ex]
        if use_traces:
            targets = [build_train_target(e, trace=generate_trace(e)) for e in train_ex]
            n_tr = sum(generate_trace(e) is not None for e in train_ex)
            log(f"{condition} seed{seed}: {n_tr}/{len(train_ex)} train rows got a reasoning chain")
        else:
            targets = [build_train_target(e) for e in train_ex]
        log(f"{condition} seed{seed}: training on {len(train_ex)} examples ...")
        model, tok, device = train(config.FINETUNE_MODEL, sources, targets, seed=seed, device=device)
        ckpt = f"checkpoints/{condition}_seed{seed}"
        model.save_pretrained(ckpt); tok.save_pretrained(ckpt)
        eval_ex = load_wtq_eval(n=s["eval_n"], seed=config.EVAL_SEED)
        res = predict_and_evaluate(
            model, tok, eval_ex, build_baseline_prompt,
            condition=condition, model_id=config.FINETUNE_MODEL, seed=seed,
            task="wtq", device=device,
        )
        log(f"{condition} seed{seed}: {res['metrics']}")


def run_generalization(s: dict, device: str) -> None:
    log("=== GENERALIZATION (TabFact) ===")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tf = load_tabfact(n=s["eval_n"], seed=config.EVAL_SEED)
    for cond in ["finetune_answers", "finetune_traces"]:
        for seed in s["seeds"]:
            gen_cond = f"generalization_{cond}"
            if results_exist(gen_cond, config.FINETUNE_MODEL, seed):
                log(f"{gen_cond} seed{seed}: results exist, skip")
                continue
            ckpt = f"checkpoints/{cond}_seed{seed}"
            if not Path(ckpt).exists():
                log(f"skip {cond} seed{seed}: no checkpoint")
                continue
            model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(device)
            tok = AutoTokenizer.from_pretrained(ckpt)
            res = predict_and_evaluate(
                model, tok, tf, build_baseline_prompt,
                condition=f"generalization_{cond}", model_id=config.FINETUNE_MODEL, seed=seed,
                task="tabfact", device=device,
            )
            log(f"generalization_{cond} seed{seed}: {res['metrics']}")


def aggregate() -> None:
    log("=== AGGREGATE ===")
    from src import analysis

    results = analysis.load_results()
    df = analysis.aggregate(results)
    paths = analysis.write_summary_table(df)
    analysis.plot_primary_metric(df, config.RESULTS_DIR / "primary_metric.png")
    if any(r["task"] == "wtq" for r in results):
        analysis.plot_error_distribution(results, config.RESULTS_DIR / "error_distribution.png")
    log(f"wrote summary table -> {paths['md']}")
    print("\n================ SUMMARY ================\n")
    print(df.to_string(index=False))
    print("\nFiles in results/: summary_table.csv, summary_table.md, primary_metric.png, error_distribution.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["quick", "full"], default="quick")
    args = ap.parse_args()

    device = config.get_device()
    s = get_settings(args.scale)
    log(f"device={device} scale={args.scale} settings={s}")
    t0 = time.time()

    run_baseline(s, device)
    run_cot(s, device)
    run_finetune(s, device, "finetune_answers", use_traces=False)
    run_finetune(s, device, "finetune_traces", use_traces=True)
    run_generalization(s, device)
    aggregate()

    log(f"ALL DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
