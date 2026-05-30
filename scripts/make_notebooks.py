"""Generate the 5 condition notebooks as valid .ipynb JSON.

Notebooks are thin drivers over `src/`: each clones the repo + installs deps on
Colab, exposes a SMOKE toggle, auto-detects the device, runs its condition, and
writes results JSON. Run: python scripts/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
REPO_URL = "https://github.com/adiseshvsanklapur/ECS111FinalProject.git"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP = f"""
# --- Setup ---
# On Colab: clone the repo and install deps INTO THE KERNEL (%pip, not !pip).
# Locally: find the repo root and put it on sys.path. No clone, no install --
# run this notebook with the project's .venv kernel ("ECS111 (.venv)"), which
# already has datasets/transformers/torch. Locally this is a true no-op.
import os
import sys

try:
    import google.colab  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    if not os.path.isdir("ECS111FinalProject"):
        !git clone {REPO_URL}
    os.chdir("ECS111FinalProject")
    %pip -q install -r requirements.txt
else:
    # Walk up from the notebook's directory to the repo root (the dir with src/).
    root = os.path.abspath(os.getcwd())
    while root != os.path.dirname(root) and not os.path.isdir(os.path.join(root, "src")):
        root = os.path.dirname(root)
    if not os.path.isdir(os.path.join(root, "src")):
        raise RuntimeError(
            "Could not find the repo root (no src/ found walking up). "
            "Open this notebook from inside the cloned ECS111FinalProject."
        )
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)

print("cwd:", os.getcwd())
"""

SMOKE = """
# Set SMOKE = False for the full, reported run. SMOKE = True does a fast real
# end-to-end pass (flan-t5-small, tiny slice) to confirm everything works first.
SMOKE = True
"""

COMMON_CONFIG = """
from src import config
device = config.get_device()
print("device:", device)

if SMOKE:
    prompt_models = [config.SMOKE_MODEL]
    seeds = [13]
    eval_n = config.SMOKE_EVAL_N
    eval_n_cot = config.SMOKE_EVAL_N
    train_n = config.SMOKE_TRAIN_N
else:
    prompt_models = config.PROMPT_MODELS      # flan-t5-base + large
    seeds = config.SEEDS                       # [13, 42]
    eval_n = config.EVAL_N                      # 1000
    eval_n_cot = config.EVAL_N_COT             # 500
    train_n = config.TRAIN_N                    # 8000
"""


def build_baseline() -> dict:
    cells = [
        md("# Condition C0 — Baseline (no examples)\n\n"
           "Flan-T5 base + large, question + serialized table only. Sets the floor "
           "for every other condition. Metric: Exact-Match + token-F1 on WikiTableQuestions."),
        code(SETUP), code(SMOKE), code(COMMON_CONFIG),
        code("""
from src.data import load_wtq_eval
from src.prompts import build_baseline_prompt
from src.evaluate import predict_and_evaluate
from src.trainer import load_model_and_tokenizer

rows = []
for model_id in prompt_models:
    for seed in seeds:
        examples = load_wtq_eval(n=eval_n, seed=seed)
        model, tok, device = load_model_and_tokenizer(model_id, device)
        res = predict_and_evaluate(
            model, tok, examples, build_baseline_prompt,
            condition="baseline", model_id=model_id, seed=seed,
            task="wtq", device=device,
        )
        print(model_id, "seed", seed, res["metrics"])
        rows.append((model_id, seed, res["metrics"]))
rows
"""),
    ]
    return notebook(cells)


def build_cot() -> dict:
    cells = [
        md("# Condition A — Chain-of-Thought Prompting\n\n"
           "6 hand-written exemplars prepended to each prompt. Two chain formats tested: "
           "`plain` (natural language) vs `structured` (numbered steps). No training. "
           "Base + large. Metric: EM + token-F1 on WTQ."),
        code(SETUP), code(SMOKE), code(COMMON_CONFIG),
        code("""
import functools
from src.data import load_wtq_eval
from src.prompts import build_cot_prompt
from src.evaluate import predict_and_evaluate
from src.trainer import load_model_and_tokenizer

rows = []
for style in ["plain", "structured"]:
    for model_id in prompt_models:
        for seed in seeds:
            examples = load_wtq_eval(n=eval_n_cot, seed=seed)
            model, tok, device = load_model_and_tokenizer(model_id, device)
            prompt_fn = functools.partial(build_cot_prompt, style=style, n_shots=config.N_SHOTS)
            res = predict_and_evaluate(
                model, tok, examples, prompt_fn,
                condition=f"cot_{style}", model_id=model_id, seed=seed,
                task="wtq", device=device,
            )
            print(style, model_id, "seed", seed, res["metrics"])
            rows.append((style, model_id, seed, res["metrics"]))
rows
"""),
    ]
    return notebook(cells)


def _finetune_cells(condition: str, use_traces: bool, title_md: str) -> list[dict]:
    target_line = (
        "targets = [build_train_target(e, trace=generate_trace(e)) for e in train_ex]"
        if use_traces
        else "targets = [build_train_target(e) for e in train_ex]"
    )
    trace_import = "from src.trace_templates import generate_trace\n" if use_traces else ""
    run = f"""
import os
from src.data import load_wtq_train, load_wtq_eval
from src.prompts import build_train_source, build_train_target, build_baseline_prompt
from src.trainer import train
from src.evaluate import predict_and_evaluate
{trace_import}
os.makedirs("checkpoints", exist_ok=True)
ft_label = config.SMOKE_MODEL if SMOKE else config.FINETUNE_MODEL

rows = []
for seed in seeds:
    train_ex = load_wtq_train(n=train_n, seed=seed)
    sources = [build_train_source(e) for e in train_ex]
    {target_line}
    model, tok, device = train(
        config.FINETUNE_MODEL, sources, targets, seed=seed, smoke=SMOKE, device=device
    )
    ckpt = f"checkpoints/{condition}_seed{{seed}}"
    model.save_pretrained(ckpt); tok.save_pretrained(ckpt)

    eval_ex = load_wtq_eval(n=eval_n, seed=config.EVAL_SEED)
    res = predict_and_evaluate(
        model, tok, eval_ex, build_baseline_prompt,
        condition="{condition}", model_id=ft_label, seed=seed,
        task="wtq", device=device,
    )
    print("seed", seed, res["metrics"])
    rows.append((seed, res["metrics"]))
rows
"""
    return [md(title_md), code(SETUP), code(SMOKE), code(COMMON_CONFIG), code(run)]


def build_finetune_answers() -> dict:
    return notebook(_finetune_cells(
        "finetune_answers", use_traces=False,
        title_md=("# Condition B — Fine-tune (answers only)\n\n"
                  "Fine-tune Flan-T5-base on WTQ; target = final answer. 2 seeds. "
                  "Saves checkpoints for the generalization test. Expected to give the "
                  "highest WTQ accuracy."),
    ))


def build_finetune_traces() -> dict:
    return notebook(_finetune_cells(
        "finetune_traces", use_traces=True,
        title_md=("# Condition C — Fine-tune (reasoning traces)\n\n"
                  "Same data + setup as Condition B, but the target is a rule-based "
                  "reasoning chain ending in the answer (where the derivation is "
                  "unambiguous; otherwise answer only). Tests whether supervising on "
                  "intermediate steps improves final answers."),
    ))


def build_generalization() -> dict:
    cells = [
        md("# Generalization Test — fine-tuned models on TabFact (zero TabFact training)\n\n"
           "Loads the Condition B and C checkpoints and evaluates them on TabFact. "
           "Strong TabFact accuracy with no TabFact training = real transfer, not "
           "memorization. Ends with the aggregated summary table + plots."),
        code(SETUP), code(SMOKE), code(COMMON_CONFIG),
        code("""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.data import load_tabfact
from src.prompts import build_baseline_prompt
from src.evaluate import predict_and_evaluate

tabfact = load_tabfact(n=eval_n, seed=config.EVAL_SEED)
ft_label = config.SMOKE_MODEL if SMOKE else config.FINETUNE_MODEL

rows = []
for cond in ["finetune_answers", "finetune_traces"]:
    for seed in seeds:
        ckpt = f"checkpoints/{cond}_seed{seed}"
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(device)
        tok = AutoTokenizer.from_pretrained(ckpt)
        res = predict_and_evaluate(
            model, tok, tabfact, build_baseline_prompt,
            condition=f"generalization_{cond}", model_id=ft_label, seed=seed,
            task="tabfact", device=device,
        )
        print(cond, "seed", seed, res["metrics"])
        rows.append((cond, seed, res["metrics"]))
rows
"""),
        md("## Aggregate everything\n\n"
           "Run after all condition notebooks have written their results JSONs to `results/`."),
        code("""
from src import analysis
results = analysis.load_results()
df = analysis.aggregate(results)
display(df)
paths = analysis.write_summary_table(df)
analysis.plot_primary_metric(df, config.RESULTS_DIR / "primary_metric.png")
import glob
wtq_results = [r for r in results if r["task"] == "wtq"]
if wtq_results:
    analysis.plot_error_distribution(results, config.RESULTS_DIR / "error_distribution.png")
print("wrote:", paths)
"""),
    ]
    return notebook(cells)


def main() -> None:
    NB_DIR.mkdir(exist_ok=True)
    out = {
        "00_baseline.ipynb": build_baseline(),
        "01_cot_prompting.ipynb": build_cot(),
        "02_finetune_answers.ipynb": build_finetune_answers(),
        "03_finetune_traces.ipynb": build_finetune_traces(),
        "04_generalization.ipynb": build_generalization(),
    }
    for name, nb in out.items():
        (NB_DIR / name).write_text(json.dumps(nb, indent=1))
        print("wrote", name)


if __name__ == "__main__":
    main()
