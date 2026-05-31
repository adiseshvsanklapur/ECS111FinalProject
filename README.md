# Chain-of-Thought Prompting vs. Fine-Tuning for Table Reasoning in Small Language Models

**ECS 111 Final Project**

A controlled comparison of **Chain-of-Thought (CoT) prompting** and **supervised fine-tuning** for
table question answering, run entirely on **small, free-tier models** under real compute limits
(free Google Colab T4, under 2 hours per condition).

> **Core question:** when you can't afford a 100B-parameter model, which helps more for table
> reasoning, prompting or fine-tuning, and where does each one break?

**Status: complete.** The full study ran end-to-end on a Colab T4 (2 seeds, 1000 eval / 500 CoT,
8000 train). `results/`, the report, and the slides are filled from those runs, and **161 pytest
cases pass**. Every reported number comes from an actual run and traces to a file in `results/`,
none are hand-typed (`scripts/finalize_report.py` exits non-zero if any number is left unmapped).

**Headline result, a negative one.** No method we tried beat the plain zero-shot baseline. Flan-T5-large
with no examples scored highest on WikiTableQuestions (**EM 0.241**). Chain-of-thought prompting *more
than halved* exact match, light fine-tuning landed *below* the baseline, and the WTQ-fine-tuned models
did **not** transfer to TabFact. Their outputs collapsed to a non-true/false format. The full read is
in [`report/REPORT.md`](report/REPORT.md), and the numbers are in
[`results/summary_table.md`](results/summary_table.md).

---

## Table of Contents

1. [Overview](#overview)
2. [Results](#results)
3. [Motivation & Goals](#motivation--goals)
4. [Problem Statement](#problem-statement)
5. [Novelty](#novelty)
6. [Datasets](#datasets)
7. [Methodology: The 5 Conditions](#methodology-the-5-conditions)
8. [Evaluation Plan](#evaluation-plan)
9. [Outcome vs. Success Criteria](#outcome-vs-success-criteria)
10. [Repository Structure](#repository-structure)
11. [Team & Task Division](#team--task-division)
12. [How We Ran It](#how-we-ran-it)
13. [Reproducibility & Setup](#reproducibility--setup)
14. [Risks & Mitigations](#risks--mitigations)
15. [References](#references)

---

## Overview

Tables are the default format for real-world data: medical records, government data, financial
reports, research results. But language models reason poorly across rows and columns. There are two
standard fixes:

- **Chain-of-Thought (CoT) prompting**: guide the model through step-by-step reasoning *at inference
  time*. No retraining, you just change the prompt.
- **Supervised fine-tuning**: *train* the model on question and answer pairs until it improves.

Almost all published evidence for these uses very large, expensive models. **This project looks at the
case most studies skip: both methods on small models that run for free.** We compare **Flan-T5-base
(250M)** and **Flan-T5-large (780M)** on **WikiTableQuestions** and **TabFact**, with a cross-dataset
generalization test and a full error-type breakdown.

---

## Results

Full run: Flan-T5-base (250M) and large (780M), 2 seeds (13, 42), 1000-example eval
(500 for CoT), 8000-example fine-tune. Every figure traces to a file in `results/`.

**WikiTableQuestions, exact match (mean ± std over 2 seeds):**

| Condition | Flan-T5-base | Flan-T5-large |
|---|---|---|
| Baseline (zero-shot) | 0.171 ± 0.011 | **0.241 ± 0.009** |
| CoT, plain | 0.015 ± 0.001 | 0.052 ± 0.008 |
| CoT, structured | 0.019 ± 0.003 | 0.147 ± 0.015 |
| Fine-tune, answers only | 0.157 ± 0.002 | n/a (OOMs on T4) |
| Fine-tune, + reasoning traces | 0.121 ± 0.009 | n/a |

**TabFact, out-of-distribution transfer (zero TabFact training, n=1000).** The majority-class
floor is **0.551**. The models almost never give a gradeable true/false answer, so these are
output-format-collapse numbers, not reasoning scores, which is why each accuracy carries its
mappable-output count:

| Model | Accuracy | Mappable outputs | Accuracy when mappable |
|---|---|---|---|
| Untrained base (floor) | 0.043 | 170 / 2000 | 0.506 (≈ chance) |
| Fine-tune, answers | 0.003 | 11 / 2000 | 0.455 (n=11) |
| Fine-tune, traces | 0.002 | 13 / 2000 | 0.308 (n=13) |

**What it means:** on a small free model the plain baseline won. CoT hurt strict exact match (the
model reasons in prose instead of naming the cell), and light fine-tuning didn't beat the baseline and
didn't transfer. Error analysis, McNemar, and chain-quality κ (0.784) are in
[`report/REPORT.md`](report/REPORT.md).

---

## Motivation & Goals

Four goals:

1. Run a **controlled comparison** of CoT prompting vs. fine-tuning on the *same* small models.
2. **Dig into how and where each method fails**, not just top-line accuracy.
3. **Test generalization** to a completely different dataset (train on WTQ, test on TabFact, zero
   TabFact training).
4. Keep **everything reproducible on free Colab in under 2 hours per condition.**

A second theme is **interpretability**: CoT produces a visible, checkable reasoning chain, while
fine-tuning produces only a final answer. Where trust matters, that difference matters.

---

## Problem Statement

Table QA looks like reading comprehension, but it is a multi-step pipeline that must *all* succeed:
understand the question, locate the right cells, pick the right operation (filter, sum, compare,
sort), and return the answer in the correct format. A failure at any step yields a confident wrong
answer.

Formally: given a dataset `D = {(qᵢ, Tᵢ, aᵢ)}` where `qᵢ` is a question, `Tᵢ` a table, and `aᵢ` the
correct answer, find a model `M` such that `M(qᵢ, Tᵢ)` returns `aᵢ`, as accurately as possible, with
minimal compute, and in a way that generalizes to new data.

- **CoT risk:** small models often lack the capacity to follow complex reasoning demos, so they copy
  surface patterns without the underlying logic.
- **Fine-tuning risk:** overfitting, where the model learns dataset-specific shortcuts instead of
  general reasoning and then collapses on a different table or question format.

---

## Novelty

The individual pieces (CoT, table QA, fine-tuning) are all established. The **specific combination is
not**: a controlled prompting-vs-fine-tuning comparison **at this model size** (250M / 780M), on these
tasks, **under strict free-Colab limits**, with a **cross-dataset generalization test** and an
**error-type breakdown**.

- Wei et al. (2022) showed CoT works mainly on 100B+ models, and noted small models benefit far less.
- UnifiedSKG (Xie et al., 2022) is the closest prior work, but it does **not** run prompting vs.
  fine-tuning as a controlled experiment, and not under strict compute limits.

---

## Datasets

Both load from HuggingFace `datasets` with one line, no account, API key, or manual download.

| Dataset | Size | Task | Role | License |
|---|---|---|---|---|
| **WikiTableQuestions** (Pasupat & Liang, 2015) | 22,033 QA pairs / 2,108 tables | Answer a question from a table (filter, sort, compare, multi-hop) | **Primary** (train + eval) | CC BY-SA 4.0 |
| **TabFact** (Chen et al., 2020) | 117,854 examples / 16,573 tables | True/false fact verification over a table | **Out-of-distribution** generalization test | MIT |

**Table serialization:** tables are flattened to text, column headers first then each row in
sequence, to stay within the small models' token limits, consistent with prior encoder-decoder work.

> **Loader note (important):** `datasets >= 4.0` removed script-based datasets, so the canonical
> `wikitablequestions` / `tab_fact` ids no longer load. We use content-identical **parquet mirrors**.
> WTQ is [`lighteval/wikitablequestions`](https://huggingface.co/datasets/lighteval/wikitablequestions)
> (inline tables, one 18,486-example pool that we split into a **fixed seeded train/eval partition**
> that never overlaps, see `src/data.py:disjoint_train_eval_indices`), and TabFact is
> [`target-benchmark/tabfact-queries`](https://huggingface.co/datasets/target-benchmark/tabfact-queries)
> joined to `tabfact-corpus` on `table_id`. All ids are verified by real download.

---

## Methodology: The 5 Conditions

Ordered so earlier conditions stand alone if time runs short.

| # | Condition | Model(s) | Training? | Key detail |
|---|---|---|---|---|
| **C0** | **Baseline** | base + large | No | Question + table only, no examples. Sets the floor. |
| **CA** | **CoT Prompting** | base + large | No | 6 to 8 hand-written reasoning exemplars prepended; **two chain formats** tested (plain English vs. structured step-by-step). |
| **CB** | **Fine-tune, answers only** | base | Yes | Target is the final answer only. Standard fine-tuning; expected highest WTQ accuracy. |
| **CC** | **Fine-tune + reasoning traces** | base | Yes | Target is a reasoning chain then the answer. Chains auto-generated by **rule-based templates** for unambiguous derivations. |
| **CG** | **Generalization test** | B & C models | No | Evaluate fine-tuned models on **TabFact** with zero TabFact training, the strongest signal of real reasoning vs. memorization. |

**Training setup (CB, CC):** AdamW, lr `3e-4`, batch 8 × grad-accum 4 (effective 32), 3 epochs,
T4 GPU. Greedy decoding, temperature 0, fixed and documented seeds. Each condition is a self-contained
Colab notebook that runs end-to-end.

---

## Evaluation Plan

Two layers: standard metrics for comparability, plus failure analysis for insight.

| Metric | Applies to | Purpose |
|---|---|---|
| **Exact-Match Accuracy** | WTQ | Headline metric (normalized: lowercase, punctuation removed) |
| **Token-level F1** | WTQ | Partial credit for list/range answers where EM is too strict |
| **Classification Accuracy** | TabFact | Binary true/false correctness |
| **Error-Type Breakdown** | all | Every wrong answer labeled **lookup** / **aggregation** / **multi-hop** |
| **Reasoning Chain Quality** | CoT conditions | 100 sampled chains, **2 independent raters**, 0/1/2 scale, **Cohen's kappa** for agreement |
| **Compute Cost** | all | Inference time/example + peak GPU memory |

**Statistics:** each condition run with **2 seeds**; report mean ± std. Headline CoT-vs-fine-tune
comparison validated with **McNemar's test** on paired per-example predictions.

---

## Outcome vs. Success Criteria

The proposal set three bars. Two were **not** met, and that is the finding, not a failure of the
experiment:

1. ❌ **A condition beats the baseline by ≥ 5 EM points** on WikiTableQuestions: **not met.** No
   condition beat the baseline at all. The best non-baseline result (fine-tune answers, 0.157)
   trailed the base baseline (0.171), and every method trailed the large baseline (0.241).
2. ❌ **A condition reaches ≥ 60% on TabFact** with zero TabFact training: **not met.** Best raw
   accuracy was 0.043, below the 0.551 majority-class floor. Transfer collapsed at the output-format
   level (the fine-tuned models stopped giving true/false).
3. ✅ **Every condition completes in < 2 hours** on a free Colab T4: **met** (per-example times in
   `results/summary_table.md`).

Deliverables complete: 5 condition notebooks plus a self-contained full-run notebook (fixed seeds),
the written report (`report/`), and the slides (`slides/`). A clean, fully reproducible negative result.

---

## Repository Structure

```
ECS111FinalProject/
├── README.md                       # this file
├── Project_Proposal.pdf            # approved proposal
├── requirements.txt                # torch, transformers, datasets, scipy, sklearn, ...
├── notebooks/                      # Colab drivers (clone, install, import, run)
│   ├── 00_baseline.ipynb            # C0
│   ├── 01_cot_prompting.ipynb       # CA
│   ├── 02_finetune_answers.ipynb    # CB
│   ├── 03_finetune_traces.ipynb     # CC
│   ├── 04_generalization.ipynb      # CG  (+ aggregation/plots)
│   └── ECS111_Colab_FullRun.ipynb   # self-contained full run, shardable across 5 accounts
├── src/                            # pure library (metrics/data import without torch)
│   ├── config.py                   # seeds, model/dataset ids, hyperparams, get_device()
│   ├── data.py                     # loaders, serialize_table, seeded disjoint split
│   ├── prompts.py                  # baseline + CoT + TabFact builders, answer extraction
│   ├── cot_exemplars.py            # hand-written exemplars (plain / structured)
│   ├── trace_templates.py          # rule-based reasoning-trace generator (CC)
│   ├── metrics.py                  # EM, token-F1, TabFact verbalizer + acc, McNemar, Cohen κ
│   ├── trainer.py                  # seq2seq fine-tune + greedy generate (device-aware)
│   ├── evaluate.py                 # predict, score, label errors, write results JSON
│   ├── analysis.py                 # aggregate seeds, McNemar, κ, tables + plots
│   └── report_fill.py              # map results JSON to every report/slide number (single source)
├── scripts/
│   ├── run_all_local.py            # run every condition for real (--scale quick|full, --shard)
│   ├── make_colab_notebook.py      # regenerate the self-contained full-run notebook
│   ├── make_notebooks.py           # regenerate the 5 condition notebooks
│   ├── rate_chains.py              # two-rubric chain-quality rating, Cohen κ
│   ├── finalize_report.py          # fill report numeric tokens from results (fails loud)
│   ├── make_slides.py              # build the slide deck from the same token map
│   ├── export_chains.py            # dump sampled CoT chains for rating
│   ├── resume_finetune_local.py    # resume a hung run (skip completed conditions)
│   └── smoke_local.py              # tiny real end-to-end pipeline check
├── tests/                          # 161 pytest cases (metrics, data, traces, analysis, report_fill)
├── results/                        # per-condition×seed JSON + summary_table.{md,csv} + plots + chain_quality.json
├── report/REPORT.md                # final report (numbers auto-filled; prose in negative-result frame)
└── slides/ECS111_slides.pptx       # generated deck
```

---

## Team & Task Division

Assignment goal: **even effort across all 5 members.** All members have free Colab, so training
parallelizes across accounts.

| Member | Primary Ownership | Key Tasks |
|---|---|---|
| **Adisesh Venkatesh** | Foundation + Baseline | Shared core (`tableqa_core.py`, `metrics.py`), a Day-1 priority; baseline notebook (C0); ~5 CoT exemplars; chain-quality **rater #1** |
| **Amar Thota** | Condition A (CoT) | CoT prompting notebook (both chain formats, base + large); **lead CoT exemplar writing**; compute log for CA; assembles slide deck |
| **Nikhil Karthikeyan** | Condition B + Trainer | Shared `trainer.py` harness; fine-tune answers-only (CB, 2 seeds); chain-quality **rater #2** |
| **Anant Madhok** | Condition C | Rule-based reasoning-trace **template generator**; fine-tune + traces (CC, 2 seeds, reuses trainer); ~5 CoT exemplars |
| **Sanjay Manivasagam** | Results & Generalization | Generalization test (CG); error-type breakdown; McNemar + mean/std; compute-cost summary; all results tables + plots; merges report |

**Shared tasks (split for fairness):**

- **CoT exemplars** (6 to 8 × 2 formats, about 12 to 16): Amar (lead ~6) + Adisesh (~5) + Anant (~5).
  **Done Day 1** so CA can run Day 2.
- **Chain-quality raters** (Cohen's kappa needs 2): **Adisesh + Nikhil**, rated independently, neither
  is the CoT exemplar lead, to avoid bias.
- **Report sections** (Day 4): each writes their own area, Sanjay merges, all proof against the rubric.
- **Slides** (Day 4): split by section; Amar assembles.

---

## How We Ran It

The project ran on a 4-day plan (below). Reported numbers come from the final full Colab T4 run, not
the day-by-day smoke passes.

### Day 1: Foundation and parallel starts
- **Adisesh:** shared core + metrics (top priority, unblocks everyone). Push by midday.
- **Nikhil:** trainer harness (uses core). Kick off CB run by evening.
- **Amar + Adisesh + Anant:** write all CoT exemplars (both formats).
- **Anant:** build reasoning-trace template generator.
- **Sanjay:** scaffold eval/analysis notebook against the metrics module.
- **Gate:** shared core + trainer working; baseline runs by end of day.

### Day 2: Run conditions (parallel Colab)
- **Adisesh:** finish baseline (base + large), log compute.
- **Amar:** run CA, both chain formats, base + large (needs Day-1 exemplars).
- **Nikhil:** CB fine-tune answers-only, **both seeds**, start early.
- **Anant:** CC fine-tune + traces, **both seeds**, start early.
- **Sanjay:** lock error-label rubric + compute logging; evaluate C0 + CA as they land.

### Day 3: Generalization, evaluation, stats
- **Sanjay:** generalization test (B + C on TabFact), check the 60% threshold; then error breakdown,
  McNemar, mean/std, compute summary, tables + plots.
- **Adisesh + Nikhil:** chain-quality rating (100 chains each, independent), Cohen's kappa.
- **Everyone:** re-run any condition failing a threshold or over 2 hr.
- **Gate:** all 3 success criteria checked; evening buffer for triage.

### Day 4: Write and present
- **All:** report sections, Sanjay merges and proofs.
- **Amar:** assemble slide deck.
- **All:** rehearse; final reproducibility pass on a fresh Colab runtime.

---

## Reproducibility & Setup

### On Colab (the official run target)
Open any `notebooks/0*.ipynb`. The first cell clones this repo and runs
`pip install -r requirements.txt`, no other setup. Each notebook has a **`SMOKE = True`**
toggle: run it once as-is for a fast real end-to-end sanity pass (flan-t5-small, tiny slice),
then set `SMOKE = False` and re-run for the reported numbers. Set the runtime to **GPU (T4)**.

### Locally (development and verification, works on CPU, CUDA, or Apple-Silicon MPS)
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q          # 161 pure-logic tests
.venv/bin/python scripts/smoke_local.py        # real data download + full pipeline on your device
```
`src/config.py:get_device()` auto-selects `cuda → mps → cpu`, so the same code runs everywhere.
This codebase was verified end-to-end on an Apple M4 Pro via the MPS backend (real fine-tune,
generation, checkpoint round-trip, and TabFact generalization).

**Models:** `google/flan-t5-base` (250M, fine-tuned + prompted), `google/flan-t5-large` (780M,
prompting only), `google/flan-t5-small` (SMOKE only).

**Reproducibility rules:**
- All random seeds **fixed** in `config.py`; each condition runs with **2 seeds** (`[13, 42]`).
- Greedy decoding (`do_sample=False`, `num_beams=1`), deterministic.
- Eval is a **seeded cap** (`EVAL_N=1000`, CoT `500`) to keep each condition **< 2 hr** on a T4.
- Fine-tune notebooks save checkpoints to `checkpoints/`; the generalization notebook reloads them.

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Training is the bottleneck (2 seeds × 2 conditions) | Start CB/CC Day-1 evening; parallelize seeds across separate Colab accounts; checkpoint to Drive |
| Flan-T5-**large** fine-tune may OOM on T4 | Per proposal, **only fine-tune base**; keep large to prompting (C0/CA), don't expand scope |
| CoT exemplars block CA | Hard deadline: exemplars finished Day 1 |
| Ambiguous reasoning-trace generation | Auto-generate traces **only** where the derivation is unambiguous; skip the rest |
| Colab disconnects | Save model checkpoints + result JSONs to Drive after each run |
| Inter-rater drift on chain quality | Agree on 0/1/2 rubric examples first; rate independently; then compute kappa |

---

## References

- Chen, W., et al. (2020). *TabFact: A Large-scale Dataset for Table-based Fact Verification.* ICLR 2020.
- Herzig, J., et al. (2020). *TAPAS: Weakly Supervised Table Parsing via Pre-training.* ACL 2020.
- Pasupat, P., & Liang, P. (2015). *Compositional Semantic Parsing on Semi-Structured Tables.* ACL 2015.
- Wang, X., et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023.
- Wei, J., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS 2022.
- Xie, T., et al. (2022). *UnifiedSKG: Unifying and Multi-Tasking Structured Knowledge Grounding.* EMNLP 2022.
- Yao, S., et al. (2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models.* NeurIPS 2023.

---

*Team: Adisesh Venkatesh · Amar Thota · Nikhil Karthikeyan · Sanjay Manivasagam · Anant Madhok*
