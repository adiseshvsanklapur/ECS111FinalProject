# Chain-of-Thought Prompting vs. Fine-Tuning for Table Reasoning in Small Language Models

**ECS 111 — Final Project**

A controlled comparison of **Chain-of-Thought (CoT) prompting** and **supervised fine-tuning** for
table question answering, run entirely on **small, free-tier models** under real compute constraints
(free Google Colab T4, < 2 hours per condition).

> **Core question:** When you can't afford a 100B-parameter model, which is the better lever for
> table reasoning — prompting or fine-tuning — and where does each one break?

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation & Goals](#motivation--goals)
3. [Problem Statement](#problem-statement)
4. [Novelty](#novelty)
5. [Datasets](#datasets)
6. [Methodology — The 5 Conditions](#methodology--the-5-conditions)
7. [Evaluation Plan](#evaluation-plan)
8. [Definition of Done (Success Criteria)](#definition-of-done-success-criteria)
9. [Repository Structure](#repository-structure)
10. [Team & Task Division](#team--task-division)
11. [4-Day Execution Timeline](#4-day-execution-timeline)
12. [Reproducibility & Setup](#reproducibility--setup)
13. [Risks & Mitigations](#risks--mitigations)
14. [References](#references)

---

## Overview

Tables are the default format for real-world data — medical records, government data, financial
reports, research results. Yet language models reason poorly across rows and columns. Two standard
fixes exist:

- **Chain-of-Thought (CoT) prompting** — guide the model through step-by-step reasoning *at inference
  time*. No retraining; just change the prompt.
- **Supervised fine-tuning** — *train* the model on question/answer pairs until it improves.

Almost all published evidence for these uses very large, expensive models. **This project studies the
under-explored regime: both methods applied to small models that run for free.** We run a controlled
comparison on **Flan-T5-base (250M)** and **Flan-T5-large (780M)** over **WikiTableQuestions** and
**TabFact**, with a cross-dataset generalization test and a full error-type breakdown.

---

## Motivation & Goals

Four goals:

1. Run a **controlled comparison** of CoT prompting vs. fine-tuning on the *same* small models.
2. **Dig into how and where each method fails** — not just top-line accuracy.
3. **Test generalization** to a completely different dataset (train on WTQ, test on TabFact, zero
   TabFact training).
4. Keep **everything reproducible on free Colab in under 2 hours per condition.**

A secondary theme is **interpretability**: CoT produces a visible, checkable reasoning chain;
fine-tuning produces only a final answer. Where trust matters, that difference matters.

---

## Problem Statement

Table QA looks like reading comprehension but is a multi-step pipeline that must *all* succeed:
understand the question → locate the right cells → pick the right operation (filter / sum / compare /
sort) → return the answer in the correct format. A failure at any step yields a confident wrong answer.

Formally: given a dataset `D = {(qᵢ, Tᵢ, aᵢ)}` where `qᵢ` is a question, `Tᵢ` a table, and `aᵢ` the
correct answer, find a model `M` such that `M(qᵢ, Tᵢ) → aᵢ` — **as accurately as possible, with
minimal compute, and in a way that generalizes to new data.**

- **CoT risk:** small models often lack the capacity to follow complex reasoning demos — they copy
  surface patterns without the underlying logic.
- **Fine-tuning risk:** overfitting — learning dataset-specific shortcuts instead of general reasoning,
  which collapses on a different table/question format.

---

## Novelty

The individual pieces (CoT, table QA, fine-tuning) are all established. The **specific combination is
not**: a controlled prompting-vs-fine-tuning comparison **at this model size** (250M / 780M), on these
tasks, **under strict free-Colab limits**, with a **cross-dataset generalization test** and an
**error-type breakdown**.

- Wei et al. (2022) showed CoT works mainly on 100B+ models — and noted small models benefit far less.
- UnifiedSKG (Xie et al., 2022) is the closest prior work but does **not** run prompting vs.
  fine-tuning as a controlled experiment, nor under strict compute limits.

---

## Datasets

Both load from HuggingFace `datasets` with one line — no account, API key, or manual download.

| Dataset | Size | Task | Role | License |
|---|---|---|---|---|
| **WikiTableQuestions** (Pasupat & Liang, 2015) | 22,033 QA pairs / 2,108 tables | Answer a question from a table (filter, sort, compare, multi-hop) | **Primary** — train + eval | CC BY-SA 4.0 |
| **TabFact** (Chen et al., 2020) | 117,854 examples / 16,573 tables | True/false fact verification over a table | **Out-of-distribution** generalization test | MIT |

**Table serialization:** tables are flattened to text — column headers first, then each row in
sequence — to stay within the small models' token limits, consistent with prior encoder-decoder work.
Official train/validation/test splits, no overlap.

---

## Methodology — The 5 Conditions

Ordered so earlier conditions stand alone if time runs short.

| # | Condition | Model(s) | Training? | Key detail |
|---|---|---|---|---|
| **C0** | **Baseline** | base + large | No | Question + table only, no examples. Sets the floor. |
| **CA** | **CoT Prompting** | base + large | No | 6–8 hand-written reasoning exemplars prepended; **two chain formats** tested (plain English vs. structured step-by-step). |
| **CB** | **Fine-tune, answers only** | base | Yes | Target = final answer only. Standard fine-tuning; expected highest WTQ accuracy. |
| **CC** | **Fine-tune + reasoning traces** | base | Yes | Target = reasoning chain → answer. Chains auto-generated by **rule-based templates** for unambiguous derivations. |
| **CG** | **Generalization test** | B & C models | No | Evaluate fine-tuned models on **TabFact** with zero TabFact training — the strongest signal of real reasoning vs. memorization. |

**Training setup (CB, CC):** AdamW · lr `3e-4` · batch 8 × grad-accum 4 (effective 32) · 3 epochs ·
T4 GPU. Greedy decoding, temperature 0, fixed + documented seeds. Each condition is a self-contained
Colab notebook that runs end-to-end.

---

## Evaluation Plan

Two layers — standard metrics for comparability, plus failure analysis for insight.

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

## Definition of Done (Success Criteria)

The project is successful **iff all three hold**:

1. ✅ **≥ 1 condition beats the baseline by ≥ 5 Exact-Match points** on WikiTableQuestions.
2. ✅ **≥ 1 condition reaches ≥ 60% on TabFact** with zero TabFact training.
3. ✅ **Every condition completes in < 2 hours** on a free Colab T4.

Plus deliverable completeness: 5 reproducible notebooks (fixed seeds), full written report, slides.

---

## Repository Structure

```
ECS111FinalProject/
├── README.md                  # this file
├── Project_Proposal.pdf       # approved proposal
├── notebooks/
│   ├── 00_baseline.ipynb       # C0
│   ├── 01_cot_prompting.ipynb  # CA
│   ├── 02_finetune_answers.ipynb     # CB
│   ├── 03_finetune_traces.ipynb      # CC
│   └── 04_generalization.ipynb # CG
├── src/
│   ├── tableqa_core.py         # data loaders, table serialization, seed util
│   ├── metrics.py              # EM, token-F1, classification acc
│   ├── trainer.py              # shared fine-tune config (CB + CC)
│   └── trace_templates.py      # rule-based reasoning-trace generator (CC)
├── prompts/
│   └── cot_exemplars.md        # 6–8 hand-written CoT examples, both formats
├── results/                    # metrics JSON, tables, plots (per condition × seed)
├── report/                     # written report
└── slides/                     # presentation deck
```

> Folders are the planned layout; they get populated as the 4-day build progresses.

---

## Team & Task Division

Assignment goal: **even effort across all 5 members.** All members have free Colab → training
parallelizes across accounts.

| Member | Primary Ownership | Key Tasks |
|---|---|---|
| **Adisesh Venkatesh** | Foundation + Baseline | Shared core (`tableqa_core.py`, `metrics.py`) — Day-1 priority; baseline notebook (C0); ~5 CoT exemplars; chain-quality **rater #1** |
| **Amar Thota** | Condition A (CoT) | CoT prompting notebook (both chain formats, base + large); **lead CoT exemplar writing**; compute log for CA; assembles slide deck |
| **Nikhil Karthikeyan** | Condition B + Trainer | Shared `trainer.py` harness; fine-tune answers-only (CB, 2 seeds); chain-quality **rater #2** |
| **Anant Madhok** | Condition C | Rule-based reasoning-trace **template generator**; fine-tune + traces (CC, 2 seeds, reuses trainer); ~5 CoT exemplars |
| **Sanjay Manivasagam** | Results & Generalization | Generalization test (CG); error-type breakdown; McNemar + mean/std; compute-cost summary; all results tables + plots; merges report |

**Shared tasks (split for fairness):**

- **CoT exemplars** (6–8 × 2 formats ≈ 12–16): Amar (lead ~6) + Adisesh (~5) + Anant (~5). **Done Day 1**
  so CA can run Day 2.
- **Chain-quality raters** (Cohen's kappa needs 2): **Adisesh + Nikhil**, rated independently — neither
  is the CoT exemplar lead, to avoid bias.
- **Report sections** (Day 4): each writes their own area → Sanjay merges → all proof against rubric.
- **Slides** (Day 4): split by section; Amar assembles.

---

## 4-Day Execution Timeline

Dependencies marked `→`.

### Day 1 — Foundation + parallel starts
- **Adisesh:** shared core + metrics (TOP PRIORITY — unblocks everyone). Push by midday.
- **Nikhil:** trainer harness (uses core). Kick off CB run by evening.
- **Amar + Adisesh + Anant:** write all CoT exemplars (both formats).
- **Anant:** build reasoning-trace template generator.
- **Sanjay:** scaffold eval/analysis notebook against the metrics module.
- **Gate:** shared core + trainer working; baseline runs by end of day.

### Day 2 — Run conditions (parallel Colab)
- **Adisesh:** finish baseline (base + large), log compute.
- **Amar:** run CA — both chain formats, base + large (needs Day-1 exemplars).
- **Nikhil:** CB fine-tune answers-only, **both seeds** — start early.
- **Anant:** CC fine-tune + traces, **both seeds** — start early.
- **Sanjay:** lock error-label rubric + compute logging; evaluate C0 + CA as they land.

### Day 3 — Generalization, evaluation, stats
- **Sanjay:** generalization test (B + C on TabFact) → 60% threshold; then error breakdown, McNemar,
  mean/std, compute summary, tables + plots.
- **Adisesh + Nikhil:** chain-quality rating (100 chains each, independent) → Cohen's kappa.
- **Everyone:** re-run any condition failing a threshold or > 2 hr.
- **Gate:** all 3 success criteria checked; evening buffer for triage.

### Day 4 — Write + present
- **All:** report sections → Sanjay merges + proofs.
- **Amar:** assemble slide deck.
- **All:** rehearse; final reproducibility pass on fresh Colab runtime.

---

## Reproducibility & Setup

Each notebook runs top-to-bottom on a **fresh free Colab T4** with no manual setup.

```python
# install (first cell of every notebook)
!pip install -q transformers datasets evaluate accelerate

# load datasets — no auth needed
from datasets import load_dataset
wtq     = load_dataset("wikitablequestions")
tabfact = load_dataset("tab_fact", "tab_fact")

# models
# google/flan-t5-base   (250M)
# google/flan-t5-large  (780M)
```

**Reproducibility rules:**
- All random seeds **fixed and documented**; each condition run with **2 seeds**.
- Greedy decoding, `temperature=0` for prompting.
- Wall-clock timer logged in each notebook (must read **< 2 hr**).
- Checkpoints + result JSONs saved to Google Drive after every run (Colab can disconnect).

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Training is the bottleneck (2 seeds × 2 conditions) | Start CB/CC Day-1 evening; parallelize seeds across separate Colab accounts; checkpoint to Drive |
| Flan-T5-**large** fine-tune may OOM on T4 | Per proposal, **only fine-tune base**; keep large to prompting (C0/CA) — don't expand scope |
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
