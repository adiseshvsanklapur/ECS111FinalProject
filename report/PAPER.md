# Chain-of-Thought Prompting versus Fine-Tuning for Table Reasoning in Small Language Models

**Adisesh Venkatesh, Amar Thota, Nikhil Karthikeyan, Sanjay Manivasagam, Anant Madhok**
*University of California, Davis — ECS 111: Machine Learning*

---

## Abstract

When only a small, free-tier language model is available, is it better to spend effort on chain-of-thought (CoT) prompting or on supervised fine-tuning to make the model answer questions about tables? We run a controlled comparison on FLAN-T5-base (250M) and FLAN-T5-large (780M) over WikiTableQuestions (WTQ) and a held-out generalization test on TabFact, with two random seeds per condition and every reported number traced to a result file rather than hand-typed. The result is negative and consistent: no method we tried beat a plain zero-shot baseline. The strongest system overall was FLAN-T5-large prompted with no examples, at 0.241 exact match; few-shot CoT *more than halved* exact match (plain CoT: 0.015 base, 0.052 large), and light fine-tuning of the base model landed below its own baseline (0.157 answers-only, 0.121 with reasoning traces, versus 0.171 baseline). On the out-of-distribution TabFact transfer, the fine-tuned models did not transfer at all: they almost never emitted a gradeable true/false token (0.5% mappable outputs), so their 0.1–0.3% accuracy reflects an output-format collapse under distribution shift, not table-fact reasoning, and sits below the 0.551 majority-class floor for that reason. We trace the CoT degradation to a format effect — small models reason in prose while strict exact match rewards the bare cell — and we confirm the headline prompting-vs-fine-tuning gap is statistically real (McNemar, p < 10⁻¹³). We conclude that, at this scale and compute budget, the highest-leverage choice is the largest model one can prompt plainly, with careful output formatting, rather than CoT or a quick fine-tune.

---

## 1. Introduction

Tables are a default container for real-world data: medical records, government statistics, financial reports, and sports results all live in rows and columns. A language model that cannot read a table reliably is of limited use regardless of its fluency on free text. Yet table question answering is deceptively hard: it is a multi-step pipeline — understand the question, locate the relevant cells, choose the right operation (filter, count, compare, sort), and return the answer in the expected format — in which a slip at any step yields a confident wrong answer.

Two standard remedies exist. **Chain-of-thought (CoT) prompting** guides the model through intermediate reasoning at inference time, changing only the prompt and requiring no training. **Supervised fine-tuning** instead updates the model's weights on question/answer pairs. The published evidence for both is dominated by very large, expensive models. Wei et al. [5] reported that CoT is an emergent benefit of scale and that small models gain far less from it; most fine-tuning evidence likewise assumes compute that a student or hobbyist on a free Colab GPU does not have.

This paper studies the under-explored regime directly: **both methods applied to small models that anyone can run for free**, under a strict budget (a single free Colab T4, under two hours per condition). We ask a sharp, practical question — *when you cannot afford a 100B-parameter model, which lever helps table reasoning more, prompting or fine-tuning, and where does each one break?* We contribute:

1. A controlled comparison of zero-shot, two CoT prompt styles, and two fine-tuning recipes on the *same* small models (FLAN-T5-base and -large), with two seeds and reproducible seeded data slices.
2. A cross-dataset generalization test (train on WTQ, evaluate on TabFact with zero TabFact training) that isolates transfer from memorization.
3. An honest decomposition of the TabFact transfer result that separates *reasoning* failure from *output-format* failure — the distinction that makes the near-zero numbers interpretable.
4. A failure analysis (error-type breakdown, chain-quality rating with inter-rater agreement, and a paired significance test) and a full provenance discipline in which every reported number is regenerated from a result file by a script that fails loudly on any unmapped figure.

Our findings are negative, and we report them as such: no condition beat the plain baseline. We argue this is itself a useful, reproducible result for the small-model regime, and we explain the mechanisms behind it.

---

## 2. Related Work

**Chain-of-thought prompting.** Wei et al. [5] introduced CoT prompting and showed it elicits reasoning primarily in models above roughly 100B parameters, explicitly noting that smaller models benefit far less and can even be hurt. Subsequent work improved CoT for large models — self-consistency [4] aggregates multiple sampled chains, and tree-of-thoughts [7] searches over reasoning branches — but these add inference cost and target large models, and none focus on tables. Our work tests the small-model boundary Wei et al. flagged, on a structured-input task, under a fixed compute cap.

**Table question answering and verification.** WikiTableQuestions [3] established compositional QA over semi-structured tables (filtering, aggregation, comparison, multi-hop). TabFact [1] framed table understanding as binary fact verification (true/false over a statement and a table). TAPAS [2] trained a table-specialized encoder via weak supervision but without explicit step-by-step reasoning. UnifiedSKG [6] unified many structured-knowledge tasks under a text-to-text interface and is the closest prior framing to ours, but it does not pit prompting against fine-tuning as a controlled experiment, nor does it operate under strict free-tier compute limits.

**Position of this work.** The individual ingredients — CoT, table QA, seq2seq fine-tuning — are established. The specific combination is not: a controlled prompting-vs-fine-tuning comparison *at 250M–780M scale*, on WTQ and TabFact, *under a free-Colab budget*, with a cross-dataset transfer test and an error-type breakdown. The gap we fill is empirical evidence for practitioners who cannot scale.

---

## 3. Methodology

### 3.1 Models

We use the FLAN-T5 family: FLAN-T5-base (250M parameters) and FLAN-T5-large (780M). The base model is both prompted and fine-tuned; the large model is prompt-only, because fine-tuning it exhausts the 16 GB of a free Colab T4. FLAN-T5-small (80M) is used only for a fast end-to-end smoke check, never for reported numbers. All generation is greedy (`do_sample=False`, `num_beams=1`), making decoding deterministic.

### 3.2 Datasets

Both datasets load from the HuggingFace Hub with no account or API key. Because `datasets >= 4.0` removed script-based datasets, the canonical `wikitablequestions` and `tab_fact` ids no longer load; we use content-identical parquet mirrors. **WTQ** comes from `lighteval/wikitablequestions` as a single 18,486-example pool, from which we carve a *fixed, seeded, disjoint* train/eval partition that never overlaps. **TabFact** comes from `target-benchmark/tabfact-queries` joined to `tabfact-corpus` on `table_id`. Every table is serialized to text — column headers first, then each row, cells delimited by a bar — to fit the small models' token limits. WTQ is the primary train-and-test task; TabFact is used *only* for evaluation, never for training, so a fine-tuned model's TabFact performance measures transfer rather than memorization.

### 3.3 Conditions

We evaluate six conditions, ordered so earlier ones stand alone:

| # | Condition | Model(s) | Training | Description |
|---|-----------|----------|----------|-------------|
| C0 | Baseline | base + large | none | Question + serialized table, no examples. The floor. |
| CA-plain | CoT, plain | base + large | none | Six hand-written exemplars with free-paragraph reasoning prepended. |
| CA-struct | CoT, structured | base + large | none | Same six exemplars, reasoning as a fixed step template. |
| CB | Fine-tune, answers-only | base | yes | Target = final answer only (standard fine-tuning). |
| CC | Fine-tune, + reasoning traces | base | yes | Target = a rule-generated reasoning chain ending in the answer. |
| CG | Generalization | CB & CC models, + base floor | none | The fine-tuned (and untrained-base) models evaluated on TabFact, zero TabFact training. |

The two CoT styles let us separate *whether* reasoning demonstrations help from *whether their format* matters. The reasoning traces in CC are produced by rule-based templates that emit a chain only when the derivation is unambiguous and otherwise fall back to the plain answer, so a wrong chain never enters training; coverage was 3,740 of 8,000 training rows (46.8%) for both seeds.

### 3.4 Training

Fine-tuning uses one recipe for both CB and CC to keep the comparison fair: AdamW, learning rate 3×10⁻⁴, batch size 8 with gradient accumulation 4 (effective batch 32), three epochs, on a T4. The only difference between CB and CC is the target text. Each condition runs twice, with seeds 13 and 42, and we report mean ± standard deviation across the two.

### 3.5 Evaluation and Statistics

The official run evaluates on 1,000 examples for the baseline, fine-tuning, and TabFact conditions, and on 500 for the slower CoT conditions; the training pool is 8,000 examples. Metrics:

- **Exact match (EM)** — primary WTQ metric; predictions and golds are normalized (lowercased, punctuation removed) before comparison.
- **Token-level F1** — partial credit where EM is too strict (lists, ranges).
- **Classification accuracy** — TabFact; the model's free-form output is mapped to true/false by a keyword verbalizer (`true/yes/entail/supported/correct` → true; `false/no/refut/contradict/incorrect` → false), and outputs matching neither (or both) are unmappable and scored wrong.
- **Error-type breakdown** — every wrong WTQ answer is labeled *lookup*, *aggregation*, or *multi-hop* from question cues.
- **Chain quality** — 100 sampled CoT chains scored 0/1/2 by two independent rubrics (one strict, one lenient), with Cohen's κ for inter-rater agreement.
- **Compute** — inference seconds per example.

For the headline comparison (best CoT vs. best fine-tune on the base model), we run **McNemar's test** on paired per-example correctness, using an exact binomial test on the discordant pairs. Provenance is enforced mechanically: `report_fill.py` maps each reported figure to a result JSON and `finalize_report.py` exits non-zero if any numeric token is left unmapped, so no number in this paper is hand-typed.

---

## 4. Experiments and Results

### 4.1 Main task: WikiTableQuestions

Table 1 reports exact match and token-F1 for every WTQ condition, as mean ± std over seeds 13 and 42.

**Table 1: WikiTableQuestions — exact match and token-F1 (mean ± std, 2 seeds).**

| Condition | Model | Exact Match | Token-F1 |
|-----------|-------|-------------|----------|
| Baseline | base | 0.171 ± 0.011 | 0.204 ± 0.011 |
| Baseline | large | **0.241 ± 0.009** | **0.279 ± 0.011** |
| CoT, plain | base | 0.015 ± 0.001 | 0.062 ± 0.001 |
| CoT, plain | large | 0.052 ± 0.008 | 0.099 ± 0.007 |
| CoT, structured | base | 0.019 ± 0.003 | 0.059 ± 0.001 |
| CoT, structured | large | 0.147 ± 0.015 | 0.197 ± 0.011 |
| Fine-tune, answers | base | 0.157 ± 0.002 | 0.207 ± 0.006 |
| Fine-tune, traces | base | 0.121 ± 0.009 | 0.158 ± 0.012 |

Three results stand out. **The plain baseline is the best system at every comparison.** FLAN-T5-large with no examples reaches 0.241 EM, the highest number anywhere in the study; the base baseline (0.171) also beats every base-model method. **CoT degrades exact match severely.** Plain CoT collapses the base model to 0.015 and the large model to 0.052 — a drop of more than 4× and 4.6× from their baselines. Structured-step CoT is less destructive on the large model (0.147) but still well under its 0.241 baseline, and on the base model it is no better than plain CoT (0.019). **Fine-tuning does not beat the baseline either:** answers-only fine-tuning reaches 0.157 (below the 0.171 base baseline), and adding reasoning traces makes it *worse* at 0.121.

### 4.2 Generalization: TabFact transfer

The TabFact eval uses a prompt that explicitly asks for true or false, and scoring maps free-form output to a label via the verbalizer above, so the test is fair. The gold split on the 1,000-example slice is 551 false / 449 true, giving a majority-class floor of 0.551. We report four quantities per model, because raw accuracy alone is misleading here:

**Table 2: TabFact transfer (zero TabFact training, pooled over 2 seeds, n = 2000). Floor = 0.551.**

| Model | Accuracy | Mappable outputs | Unmappable | Accuracy \| mappable |
|-------|----------|------------------|-----------|----------------------|
| Untrained base (floor) | 0.043 ± 0.000 | 170 / 2000 | 0.915 | 0.506 (n = 170) |
| Fine-tune, answers | 0.003 ± 0.001 | 11 / 2000 | 0.995 | 0.455 (n = 11) |
| Fine-tune, traces | 0.002 ± 0.001 | 13 / 2000 | 0.994 | 0.308 (n = 13) |

The accuracy numbers (0.1–4.3%) are far below both the 60% target set in our proposal and the 0.551 majority floor. The decomposition explains why: the models almost never emit a gradeable true/false token. The untrained base produces a mappable answer only 8.5% of the time; the WTQ-fine-tuned models produce one 0.5% of the time, having locked into short WTQ-style outputs (the answers model emitted the literal token `0` 153 times in a single seed). Critically, *when* the untrained base does answer in-format, it is correct 0.506 of the time — exactly chance. The fine-tuned conditional rates rest on 11 and 13 examples and are not interpretable. This is an output-format collapse under distribution shift, not anti-correlation with truth: the models did not learn to verify facts, and fine-tuning on WTQ made the format mismatch strictly worse.

### 4.3 Error analysis

Every wrong WTQ answer is labeled by reasoning type. Table 3 gives the share of each error type among wrong answers, pooled over both seeds.

**Table 3: WTQ error-type share among wrong answers (pooled, 2 seeds).**

| Condition (model) | Lookup | Aggregation | Multi-hop |
|-------------------|--------|-------------|-----------|
| Baseline (base) | 26% | 61% | 13% |
| Baseline (large) | 24% | 64% | 12% |
| CoT plain (base) | 31% | 57% | 13% |
| CoT structured (large) | 28% | 60% | 13% |
| Fine-tune answers (base) | 29% | 57% | 14% |
| Fine-tune traces (base) | 30% | 55% | 15% |

Aggregation questions — *how many*, *highest/lowest*, *total* — are the dominant failure for every condition (55–64% of all errors), with lookups a distant second and multi-hop steady near an eighth. The error mix barely shifts across conditions: CoT did not change *what* the model gets wrong, it simply got far fewer answers right overall. The reasoning-traces fine-tune is the clearest negative: it was designed to reduce reasoning errors, yet its aggregation share (55%) is essentially unchanged from the answers-only model (57%) and its multi-hop share even rose — the chains added length without fixing operations.

### 4.4 Chain quality

Two rubrics scored 100 sampled CoT chains on a 0/1/2 scale: a strict rubric (full credit only for a chain that names a concrete table operation and reaches the gold answer) and a lenient rubric (credit for any multi-step chain that reaches an answer). Mean scores were 0.11 (strict) and 0.15 (lenient) out of 2 — both near the floor — and the two rubrics agreed at Cohen's κ = 0.784, substantial agreement. The chains are mostly empty or degenerate by either standard, and the strong κ confirms this read is not one rubric's quirk.

### 4.5 Significance

For the headline comparison — best CoT (structured) versus best fine-tune (answers) on the base model — McNemar's exact test on paired per-example correctness gives p < 10⁻¹³ (p = 2.2×10⁻¹⁴, seed 13). The gap is real, not seed noise: fine-tuning beats prompting on the base model. It does not change the headline, because both still lose to the plain baseline.

### 4.6 Compute

Inference cost (seconds per example) tracks prompt length as expected. Baseline inference is cheapest (0.126 s base, 0.200 s large); CoT, with six prepended exemplars, is 2–3.5× slower (0.290–0.308 s base, 0.715–0.718 s large); fine-tuned base inference is fastest of all at 0.091 s (answers) because it emits short outputs. Every condition completed well under the two-hour-per-condition budget on a free T4.

---

## 5. Discussion and Analysis

**Why CoT hurts.** The damage is a format effect, not a thinking effect. Under CoT, FLAN-T5 writes the reasoning out and frequently never emits the bare cell that exact match requires — e.g., it answers "The total number of senators … is 130." when the gold is "36". Token-F1, which gives partial credit, falls less steeply than EM (e.g., structured-large F1 0.197 vs. EM 0.147), confirming that some answer content survives but the surface form does not match. This is consistent with Wei et al.'s warning that models at this scale benefit little from CoT; we add that for strict-match table QA, the format mismatch turns "little benefit" into active harm.

**Why fine-tuning does not transfer.** Fine-tuning on WTQ optimizes the model to emit short factoid answers. That objective is directly at odds with TabFact, which needs a true/false token. The fine-tuned models therefore produce WTQ-style outputs on TabFact (numbers, table spans) that the verbalizer cannot map, collapsing coverage to 0.5%. The reasoning-traces variant does not rescue this: it neither improved WTQ EM nor TabFact transfer, indicating the rule-generated chains taught surface verbosity rather than transferable operations. The transfer result is thus best read as evidence of *format lock-in*, the fine-tuning analogue of the prompting format effect.

**What a practitioner should take away.** At 250M–780M on a free GPU, the highest-leverage choice is the largest model one can prompt plainly, plus attention to output format — not few-shot CoT and not a quick fine-tune, both of which cost accuracy here. The most reliable signal in our study is the simplest baseline.

**Limitations and threats to validity.** (1) Eval is a seeded 1,000-example slice (500 for CoT), not the full test set, to stay under budget; the small standard deviations across two seeds suggest the slice is stable, but it is still a sample. (2) Only the base model was fine-tuned; the large model's fine-tuning behavior is unknown because it OOMs on a T4. (3) The error-type labels are heuristic (question-cue based), so the breakdown is approximate. (4) The trace generator covers only unambiguous derivations (46.8% of rows), so CC leans on plain answers for the remainder, which may dilute any trace effect. (5) The TabFact accuracies are format-compliance numbers under distribution shift and should not be read as reasoning scores; we report the decomposition precisely so they are not misused.

---

## 6. Conclusion

We asked whether chain-of-thought prompting or supervised fine-tuning better helps a small, free-tier model read tables, and the answer at this scale is *neither*. The plain zero-shot baseline on FLAN-T5-large was the best system at 0.241 exact match; CoT more than halved exact match, light fine-tuning landed below the baseline, and the WTQ-fine-tuned models did not transfer to TabFact — their outputs collapsed to a non-true/false format, leaving accuracy below the 0.551 majority floor. The mechanisms are two faces of the same problem: small models manage table content but not the required output format, and both CoT and fine-tuning worsen that mismatch. For anyone on a small compute budget, the practical recommendation is to prompt the largest available model plainly and invest in output formatting, rather than in few-shot reasoning or a quick fine-tune. As a negative result, fully reproducible and with every number traced to a file, this is a useful data point for the small-model table-reasoning regime that the literature has largely left unexamined.

---

## Contribution Statement

- **Adisesh Venkatesh** built the shared core library and the baseline condition, wrote the dataset and experimental-setup components, and co-designed and applied the chain-quality rating rubrics.
- **Amar Thota** implemented and ran both chain-of-thought conditions (plain and structured) on both models, led the hand-written CoT exemplars, and assembled the slide deck.
- **Nikhil Karthikeyan** built the seq2seq training harness, ran the answers-only fine-tuning across both seeds, and co-designed and applied the chain-quality rating rubrics.
- **Anant Madhok** built the rule-based reasoning-trace generator and ran the reasoning-traces fine-tuning across both seeds.
- **Sanjay Manivasagam** led and coordinated the project, ran the TabFact generalization test, and produced the error analysis, the statistical tests (McNemar, Cohen's κ, mean/std), the results tables and plots, and the merged report.

---

## References

[1] Chen, W., Wang, H., Chen, J., Zhang, Y., Wang, H., Li, S., Zhou, X., & Wang, W. Y. (2020). *TabFact: A Large-scale Dataset for Table-based Fact Verification.* ICLR 2020.

[2] Herzig, J., Nowak, P. K., Müller, T., Piccinno, F., & Eisenschlos, J. M. (2020). *TAPAS: Weakly Supervised Table Parsing via Pre-training.* ACL 2020.

[3] Pasupat, P., & Liang, P. (2015). *Compositional Semantic Parsing on Semi-Structured Tables.* ACL 2015.

[4] Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023.

[5] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS 2022.

[6] Xie, T., Wu, C. H., Shi, P., Zhong, R., Scholak, T., Yasunaga, M., et al. (2022). *UnifiedSKG: Unifying and Multi-Tasking Structured Knowledge Grounding with Text-to-Text Language Models.* EMNLP 2022.

[7] Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models.* NeurIPS 2023.
