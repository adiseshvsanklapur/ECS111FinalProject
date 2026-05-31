# Chain-of-Thought Prompting versus Fine-Tuning for Table Reasoning in Small Language Models

**Adisesh Venkatesh, Amar Thota, Nikhil Karthikeyan, Sanjay Manivasagam, Anant Madhok**
*University of California, Davis · ECS 111: Machine Learning*

---

## Abstract

If the only model you can run is a small, free one, what helps it answer questions about tables more: chain-of-thought (CoT) prompting or supervised fine-tuning? We compare both on FLAN-T5-base (250M) and FLAN-T5-large (780M), using WikiTableQuestions (WTQ) as the main task and TabFact as a held-out generalization test. Every condition runs on two random seeds, and every number in this paper is read from a result file rather than typed by hand. The answer came back negative, and it was consistent: nothing we tried beat a plain zero-shot baseline. The best system was FLAN-T5-large with no examples in the prompt, at 0.241 exact match. Few-shot CoT cut exact match by more than half (plain CoT scored 0.015 on base and 0.052 on large), and fine-tuning the base model landed below its own baseline (0.157 with answers only, 0.121 with reasoning traces, against a 0.171 baseline). On the TabFact transfer the fine-tuned models did not carry over at all. They almost never produced a usable true or false token (0.5% of outputs could be graded), so their accuracy of 0.1% to 0.3% really measures whether the output was in the right format on a new task, not whether the model can reason about tables, and that is why it sits below the 0.551 majority-class floor. We tie the CoT drop to a format problem: small models write their reasoning out in prose, while exact match only rewards the bare cell value. We also check that the gap between the best prompt and the best fine-tune is real and not noise (McNemar, p < 10⁻¹³). The short version: at this size and budget, you are better off prompting the biggest model you can run plainly and getting the answer format right than reaching for CoT or a quick fine-tune.

---

## 1. Introduction

Tables hold a lot of the data people actually care about: medical records, government statistics, financial reports, sports results. A model that cannot read a table well is not very useful, no matter how fluent it sounds on plain text. And table question answering is harder than it looks. The model has to understand the question, find the right cells, pick the right operation (filter, count, compare, sort), and then give the answer in the form the grader expects. Miss any one of those steps and you get a confident wrong answer.

There are two common fixes. **Chain-of-thought (CoT) prompting** walks the model through the reasoning at answer time. You only change the prompt, no training needed. **Supervised fine-tuning** trains the model on question and answer pairs instead. Almost all of the published evidence for both comes from very large, expensive models. Wei et al. [5] found that CoT mostly helps once models get very big, and that small models get much less out of it. Most fine-tuning results assume more compute than a student on a free Colab GPU has.

This paper looks at the case the literature mostly skips: both methods on small models that anyone can run for free, on a tight budget (one free Colab T4, under two hours per condition). The question we care about is simple. When you cannot afford a 100B-parameter model, does prompting or fine-tuning help a small model read tables more, and where does each one fall apart? We do four things:

1. We compare zero-shot, two CoT prompt styles, and two fine-tuning recipes on the same small models (FLAN-T5-base and large), with two seeds and fixed, seeded data splits.
2. We test transfer with a second dataset (train on WTQ, evaluate on TabFact with no TabFact training), which separates real reasoning from memorizing one dataset.
3. We break the TabFact transfer number into its parts so a reasoning failure is not confused with an output-format failure. That split is what makes the near-zero numbers mean something.
4. We dig into the failures (error types, chain-quality rating with two raters, and a paired significance test), and we keep every number honest by reading it back from a result file with a script that errors out if any number is missing.

Our results are negative: nothing beat the plain baseline. We think that is still a useful and reproducible finding for the small-model case, and we explain why it happens.

---

## 2. Related Work

**Chain-of-thought prompting.** Wei et al. [5] introduced CoT prompting and showed it mainly helps models above about 100B parameters. They also said plainly that smaller models get much less from it and can even do worse. Later work made CoT stronger for large models: self-consistency [4] samples several chains and takes a majority vote, and tree-of-thoughts [7] searches over reasoning branches. Both cost more at inference, both target large models, and neither focuses on tables. We test the small-model edge that Wei et al. pointed at, on a table task, with a fixed compute limit.

**Table question answering and verification.** WikiTableQuestions [3] set up question answering over semi-structured tables that needs filtering, aggregation, comparison, and multi-hop steps. TabFact [1] turned table understanding into a true or false check over a statement and a table. TAPAS [2] trained a table-specific encoder with weak supervision but no step-by-step reasoning. UnifiedSKG [6] put many structured-data tasks into one text-to-text setup and is the closest prior work to ours, but it does not run prompting against fine-tuning as a controlled experiment, and it was not built for tight free-tier compute.

**Where this fits.** The pieces are all known: CoT, table QA, and seq2seq fine-tuning. The combination is not. We run a controlled prompting-versus-fine-tuning comparison at 250M to 780M scale, on WTQ and TabFact, on a free-Colab budget, with a transfer test and an error breakdown. What we add is evidence for people who cannot scale up.

---

## 3. Methodology

### 3.1 Models

We use two FLAN-T5 models: FLAN-T5-base (250M parameters) and FLAN-T5-large (780M). We prompt and fine-tune the base model, and we only prompt the large one, because fine-tuning it runs out of the 16 GB on a free Colab T4. FLAN-T5-small (80M) is used only for a quick end-to-end check, never for reported numbers. All decoding is greedy (`do_sample=False`, `num_beams=1`), so the outputs are the same on every run.

### 3.2 Datasets

Both datasets load from the HuggingFace Hub with no account or API key. Because `datasets >= 4.0` removed script-based datasets, the canonical `wikitablequestions` and `tab_fact` ids no longer load, so we use content-identical parquet mirrors. **WTQ** comes from `lighteval/wikitablequestions` as a single 18,486-example pool, from which we carve a fixed, seeded, disjoint train and eval partition that never overlaps. **TabFact** comes from `target-benchmark/tabfact-queries` joined to `tabfact-corpus` on `table_id`. We turn each table into plain text, with the column headers first, then each row, and cells separated by a bar, so it fits the small models' token limits. WTQ is the task we train and test on. TabFact is only ever used for evaluation, never for training, so a fine-tuned model's TabFact score tells us about transfer rather than memorization.

### 3.3 Conditions

We evaluate six conditions, ordered so earlier ones stand alone:

| # | Condition | Model(s) | Training | Description |
|---|-----------|----------|----------|-------------|
| C0 | Baseline | base + large | none | Question + serialized table, no examples. The floor. |
| CA-plain | CoT, plain | base + large | none | Six hand-written exemplars with free-paragraph reasoning prepended. |
| CA-struct | CoT, structured | base + large | none | Same six exemplars, reasoning as a fixed step template. |
| CB | Fine-tune, answers-only | base | yes | Target is the final answer only (standard fine-tuning). |
| CC | Fine-tune, + reasoning traces | base | yes | Target is a rule-generated reasoning chain ending in the answer. |
| CG | Generalization | CB & CC models, + base floor | none | The fine-tuned (and untrained-base) models evaluated on TabFact, zero TabFact training. |

The two CoT styles let us separate two questions: does showing the reasoning help at all, and does the shape of that reasoning matter. The traces in CC come from rule-based templates that only write a chain when the steps are clear, and fall back to the plain answer otherwise, so a wrong chain never gets into training. That covered 3,740 of 8,000 training rows (46.8%) for both seeds.

### 3.4 Training

Both fine-tuning runs use the same recipe so the comparison stays fair: AdamW, learning rate 3×10⁻⁴, batch size 8 with gradient accumulation of 4 (effective batch 32), three epochs, on a T4. The only thing that changes between CB and CC is the target text. Each condition runs twice, on seeds 13 and 42, and we report the mean and standard deviation across the two.

### 3.5 Evaluation and Statistics

The full run evaluates on 1,000 examples for the baseline, fine-tuning, and TabFact conditions, and on 500 for the slower CoT conditions. The training pool is 8,000 examples. We track a few measures. **Exact match** is the main WTQ metric, after lowercasing and stripping punctuation. **Token-level F1** gives partial credit when exact match is too strict. **Classification accuracy** is for TabFact, where we map the model's free-form output to true or false with a small keyword rule (`true`, `yes`, `entail`, `supported`, `correct` count as true; `false`, `no`, `refut`, `contradict`, `incorrect` count as false), and anything that matches neither is counted as unmappable and wrong. The **error-type breakdown** labels every wrong WTQ answer as a lookup, aggregation, or multi-hop miss. **Chain quality** has 100 sampled CoT chains scored 0, 1, or 2 by two separate rubrics, with Cohen's κ for agreement. **Compute** is the inference time per example. For the headline comparison we run **McNemar's exact test** on paired per-example correctness. We keep the numbers honest with code: `report_fill.py` maps each reported figure back to a result file, and `finalize_report.py` stops with an error if any number is left unfilled, so nothing here is typed by hand.

---

## 4. Experiments and Results

### 4.1 Main task: WikiTableQuestions

Table 1 reports exact match and token-F1 for every WTQ condition, as mean ± std over seeds 13 and 42.

**Table 1: WikiTableQuestions, exact match and token-F1 (mean ± std, 2 seeds).**

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

Three things stand out. First, the plain baseline wins every comparison. FLAN-T5-large with no examples reaches 0.241 EM, the best number anywhere in the study, and the base baseline (0.171) beats every method that runs on the base model. Second, CoT hurts exact match a lot. Plain CoT drops the base model to 0.015 and the large model to 0.052, more than four times worse than their baselines. Structured-step CoT is less bad on the large model (0.147) but still well under its 0.241 baseline, and on the base model it is no better than plain CoT (0.019). Third, fine-tuning does not beat the baseline either: answers-only fine-tuning reaches 0.157, below the 0.171 base baseline, and adding reasoning traces makes it worse at 0.121. The token-F1 column shows why CoT looks so bad under exact match: F1 stays well above EM, which means the model does produce relevant content, just not the bare cell value that exact match wants.

### 4.2 Generalization: TabFact transfer

For TabFact we prompt the model to answer true or false, and we map its output to a label with the keyword rule above, so the test is fair. The gold split on the 1,000-example slice is 551 false and 449 true, so always guessing the majority class would score 0.551. We report four numbers per model, because the raw accuracy on its own is misleading here:

**Table 2: TabFact transfer (zero TabFact training, pooled over 2 seeds, n = 2000). Floor = 0.551.**

| Model | Accuracy | Mappable outputs | Unmappable | Accuracy \| mappable |
|-------|----------|------------------|-----------|----------------------|
| Untrained base (floor) | 0.043 ± 0.000 | 170 / 2000 | 0.915 | 0.506 (n = 170) |
| Fine-tune, answers | 0.003 ± 0.001 | 11 / 2000 | 0.995 | 0.455 (n = 11) |
| Fine-tune, traces | 0.002 ± 0.001 | 13 / 2000 | 0.994 | 0.308 (n = 13) |

The accuracy numbers (0.1% to 4.3%) sit far below both the 60% target from our proposal and the 0.551 majority floor. Breaking them apart shows why: the models almost never give a gradeable true or false. The untrained base produces a usable answer only 8.5% of the time, and the WTQ-fine-tuned models do so 0.5% of the time, because they have locked into short WTQ-style outputs (the answers model wrote the literal token `0` 153 times in a single seed). The telling part is that when the untrained base does answer in the right format, it is correct 0.506 of the time, which is just chance. The fine-tuned conditional rates rest on only 11 and 13 examples, so they do not mean much. So this is a failure to produce the right output format on a new task, not the model being wrong on purpose. The models did not learn to check facts, and fine-tuning on WTQ made the format mismatch worse.

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

Aggregation questions (the how-many, highest-or-lowest, and total kind) are the biggest source of errors in every condition, between 55% and 64% of all wrong answers. Lookups are a distant second, and multi-hop questions stay near an eighth. The mix barely changes across conditions, which means CoT did not change what the model gets wrong, it just got far fewer answers right overall. The reasoning-traces fine-tune is the clearest miss. It was meant to cut reasoning errors, but its aggregation share (55%) is about the same as the answers-only model (57%) and its multi-hop share even went up. The chains added length without fixing the operations.

### 4.4 Chain quality

Two rubrics scored 100 sampled CoT chains on a 0, 1, 2 scale. The strict rubric gives full credit only when a chain names a real table operation and reaches the gold answer. The lenient rubric gives credit to any multi-step chain that reaches an answer. Mean scores were 0.11 (strict) and 0.15 (lenient) out of 2, both near the floor, and the two rubrics agreed at Cohen's κ = 0.784, which is substantial agreement. By either standard the chains are mostly empty or degenerate, and the high κ says this is not just one rubric's opinion.

### 4.5 Significance

For the headline comparison, best CoT (structured) against best fine-tune (answers) on the base model, McNemar's exact test on paired per-example correctness gives p < 10⁻¹³ (p = 2.2×10⁻¹⁴, seed 13). The gap is real and not seed noise: fine-tuning beats prompting on the base model. It does not change the bigger picture, because both still lose to the plain baseline.

### 4.6 Compute

Inference cost (seconds per example) follows the prompt length, as expected. The baseline is cheapest (0.126 s base, 0.200 s large). CoT, with six prepended exemplars, is 2 to 3.5 times slower (0.290 to 0.308 s base, 0.715 to 0.718 s large). The fine-tuned base model is fastest at 0.091 s (answers) because it writes short outputs. Every condition finished well under the two-hour-per-condition budget on a free T4.

---

## 5. Discussion and Analysis

**Why CoT hurts.** The problem is the output format, not the thinking. With CoT, FLAN-T5 writes the reasoning out and often never gives the bare cell value that exact match needs. For example, it answers "The total number of senators is 130." when the gold answer is "36". Token-F1, which gives partial credit, drops less than exact match does (for instance, structured-large scores 0.197 F1 but only 0.147 EM), which tells us some of the right content is there but the surface form does not match. This lines up with Wei et al.'s point that models this size get little from CoT. We would add that for strict-match table QA, the format mismatch turns "little benefit" into real harm.

**Why fine-tuning does not transfer.** Fine-tuning on WTQ trains the model to write short factoid answers. That goal works against TabFact, which needs a true or false token. So on TabFact the fine-tuned models keep writing WTQ-style outputs, like numbers and table spans, that the keyword rule cannot map, and coverage drops to 0.5%. The reasoning-traces version does not fix this. It did not help WTQ EM or TabFact transfer, which suggests the rule-built chains taught the model to write more rather than to reason in a way that carries over. The cleanest way to read the transfer result is that fine-tuning locked the model into one output format, the training-time version of the same format problem we saw with prompting.

**What this means in practice.** At 250M to 780M on a free GPU, the best move is to prompt the biggest model you can run plainly and get the output format right. Few-shot CoT and a quick fine-tune both cost accuracy here. The most reliable thing in our study was the simplest baseline.

**Limitations.** (1) We evaluate on a seeded 1,000-example slice (500 for CoT), not the full test set, to stay under budget. The small standard deviations across two seeds suggest the slice is stable, but it is still a sample. (2) We only fine-tuned the base model, since the large one runs out of memory on a T4, so we do not know how it would behave fine-tuned. (3) The error-type labels come from simple question cues, so the breakdown is approximate. (4) The trace generator only writes chains for the clear cases (46.8% of rows), so CC falls back to plain answers for the rest, which could wash out any effect from the traces. (5) The TabFact accuracies measure output-format compliance on a new task, not reasoning, so they should not be read as reasoning scores. We report the full breakdown so they are not misused.

---

## 6. Conclusion

We asked whether chain-of-thought prompting or supervised fine-tuning helps a small, free model read tables more, and at this scale the answer is neither. The plain zero-shot baseline on FLAN-T5-large was the best system at 0.241 exact match. CoT cut exact match by more than half, light fine-tuning landed below the baseline, and the WTQ-fine-tuned models did not transfer to TabFact. Their outputs stopped being true or false, which left accuracy below the 0.551 majority floor. Both failures come from the same place: these small models handle the table content but not the output format the task wants, and both CoT and fine-tuning make that worse. If you are on a small budget, the practical advice is to prompt the largest model you can run plainly and spend your effort on getting the answer format right, not on few-shot reasoning or a quick fine-tune. As a negative result that is fully reproducible, with every number traced back to a file, this is a useful data point for the small-model table-reasoning case that the literature has mostly skipped.

---

## Contribution Statement

- **Adisesh Venkatesh** built the shared core library and the baseline condition, wrote the dataset and setup sections, and helped design and run the chain-quality rating.
- **Amar Thota** built and ran both chain-of-thought conditions (plain and structured) on both models, led the hand-written CoT examples, and put the slide deck together.
- **Nikhil Karthikeyan** built the training harness, ran the answers-only fine-tuning on both seeds, and helped design and run the chain-quality rating.
- **Anant Madhok** built the rule-based reasoning-trace generator and ran the reasoning-traces fine-tuning on both seeds.
- **Sanjay Manivasagam** led and coordinated the project, ran the TabFact generalization test, and did the error analysis, the statistics (McNemar, Cohen's κ, mean and standard deviation), the results tables and plots, and the final report.

---

## References

[1] Chen, W., Wang, H., Chen, J., Zhang, Y., Wang, H., Li, S., Zhou, X., & Wang, W. Y. (2020). *TabFact: A Large-scale Dataset for Table-based Fact Verification.* ICLR 2020.

[2] Herzig, J., Nowak, P. K., Müller, T., Piccinno, F., & Eisenschlos, J. M. (2020). *TAPAS: Weakly Supervised Table Parsing via Pre-training.* ACL 2020.

[3] Pasupat, P., & Liang, P. (2015). *Compositional Semantic Parsing on Semi-Structured Tables.* ACL 2015.

[4] Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023.

[5] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS 2022.

[6] Xie, T., Wu, C. H., Shi, P., Zhong, R., Scholak, T., Yasunaga, M., et al. (2022). *UnifiedSKG: Unifying and Multi-Tasking Structured Knowledge Grounding with Text-to-Text Language Models.* EMNLP 2022.

[7] Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models.* NeurIPS 2023.
