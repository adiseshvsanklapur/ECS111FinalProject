# Chain of Thought Prompting versus Fine Tuning for Table Reasoning in Small Language Models

ECS 111 Final Report

Team: Adisesh Venkatesh, Amar Thota, Nikhil Karthikeyan, Sanjay Manivasagam, Anant Madhok


## Abstract

We ask a simple practical question. If all you can run is a small free language model, what helps it answer questions about tables more, chain of thought prompting or supervised fine tuning, and where does each one break. We test two Flan T5 models, the 250M base and the 780M large. We prompt both in three ways, a plain baseline with no examples, a few shot chain of thought in a paragraph style, and a few shot chain of thought in a structured step style. We also fine tune the base model on WikiTableQuestions two ways, once with only the answer as the target and once with a short rule built reasoning chain as the target. We score everything with exact match and token F1 on WikiTableQuestions and we test the two fine tuned models on TabFact, a different task we never trained on, to see if any of this carries over. On WikiTableQuestions the plain baseline on the large model reached an exact match of 0.241 ± 0.009, the best score anywhere in the study, and on the TabFact transfer test the best model scored only 0.043, far under the 0.551 majority-class floor. We report mean and spread over two seeds and we check the main gap with a McNemar test. The short version is that on a small free model no trick we tried beat just asking the plain question: chain of thought cut exact match by more than half, fine tuning landed below the baseline, and neither carried over to TabFact.


## 1. What we are doing and why

*Who writes this: the group. Sanjay merges.*

We wanted to know one thing. When all you can run is a small free model, what helps it read a table better, prompting or fine tuning. Tables show up everywhere. Medical records, sports stats, money reports, they all live in tables. So a model that cannot handle them is not very useful no matter how good it sounds on normal text.

There are two main tricks people use. One is chain of thought prompting where you show the model how to reason step by step right inside the prompt. The other is fine tuning where you actually train the model on questions and answers until it gets better. Both work in papers but those papers use giant models that cost a lot. We do not have that. So we tested both tricks on small models that anyone can run for free and we looked at which one wins and where each one falls apart.

Short answer up front: neither trick helped at this scale. The plain baseline won, chain of thought hurt, and fine tuning did not carry over to a new task.

## 2. The problem

*Who writes this: Adisesh.*

Reading a table looks easy but it is not. The model has to understand the question. Then it has to find the right cells. Then it has to pick the right move, like adding or sorting or comparing. Then it has to give the answer in the right form. If it slips on any one of those steps the answer comes out wrong and the model still sounds sure of itself.

Prompting helps by showing examples first. The risk is a small model just copies the look of the examples and misses the actual logic. Fine tuning helps by training on real answers. The risk there is the model memorizes shortcuts for one dataset and then breaks on a new one.

## 3. What others did before us

*Who writes this: Anant.*

Chain of thought is not new. Wei and the others showed it works in 2022 but mostly on models over a hundred billion parameters and they even said small models get a lot less out of it. Later work added things like self consistency and tree of thoughts but those cost even more compute and none of them focus on tables.

On the table side WikiTableQuestions and TabFact set up the tasks we use. TAPAS trained a model on tables but with no step by step reasoning. UnifiedSKG is the closest thing to what we do but it never put prompting against fine tuning in a clean head to head and it was not built for tight compute limits. That gap is the spot we are working in.

## 4. The data

*Who writes this: Adisesh.*

We use two datasets and both download straight from Hugging Face with no login.

WikiTableQuestions is our main one. The questions need real work, filtering rows, finding a max or a min, comparing across columns, sometimes two of those at once. That spread is why we picked it. We train and test on it.

TabFact is our second one and we only use it to test, never to train. There the model reads a statement and a table and says true or false. The format is different on purpose. If a model we trained on WikiTableQuestions still does okay on TabFact then it actually learned to reason instead of just memorizing.

One thing worth saying straight. The newer Hugging Face datasets library stopped loading the old script datasets so the plain wikitablequestions and tab_fact names do not work anymore. We swapped in parquet copies that hold the same data. For WikiTableQuestions that copy comes as one big pool so we split it ourselves into a train part and a test part with a fixed seed and we made sure the two parts never share an example. For TabFact we join the questions file to the tables file by the table id. We checked all of this by really downloading it.

For the official run the train pool is eight thousand examples and the eval slice is one thousand examples for the baseline, fine tune, and TabFact tests, with a smaller cap for the chain of thought tests since those prompts are much longer. The run that produced these numbers used eight thousand training examples, a one thousand example eval slice for baseline, fine tuning, and TabFact, and a five hundred example slice for the chain of thought tests, with every setup run twice, on seed thirteen and seed forty two.

## 5. The six setups

*Who writes this: Adisesh with Nikhil on the training details.*

Before anything else we turn every table into plain text. Headers first then each row, cells split by a bar. We keep it simple so it fits inside the small model token limit.

We test six setups in all. Three are prompting only and run on both models. Two are fine tuning and run on the base model only. The last one reuses the two fine tuned models on a brand new task. Here is what each one is.

The baseline, our setup C0, just hands the model the question and the serialized table with no examples in front of it. This is the floor.

The plain chain of thought setup puts six worked examples in front of the question and each example spells out the reasoning as a normal paragraph before giving the answer.

The structured chain of thought setup uses the same six examples but writes the reasoning as a fixed step template instead of a free paragraph. Same content, tighter shape. We test both styles so we can see if the shape of the reasoning matters for a small model.

The answers only fine tune trains the base model on WikiTableQuestions with just the final answer as the target. No reasoning in the training, only the answer. This is the normal way people fine tune.

The reasoning traces fine tune uses the same data and the same recipe but swaps the target. Instead of only the answer we put a short reasoning chain that ends in the answer. We build those chains with rules, not by hand, and the rules skip an example when the steps are not clear, so a wrong chain never gets into the training set.

The generalization setup is not a new training run. We take the two fine tuned models and we run them on TabFact, a task they never saw, to see what carried over.

## 6. How we ran it

*Who writes this: Nikhil with Adisesh on the eval details.*

For prompting we wrote out a handful of examples by hand with the reasoning spelled out and we stuck six of them in front of each question, in both the paragraph style and the step style.

For fine tuning we use the same training recipe everywhere so the comparison stays fair. AdamW, learning rate of three times ten to the minus four, batch of eight with four steps of accumulation so the real batch is thirty two, three passes over the data. We decode greedy with no sampling so the output is the same every time we run it. We fix the seeds and we run every setup twice, with seed thirteen and seed forty two, so we can report a mean and a spread.

The models are Flan T5 base at two hundred fifty million parameters and Flan T5 large at seven hundred eighty million. We fine tune the base model only. The large one runs out of memory when you try to train it on the free Colab T4, so for the large model we only prompt. The official numbers come from a full Colab T4 run on one thousand eval examples and eight thousand train examples. We built all of this as a small set of python files and a notebook that runs top to bottom on a free Colab GPU.

## 7. Baseline

*Who writes this: Adisesh.*

We started simple. Before any tricks we just handed the model the question and the table and nothing else and we wrote down what it got. This is our floor. Everything later has to beat this to mean anything. We ran it on both the base model and the larger one.

I expected it to do poorly and mostly it did, but it held up best on plain lookup questions and fell apart on aggregation, which is where most of its misses came from: on the base model about three in five wrong answers were aggregation questions and only about a quarter were lookups. The numbers are below.

Baseline exact match, base model: 0.171 ± 0.011. Large model: 0.241 ± 0.009. Token F1, base model: 0.204 ± 0.011. Large model: 0.279 ± 0.011.

## 8. Chain of thought prompting

*Who writes this: Amar.*

Here we put six worked examples in front of the question and asked the model to reason before answering. We ran both styles, the paragraph one and the numbered steps one, on both models.

We wanted to see two things. First does showing the reasoning help a small model at all. Second does the style of the reasoning matter. Both answers are negative. Showing the reasoning did not help, it hurt: plain chain of thought dropped the base model from 0.171 to 0.015 exact match and the large model from 0.241 to 0.052. Style mattered only in that the structured steps were less bad than the paragraph on the large model, 0.147 against 0.052, but both still trailed the no-example baseline. The cause is format, not thinking. The model writes the reasoning out and never emits the bare cell that exact match wants, which is why token F1 falls less hard than exact match does.

Plain style exact match: base 0.015 ± 0.001, large 0.052 ± 0.008. Structured step exact match: base 0.019 ± 0.003, large 0.147 ± 0.015. Token F1 for the plain style: base 0.062 ± 0.001, large 0.099 ± 0.007. For the structured style: base 0.059 ± 0.001, large 0.197 ± 0.011. We also looked at how long the reasoning made each prompt and what that cost us in time, see the compute part later.

## 9. Fine tuning on answers only

*Who writes this: Nikhil.*

Now we trained the base model on WikiTableQuestions with just the final answer as the target. No reasoning in the training, only the answer. This is the normal way people fine tune and we figured it would give us the best score on the WikiTableQuestions test since the model now saw the kind of questions and tables it gets asked about.

We trained it twice with two seeds and saved both models so we can test them on TabFact later.

Exact match across the two seeds: 0.157 ± 0.002. Token F1: 0.207 ± 0.006. We compare this straight against the traces version next.

## 10. Fine tuning with reasoning traces

*Who writes this: Anant.*

This setup uses the same data and the same recipe as the answers only one. The only change is the target. Instead of only the answer we put a short reasoning chain that ends in the answer.

We did not write those chains by hand for thousands of examples. We wrote rules that build a chain only when the steps are clear and there is no guessing. When a rule is not sure it skips the example and we just use the plain answer there. We did that on purpose. A wrong chain in the training set teaches the model the wrong thing so we would rather have fewer chains than bad ones.

The real question we are asking. Does training on the steps make the final answers better or does the model just learn to write more words without getting more right. It added words without adding correctness. Traces scored 0.121 exact match against 0.157 for answers only, and the error mix barely moved, aggregation stayed the dominant failure at 55 percent of misses against 57 percent, so the steps did not cut the reasoning mistakes they were meant to fix.

Exact match across two seeds: 0.121 ± 0.009. Token F1: 0.158 ± 0.012. How many training examples actually got a chain: 3740 of 8000, about 47 percent; the rest fell back to the plain answer because the rules would not commit to a chain.

## 11. Testing on a different dataset

*Who writes this: Sanjay.*

This is the part I care about most. We took the two fine tuned models from the answers setup and the traces setup and we ran them on TabFact without training them on TabFact at all. Different task, different tables, different answer format.

If a model really learned to reason it should still do okay here. If it only memorized WikiTableQuestions it will fall flat. So this number tells us more than the main score does.

Here is the whole picture, not one number. Untrained base floor: TabFact accuracy 0.043 ± 0.000, but it produced a mappable true-or-false on only 170 of 2000 outputs (0.915 unmappable), and among those it scored 0.506 (n=170). Answers model: accuracy 0.003 ± 0.001, coverage 11 of 2000, unmappable 0.995, accuracy when mappable 0.455 (n=11). Traces model: accuracy 0.002 ± 0.001, coverage 13 of 2000, unmappable 0.994, accuracy when mappable 0.308 (n=13). The majority-class floor on this slice is 0.551. Our target was sixty percent or better with no TabFact training.

These are floor-effect numbers and they need reading carefully. Under zero-shot transfer to TabFact the models emit 0, 1, or table spans instead of true or false, so 91 to 99 percent of outputs are unmappable and scored wrong. The reported accuracy of 0.1 to 4.3 percent therefore measures label-format compliance under distribution shift, not table-fact reasoning, and it sits below the 0.551 majority-class floor for that reason, not because the model is anti-correlated with truth. The tell is the conditional accuracy: when the untrained base does emit a true or false it is right 0.506 of the time, dead level with chance. Fine tuning on WikiTableQuestions makes this worse, not better, collapsing coverage to 11 and 13 mappable outputs out of 2000, so the answers and traces conditional rates carry n of 11 and 13 and are too thin to read as anything but a format collapse. We did not clear the sixty percent bar; on this evidence the fine tuned models did not transfer to TabFact at all.

## 12. How we measured things

*Who writes this: Sanjay.*

We used a few measures and each one is here for a reason.

Exact match is the main one for WikiTableQuestions. We lowercase and strip punctuation and then the answer has to match. Token F1 sits next to it and gives partial credit when the answer is a list or a range and exact match is too strict. For TabFact we just check the true or false call.

We also did not stop at the top line. For every wrong answer we tagged what kind of mistake it was. A lookup mistake means it grabbed the wrong cell. An aggregation mistake means it did the wrong operation. A multi hop mistake means it failed on a question that needed two steps. Two of us also read a hundred reasoning chains each and scored them zero one or two and then we checked how much we agreed using Cohen kappa. And we logged how long each setup took per example and how much memory it used since the prompting one writes much longer inputs.

## 13. Results

*Who writes this: Sanjay pulls the table together, everyone checks their row.*

The summary across all setups, mean and spread over two seeds, with time per example:

| condition                       | model         | task    | metric                  |   mean |    std |   n_seeds |   sec_per_example |
|:--------------------------------|:--------------|:--------|:------------------------|-------:|-------:|----------:|------------------:|
| generalization_baseline         | flan-t5-base  | tabfact | classification_accuracy | 0.043  | 0      |         2 |             0.208 |
| generalization_finetune_answers | flan-t5-base  | tabfact | classification_accuracy | 0.0025 | 0.0005 |         2 |             0.113 |
| generalization_finetune_traces  | flan-t5-base  | tabfact | classification_accuracy | 0.002  | 0.001  |         2 |             0.187 |
| baseline                        | flan-t5-base  | wtq     | exact_match             | 0.171  | 0.011  |         2 |             0.126 |
| baseline                        | flan-t5-large | wtq     | exact_match             | 0.241  | 0.009  |         2 |             0.200 |
| cot_plain                       | flan-t5-base  | wtq     | exact_match             | 0.015  | 0.001  |         2 |             0.308 |
| cot_plain                       | flan-t5-large | wtq     | exact_match             | 0.052  | 0.008  |         2 |             0.715 |
| cot_structured                  | flan-t5-base  | wtq     | exact_match             | 0.019  | 0.003  |         2 |             0.290 |
| cot_structured                  | flan-t5-large | wtq     | exact_match             | 0.147  | 0.015  |         2 |             0.718 |
| finetune_answers                | flan-t5-base  | wtq     | exact_match             | 0.157  | 0.002  |         2 |             0.091 |
| finetune_traces                 | flan-t5-base  | wtq     | exact_match             | 0.1215 | 0.0085 |         2 |             0.161 |

The full table is in `results/summary_table.md`, rebuilt straight from the result JSONs.

A few sentences on what jumps out. The plain baseline on the large model won outright at 0.241 exact match, and nothing beat it: every prompting and fine tuning setup scored lower. Ranked by family, the no-trick baseline came first, fine tuning second, prompting last. The large model beat the base model in every condition they shared, but adding any trick on top of the large model still left it short of its own plain baseline.

## 14. Where the models went wrong

*Who writes this: Sanjay.*

Two setups can land on the same score and still be bad at completely different things.

Aggregation questions, the how-many and the highest-and-lowest kind, were the dominant failure for every setup, between 55 and 64 percent of all wrong answers. Lookups were a distant second at a quarter to a third, and multi-hop questions stayed around an eighth. Chain of thought did not change that mix, it just got far fewer right overall. The reasoning-traces fine tune is the clearest miss: it was supposed to cut the reasoning errors and did not, aggregation stayed at 55 percent of its misses and multi-hop even ticked up, so the chains added length without fixing the operations.

The chain quality scores go here too. Rater A averaged 0.11 out of two, rater B averaged 0.15, and the two rubrics agreed at Cohen kappa 0.784. Both rubrics scored the sampled chains near the floor, so the chains were mostly empty or degenerate by either standard, and the strong kappa says that low read is not one grader's quirk: the two independent rubrics agreed the reasoning was thin.

## 15. Did the difference matter

*Who writes this: Sanjay.*

A gap in scores is only worth talking about if it is real and not just luck. So for the main fight, the best prompting setup against the best fine tuning setup, we ran McNemar on the paired answers.

The McNemar test on the paired per-example answers gives p = 0.000 (cot_structured vs finetune_answers, seed 13). That is well under 0.05, so the gap between the best fine tune and the best prompt is real and not seed noise: fine tuning beats prompting on the base model. It does not change the headline, though, because both still lose to the plain baseline.

## 16. What we think it means

*Who writes this: the group, Sanjay merges.*

The honest read is that this is a negative result, and a useful one. Chain of thought did not help the small model, it hurt it, exactly the failure mode Wei flagged for models this size, and most of the damage is that strict exact match punishes a model that reasons out loud instead of naming the cell. Fine tuning did not win either: it landed below its own baseline on WikiTableQuestions and then collapsed on TabFact, producing almost no gradeable true-or-false answers, which is the opposite of carrying over. The reasoning traces added words without adding correctness. What surprised us is that the simplest thing, just asking the plain question on the bigger model, beat everything we built on top of it. What we would not trust if someone tried to use this for real: any of the TabFact numbers as reasoning scores, since they are format-compliance numbers, not reasoning ones; and any claim resting on a single seed, since the spreads are small but real.

We also want to be straight about the limits. We capped the eval size to stay under the two hour budget so these numbers are on a sample not the whole test set. We only fine tuned the base model since the large one does not fit on the free GPU for training. And our trace rules only cover the clear cases so the traces setup leans on plain answers for the rest.

## 17. Conclusion

*Who writes this: Sanjay.*

We asked whether prompting or fine tuning helps a small free model read tables, and the answer at this scale is neither. The plain zero-shot baseline on Flan-T5-large was the best system at 0.241 exact match; chain of thought more than halved that, and light fine tuning on the base model landed under its own baseline and did not transfer to TabFact. Someone on a small budget should spend their effort on the biggest model they can prompt plainly and on getting the answer format right, not on few-shot reasoning or a quick fine tune, because on this evidence both cost accuracy rather than adding it.

## 18. Who did what

*Who writes this: the group.*

We split the work so everyone carried about the same load.

Adisesh built the shared core and the baseline and wrote the data and setup parts. Amar ran the chain of thought setup in both styles and led the hand written examples and put the slides together. Nikhil built the training code and ran the answers only fine tuning. Anant built the rule based trace generator and ran the traces fine tuning. Sanjay led and coordinated the project as lead author, and ran the generalization test, the error analysis, and the statistics, and pulled the report together. Two of us, Adisesh and Nikhil, scored the reasoning chains.

## References

Chen, W., et al. (2020). TabFact: A Large scale Dataset for Table based Fact Verification. ICLR 2020.

Herzig, J., et al. (2020). TAPAS: Weakly Supervised Table Parsing via Pre training. ACL 2020.

Pasupat, P., and Liang, P. (2015). Compositional Semantic Parsing on Semi Structured Tables. ACL 2015.

Wang, X., et al. (2023). Self Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023.

Wei, J., et al. (2022). Chain of Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022.

Xie, T., et al. (2022). UnifiedSKG: Unifying and Multi Tasking Structured Knowledge Grounding. EMNLP 2022.

Yao, S., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. NeurIPS 2023.
