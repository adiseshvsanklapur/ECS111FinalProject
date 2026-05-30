# Chain of Thought Prompting versus Fine Tuning for Table Reasoning in Small Language Models

ECS 111 Final Report

Team: Adisesh Venkatesh, Amar Thota, Nikhil Karthikeyan, Sanjay Manivasagam, Anant Madhok

> Note to the team. This is a skeleton. The words are here to get us going and to keep the voice the same across the whole thing. Anything inside square brackets like [FILL ...] is a number or a fact we drop in after the Colab runs finish. Each section says who writes it. Keep the writing plain. No dashes. Keep punctuation light. Write in active voice. Mix we and I. Read your part out loud and if it sounds like a robot wrote it then fix it.

## 1. What we are doing and why

*Who writes this: the group. Sanjay merges.*

We wanted to know one thing. When all you can run is a small free model, what helps it read a table better, prompting or fine tuning. Tables show up everywhere. Medical records, sports stats, money reports, they all live in tables. So a model that cannot handle them is not very useful no matter how good it sounds on normal text.

There are two main tricks people use. One is chain of thought prompting where you show the model how to reason step by step right inside the prompt. The other is fine tuning where you actually train the model on questions and answers until it gets better. Both work in papers but those papers use giant models that cost a lot. We do not have that. So we tested both tricks on small models that anyone can run for free and we looked at which one wins and where each one falls apart.

[FILL one sentence summary of the final takeaway once results are in.]

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

[FILL the exact counts we end up using, train size and eval size, once set.]

## 5. How we set things up

*Who writes this: Adisesh with Nikhil on the training details.*

We turn every table into plain text. Headers first then each row, cells split by a bar. We keep it simple so it fits inside the small model token limit.

For prompting we wrote out a handful of examples by hand with the reasoning spelled out and we stuck six of them in front of each question. We wrote each example in two styles. One reads like a normal paragraph and one reads like numbered steps. We test both to see if the style changes anything for a small model.

For fine tuning we use the same training recipe everywhere so the comparison stays fair. AdamW, learning rate of three times ten to the minus four, batch of eight with four steps of accumulation so the real batch is thirty two, three passes over the data. We decode greedy with no sampling so the output is the same every time we run it. We fix the seeds and we run every setup twice with two seeds so we can report a mean and a spread.

We built all of this as a small set of python files and five notebooks, one per setup, and each notebook runs top to bottom on a free Colab GPU.

## 6. Baseline

*Who writes this: Adisesh.*

We started simple. Before any tricks we just handed the model the question and the table and nothing else and we wrote down what it got. This is our floor. Everything later has to beat this to mean anything. We ran it on both the base model and the larger one.

I expected it to do poorly and mostly it did, but [FILL note any question types where the baseline was surprisingly okay]. The numbers are below.

Baseline exact match, base model: [FILL]. Large model: [FILL]. Token F1: [FILL].

## 7. Chain of thought prompting

*Who writes this: Amar.*

Here we put six worked examples in front of the question and asked the model to reason before answering. We ran both styles, the paragraph one and the numbered steps one, on both models.

We wanted to see two things. First does showing the reasoning help a small model at all. Second does the style of the reasoning matter. [FILL what we found on both.]

Plain style exact match: base [FILL], large [FILL]. Numbered steps exact match: base [FILL], large [FILL]. We also looked at how long the reasoning made each prompt and what that cost us in time, see the compute part later.

## 8. Fine tuning on answers only

*Who writes this: Nikhil.*

Now we trained the base model on WikiTableQuestions with just the final answer as the target. No reasoning in the training, only the answer. This is the normal way people fine tune and we figured it would give us the best score on the WikiTableQuestions test since the model now saw the kind of questions and tables it gets asked about.

We trained it twice with two seeds and saved both models so we can test them on TabFact later.

Exact match across the two seeds: [FILL mean and spread]. Token F1: [FILL]. We compare this straight against the traces version next.

## 9. Fine tuning with reasoning traces

*Who writes this: Anant.*

This setup uses the same data and the same recipe as the answers only one. The only change is the target. Instead of only the answer we put a short reasoning chain that ends in the answer.

We did not write those chains by hand for thousands of examples. We wrote rules that build a chain only when the steps are clear and there is no guessing. When a rule is not sure it skips the example and we just use the plain answer there. We did that on purpose. A wrong chain in the training set teaches the model the wrong thing so we would rather have fewer chains than bad ones.

The real question we are asking. Does training on the steps make the final answers better or does the model just learn to write more words without getting more right. [FILL the answer once we see it.]

Exact match across two seeds: [FILL]. How many training examples actually got a chain: [FILL coverage from the trace generator].

## 10. Testing on a different dataset

*Who writes this: Sanjay.*

This is the part I care about most. We took the two fine tuned models from the answers setup and the traces setup and we ran them on TabFact without training them on TabFact at all. Different task, different tables, different answer format.

If a model really learned to reason it should still do okay here. If it only memorized WikiTableQuestions it will fall flat. So this number tells us more than the main score does.

TabFact accuracy, answers model: [FILL]. Traces model: [FILL]. Our target was sixty percent or better with no TabFact training and [FILL whether we hit it].

## 11. How we measured things

*Who writes this: Sanjay.*

We used a few measures and each one is here for a reason.

Exact match is the main one for WikiTableQuestions. We lowercase and strip punctuation and then the answer has to match. Token F1 sits next to it and gives partial credit when the answer is a list or a range and exact match is too strict. For TabFact we just check the true or false call.

We also did not stop at the top line. For every wrong answer we tagged what kind of mistake it was. A lookup mistake means it grabbed the wrong cell. An aggregation mistake means it did the wrong operation. A multi hop mistake means it failed on a question that needed two steps. Two of us also read a hundred reasoning chains each and scored them zero one or two and then we checked how much we agreed using Cohen kappa. And we logged how long each setup took per example and how much memory it used since the prompting one writes much longer inputs.

## 12. Results

*Who writes this: Sanjay pulls the table together, everyone checks their row.*

The full table lives in the results folder and the notebook builds it for us. Drop the summary here.

[FILL the summary table: each setup, the model, exact match or TabFact accuracy as mean and spread over two seeds, and the time per example.]

A few sentences on what jumps out. [FILL which setup won on WikiTableQuestions, whether prompting or fine tuning came out ahead, and how the large model compared to the base one.]

## 13. Where the models went wrong

*Who writes this: Sanjay.*

This is the interesting part. Two setups can land on the same score and still be bad at completely different things.

[FILL the error breakdown read. Which setup made mostly lookup mistakes, which one choked on the two step questions, and whether the chains actually cut down the reasoning mistakes or not.]

The chain quality scores go here too. [FILL the average score and the Cohen kappa between the two of us so people know how much to trust the read.]

## 14. Did the difference matter

*Who writes this: Sanjay.*

A gap in scores is only worth talking about if it is real and not just luck. So for the main fight, the best prompting setup against the best fine tuning setup, we ran McNemar on the paired answers.

[FILL the p value and one plain sentence saying whether the difference is real or could just be noise.]

## 15. What we think it means

*Who writes this: the group, Sanjay merges.*

[FILL the honest read. Did prompting help the small model or barely move it like Wei warned. Did fine tuning win on WikiTableQuestions but fall apart on TabFact. Did the reasoning traces help or just add words. Say what surprised us and say what we would not trust if someone tried to use this for real.]

We also want to be straight about the limits. We capped the eval size to stay under the two hour budget so these numbers are on a sample not the whole test set. We only fine tuned the base model since the large one does not fit on the free GPU for training. And our trace rules only cover the clear cases so the traces setup leans on plain answers for the rest.

## 16. Conclusion

*Who writes this: Sanjay.*

[FILL three or four sentences. The question we asked, the answer we got, and what someone on a small budget should actually pick after reading this.]

## 17. Who did what

*Who writes this: the group.*

We split the work so everyone carried about the same load.

Adisesh built the shared core and the baseline and wrote the data and setup parts. Amar ran the chain of thought setup in both styles and led the hand written examples and put the slides together. Nikhil built the training code and ran the answers only fine tuning. Anant built the rule based trace generator and ran the traces fine tuning. Sanjay ran the generalization test and the error analysis and the stats and pulled the report together. Two of us, [FILL the two names], scored the reasoning chains.

## References

Chen, W., et al. (2020). TabFact: A Large scale Dataset for Table based Fact Verification. ICLR 2020.

Herzig, J., et al. (2020). TAPAS: Weakly Supervised Table Parsing via Pre training. ACL 2020.

Pasupat, P., and Liang, P. (2015). Compositional Semantic Parsing on Semi Structured Tables. ACL 2015.

Wang, X., et al. (2023). Self Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023.

Wei, J., et al. (2022). Chain of Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022.

Xie, T., et al. (2022). UnifiedSKG: Unifying and Multi Tasking Structured Knowledge Grounding. EMNLP 2022.

Yao, S., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. NeurIPS 2023.
