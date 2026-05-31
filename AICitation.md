# AI Citations

---

## Adisesh Venkatesh Sanklapur

**Contribution:** Built the shared core library and the baseline condition, wrote the dataset and experimental setup components, and co-designed and applied the chain-quality rating rubrics.

**AI use:** I used Claude (Anthropic) to help me understand the project from the ground up after reading our approved proposal. I shared the proposal and asked questions about table question answering, chain-of-thought prompting vs. fine-tuning, the datasets (WikiTableQuestions and TabFact), what the baseline and shared library needed to include, and how the chain-quality rubrics and Cohen's kappa worked. I used this to clarify concepts and plan my work, not to generate project code or write report text.

**Shared conversation:** [Claude chat](https://claude.ai/share/0e047213-4091-46af-b884-0f9cd691f00c)

---

## Nikhil

**Contribution:** Built the seq2seq training harness, ran the answers-only fine-tuning across both seeds, and co-designed and applied the chain-quality rating rubrics.

**AI use:** I used Claude (Anthropic) to help me understand the project from the ground up after reading our approved proposal. I shared the proposal and asked questions about what seq2seq and fine-tuning mean mechanically, what a "training harness" consists of in practice, the optimizer and hyperparameter settings (AdamW, learning rate, gradient accumulation), why running with two random seeds matters for reporting results, what chain-quality ratings are measuring and why exact match alone is insufficient, and why chain-of-thought prompting can hurt smaller models. I used this to clarify concepts and plan my work, not to generate project code or write report text.

**Shared conversation:** [Claude chat](https://claude.ai/share/4b12fc2b-47ae-4c39-8ecc-d86c6d914170)

---

## Anant Madhok

**Contribution:** Contributed to the project design, model selection analysis, evaluation methodology, chain-quality assessment framework, statistical testing plan, and report development. Also helped investigate fine-tuning strategies, reasoning-trace generation approaches, and methods for comparing baseline and trained models.

**AI use:** I used ChatGPT to help me understand and refine the project after reviewing our approved proposal. I discussed model selection (including the Flan-T5 family), fine-tuning approaches, reasoning-trace generation, chain-quality evaluation, Cohen's kappa, McNemar's test, and methods for comparing trained versus untrained models. I also used ChatGPT to explore potential evaluation metrics, dataset considerations, and experimental design decisions. I used these conversations to clarify concepts, evaluate methodological choices, and plan my work. I did not use ChatGPT to generate project code or write report content that was submitted as my own work.

**Shared conversation:** [ChatGPT chat](https://chatgpt.com/share/6a1cb605-22f0-83e8-a3fd-7a2233a9d037)

---

## Amar Thota

**Contribution:** Implemented and ran both chain-of-thought conditions (plain and structured) across both models, and led the hand-written CoT exemplars.

**AI use:** I used Claude (Anthropic) to build up my understanding of the project from the ground up after reading our approved proposal. I shared the proposal and asked questions about table question answering, chain-of-thought prompting versus fine-tuning, the datasets (WikiTableQuestions and TabFact), what the baseline and shared library needed to include, and how the chain-quality rubrics and Cohen's kappa worked. I used this to clarify concepts and plan my work, not to generate project code or write report text.

**Shared conversation:** [Claude chat](https://claude.ai/share/883fcc26-bfd6-4761-84b6-24619cdf0d02)

---

## Sanjay Manivasagam

**Contribution:** Led and coordinated the project as lead author. Ran the generalization test, the error analysis, and the statistics, and pulled the final report together.

**AI use:** I used Claude (Anthropic) to help me understand the project after reading our approved proposal, and to help with some of my analysis code. I shared the proposal and asked questions about table question answering, the generalization test from WikiTableQuestions to TabFact, how to categorize the model's errors, and how to read the statistics (significance testing and Cohen's kappa). Claude also helped me draft and debug some of the analysis and statistics code. I did not use it to generate the core project code or to write report text.

**Shared conversation:** [Claude chat](https://claude.ai/share/c4d80292-a61f-4615-ac49-996bd1691080)
