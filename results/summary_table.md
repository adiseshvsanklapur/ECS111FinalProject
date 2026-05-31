| condition                       | model         | task    | metric                  |   mean |    std |   n_seeds |   sec_per_example |
|:--------------------------------|:--------------|:--------|:------------------------|-------:|-------:|----------:|------------------:|
| generalization_baseline         | flan-t5-base  | tabfact | classification_accuracy | 0.043  | 0      |         2 |         0.207939  |
| generalization_finetune_answers | flan-t5-base  | tabfact | classification_accuracy | 0.0025 | 0.0005 |         2 |         0.11265   |
| generalization_finetune_traces  | flan-t5-base  | tabfact | classification_accuracy | 0.002  | 0.001  |         2 |         0.187188  |
| baseline                        | flan-t5-base  | wtq     | exact_match             | 0.171  | 0.011  |         2 |         0.125557  |
| baseline                        | flan-t5-large | wtq     | exact_match             | 0.241  | 0.009  |         2 |         0.200035  |
| cot_plain                       | flan-t5-base  | wtq     | exact_match             | 0.015  | 0.001  |         2 |         0.307899  |
| cot_plain                       | flan-t5-large | wtq     | exact_match             | 0.052  | 0.008  |         2 |         0.714749  |
| cot_structured                  | flan-t5-base  | wtq     | exact_match             | 0.019  | 0.003  |         2 |         0.290231  |
| cot_structured                  | flan-t5-large | wtq     | exact_match             | 0.147  | 0.015  |         2 |         0.717532  |
| finetune_answers                | flan-t5-base  | wtq     | exact_match             | 0.157  | 0.002  |         2 |         0.0910567 |
| finetune_traces                 | flan-t5-base  | wtq     | exact_match             | 0.1215 | 0.0085 |         2 |         0.16083   |