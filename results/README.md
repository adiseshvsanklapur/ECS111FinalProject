# Results

Every file here is produced by the pipeline, not edited by hand. The numbers in the report and slides
are filled from these JSONs by `scripts/finalize_report.py`, which exits non-zero if any number is left
unmapped, so the prose can never drift from the data.

## Per-condition results: `<condition>_<model>_seed<seed>.json`

One file per (condition × model × seed). Schema (see `src/evaluate.py`):

- `condition`, `model`, `seed`, `task` (`wtq` | `tabfact`), `n`
- `metrics`: `exact_match` + `token_f1` (WTQ), or `classification_accuracy` (TabFact)
- `error_distribution`: lookup / aggregation / multi_hop / correct counts (WTQ only)
- `compute`: seconds/example, peak memory, device
- `predictions`: per example, `id`, `pred`, `gold`, `correct`, `error_type` (plus the `raw` chain for CoT)

Conditions: `baseline`, `cot_plain`, `cot_structured`, `finetune_answers`, `finetune_traces`, and
`generalization_*` (the base and fine-tuned models evaluated on TabFact with zero TabFact training).

## Aggregates and artifacts

- `summary_table.md` / `summary_table.csv`: mean ± std of the primary metric per condition, over seeds
- `primary_metric.png`: bar chart of the primary metric per condition
- `error_distribution.png`: WTQ error-type stack per condition
- `chain_quality.json`: Cohen κ plus the mean scores from the two rubric raters
- `chains_rated_a.csv` / `chains_rated_b.csv`: each rater's per-chain 0/1/2 scores
- `_archive_quick_seed13/`: superseded seed-13 quick run, kept for provenance
- `_smoke_archive/`: tiny smoke-run outputs

## Regenerate

```bash
python scripts/run_all_local.py --scale full      # per-condition JSONs (needs a GPU)
python scripts/rate_chains.py results/cot_structured_flan-t5-base_seed13.json
python scripts/finalize_report.py                 # fill the report numbers from these files
python scripts/make_slides.py                      # rebuild the deck from the same token map
```
