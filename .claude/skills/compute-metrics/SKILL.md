---
name: compute-metrics
description: Recompute NMAE/QWK/Pearson/Spearman metrics from a results CSV, compare two runs with a paired Wilcoxon test, build a pool-average baseline from prior assessments, or count DA/RA adjudication outcomes and review tokens
---

# Compute Metrics

Wraps the framework's `metrics`/`compare`/`baseline`/`adjudication` subcommands — all four reload the rubrics and datasets declared in a `benchmark.yaml` and reuse the same scoring code `run` calls automatically after grading.

## Usage

```
/compute-metrics results.csv --config study/benchmark.yaml
/compute-metrics --compare baseline_results.csv variant_results.csv --config study/benchmark.yaml
/compute-metrics --baseline-average results_a.csv results_b.csv --config study/benchmark.yaml
```

## Steps

1. Ask which config to use if not given — it determines which rubrics/datasets the CSV is evaluated against.
2. **One results CSV** → recompute its metrics standalone (e.g. after fixing a scoring/rubric bug):
   ```bash
   abaes-eval-fwk metrics <config> <results.csv> [--output DIR]
   ```
   Writes `<DIR or output_dir/analysis>/<results_stem>/<experiment_id>.csv`: one row per dataset (NMAE, NRMSE, QWK, Pearson, Spearman, plus any configured custom metrics) plus a trailing Average row. NMAE (absolute errors rescaled by `10 / (score_max − score_min)`, then averaged) is the primary metric.
3. **Two CSVs (baseline vs. variant)** → paired comparison:
   ```bash
   abaes-eval-fwk compare <config> <baseline.csv> <variant.csv> [--allow-partial] [--alpha 0.05]
   ```
   Requires identical essay IDs and human targets between the two by default — pass `--allow-partial` to explicitly allow comparing only the common subset. Always writes both sides' own metrics under `<output_dir>/analysis/comparison_metrics/{baseline,variant}/`, plus a verdict file (default `<output_dir>/analysis/comparison.csv`, or the exact path given to `--output`): a per-dataset two-sided Wilcoxon test on normalized absolute error, with sample overlap, p-value, direction, and effect size (no multiple-comparison correction).
4. **Multiple prior-assessment CSVs → one pooled baseline** (e.g. averaging several independent graders before comparing against a model variant):
   ```bash
   abaes-eval-fwk baseline <config> <source1.csv> <source2.csv> ... [--output PATH]
   ```
   Writes a synthetic results CSV usable as input to `metrics`/`compare`, to `<output_dir>/analysis/pool_average_results.csv` or the exact path given to `--output`.
5. **A pool's sources and/or its reviews → final-agent procedure outcomes and review cost**:
   ```bash
   abaes-eval-fwk adjudication <config> <source1.csv> <source2.csv> ... [--reviews <reviews.csv> ...] [--threshold 0.2] [--output DIR]
   ```
   DA outcomes (agreement/disagreement) are recomputed from the pool sources with the run-time disagreement rule; RA outcomes (positive/mixed/negative) and review completion tokens per essay come from the reviews CSVs (`is_admissible` must be `1`/`0`). Writes `outcomes.csv`, `outcome_items.csv`, and — when the reviews record tokens — `review_tokens.csv` to `<DIR or output_dir/analysis/adjudication>/`.
6. Report where each output file landed.
7. Offer to display a summary table of the metrics/verdict in the terminal.

## Gotcha: `--output` means different things per subcommand

For `metrics` and `adjudication`, `--output` is a **directory** (for `metrics` the file becomes `<DIR>/<results_stem>/<experiment_id>.csv`). For `compare` and `baseline`, `--output` is the **exact destination file path**. Passing a bare filename to `metrics --output` or a bare directory to `compare --output`/`baseline --output` produces the wrong shape — always show the resolved command before running it.

## Gotcha: dataset coverage and extensions

- A results CSV naming a dataset not declared in the config's `datasets` component raises `Dataset configurations missing for: [...]` — point the user at `config/datasets.yaml` if that happens.
- Any `metrics.custom.<name>` extension named in the config is resolved the same way during `metrics`/`compare` as during a live `run` — a broken extension fails the same whether or not a model was ever called.
