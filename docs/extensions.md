# Extensions

Six parts of the pipeline accept your own code instead of a built-in, named in configuration as
`module:function`. No package install or fork is needed.

## Contents

- [How extensions are found](#how-extensions-are-found)
- [Extension points](#extension-points)
- [Retrieval selector](#retrieval-selector)
- [Evaluation and calibration sampling](#evaluation-and-calibration-sampling)
- [Dataset loader](#dataset-loader)
- [Custom metric](#custom-metric)
- [Criteria aggregation](#criteria-aggregation)
- [Model adapter](#model-adapter)
- [Extensions and reproducibility](#extensions-and-reproducibility)

## How extensions are found

Put the module beside `benchmark.yaml`. That directory is added to `sys.path` for every command,
so `extensions.metrics:adjacent_agreement` imports `extensions/metrics.py` from your project. An
importable installed package works too.

```
my-project/
├── benchmark.yaml
├── config/...
└── extensions/
    ├── __init__.py
    └── metrics.py
```

The playground ships a working example of each point in `extensions/` (run
`abaes-eval-fwk init playground` to get a copy).

## Extension points

| Point | Configured at | Called as | Returns |
|---|---|---|---|
| Retrieval selector | `retrieval.strategy`, or an experiment's `retrieval` | `f(*, rubric, calibration_pool, student_text, criterion)` | `list[HumanAssessment]` |
| Evaluation sampling | `sampling.evaluation_strategy` | `f(df, n_per_set=...)` | DataFrame of rows to grade |
| Calibration sampling | `sampling.calibration_strategy` | `f(dataset, df_full, rubric)` | DataFrame pool (may be empty) |
| Dataset loader | `datasets[].loader` | `f(*, dataset, data_path)` | DataFrame with `essay_id`, `set_id`, `essay`, `human_score` |
| Custom metric | `metrics.custom.<column>` | `f(*, preds, humans, scale)` | number, or NaN where undefined |
| Criteria aggregation | rubric `criteria_aggregation` | `f(*, rubric, scores)` | overall score |
| Model adapter | `models.<profile>.adapter_factory` | `f(config)` | LangChain-compatible chat model |

Check the call shape carefully. The two sampling hooks receive their data positionally (evaluation
sampling gets `n_per_set` by keyword); the others receive keywords only.

## Retrieval selector

Replaces example selection. It receives the whole calibration pool for the dataset and returns
the examples to show, in order.

```python
def lowest_score_first(*, rubric, calibration_pool, student_text, criterion):
    matching = [a for a in calibration_pool if a.criterion.name == criterion.name]
    if not matching:  # criterion pools fall back to holistic entries
        matching = [a for a in calibration_pool if a.criterion.name == "holistic"]
    return sorted(matching, key=lambda a: (a.human_score, a.essay_id))[:3]
```

```yaml
retrieval: extensions.selectors:lowest_score_first
```

`n_ex_per_score`, `max_examples`, and the embedding settings don't apply; your function decides.
Return an empty list for zero-shot behavior on some essays.

## Evaluation and calibration sampling

Evaluation sampling picks the essays to grade from the essays left after calibration:

```python
def lowest_scores_first(df, n_per_set, group_col="set_id", exclude_ids=None):
    ordered = df.sort_values(["human_score", "essay_id"], kind="stable")
    return ordered.groupby(group_col, group_keys=False).head(n_per_set).reset_index(drop=True)
```

It must return a nonempty DataFrame. Calibration sampling builds the few-shot pool and must
return columns `rubric_id`, `essay_id`, `human_score`, plus one column per rubric criterion (use
`pd.NA` where there's no score). See `one_per_score` in the playground's `extensions/sampling.py`.
An empty pool is allowed, and is only a problem for few-shot retrieval.

Pools are cached with a fingerprint of the data and strategy, and `prepare` writes the resulting
splits without grading anything, so you can inspect a new strategy's choices for free.

## Dataset loader

See [datasets](datasets.md#your-own-essays-loader). The returned frame is validated exactly as a
built-in dataset's is.

## Custom metric

```python
def adjacent_agreement(*, preds, humans, scale):
    """Share of essays scored within one point of the human rater."""
    return sum(abs(p - h) <= 1 for p, h in zip(preds, humans, strict=True)) / len(preds)
```

```yaml
metrics:
  custom:
    adjacent: extensions.metrics:adjacent_agreement
```

`preds` and `humans` are paired rounded scores; `scale` has `min` and `max`. The key becomes the
column header, after the built-in columns. Return `math.nan` when the metric is undefined for a
sample rather than raising. Names that shadow a built-in (`nmae`, `qwk`, …) are rejected. Custom
metrics also apply when recomputing with `abaes-eval-fwk metrics`, which makes them cheap to
develop against an existing results CSV.

## Criteria aggregation

Only needed for formulas `mean`, `sum`, and `weighted` can't express:

```python
def capped_mean(*, rubric, scores):
    """Mean, but no more than one point above the weakest criterion."""
    values = list(scores.values())
    return min(sum(values) / len(values), min(values) + 1)
```

```yaml
# in the rubric YAML
criteria_aggregation: extensions.aggregation:capped_mean
```

`scores` maps criterion name to score. The result is rounded half-up.

## Model adapter

See [models](models.md#custom-adapters).

## Extensions and reproducibility

Every run hashes the source of each extension its configuration names and records the hashes in
`manifest.json` under `environment.extension_sha256`. Changing an extension's code is treated as
a new experiment: `--resume` of an earlier run is rejected. Commit your `extensions/` directory
alongside the config so results stay traceable.
