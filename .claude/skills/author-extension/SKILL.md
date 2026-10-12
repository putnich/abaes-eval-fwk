---
name: author-extension
description: Write and wire a module:function extension — retrieval selector, sampling strategy, dataset loader, custom metric, criteria aggregation, or model adapter — into a benchmark project
---

# Author Extension

Six pipeline steps accept user code named as `module:function` in configuration. The module lives beside `benchmark.yaml`; that directory is put on `sys.path` for every command, so nothing needs installing.

## Usage

```
/author-extension
/author-extension metric
/author-extension selector --config playground/benchmark.yaml
```

## Extension points

| Kind | Configured at | Called with | Returns | Example in playground `extensions/` |
|---|---|---|---|---|
| Retrieval selector | `retrieval.strategy` (or an experiment's `retrieval:`) | keywords `rubric`, `calibration_pool`, `student_text`, `criterion` | `list[HumanAssessment]` | `selectors.py` |
| Evaluation sampling | `sampling.evaluation_strategy` | `(df, n_per_set=...)` — positional frame | DataFrame | `sampling.py:lowest_scores_first` |
| Calibration sampling | `sampling.calibration_strategy` | `(dataset, df_full, rubric)` — positional | DataFrame (pool) | `sampling.py:one_per_score` |
| Dataset loader | `datasets[].loader` | `dataset`, `data_path` | DataFrame with `essay_id`, `set_id`, `essay`, `human_score` | `datasets.py` |
| Custom metric | `metrics.custom.<column>` | `preds`, `humans`, `scale` | number, or NaN when undefined | `metrics.py` |
| Criteria aggregation | rubric YAML `criteria_aggregation` | `rubric`, `scores` (dict by criterion) | overall score (rounded half-up afterwards) | `aggregation.py` |
| Model adapter | `models.<profile>.adapter_factory` | the resolved `LLMConfig` | LangChain-compatible chat model | `abaes_eval_fwk.playground:create_model` |

## Steps

1. Ask which kind if not given. Run `abaes-eval-fwk init` for a playground in a scratch directory if the user wants the worked examples to read.
2. Check a built-in doesn't already cover it: `criteria_aggregation` has `mean`, `sum`, `weighted`; retrieval has `zero_shot`, `random`, `lexical`, `semantic`, `hybrid`. Only write code for what configuration can't express.
3. Create or extend `<project>/extensions/<module>.py` (with an `extensions/__init__.py`). Match the call shape in the table exactly — keyword-only (`*,`) where the framework passes keywords, positional for the two sampling hooks. Keep the module free of import-time side effects.
4. Name it in configuration, e.g.:
   ```yaml
   metrics:
     custom:
       adjacent: extensions.metrics:adjacent_agreement
   ```
5. Validate, then dry-run on the playground or a small `--set sampling.sample_size=...`:
   ```bash
   abaes-eval-fwk validate <config> --check-data
   abaes-eval-fwk run <config> --experiment <name> --dry-run
   ```
   A metric can be tested with no model calls at all by running `abaes-eval-fwk metrics <config> <existing results.csv>`.

## Gotcha: extensions are part of the experiment's identity

Each run records a SHA-256 of every extension module its config names (`environment.extension_sha256` in `manifest.json`). Editing an extension after a run started means `--resume` on that run is rejected. That's intentional — changed code is a different experiment — so start a new run instead of working around it.
