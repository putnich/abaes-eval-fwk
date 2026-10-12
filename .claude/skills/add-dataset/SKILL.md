---
name: add-dataset
description: Add a new essay dataset and its rubric to a benchmark project, via a normalized CSV or a custom loader, and validate it
---

# Add Dataset

A dataset is one entry in the project's `config/datasets.yaml` plus a rubric YAML. Built-in families (ASAP AES sets, ASAP 2.0 sets, ELLIPSE) are selected by name; everything else is a normalized CSV or a `module:function` loader.

## Usage

```
/add-dataset
/add-dataset my_essays.csv --config study/benchmark.yaml
```

## Steps

1. Ask for the config and the data source. Pick the route:
   - **CSV already has** `essay_id`, `set_id`, `essay`, `human_score` (plus one column per analytic criterion, if any) → `csv:` route.
   - **Different shape or not a CSV** → loader route (step 3). Prefer this over hand-normalizing a file that will be regenerated.
2. **CSV route** — add to `config/datasets.yaml`:
   ```yaml
   - name: My essays            # label used in results CSVs and metrics tables
     csv: data/my_essays.csv    # relative to the YAML file
     rubric: rubrics/my_essays.yaml
     set_id: prompt-1           # optional: keep only rows with this set_id
     criteria_columns:          # optional: columns holding per-criterion human scores
     - ideas
     - organization
   ```
   `set_id` is only valid with `csv:`/`loader:`. `csv:` and a raw `data:` path are mutually exclusive.
3. **Loader route** — write a function beside `benchmark.yaml` (see `/author-extension`) and name it:
   ```yaml
   - name: My essays
     loader: extensions.datasets:my_loader
     csv: data/raw_export.csv   # optional; passed to the loader as data_path
     rubric: rubrics/my_essays.yaml
   ```
   The loader receives `dataset=` and `data_path=` (None if no file is configured) and returns a DataFrame with the four base columns. The playground's `extensions/datasets.py` has two working examples.
4. **Rubric** — create the YAML (copy one from `rubrics/` or the packaged `resources/rubrics/` as a starting point):
   ```yaml
   assignment_type: Narrative essay   # required
   score_ranges:
     overall: [0, 12]
     per_criterion: [0, 3]
   criteria_aggregation: sum      # mean | sum | weighted | module:function
   source_dependent: false
   prompt: |
     The essay prompt shown to students.
   criteria:
     ideas:
       score_levels:
       - score: 3
         guideline: ...
   ```
   `weighted` requires `criteria_weights` naming every criterion exactly once (finite, ≥ 0, not all zero). Weights are rejected with any other aggregation.
5. Validate — this loads the frame and checks it against the rubric, the same way a built-in dataset is checked:
   ```bash
   abaes-eval-fwk validate <config> --check-data
   ```
   It prints `<label>: <n> validated submissions` per dataset.
6. If experiments restrict `datasets:`, tell the user which experiments will now include the new dataset and which won't.

## Gotcha: identifiers

Essay and set IDs are read as literal strings. Leading zeros and values like `NA` are preserved. Don't "clean" them in a loader with numeric casts — that silently changes which essays match across runs.
