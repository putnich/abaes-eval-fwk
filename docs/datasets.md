# Datasets

How to get the benchmark corpora, use your own essays, and write rubrics. Student essays are not
distributed with the framework; each corpus stays under its provider's terms.

## Contents

- [Built-in datasets](#built-in-datasets)
- [Your own essays: normalized CSV](#your-own-essays-normalized-csv)
- [Your own essays: loader](#your-own-essays-loader)
- [Rubrics](#rubrics)
- [Validation rules](#validation-rules)
- [Restricting experiments to datasets](#restricting-experiments-to-datasets)

## Built-in datasets

Six benchmark sets are registered by name. Download each corpus from its provider and place the
files under `<data_root>/datasets/<family>/` (`data_root` defaults to `data`, relative to
`benchmark.yaml`):

| `name` | Family folder | File(s) | Source | Scores |
|---|---|---|---|---|
| `ASAP AES Set 1` | `ASAP_AES/` | `training_set_rel3.tsv` | [Kaggle ASAP AES](https://www.kaggle.com/c/asap-aes) | holistic (`rater1_domain1`) |
| `ASAP AES Set 4` | `ASAP_AES/` | `training_set_rel3.tsv` | same | holistic |
| `ASAP AES Set 7` | `ASAP_AES/` | `training_set_rel3.tsv` | same | holistic + ideas, organization, style, conventions (rater 1 traits) |
| `ASAP 2.0 Set 1` | `ASAP_2.0/` | `ASAP2_train_sourcetexts.csv` | [Kaggle ASAP 2.0](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2) | holistic; prompt "Exploring Venus" |
| `ASAP 2.0 Set 6` | `ASAP_2.0/` | `ASAP2_train_sourcetexts.csv` | same | holistic; prompt "Does the electoral college work?" |
| `ELLIPSE` | `ELLIPSE/` | `ELLIPSE_Final_github_train.csv` + `ellipsis_raw_rater_scores_anon_all_essay.csv` | [ELLIPSE Corpus](https://github.com/scrosseye/ELLIPSE-Corpus) | `Overall_1` + six analytic dimensions (rater 1) |

```
data/datasets/
├── ASAP_AES/training_set_rel3.tsv
├── ASAP_2.0/ASAP2_train_sourcetexts.csv
└── ELLIPSE/
    ├── ELLIPSE_Final_github_train.csv
    └── ellipsis_raw_rater_scores_anon_all_essay.csv
```

The `study` template's `config/datasets.yaml` already declares all six with the right column
mappings. Delete the entries you don't have files for. Every built-in uses the first rater's
score as the human target, so scores are a single rater's judgment, not an average of raters.

To use a file stored elsewhere, set `data: path/to/file` on the entry, relative to
`benchmark.yaml`.

## Your own essays: normalized CSV

The quickest route is a CSV that already has the framework's columns:

| Column | Required | Meaning |
|---|---|---|
| `essay_id` | yes | Unique per dataset; read as text, so leading zeros survive |
| `set_id` | yes | Prompt or subset label |
| `essay` | yes | The student text |
| `human_score` | yes | Overall score on the rubric's `overall` range |
| one column per criterion | for criterion grading | Analytic scores on the `per_criterion` range; blanks allowed |

```yaml
datasets:
- name: My essays                # the label shown in results and metrics
  csv: data/my_essays.csv
  rubric: rubrics/my_essays.yaml
  set_id: prompt-1               # optional: keep only this set
  criteria_columns: [ideas, organization]
```

The same CSV can back several datasets by giving each entry its own `set_id`, as the playground
does. `csv:` and `data:` can't be combined.

## Your own essays: loader

When a corpus isn't in that shape (different column names, several files, a database export),
write a loader instead of re-normalizing the file by hand before every run:

```yaml
datasets:
- name: My essays
  loader: extensions.datasets:my_loader
  csv: data/export.csv           # optional; passed to the loader as data_path
  rubric: rubrics/my_essays.yaml
```

```python
# extensions/datasets.py, beside benchmark.yaml
import pandas as pd


def my_loader(*, dataset, data_path):
    frame = pd.read_csv(data_path, dtype={"id": "string"}, keep_default_na=False, na_values=[""])
    return pd.DataFrame(
        {
            "essay_id": frame["id"],
            "set_id": dataset.subset_id or "default",
            "essay": frame["response"],
            "human_score": pd.to_numeric(frame["grade"]),
        }
    )
```

The returned frame goes through exactly the same validation as a built-in dataset's. The
playground's `extensions/datasets.py` has two worked examples. See also
[extensions](extensions.md).

## Rubrics

A rubric is a YAML file. The study rubrics are copied into `rubrics/` by `init --template study`
and are good starting points.

```yaml
assignment_type: Narrative essay        # required; shown to the model
score_ranges:
  overall: [0, 12]                      # the human_score range
  per_criterion: [0, 3]                 # the criterion score range
criteria_aggregation: sum               # mean (default) · sum · weighted · module:function
source_dependent: false                 # true adds source_text to prompts and a verification step
source_text: ''
prompt: |
  The prompt the students answered.
rubric_guidelines: |                    # optional overall guidance for holistic grading
  ...
criteria:
  ideas:
    score_levels:
    - score: 3
      guideline: Ideas are clearly focused and thoroughly developed.
    - score: 2
      guideline: ...
  organization:
    score_levels: [...]
```

### Criteria aggregation

Criterion-specialized grading produces one score per criterion. The rubric's
`criteria_aggregation` combines them into the overall score that's evaluated against
`human_score`:

| Value | Overall score |
|---|---|
| `mean` | Mean of the criterion scores (default) |
| `sum` | Sum, for rubrics whose total is the sum of traits (e.g. ASAP AES Set 7) |
| `weighted` | Weighted mean; needs `criteria_weights` |
| `module:function` | Your callable, for formulas the others can't express |

```yaml
criteria_aggregation: weighted
criteria_weights:          # every criterion exactly once; finite, ≥ 0, not all zero
  ideas: 0.5
  organization: 0.3
  conventions: 0.2
```

The result is rounded half-up to a score level. `criteria_weights` are rejected with any
aggregation other than `weighted`.

## Validation rules

`abaes-eval-fwk validate benchmark.yaml --check-data` loads every dataset and rejects:

- missing required columns, or an empty subset after `set_id` filtering
- missing or empty `essay_id`/`set_id`, and duplicate essay IDs
- missing or empty essay text
- `human_score` values that aren't finite or fall outside `overall`
- criterion scores outside `per_criterion` (blank criterion cells are allowed)
- `criteria_columns` that don't name a rubric criterion

On success it prints `<name>: <n> validated submissions` for each dataset.

## Restricting experiments to datasets

By default an experiment grades every configured dataset. Name a subset with `datasets:`:

```yaml
- name: rq2-a-holistic-one-dataset
  datasets: [Synthetic demo]
```

Metrics are reported per dataset, plus an average row that weights datasets equally regardless
of their size.
