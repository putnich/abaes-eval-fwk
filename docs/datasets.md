# Datasets

How to get the benchmark corpora, use your own essays, and write rubrics. Student essays are not
distributed with the framework; each corpus stays under its provider's terms.

## Contents

- [Built-in datasets](#built-in-datasets)
- [Downloading the benchmark corpora](#downloading-the-benchmark-corpora)
- [Your own essays: normalized CSV](#your-own-essays-normalized-csv)
- [Your own essays: loader](#your-own-essays-loader)
- [Rubrics](#rubrics)
- [Validation rules](#validation-rules)
- [Restricting experiments to datasets](#restricting-experiments-to-datasets)

## Built-in datasets

Six benchmark sets are registered by name. Their files go under `<data_root>/datasets/<family>/`
(`data_root` defaults to `data`, relative to `benchmark.yaml`):

| `name` | Folder | File(s) | Human target |
|---|---|---|---|
| `ASAP AES Set 1`, `ASAP AES Set 4` | `ASAP_AES/` | `training_set_rel3.tsv` | `rater1_domain1` |
| `ASAP AES Set 7` | `ASAP_AES/` | `training_set_rel3.tsv` | `rater1_domain1` + four rater-1 traits |
| `ASAP 2.0 Set 1`, `ASAP 2.0 Set 6` | `ASAP_2.0/` | `ASAP2_train_sourcetexts.csv` | `score`; prompts "Exploring Venus" and "Does the electoral college work?" |
| `ELLIPSE` | `ELLIPSE/` | `ELLIPSE_Final_github_train.csv`, `ellipsis_raw_rater_scores_anon_all_essay.csv` | `Overall_1` + six rater-1 analytic scores |

```
data/datasets/
├── ASAP_AES/training_set_rel3.tsv
├── ASAP_2.0/ASAP2_train_sourcetexts.csv
└── ELLIPSE/
    ├── ELLIPSE_Final_github_train.csv
    └── ellipsis_raw_rater_scores_anon_all_essay.csv
```

The `study` template's `config/datasets.yaml` declares all six; delete entries you have no files
for. Human targets are the first rater's scores, not an average of raters. To keep a file
elsewhere, set `data: path/to/file` on its entry (relative to `benchmark.yaml`).

## Downloading the benchmark corpora

The Kaggle commands need the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) (`pip install
kaggle`) with an API token; downloading through the browser works as well. Run them from the
directory that holds `benchmark.yaml`.

**ASAP AES** comes from Kaggle's [The Hewlett Foundation: Automated Essay
Scoring](https://www.kaggle.com/c/asap-aes/data) competition. Accept the competition rules on
that page first. Only `training_set_rel3.tsv` is read.

```bash
kaggle competitions download -c asap-aes -p data/datasets/ASAP_AES
unzip -o data/datasets/ASAP_AES/asap-aes.zip -d data/datasets/ASAP_AES
```

**ASAP 2.0** is the Kaggle dataset [ASAP 2.0: Automated Student Assessment
Prize](https://www.kaggle.com/datasets/lburleigh/asap-2-0) (CC BY 4.0, about 200 MB). Don't use the
`learning-agency-lab-automated-essay-scoring-2` competition files instead: they lack the
`prompt_name` column the loader selects sets by.

```bash
kaggle datasets download -d lburleigh/asap-2-0 -p data/datasets/ASAP_2.0 --unzip
```

**ELLIPSE** comes from the [ELLIPSE-Corpus](https://github.com/scrosseye/ELLIPSE-Corpus) repository
(CC BY-NC-SA 4.0). The per-rater scores ship as a password-protected zip, and the password is in
that repository's README.

```bash
git clone --depth 1 https://github.com/scrosseye/ELLIPSE-Corpus.git ELLIPSE-Corpus
mkdir -p data/datasets/ELLIPSE
cp ELLIPSE-Corpus/ELLIPSE_Final_github_train.csv data/datasets/ELLIPSE/
unzip -P '<password from README>' ELLIPSE-Corpus/ellipsis_raw_rater_scores_anon_all_essay.zip \
  -d data/datasets/ELLIPSE
```

Then check that every set loads:

```bash
abaes-eval-fwk validate benchmark.yaml --check-data
```

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

One CSV can back several datasets, each with its own `set_id`, as in the playground. `csv:` and
`data:` can't be combined.

## Your own essays: loader

For a corpus in another shape (other column names, several files, a database export), write a
loader instead of re-normalizing the file before every run:

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

The returned frame is validated exactly like a built-in dataset's. The playground's
`extensions/datasets.py` has two worked examples; see also [extensions](extensions.md).

## Rubrics

A rubric is a YAML file. `init --template study` copies the study rubrics into `rubrics/`; they
make good starting points.

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

`criteria_aggregation` combines criterion-specialized scores into the overall score that is
evaluated against `human_score`:

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

The result is rounded half-up to a score level. `criteria_weights` is rejected with any
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

An experiment grades every configured dataset unless it names a subset:

```yaml
- name: rq2-a-holistic-one-dataset
  datasets: [Synthetic demo]
```

Metrics are reported per dataset, plus an average row that weights datasets equally.
