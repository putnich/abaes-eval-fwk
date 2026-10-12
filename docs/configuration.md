# Configuration reference

Every setting a project can use, its default, and what it changes. For how settings flow into a
run, see [architecture](architecture/02-configuration-resolution.md).

## Contents

- [benchmark.yaml and components](#benchmarkyaml-and-components)
- [Precedence and overrides](#precedence-and-overrides)
- [paths](#paths)
- [datasets](#datasets)
- [sampling](#sampling)
- [calibration](#calibration)
- [retrieval](#retrieval)
- [prompts](#prompts)
- [runtime](#runtime)
- [models](#models)
- [metrics](#metrics)
- [experiments](#experiments)
- [metadata](#metadata)
- [Validation](#validation)

## benchmark.yaml and components

```yaml
schema_version: 1
components:
  paths: config/paths.yaml
  datasets: config/datasets.yaml
  sampling: config/sampling.yaml
  calibration: config/calibration.yaml
  retrieval: config/retrieval.yaml
  metrics: config/metrics.yaml
  prompts: config/prompts.yaml
  runtime: config/runtime.yaml
  models: config/models.yaml
  experiments:            # a list is concatenated, so experiments can be split by RQ
  - config/rq1.yaml
  - config/rq2.yaml
```

`schema_version: 1` is required. Every component is optional; an omitted one keeps the packaged
default. A component file holds its own top-level key (`sampling:` in `sampling.yaml`, and so
on), and a `paths` file holds the root keys `data_root`, `output_dir`, and `subdirs`. Any of
these keys can also be written inline in `benchmark.yaml` instead of in a component.

Paths inside any of these files resolve relative to `benchmark.yaml`'s directory, not the current
working directory.

## Precedence and overrides

From lowest to highest: packaged defaults → component files (in the listed order) → inline keys
in `benchmark.yaml` → `--set` overrides. Mappings merge key by key; lists and scalars replace.

```bash
abaes-eval-fwk run benchmark.yaml --experiment rq1-hybrid \
  --set sampling.sample_size=20 \
  --set retrieval.max_examples=3 \
  --set 'sampling.essay_ids=["101","102"]'
```

`--set` values are parsed as YAML, so `20` is an integer, `true` a boolean, and `[...]` a list.
Quote IDs that look numeric if their leading zeros matter.

## paths

| Key | Default | Meaning |
|---|---|---|
| `data_root` | `data` | Where built-in datasets are found, under `<data_root>/datasets/<family>/` |
| `output_dir` | `runs` | Root for runs, suites, splits, calibration pools, retrieval caches, and `analysis/` |
| `subdirs.metrics` | per run | Redirect metrics output; relative to `output_dir` |
| `subdirs.conversations` | per run | Redirect conversation logs; relative to `output_dir` |

Leave `subdirs` unset unless you need a shared location. By default each run writes its metrics
and conversations inside its own directory, so independent runs can't overwrite each other.

## datasets

A list of dataset entries. Full guide: [datasets](datasets.md).

```yaml
datasets:
- name: ASAP AES Set 1                 # built-in: name selects the family and set
  rubric: rubrics/asap_aes_set_1.yaml
  file: training_set_rel3.tsv
- name: My essays                      # custom: normalized CSV
  csv: data/my_essays.csv
  rubric: rubrics/my_essays.yaml
  set_id: prompt-1
  criteria_columns: [ideas, organization]
```

## sampling

Which essays are graded.

| Key | Default | Meaning |
|---|---|---|
| `sample_size` | `200` | Essays sampled per set (integer ≥ 1) |
| `seed` | `42` | Seed for sampling, calibration, and random retrieval |
| `evaluation_strategy` | `quantile_rank` | Built-in score-diverse, rank-spaced sampler, or `module:function` |
| `calibration_strategy` | `stratified` | Built-in score-stratified pool, or `module:function` |
| `essay_ids` | none | Grade exactly these IDs instead of sampling; must be unique and outside the calibration pool |

An experiment can override any of these with its own `sampling:` mapping.

## calibration

| Key | Default | Meaning |
|---|---|---|
| `max_per_score` | `10` | Maximum calibration essays kept per human score level |

A dataset's `calibration_target_n` sets its target pool size separately. The calibration
strategy is chosen in `sampling.calibration_strategy`, not here.

## retrieval

How few-shot examples are chosen. See [architecture](architecture/03-setup-and-retrieval.md#example-retrieval).

| Key | Default | Meaning |
|---|---|---|
| `strategy` | `hybrid` | `zero_shot`, `random`, `lexical` (BM25), `semantic` (embeddings), `hybrid`, or `module:function` |
| `n_ex_per_score` | `1` | Examples per score level; `0` means zero-shot |
| `max_examples` | `6` | Total cap; `0` means zero-shot |
| `hybrid_dense_weight` | `0.5` | Dense share of the hybrid score |
| `hybrid_bm25_weight` | `0.5` | BM25 share; the two weights must sum to 1 |
| `seed` | `sampling.seed` | Retrieval seed |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face ID or local path |
| `embedding_device` | `cpu` | e.g. `cuda:0`, `mps` |
| `embedding_batch_size` | `32` | Encoding batch size |
| `embedding_threads` | `1` | CPU threads for encoding |

An experiment's `retrieval:` is either a strategy name or a mapping merged over these:

```yaml
retrieval: {strategy: random, n_ex_per_score: 2, max_examples: 4}
```

`semantic` and `hybrid` need `pip install -e '.[dense]'`. The embedding model downloads on first
use.

## prompts

The grading templates sent to the model. There are two slots, `holistic` and
`criterion_specialized`. A slot you leave out keeps the packaged wording.

```yaml
prompts:
  holistic:
    file: prompts/holistic.txt     # or an inline |- block
```

| Placeholders | Available in |
|---|---|
| `assignment_type`, `assignment_prompt`, `source_essay`, `student_work`, `score_min`, `score_max`, `examples_block`, `examples_comparison_step`, `source_verification_step`, `brevity_instruction` | both |
| `rubric_guidelines` | `holistic` |
| `criterion_label`, `criterion_guidelines` | `criterion_specialized` |

`student_work`, `score_min`, `score_max`, `examples_block`, and the slot's own guidelines are
required. Unknown or missing placeholders fail at load time. The resolved text is part of the run
identity, so editing a prompt makes a new experiment and blocks `--resume` of earlier runs.
Final-agent and review prompts aren't configurable.

## runtime

Applies to the whole run; experiments can't override it.

| Key | Default | Meaning |
|---|---|---|
| `max_workers` | `1` | Concurrent grading jobs |
| `max_attempts` | `3` | Attempts per model call, including the first |
| `compute_metrics` | `true` | Compute metrics after grading |
| `save_conversations` | `true` | Write per-essay transcripts |
| `context_overflow` | `error` | `error`, or `truncate_examples` to retry with progressively shortened examples |

## models

Named profiles, merged over `model_defaults`. Full guide: [models](models.md).

```yaml
models:
  model_a:
    provider: openai_compatible
    model_name: neuralmagic-ent/Llama-3.3-70B-Instruct-FP8-dynamic
    base_url: http://localhost:8000/v1
```

## metrics

| Key | Default | Meaning |
|---|---|---|
| `compute.{nmae,nrmse,pearson,spearman,qwk}` | `true` | Columns in the saved table |
| `custom.<column>` | none | `module:function` metric; can't reuse a built-in name |
| `significance_thresholds` | `0.001`, `0.01`, `0.05` | p-values for `***`, `**`, `*` |
| `wilcoxon.alternative` | `two-sided` | `two-sided`, `less`, `greater` |
| `wilcoxon.alpha` | `0.05` | Significance level |
| `wilcoxon.zero_method` | `wilcox` | `wilcox` drops zero differences; `pratt` keeps them |
| `wilcoxon.method` | `auto` | `auto`, `approx`, `exact` |
| `comparison.allow_partial_overlap` | `false` | Compare only the datasets and essays both runs share |
| `comparison.alpha` | `0.05` | Level used by `compare` |

`compare --alpha` and `--allow-partial` override the configured values.

## experiments

```yaml
experiments:
- name: rq1-semantic             # unique; letters, digits, _ and -; ≤ 64 chars
  rq: RQ1                        # RQ1–RQ4; a grouping label
  model: model_a                 # a profile in models
  architecture: holistic         # holistic · criterion_specialized · final_agent · non_grading
  retrieval: semantic            # name or mapping (see retrieval)
  reasoning: ""                  # "" · 'yes' · 'no' · no_think — quote yes/no
  procedure: none                # none · pool_average · disagreement · review_based · review
  datasets: [ASAP AES Set 1]     # optional subset of dataset names
  sampling: {sample_size: 50}    # optional override
  control:                       # optional controlled comparison
    baseline: rq1-zero_shot
    "on": [retrieval]            # quote "on": YAML reads a bare on as true
```

### Architecture and procedure

| `architecture` | Allowed `procedure` | `reuse` |
|---|---|---|
| `holistic`, `criterion_specialized` | `none` | not allowed |
| `final_agent` | `pool_average`, `disagreement`, `review_based` | `results` ≥ 2; `independent` required except for `pool_average` |
| `non_grading` | `review` | exactly one `results` file |

`disagreement_threshold` (default `0.2`, a fraction of the score range) applies to
`disagreement`. See [architecture](architecture/05-final-agent-procedures.md) for what each
procedure does.

### reuse

```yaml
reuse:
  results: ['@rq2-a-holistic', '@rq2-b-holistic']   # candidate assessments
  independent: '@rq2-final-holistic'                # the final model's own prior run
  reviews: ['@rq4-holistic-review']                 # cached review verdicts (review_based)
```

An `@name` reference schedules that experiment first and uses the results file its manifest
records. A plain path is relative to `benchmark.yaml`. Candidates must come from distinct models.

### reasoning

`reasoning` chooses which request options from the model profile are sent:

| Value | Sends |
|---|---|
| `""` | Neither; the model's native behavior |
| `'yes'` | `reasoning_on` (required in the profile) |
| `'no'` | `reasoning_off` (required in the profile) |
| `no_think` | `reasoning_off` plus prefill options that suppress a thinking block |

### control

A controlled comparison declares its `baseline` (same RQ) and the dimensions it's allowed to
change. Validation compares the two experiments' effective settings on `model`, `datasets`,
`sampling`, `retrieval`, `architecture`, `reasoning`, `procedure`, and `reuse`. It rejects the
config if they differ on anything other than exactly the declared dimensions. This catches, for
example, a retrieval comparison that accidentally also changes the model.

## metadata

A free-form mapping recorded in each run's manifest, for example a study name or hardware notes.
Values must be JSON-compatible.

## Validation

`abaes-eval-fwk validate benchmark.yaml` loads and checks everything without contacting a model:
unknown keys (catching typos), types and ranges, model profiles, extension references,
experiment rules, dependency cycles, and controlled comparisons. `--check-data` additionally
loads every dataset and validates it against its rubric. Run both before a long or paid run.
