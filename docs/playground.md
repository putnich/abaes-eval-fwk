# Playground guide

The playground is the framework's offline, zero-credentials demo mode. It swaps the real
LLM calls for a deterministic synthetic grader, so you can exercise every retrieval
policy, grading architecture, reasoning mode, and adjudication procedure the engine
supports without an API key, network access, or real essay data.

Its scores are illustrative, not benchmark evidence — see
[Why the scores aren't real grading](#why-the-scores-arent-real-grading) below.
For what each option does inside the engine, see [architecture](architecture/README.md); for every
setting, see [configuration](configuration.md).

## Contents

- [Quick start](#quick-start)
- [What `init` generates](#what-init-generates)
- [Project layout](#project-layout)
- [The CLI commands](#the-cli-commands)
- [The experiment matrix](#the-experiment-matrix)
- [Reading the output](#reading-the-output)
- [Overriding settings from the CLI](#overriding-settings-from-the-cli)
- [Writing your own experiment](#writing-your-own-experiment)
- [Extension points](#extension-points)
- [Resuming a run](#resuming-a-run)
- [Why the scores aren't real grading](#why-the-scores-arent-real-grading)

## Quick start

From an editable install (`pip install -e '.[dev]'`) or an installed wheel:

```bash
abaes-eval-fwk init playground
abaes-eval-fwk validate playground/benchmark.yaml --check-data
abaes-eval-fwk run playground/benchmark.yaml --dry-run    # preview the plan, no execution
abaes-eval-fwk run playground/benchmark.yaml --experiment rq1-zero_shot
```

The single experiment above finishes in a few seconds and writes a results CSV under
`playground/runs/`. No network call is made and no API key is read.

A generated `playground/` at the repository root is gitignored scratch; regenerate it
whenever you want a clean copy (`make playground` does all four steps).

## What `init` generates

`abaes-eval-fwk init <directory>` (template defaults to `playground`; `demo` and
`simulation` are aliases for the same template) requires `<directory>` to not exist or be
empty, then writes:

| Path | Contents |
|---|---|
| `benchmark.yaml` | The entry point every CLI command takes; points at the component files below |
| `config/paths.yaml` | Where runs, metrics, and conversation logs are written |
| `config/datasets.yaml` | Dataset definitions (CSV + rubric + criteria columns) |
| `config/sampling.yaml` | Evaluation/calibration split sizes and strategies |
| `config/calibration.yaml` | How many examples per score the calibration pool keeps |
| `config/retrieval.yaml` | Default retrieval policy and embedding settings |
| `config/metrics.yaml` | Which metrics to compute and Wilcoxon test settings |
| `config/prompts.yaml` | The holistic and criterion grading templates |
| `config/runtime.yaml` | Worker concurrency, retries, conversation logging |
| `config/models.yaml` | Model profiles — in the playground, all pointed at the synthetic adapter |
| `config/experiments.yaml` | The full RQ1–RQ4 experiment matrix (see below) |
| `demo.csv` / `demo-rubric.yaml` | The bundled synthetic dataset and its rubric |
| `extensions/` | One worked example per [extension point](extensions.md) |

Everything is a plain, editable file copied from the installed package's templates —
nothing is generated dynamically, and nothing under the project directory is required to
stay where `init` put it.

## Project layout

```
playground/
├── benchmark.yaml          # component map; every CLI command's "config" argument
├── demo.csv                # 20 synthetic essays × 2 set_ids (demo, demo-b)
├── demo-rubric.yaml         # holistic + 2-criterion (ideas, organization) rubric
├── config/
│   ├── paths.yaml
│   ├── datasets.yaml
│   ├── sampling.yaml
│   ├── calibration.yaml
│   ├── retrieval.yaml
│   ├── metrics.yaml
│   ├── prompts.yaml
│   ├── runtime.yaml
│   ├── models.yaml
│   └── experiments.yaml
├── extensions/
│   ├── selectors.py         # retrieval: lowest_score_first
│   ├── sampling.py          # sampling: lowest_scores_first / one_per_score
│   ├── datasets.py          # dataset loaders: transposed_csv / single_set_demo
│   ├── metrics.py           # custom metrics: exact/adjacent agreement, …
│   └── aggregation.py       # criteria aggregation: median, capped_mean, …
└── runs/                     # created on first `run`; gitignored
    └── <experiment>/<suite-id>/
        ├── manifest.json     # resolved config, versions, code hash, status
        ├── request.json      # exact input handed to the worker subprocess
        ├── splits.json       # evaluation/calibration essay IDs + rubric
        ├── results/*.csv     # per-essay scores, justifications, timing, tokens
        ├── checkpoints/      # resumable progress
        └── logs/
```

`benchmark.yaml` only maps component names to file paths — all paths inside the
component files resolve relative to `benchmark.yaml`'s own directory, so the whole
project is portable: copy the folder anywhere and it still works.

## The CLI commands

Every subcommand except `init` takes a `benchmark.yaml` path as its first argument, and
accepts repeatable `--set PATH=VALUE` overrides (see
[Overriding settings](#overriding-settings-from-the-cli)).

| Command | Purpose |
|---|---|
| `init <dir> [--template playground\|study]` | Scaffold a new project |
| `validate <config> [--experiment NAME...] [--check-data]` | Check the config resolves; `--check-data` also loads and validates the datasets |
| `prepare <config>` | Precompute calibration pools and evaluation/calibration splits |
| `run <config> [--experiment NAME...] [--dry-run] [--resume DIR]` | Execute experiments (default: the whole matrix) |
| `metrics <config> <results.csv> [--output DIR]` | Recompute NMAE/NRMSE/QWK/Pearson/Spearman from a results CSV |
| `compare <config> <baseline.csv> <variant.csv> [--allow-partial] [--alpha 0.05]` | Paired two-sided Wilcoxon test between two result sets |
| `baseline <config> <csv>... [--output DIR]` | Average several prior assessments into a pool-average baseline |

`run` without `--experiment` executes every experiment in `config/experiments.yaml`, in
declared order, tracking the whole batch under one `suite_id` at
`runs/suites/<suite_id>.json`. Pass `--experiment NAME` one or more times to run a subset;
any experiment that `reuse`s another (the RQ4 final-agent cases) automatically schedules
its dependency first.

`--dry-run` prints `name: rq / architecture / procedure / model / reasoning` for the
resolved plan and performs no model calls at all — always run this before a full matrix
run on an unfamiliar config.

## The experiment matrix

`config/experiments.yaml` is deliberately exhaustive: it is a worked example of every
option the engine accepts, organized under four research questions.

- **RQ1 — retrieval policy** (`rq1-*`): `zero_shot`, `random`, `lexical` (BM25),
  `semantic` (embeddings), `hybrid`; a tuned inline retrieval mapping
  (`rq1-tuned-random`); and a reader-supplied ranking function resolved from
  `extensions/selectors.py` (`rq1-reader-selector`).
- **RQ2 — architecture** (`rq2-*`): `holistic` vs. `criterion_specialized` grading,
  repeated for all three model profiles (`a`, `b`, `final`); plus a single-dataset
  override, a reduced-sample override, and reader-supplied evaluation/calibration splits
  from `extensions/sampling.py`.
- **RQ3 — reasoning mode** (`rq3-*`): native (no directive), forced `'yes'`, forced
  `'no'`, and `no_think`; the synthetic adapter ignores these, but the matrix shows how a
  real reasoning-capable model would be configured.
- **RQ4 — final-agent adjudication** (`rq4-*`): `pool_average`, `disagreement` (with a
  tunable `disagreement_threshold`), `review_based`, a `review`-only non-grading pass, and
  a cached variant that replays stored review verdicts instead of asking again — each run
  twice, once over `holistic` candidates and once over `criterion_specialized` candidates.

Entries that compare against a baseline declare `control: {baseline: <name>, "on":
[dimension]}`; validation then rejects any effective configuration difference outside the
declared dimension, so a retrieval comparison can't accidentally also vary the model.

RQ4 entries declare `reuse.results` (and sometimes `reuse.independent` /
`reuse.reviews`) as `'@experiment-name'` references — `run` resolves these by scheduling
the referenced experiment first if it hasn't completed yet in the same invocation.

## Reading the output

Each `run` writes to `runs/<experiment>/<suite-id>/`:

- **`manifest.json`** — resolved configuration, software versions, a source-code hash,
  input fingerprints, output file locations, and `status` (`complete`/`failed`).
- **`splits.json`** — which essay IDs went to evaluation vs. calibration, and the rubric
  used.
- **`results/*.csv`** — one row per essay (or per essay × criterion for
  `criterion_specialized`): `human_score`, `pred_score_raw`, `pred_score_rounded`,
  `confidence`, `justification`, `reasoning` text if any, retrieved `examples`, timing,
  token counts, retry/error info, and adjudication `branch` for RQ4 procedures.

Feed a results CSV to `metrics` for NMAE (primary metric, `10 × MAE / (score_max −
score_min)`), NRMSE, QWK, Pearson, and Spearman, averaged across datasets. Feed two
CSVs to `compare` for a paired Wilcoxon test — note it requires identical essay IDs and
human targets by default; pass `--allow-partial` to compare just the overlapping subset.

## Overriding settings from the CLI

Every command that takes a config accepts repeatable `--set PATH=VALUE`, a dotted path
into the merged YAML:

```bash
abaes-eval-fwk run playground/benchmark.yaml --experiment rq1-hybrid \
  --set sampling.sample_size=10 \
  --set retrieval.max_examples=3 \
  --set runtime.max_workers=2
```

This is the fastest way to try a one-off variation without editing or duplicating a YAML
file.

## Writing your own experiment

Add an entry to `config/experiments.yaml` with at minimum `name`, `rq` (just a label),
`model`, `architecture` (`holistic` / `criterion_specialized` / `final_agent` /
`non_grading`), and `retrieval` (a policy name, or an inline mapping like
`{strategy: random, n_ex_per_score: 2, max_examples: 4}`). Then:

```bash
abaes-eval-fwk validate playground/benchmark.yaml --experiment my-new-experiment
abaes-eval-fwk run playground/benchmark.yaml --experiment my-new-experiment --dry-run
abaes-eval-fwk run playground/benchmark.yaml --experiment my-new-experiment
```

Copy the surrounding RQ1/RQ2 entries as templates — most new experiments are a small diff
against an existing one, often with a `control.baseline` back-reference.

## Extension points

Six points accept your own code; [extensions](extensions.md) covers all of them. The two
most common: retrieval and sampling accept a `module:function` reference instead of a built-in
policy name. The named module is imported with the config file's own directory on
`sys.path`, so it can live beside `benchmark.yaml` — `extensions/` in the generated
project is only an example location, not a required one.

- **Custom retrieval** — `retrieval: my_pkg.my_module:select_examples`. The function
  receives `rubric`, `calibration_pool`, `student_text`, and `criterion` as keyword
  arguments and returns a list of `HumanAssessment`. See
  `extensions/selectors.py:lowest_score_first` for a minimal example.
- **Custom sampling** — `sampling.evaluation_strategy` /
  `sampling.calibration_strategy: my_pkg.my_module:function_name`. See
  `extensions/sampling.py` for the two signatures (`lowest_scores_first` for evaluation
  splits, `one_per_score` for calibration pools).

Neither extension point requires touching the grading/scoring code — the engine calls
whatever you name.

## Resuming a run

A failed or interrupted run keeps its checkpoints. Resume the exact run directory with:

```bash
abaes-eval-fwk run playground/benchmark.yaml --experiment rq1-hybrid \
  --resume playground/runs/rq1-hybrid/<suite-id>
```

`--resume` re-validates that the effective configuration, evaluated submissions,
calibration examples, rubrics, reuse files, and code/dependency versions all still match
before reusing any completed work. Running without `--resume` always starts a fresh,
separate run directory.

## Why the scores aren't real grading

The playground's `models.yaml` points every model profile (`a`, `b`, `final`) at
`abaes_eval_fwk.playground:create_model` instead of a real provider. That adapter
(`src/abaes_eval_fwk/playground.py`):

- Never calls an LLM or the network. It regexes the submission text out of the rendered
  prompt and derives a 0–4 score purely from the submission's **word count**, calibrated
  against the bundled `demo.csv`'s shortest and longest essays.
- Gives each profile a different rounding rule so independent assessments disagree
  realistically instead of by a constant gap: `demo-a` floors (harsh), `demo-b` ceils
  (lenient), `demo-final` rounds to nearest (the value the other two should converge
  toward) — which is what makes the RQ4 disagreement/adjudication procedures have
  something genuine to resolve.
- Reports `confidence=0.5` and a fixed `"Synthetic playground response; this is not a
  model assessment"` justification on every row, and zero token counts.

Treat the playground as a way to learn the CLI, config surface, retrieval/architecture/
reasoning/adjudication options, and output format end-to-end — never as evidence about
grading quality. For a real benchmark, see the README's ["Benchmark a
model"](../README.md#benchmark-a-model) section, which uses `--template study` and real
model endpoints instead.
