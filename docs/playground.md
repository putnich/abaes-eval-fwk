# Playground guide

The playground is an offline demo project. A deterministic synthetic grader stands in for the
model, so every retrieval policy, grading architecture, reasoning mode, and final-agent procedure
runs without an API key or real essays. The `semantic` and `hybrid` experiments need the `dense`
extra (`pip install -e '.[dev,dense]'`) and download a small embedding model the first time they
run. Its scores are illustrative, not benchmark evidence (see
[why](#why-the-scores-arent-real-grading)).

For what each option does, see [architecture](architecture/README.md); for every setting, see
[configuration](configuration.md).

## Contents

- [Quick start](#quick-start)
- [Project layout](#project-layout)
- [CLI commands](#cli-commands)
- [The experiment matrix](#the-experiment-matrix)
- [Reading the output](#reading-the-output)
- [Writing your own experiment](#writing-your-own-experiment)
- [Resuming a run](#resuming-a-run)
- [Why the scores aren't real grading](#why-the-scores-arent-real-grading)

## Quick start

```bash
abaes-eval-fwk init playground
abaes-eval-fwk validate playground/benchmark.yaml --check-data
abaes-eval-fwk run playground/benchmark.yaml --dry-run    # print the plan only
abaes-eval-fwk run playground/benchmark.yaml --experiment rq1-zero_shot
```

One experiment takes a few seconds and writes a results CSV under `playground/runs/`. `make
playground` regenerates a clean copy and runs the whole matrix; `playground/` at the repository
root is gitignored.

`init <dir>` requires an empty or missing directory. `--template` defaults to `playground`
(`demo` and `simulation` are aliases); `study` creates the research template. Every generated
file is a plain copy you can edit or move.

## Project layout

```
playground/
├── benchmark.yaml          # component map; the config argument of every command
├── demo.csv                # 20 synthetic essays × 2 sets (demo, demo-b)
├── demo-rubric.yaml        # holistic + 2-criterion (ideas, organization) rubric
├── config/                 # paths, datasets, sampling, calibration, retrieval, metrics,
│                           # prompts, runtime, models, experiments
├── extensions/             # one worked example per extension point
│   ├── selectors.py        # retrieval: lowest_score_first
│   ├── sampling.py         # sampling: lowest_scores_first, one_per_score
│   ├── datasets.py         # loaders: transposed_csv, single_set_demo
│   ├── metrics.py          # custom metrics: exact/adjacent agreement, …
│   └── aggregation.py      # criteria aggregation: median, capped_mean, …
└── runs/<experiment>/<suite-id>/
    ├── manifest.json       # resolved config, versions, code hash, status
    ├── request.json        # input handed to the worker process
    ├── splits.json         # evaluation/calibration essay IDs + rubric
    ├── results/*.csv       # per-essay scores, justifications, timing, tokens
    ├── checkpoints/        # resumable progress
    └── logs/
```

Paths in the component files resolve relative to `benchmark.yaml`, so the folder is portable.

## CLI commands

Every command except `init` takes `benchmark.yaml` first and accepts repeatable
`--set PATH=VALUE` overrides (a dotted path, value parsed as YAML).

| Command | Purpose |
|---|---|
| `init <dir> [--template playground\|study]` | Create a project |
| `validate <config> [--experiment NAME...] [--check-data]` | Check the configuration; `--check-data` also loads the datasets |
| `prepare <config>` | Build calibration pools and evaluation splits |
| `run <config> [--experiment NAME...] [--dry-run] [--resume DIR]` | Run experiments (default: all) |
| `metrics <config> <results.csv> [--output DIR]` | Recompute metrics from a results CSV |
| `compare <config> <baseline.csv> <variant.csv> [--allow-partial] [--alpha A]` | Paired Wilcoxon test between two runs |
| `baseline <config> <csv>... [--output PATH]` | Pool-average baseline from prior assessments |
| `adjudication <config> [<csv>...] [--reviews CSV...] [--output DIR]` | Count DA/RA outcomes and review tokens |

`run` without `--experiment` runs every experiment in declared order as one suite, tracked in
`runs/suites/<suite_id>.json`. An experiment that reuses another schedules that dependency first.
`--dry-run` prints `name: rq / architecture / procedure / model=… / reasoning=…` for each
planned run and makes no model calls.

```bash
abaes-eval-fwk run playground/benchmark.yaml --experiment rq1-hybrid \
  --set sampling.sample_size=10 --set retrieval.max_examples=3 --set runtime.max_workers=2
```

## The experiment matrix

`config/experiments.yaml` shows every option the engine accepts, grouped as four research
questions:

- **RQ1, retrieval** (`rq1-*`): `zero_shot`, `random`, `lexical`, `semantic`, `hybrid`, an inline
  retrieval mapping (`rq1-tuned-random`), and a custom selector (`rq1-reader-selector`).
- **RQ2, architecture** (`rq2-*`): `holistic` vs. `criterion_specialized` for each model profile
  (`a`, `b`, `final`), plus a single-dataset run, a smaller sample, and custom sampling.
- **RQ3, reasoning** (`rq3-*`): native, `'yes'`, `'no'`, and `no_think`. The synthetic grader
  ignores these; the entries show how a real model would be configured.
- **RQ4, final agent** (`rq4-*`): `pool_average`, `disagreement` (with a tunable threshold),
  `review_based`, a `review`-only pass, and a variant that replays cached reviews, each over
  holistic and over criterion candidates.

Entries with `control: {baseline: <name>, "on": [dimension]}` fail validation if they differ from
their baseline in anything else. RQ4 entries reference their inputs as `'@experiment-name'`.

## Reading the output

- **`manifest.json`**: resolved configuration, software versions, source-code hash, input
  fingerprints, output locations, and `status`.
- **`splits.json`**: evaluation vs. calibration essay IDs, and the rubric.
- **`results/*.csv`**: one row per essay (per essay × criterion for `criterion_specialized`) with
  `human_score`, `pred_score_raw`, `pred_score_rounded`, `confidence`, `justification`,
  `reasoning`, retrieved `examples`, timing, tokens, retries, and the RQ4 `branch`.

Pass a results CSV to `metrics`, or two to `compare`. `compare` requires identical essay IDs and
human targets unless `--allow-partial` is given.

## Writing your own experiment

Add an entry to `config/experiments.yaml` with at least `name`, `rq` (a label), `model`,
`architecture`, and `retrieval`. Copying a nearby entry is usually easiest. Then:

```bash
abaes-eval-fwk validate playground/benchmark.yaml --experiment my-experiment
abaes-eval-fwk run playground/benchmark.yaml --experiment my-experiment --dry-run
abaes-eval-fwk run playground/benchmark.yaml --experiment my-experiment
```

To plug in your own code (selectors, sampling, loaders, metrics, aggregation, adapters), see
[extensions](extensions.md).

## Resuming a run

```bash
abaes-eval-fwk run playground/benchmark.yaml --experiment rq1-hybrid \
  --resume playground/runs/rq1-hybrid/<suite-id>
```

`--resume` reuses completed work only if the configuration, essays, calibration examples,
rubrics, reused files, and code and dependency versions are unchanged. Without `--resume`, a new
run directory is created.

## Why the scores aren't real grading

Every playground profile uses `abaes_eval_fwk.playground:create_model`
([source](../src/abaes_eval_fwk/playground.py)), which:

- never calls a model or the network; it scores 0–4 from the submission's **word count**,
  calibrated to the shortest and longest essays in `demo.csv`;
- rounds differently per profile, so assessments genuinely disagree: `demo-a` floors, `demo-b`
  ceils, and `demo-final` rounds to nearest;
- marks every assessment admissible in reviews, and returns `confidence=0.5`, a fixed
  justification, and zero token counts.

Use it to learn the CLI, configuration, and output format, never as evidence of grading quality.
For a real benchmark, see [Running new experiments](../README.md#running-new-experiments).
