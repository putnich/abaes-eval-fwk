# abaes-eval-fwk

`abaes-eval-fwk` is a configuration-driven Python framework for controlled experiments in LLM-based agentic automated essay scoring (AES). It enables researchers to investigate how reference examples, agent specialization, reasoning modes, and final-agent procedures affect scoring performance.

Each design dimension can be varied independently while holding the model, dataset, evaluation sample, and other settings fixed. The framework validates experiment comparisons to ensure that only the intended dimension differs and records each run's configuration, inputs, and software environment to support reproducibility and analysis.

## Design dimensions

Each experiment is a YAML entry that sets four dimensions:

| Dimension | Options | Setting |
|---|---|---|
| Reference-example retrieval | Zero-shot, or human-scored essays chosen as few-shot references by random, lexical (BM25), semantic (embedding), or hybrid retrieval | `retrieval` |
| Rubric-based specialization | One holistic agent that scores against the whole rubric, or one agent per rubric criterion, with the criterion scores combined by the rubric's formula | `architecture` |
| Reasoning | The model's default behavior, or reasoning explicitly switched on or off, with the same prompts | `reasoning` |
| Final-agent deliberation | A final agent that turns a pool of initial assessments into one final assessment | `architecture: final_agent`, `procedure` |

The final agent can follow one of three procedures:

- **Pool average**: the mean of the initial scores. This is the baseline.
- **Disagreement-based adjudication (DA)**: if the initial scores lie close together (their mean
  pairwise distance is at most 20% of the score scale), their mean is kept. Otherwise the final
  agent's own independent assessment of the essay is used.
- **Review-based adjudication (RA)**: the final agent reviews every initial assessment for rubric
  consistency and evidence grounding. If all of them are admissible, their mean is kept; if only
  some are, the admissible scores are averaged together with the independent assessment; if none
  are, the independent assessment is used.

The adjudication rule always settles the score; the final agent's model is asked only to justify
it. Pools of criterion-specialized assessments are adjudicated criterion by criterion.

Any experiment can use any model profile, and the framework keeps no model allowlist. An
experiment can name a baseline and the dimensions it changes,
`control: {baseline: <experiment>, "on": [<dimensions>]}`, and then fails validation if it
differs from that baseline in anything else. This keeps every comparison controlled.

## Checking the shipped results

[`results/study/`](results/study/README.md) holds the per-essay output of every experiment in the
study, grouped by research question: RQ1 retrieval, RQ2 specialization, RQ3 reasoning, and RQ4
final-agent deliberation. Each row holds the dataset, essay ID, criterion, model, human and
predicted scores, and the model's confidence, plus completion tokens (RQ1–RQ3) or the
pool-average score (RQ4).

From these files alone, without calling any model, the framework recomputes:

- the evaluation metrics of each experiment, per dataset and averaged;
- the paired significance tests between experiments, with effect sizes;
- the RQ4 pool-average baselines;
- the RQ4 DA and RA outcome shares and review token costs.

Metrics and tests compare predictions with the official human scores, so first download the
three public corpora ([instructions](docs/datasets.md#downloading-the-benchmark-corpora)).
[results/study/README.md](results/study/README.md) gives every command, for example:

```bash
abaes-eval-fwk init study --template study      # then set data_root in study/config/paths.yaml
abaes-eval-fwk metrics study/benchmark.yaml \
  results/study/RQ1/holistic/RQ1_holistic_hybrid_llama.csv --output analysis/rq1-hybrid
abaes-eval-fwk compare study/benchmark.yaml \
  results/study/RQ1/holistic/RQ1_holistic_zero_shot_llama.csv \
  results/study/RQ1/holistic/RQ1_holistic_hybrid_llama.csv --output analysis/rq1.csv
```

`tests/test_replication_package.py` pins the RQ4 outcome shares and review token costs, and checks
that every shipped final score follows from its counted outcome.

## Install

Python 3.11–3.14 (3.12 recommended):

```bash
python3.12 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
```

Optional extras: `dense` for semantic and hybrid retrieval (embeddings run on CPU by default and
download on first use) and `deepseek` for the DeepSeek client.

## Try it offline

```bash
abaes-eval-fwk init playground
abaes-eval-fwk validate playground/benchmark.yaml --check-data
abaes-eval-fwk run playground/benchmark.yaml
```

`init` writes a self-contained demo project with synthetic essays, a rubric, and example
extensions. A deterministic stand-in replaces the model, so every retrieval strategy,
architecture, reasoning mode, and final-agent procedure runs without an API key, a GPU, or real
student data. Its scores only illustrate the output format. See the
[playground guide](docs/playground.md).

## Running new experiments

```bash
abaes-eval-fwk init study --template study
# 1. Download the corpora: docs/datasets.md#downloading-the-benchmark-corpora
# 2. Point the profiles in study/config/models.yaml at your model endpoints
abaes-eval-fwk validate study/benchmark.yaml --check-data
abaes-eval-fwk prepare study/benchmark.yaml        # draw samples and calibration pools only
abaes-eval-fwk run study/benchmark.yaml --experiment rq1-hybrid --dry-run
abaes-eval-fwk run study/benchmark.yaml --experiment rq1-hybrid
```

The `study` template declares six benchmark sets (ASAP AES Sets 1, 4, and 7; ASAP 2.0 Sets 1 and
6; ELLIPSE) and the RQ1–RQ4 experiment matrix. Its profiles expect OpenAI-compatible endpoints,
such as self-hosted models served with vLLM; hosted APIs and custom clients work too (see
[models](docs/models.md)). Any setting can be overridden for one run:

```bash
abaes-eval-fwk run study/benchmark.yaml --experiment rq1-hybrid \
  --set sampling.sample_size=20 --set runtime.max_workers=2
```

## What a run records

Each run writes `<output_dir>/<experiment>/<suite-id>/`:

- `manifest.json`: the resolved configuration, Python and package versions, hashes of the
  framework and extension source, fingerprints of every input, and the run status.
- `splits.json`: the evaluated and calibration essay IDs of each dataset, with its rubric.
- `results/*.csv`: one row per essay, or per essay and criterion.

Sampling, calibration pools, and random retrieval are seeded, so a rerun selects the same essays
and reference examples. A final-agent experiment names its inputs as `@experiment-name`, and
selecting it runs those experiments first. A failed run keeps its checkpoint and continues with
`--resume <run-dir>`; the resume is refused if the configuration, data, code, or dependency
versions changed.

## Evaluation

- **NMAE** is the primary metric. Each absolute error is rescaled to a 0–10 scale
  (`× 10 / (score_max − score_min)`) and the rescaled errors are averaged, so datasets with
  different score ranges can be compared. Lower is better.
- **NRMSE, QWK, Pearson, and Spearman** are reported alongside it.
- Averages over datasets weight every dataset equally.
- **Significance** uses two-sided paired Wilcoxon signed-rank tests on per-essay normalized
  absolute errors, per dataset, at α = 0.05, with the rank-biserial correlation as effect size
  and no multiple-comparison correction.
- Criterion scores are combined with the rubric's aggregation formula and compared with the
  overall human score (for ELLIPSE, rater 1's `Overall_1`).

| Command | Computes |
|---|---|
| `metrics <config> <results.csv>` | Metrics of one results file |
| `compare <config> <first.csv> <second.csv>` | Per-dataset Wilcoxon tests between two runs |
| `baseline <config> <csv>...` | A pool-average baseline from initial assessments |
| `adjudication <config> <csv>... --reviews <csv>...` | DA and RA outcome counts and review tokens |

## Extensions

Six parts of the pipeline accept your own `module:function`; a module next to `benchmark.yaml` is
imported without installing anything.

| Setting | Replaces |
| --- | --- |
| `retrieval.strategy` | Reference-example selection |
| `sampling.evaluation_strategy` / `.calibration_strategy` | The evaluation sample / calibration pool |
| `datasets[].loader` | Reading a corpus |
| `metrics.custom.<name>` | An extra metric |
| rubric `criteria_aggregation` | Combining criterion scores |
| `models.<profile>.adapter_factory` | The model client |

Grading prompts are editable in `config/prompts.yaml`. Extension code and prompt text are part of
a run's identity, so changing either starts a new experiment. See [extensions](docs/extensions.md).

## Repository layout

```
src/abaes_eval_fwk/        the package: CLI, configuration, agents, retrieval, evaluation
  config/config_files/     packaged defaults and the playground and study templates
  resources/               rubrics and the synthetic playground data
docs/                      user guides and architecture chapters
results/study/             per-essay results of the study, with recomputation steps
tests/                     unit tests, and integration tests that run the framework
```

## Documentation

| Guide | Covers |
|---|---|
| [Playground](docs/playground.md) | The offline demo: every option, no model or data |
| [Architecture](docs/architecture/README.md) | How a run works, with diagrams |
| [Configuration](docs/configuration.md) | Every setting, its default, and the experiment rules |
| [Models](docs/models.md) | Providers, credentials, reasoning modes, structured output, hosting |
| [Datasets](docs/datasets.md) | Downloading the corpora, your own essays, rubrics, validation |
| [Extensions](docs/extensions.md) | Custom selectors, sampling, loaders, metrics, aggregation, adapters |
| [Packaging](docs/packaging.md) | Wheel contents, versioning, releases, CI |

## Development

```bash
python -m pytest                  # unit tests
python -m pytest -m integration   # tests that run the framework
python -m build
```

See [contributing](.github/CONTRIBUTING.md). Student datasets, credentials, and model weights are
not distributed with the package; dataset access is subject to each provider's terms.

## License

[CC BY-NC 4.0](LICENSE): copy, modify, and redistribute for non-commercial purposes with
attribution. Copyright is held solely by the author, who retains all commercial rights; a
commercial license is available on request. Contributions are accepted only under terms that
permit relicensing. Runtime dependencies are MIT, BSD, or Apache-2.0 licensed and are installed,
not vendored.
