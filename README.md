# abaes-eval-fwk

A reusable framework for experiments in agent-based automated essay scoring. Configure the model, dataset, sampling, retrieval strategy, agent architecture, reasoning requests, and final-agent procedure in YAML; each grading run records its settings, inputs, scores, timing, token usage, and optional evaluation metrics.

## Install and verify

Use Python 3.11–3.14; Python 3.12 is the recommended starting point. From a checkout of this
repository, an editable install gives you the `abaes-eval-fwk` command and the source tree:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
abaes-eval-fwk init playground
abaes-eval-fwk validate playground/benchmark.yaml --check-data
abaes-eval-fwk run playground/benchmark.yaml
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. The playground contains synthetic essays and
content-aware authored responses, makes no model requests, and tests every retrieval policy, both
grading architectures, all reasoning modes, both adjudication procedures, and pool-average
baselines; its scores are illustrative demo output, not benchmark evidence.

Installing a built wheel instead, from outside the checkout:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install dist/abaes_eval_fwk-*-py3-none-any.whl
abaes-eval-fwk init playground
abaes-eval-fwk validate playground/benchmark.yaml --check-data
abaes-eval-fwk run playground/benchmark.yaml
```

`abaes-eval-fwk init playground` generates a demo project — `demo.csv`, `demo-rubric.yaml`,
`benchmark.yaml`, and `config/*.yaml` — from templates inside the installed package, so these commands
behave the same either way and need no checkout present. Build the wheel with `python -m build`
first; see [development](#development-and-provenance).

### Extension points

Six parts of the pipeline take your own code or wording instead of the built-in one. A code
extension is a `module:function` named in configuration; put the module beside `benchmark.yaml` —
its directory joins `sys.path` for the run — and the framework imports it with no install step.
Every reference a run names is hashed into the run manifest, so a resume refuses to continue an
experiment whose extension code changed underneath it.

| Setting | Replaces | Receives | Returns |
| --- | --- | --- | --- |
| `retrieval.strategy` | few-shot example selection | `rubric`, `calibration_pool`, `student_text`, `criterion` | `list[HumanAssessment]` |
| `sampling.evaluation_strategy` / `.calibration_strategy` | which essays are graded / calibrate | the frame and sizes (see the playground example) | a `DataFrame` |
| `datasets[].loader` | reading and normalizing a corpus | `dataset`, `data_path` | a `DataFrame` of `essay_id`, `set_id`, `essay`, `human_score` |
| `metrics.custom.<name>` | an extra reported metric | `preds`, `humans`, `scale` | a number, or NaN where undefined |
| `criteria_aggregation` (rubric) | how criteria combine into a total | `rubric`, `scores` | the overall score |
| `models.<profile>.adapter_factory` | the provider client | the resolved `LLMConfig` | a LangChain-compatible chat model |

A dataset loader is the one that removes a manual step: a corpus no built-in family reads no
longer has to be normalized into a CSV before every run. What it returns is validated exactly as a
packaged dataset's frame is, against that dataset's own rubric, so nothing enters a run less
checked than ASAP or ELLIPSE would be.

Two more take configuration rather than code. The **grading prompts** live in `config/prompts.yaml`
as the text that is sent, so the prompt is an editable part of an experiment rather than a constant
inside the agents; a template may be inlined or kept in its own file (`holistic: {file:
prompts/holistic.txt}`), and its placeholders are checked at load against what the grader supplies,
so a typo fails before the first model call rather than mid-run. The resolved text enters the run
fingerprint: an edited prompt is a new experiment, not a silent one. **Criteria aggregation**
covers the common analytic case declaratively — `criteria_aggregation: weighted` plus
`criteria_weights` per criterion in the rubric YAML — and falls back to a callable for a formula
weights cannot state.

Working examples of all five code points ship with the playground, in
[`extensions/`](src/abaes_eval_fwk/config/config_files/playground/extensions/): `selectors.py`,
`sampling.py`, `datasets.py`, `metrics.py`, and `aggregation.py`. Each is selected by naming it in
configuration, e.g. `retrieval: extensions.selectors:lowest_score_first` or
`metrics.custom.adjacent: extensions.metrics:adjacent_agreement`.

For semantic or hybrid retrieval, install `python -m pip install -e '.[dense]'`; embeddings default
to CPU and download the configured embedding model on first use. Random, lexical, and zero-shot runs
need no embedding model or GPU.

## Configuration files

The checked-in YAML definitions live in [`src/abaes_eval_fwk/config/config_files/`](src/abaes_eval_fwk/config/config_files/): the [research study components](src/abaes_eval_fwk/config/config_files/study/), [playground components](src/abaes_eval_fwk/config/config_files/playground/), and [shared defaults](src/abaes_eval_fwk/config/config_files/defaults/). Each generated project has a `benchmark.yaml` plus separate component files for paths, datasets, sampling, retrieval, runtime, models, and experiments. The playground's synthetic dataset and rubric are packaged data rather than configuration, so they live in [`src/abaes_eval_fwk/resources/playground/`](src/abaes_eval_fwk/resources/playground/) beside the [study rubrics](src/abaes_eval_fwk/resources/rubrics/).

## Benchmark a model

```bash
abaes-eval-fwk init study --template study
# Edit study/config/models.yaml and study/config/datasets.yaml: set endpoints and select datasets.
# Obtain the benchmark files described in docs/datasets.md.
abaes-eval-fwk validate study/benchmark.yaml --check-data
abaes-eval-fwk prepare study/benchmark.yaml
abaes-eval-fwk run study/benchmark.yaml --experiment rq1-hybrid --dry-run
abaes-eval-fwk run study/benchmark.yaml --experiment rq1-hybrid
```

The study template includes the six benchmark sets and these configurable comparisons:

- RQ1: zero-shot, random, lexical/BM25, semantic, and hybrid example retrieval.
- RQ2: holistic grading and independent specialists for every rubric criterion.
- RQ3: explicit reasoning on/off request settings with identical grading prompts, in both architectures.
- RQ4: pool-average baselines, disagreement-based adjudication, and review-based adjudication, in both architectures.

The template supplies three initial model profiles and a final model profile for RQ4. Edit them to identify your actual models; the framework does not maintain an allowlist of models or assume that a future model supports today's reasoning parameters.

A single experiment can use any combination of retrieval, architecture, and supported reasoning settings. RQ labels organize experiments; they do not restrict models or freeze the experimental design.

An experiment may declare `control: {baseline: <experiment>, "on": [<dimensions>]}`; validation then rejects any effective difference outside the declared dimensions. The study template uses these checks for the RQ1 retrieval, RQ2 architecture, RQ3 reasoning, and RQ4 procedure comparisons.

```bash
abaes-eval-fwk run study/benchmark.yaml --experiment rq1-hybrid \
  --set sampling.sample_size=20 --set retrieval.max_examples=3 \
  --set retrieval.n_ex_per_score=1 --set runtime.max_workers=2
```

See [configuration](docs/configuration.md) for all settings, [model connections and hardware](docs/models.md) for API and local hosting, and [datasets](docs/datasets.md) for acquisition and custom CSVs. YAML paths are relative to the configuration file; output never needs to be written inside the installed package.

## Documentation

| Guide | Covers |
|---|---|
| [Playground](docs/playground.md) | The offline demo: every option exercised with no model or data |
| [Architecture](docs/architecture/README.md) | How a run works: configuration resolution, sampling, retrieval, grading architectures, final-agent procedures, provenance (with diagrams) |
| [Configuration](docs/configuration.md) | Every setting, its default, and the experiment rules |
| [Models](docs/models.md) | Providers, credentials, reasoning modes, structured output, hosting |
| [Datasets](docs/datasets.md) | Built-in corpora, your own essays, rubrics, validation |
| [Extensions](docs/extensions.md) | Plugging in your own selectors, sampling, loaders, metrics, aggregation, adapters |
| [Packaging](docs/packaging.md) | Wheel contents, versioning, releases, CI |

## Results and comparisons

Each experiment writes a unique directory under `output_dir/<experiment>/<suite-id>/`. Its `manifest.json` includes the resolved configuration, software versions, source-code hash, input fingerprints, output locations, and completion status; `splits.json` records evaluation and calibration essay IDs and rubric definitions.

Result CSVs retain scores, justifications, available reasoning text, retrieved examples, token counts, timing, retries, and adjudication branches. Criterion results remain available individually and are aggregated for overall evaluation; every architecture uses the configured overall human target, including raw rater-1 `Overall_1` for ELLIPSE.

```bash
abaes-eval-fwk metrics study/benchmark.yaml path/to/run_results.csv --output analysis/model-a
abaes-eval-fwk compare study/benchmark.yaml path/to/baseline_results.csv \
  path/to/variant_results.csv --output analysis/comparison.csv
abaes-eval-fwk baseline study/benchmark.yaml path/to/model-a_results.csv \
  path/to/model-b_results.csv --output analysis/pool_results.csv
abaes-eval-fwk adjudication study/benchmark.yaml path/to/model-a_results.csv \
  path/to/model-b_results.csv --reviews path/to/*_reviews.csv --output analysis/adjudication
```

`adjudication` counts how disagreement- and review-based adjudication would resolve a pool:
DA outcomes from the pooled source scores, RA outcomes and the review's completion tokens per
essay from the pool's reviews CSVs.

NMAE is the primary metric: `10 × MAE / (score_max − score_min)`. Outputs also include MAE, RMSE, NRMSE on the same 0–10 reference scale, QWK, Pearson, and Spearman; dataset averages weight datasets equally. Paired two-sided Wilcoxon tests compare absolute scoring errors, require identical essay IDs and human targets by default, and report sample overlap, p-values, direction, and effect size; `--allow-partial` explicitly allows the common subset. Tests are reported per dataset without a multiple-comparison correction.

## Reuse and resume

A final-agent experiment references source CSV paths or earlier experiment names such as `@model_a-holistic`. Selecting that final experiment automatically schedules its declared dependencies first; `--dry-run` shows the complete plan before any requests.

The final model's independent assessment must come from a prior run over the same essays and with the same architecture. It is formed before the final model sees the candidate assessments, and its CSV model ID must match the final model.

Failed or partial runs exit unsuccessfully and retain checkpoints. Resume the specific run directory with the same configuration and overrides:

```bash
abaes-eval-fwk run study/benchmark.yaml --experiment rq1-hybrid \
  --resume study/runs/rq1-hybrid/<suite-id>
```

Resume checks the effective configuration, evaluated submissions, calibration examples, rubrics, reuse files, code, and dependency versions before reusing completed work. A new invocation without `--resume` starts a separate run.

Pool-average runs use the same resume checks and record the evaluated essay IDs in `splits.json`.

## Development and provenance

```bash
python -m pytest
python -m build
```

[Extensions](docs/extensions.md) covers custom datasets, splits, score policies, and plugins. [Packaging](docs/packaging.md) records what the wheel and sdist each contain, how versions and releases work, and where an installed reader finds the docs.

Student datasets, results, credentials, and downloaded model weights are not distributed in the package; dataset access remains subject to the original providers' terms.

## License

This framework is distributed under [CC BY-NC 4.0](LICENSE): copy, modify, and redistribute it for non-commercial purposes with attribution. Copyright is held solely by the author, who retains full commercial rights in the work; a commercial license is available on request.

Contributions are accepted only under terms that permit relicensing, so that dual licensing remains available. Runtime dependencies are MIT, BSD, or Apache-2.0 licensed and are installed rather than vendored, so they impose no further conditions on use of this code.
