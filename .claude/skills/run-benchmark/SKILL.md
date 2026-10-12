---
name: run-benchmark
description: Validate and run one or more named experiments against the grading framework (playground or a research study)
---

# Run Benchmark

Runs `abaes-eval-fwk validate`/`run` for one or more experiments declared in a `benchmark.yaml`, created by `abaes-eval-fwk init`.

## Usage

```
/run-benchmark
/run-benchmark --config playground/benchmark.yaml
/run-benchmark --config study/benchmark.yaml --experiment rq1-hybrid
```

## Steps

1. Ask which config to run if not given — default to `playground/benchmark.yaml`. If the directory doesn't exist yet, scaffold it first:
   ```bash
   abaes-eval-fwk init playground          # or: init study --template study
   ```
   There are only two templates: `demo`/`simulation` are aliases for `playground`, `full` for `study`. `init` refuses a non-empty directory — ask before deleting anything other than the gitignored `/playground/` or `/study/` scratch dirs.
2. Validate before running anything:
   ```bash
   abaes-eval-fwk validate <config> --check-data
   ```
3. Ask which experiment(s) to run, or run every experiment declared in the config. The CLI has no list command, so if the user doesn't know the names, load the config (from the project directory, framework installed) and summarize the output as a table:
   ```bash
   python3 -c "
   from pathlib import Path
   from abaes_eval_fwk.configuration import load_config, select_experiments

   for spec in select_experiments(load_config(Path('benchmark.yaml'))):
       control = f' controls={spec.control.baseline}/on={list(spec.control.on)}' if spec.control else ''
       print(f'{spec.name}: rq={spec.rq} model={spec.model} arch={spec.architecture} '
             f'retrieval={spec.retrieval} reasoning={spec.reasoning or \"default\"} '
             f'procedure={spec.procedure}{control}')
   "
   ```
   If the user named a keyword ("semantic", "criterion"), point at the matching names directly.
4. **Gotcha:** what costs money is `models.yaml`, not the template name. The `playground` template (and its `demo`/`simulation` aliases) ships `provider: demo` profiles that answer offline. The `study` template (and its `full` alias) points at real endpoints. If any profile the selected experiments use isn't `provider: demo`, show the dry-run plan and get explicit confirmation before the real run:
   ```bash
   abaes-eval-fwk run <config> --experiment NAME [--experiment NAME2 ...] --dry-run
   ```
5. Build and show the real command, including any `--set PATH=VALUE` overrides the user wants:
   ```bash
   abaes-eval-fwk run <config> --experiment rq1-hybrid \
     --set sampling.sample_size=20 --set runtime.max_workers=2
   ```
6. Run it and stream output.
7. Report where results landed:
   - Suite manifest: `<output_dir>/suites/<suite_id>.json` (`output_dir` is `runs/` by default — see `config/paths.yaml`)
   - Per-experiment run directory: `<output_dir>/<experiment_name>/<suite_id>/`, containing `manifest.json`, `request.json`, `results/`, `metrics/`, `conversations/`, `checkpoints/`, and `logs/run.log`
   - Run status from the manifest: `complete`, `partial` (some essays failed — they're listed in the manifest's `failures`), or `failed`. For anything other than `complete`, hand off to `/debug-run`.

## Key CLI flags

| Flag | Applies to | Notes |
|---|---|---|
| `--experiment NAME` | `validate`, `run` | Repeat to select several; omit to select every declared experiment (plus any `@name` reuse dependency, auto-ordered) |
| `--set PATH=VALUE` | all config subcommands | Override a dotted YAML setting; repeatable |
| `--check-data` | `validate` | Also loads and validates datasets/rubrics, not just config structure |
| `--dry-run` | `run` | Prints the ordered plan (rq/architecture/procedure/model/reasoning per experiment); makes no model calls |
| `--resume PATH` | `run` | Resumes one existing run directory with identical config/inputs — the selection must resolve to exactly one experiment matching that directory's prior manifest |

Run `abaes-eval-fwk <command> -h` for the authoritative flag list.
