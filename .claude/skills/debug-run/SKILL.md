---
name: debug-run
description: Diagnose a failed or partial experiment run from its suite/run manifests and logs, and resume it safely
---

# Debug Run

Every `run` writes a suite manifest plus one directory per experiment. This skill reads them to explain what happened and whether `--resume` is possible.

## Usage

```
/debug-run
/debug-run runs/suites/20261011T120000Z_ab12cd34.json
/debug-run runs/rq1-hybrid/20261011T120000Z_ab12cd34
```

## Layout

```
<output_dir>/                         # runs/ by default (config/paths.yaml)
├── suites/<suite_id>.json            # status, error, and one entry per experiment with its manifest path
└── <experiment>/<suite_id>/
    ├── request.json                  # exact config + spec handed to the worker process
    ├── manifest.json                 # status, timestamps, error, environment, resolved_reuse, results, failures
    ├── logs/run.log
    ├── results/                      # results CSV(s); reviews for review-based procedures
    ├── metrics/                      # unless paths.subdirs.metrics overrides it
    ├── conversations/                # per-essay transcripts when runtime.save_conversations
    └── checkpoints/grader.sqlite     # what --resume continues from
```

## Steps

1. If given nothing, list the newest files in `<output_dir>/suites/` and ask which suite. A suite's `status` is `running`, `complete`, or `failed`; `error` names the experiment that stopped it.
2. Open the failing experiment's `manifest.json`. `status` is `preparing`, `running`, `complete`, `partial` (graded, but `failures` lists essays that didn't), or `failed` (see `error`). `previous_attempts` shows earlier resumes.
3. Read the tail of `logs/run.log` for the traceback or the last provider error. Common causes:
   - Connection/timeout errors → the endpoint in `models.yaml` is down or `timeout` is too low.
   - `context_overflow` errors → lower `retrieval.max_examples` or keep `runtime.context_overflow: truncate_examples`.
   - `Unknown experiment reference` / `has no complete results artifact` → a `@name` reuse dependency didn't finish. Fix and run the dependency first.
   - `Reuse file not found` → a reuse path is relative to `benchmark.yaml`, not the CWD.
4. For a specific essay, read `conversations/` for that essay ID to see exactly what the model was sent and returned.
5. Resume when the cause was transient and nothing in the config changed:
   ```bash
   abaes-eval-fwk run <config> --experiment <same name> --resume <run_dir>
   ```
   `--resume` reuses the run directory and checkpoint. It is rejected if the run's identity changed: spec, model profile, retrieval, runtime, prompt text, sampling, dataset snapshots, reuse input files, framework source, or extension source. If it's rejected, explain which input differs and start a fresh run rather than forcing it.
6. Summarize: what failed, why, and either the resume command or what to change first.
