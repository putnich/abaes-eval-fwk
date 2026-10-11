# Run lifecycle

What happens between `abaes-eval-fwk run` and a finished suite, and how every run records its inputs.

`run` keeps the parent process thin. Each experiment executes in its own Python process, so a
crash, a leaked thread, or global state in one experiment can't contaminate the next.

```mermaid
sequenceDiagram
    actor User
    participant CLI as cli.py (parent)
    participant Cfg as configuration.py
    participant W as worker process<br/>(python -m abaes_eval_fwk _worker)
    participant Eng as grading engine
    participant FS as output_dir

    User->>CLI: abaes-eval-fwk run benchmark.yaml --experiment X
    CLI->>Cfg: load_config() + --set overrides
    Cfg-->>CLI: BenchmarkConfig (validated)
    CLI->>Cfg: select_experiments(): order X after its @dependencies
    CLI->>FS: suites/<suite_id>.json (status: running)
    loop each selected experiment, in dependency order
        CLI->>FS: <name>/<suite_id>/request.json
        CLI->>W: spawn with request.json
        W->>FS: manifest.json (status: preparing, environment hashes)
        W->>Eng: build setups, verify provenance, grade
        Eng->>FS: results/*.csv, checkpoints/, conversations/, logs/
        W->>FS: manifest.json (complete · partial · failed)
        W-->>CLI: exit code
        CLI->>FS: append run to suite (stop the suite on failure)
    end
    CLI-->>User: Suite: runs/suites/<suite_id>.json
```

A dependency's manifest feeds the next experiment: `@name` references in `reuse` resolve to the
results file the dependency's manifest recorded, and a dependency that didn't reach `complete`
stops the suite.

## Provenance and resume

Every run records what it was given, so a result can be traced and a resume can't silently mix
inputs.

```mermaid
flowchart TB
    subgraph identity["run identity (checked on --resume)"]
        spec["experiment spec"]
        model["resolved model profile"]
        ret["resolved retrieval"]
        rt["runtime"]
        pr["prompt text"]
        smp["sampling"]
        ds["dataset snapshots<br/>sample + calibration IDs, rubric"]
        src["hashes of reused input files"]
        code["framework source SHA-256"]
        ext["extension source SHA-256"]
    end
    identity --> m["manifest.json"]
    m --> q{"--resume DIR:<br/>identity unchanged?"}
    q -- yes --> cont["continue from checkpoints/grader.sqlite;<br/>completed essays are not regraded"]
    q -- no --> rej["rejected: start a new run"]
```

The manifest also records the Python version, platform, installed package versions, artifact
paths, timestamps, and earlier attempts. Checkpoints are keyed by (rubric, essay, criterion), and
the checkpoint file is specific to the variant, model, and source experiments, so two runs can't
read each other's records.

---

[← Overview](README.md) · [Overview](README.md) · [Configuration resolution →](02-configuration-resolution.md)
