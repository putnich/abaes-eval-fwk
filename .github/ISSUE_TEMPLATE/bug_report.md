---
name: Bug Report
about: Report a bug or issue
title: "[BUG] "
labels: bug
assignees: ''
---

## Description
A clear and concise description of what went wrong during configuration, setup, or a run.

## Steps to Reproduce
1. Command(s) run:
```bash
abaes-eval-fwk validate study/benchmark.yaml --check-data
abaes-eval-fwk run study/benchmark.yaml --experiment rq1-hybrid
```
2. Config/template used: (e.g., `playground`, `study`, `simulation`, `demo`, `full`, or a custom `benchmark.yaml`)
3. Dataset(s) involved: (e.g., ASAP AES Set 1/4/7, ELLIPSE, or a custom CSV loader)
4. Observe '...'

## Expected Behavior
What you expected the framework to produce (e.g., a completed run directory with `results.csv`, a specific metric value, a successful `--dry-run` plan).

## Actual Behavior
What actually happened (e.g., crash, validation error, wrong/implausible score, malformed output, resume rejected).

## Environment
- Python version: (e.g., 3.11, 3.12, 3.13, 3.14)
- OS: (e.g., Ubuntu 22.04, macOS 13, Windows 11)
- Install method: (editable `-e '.[dev]'`, built wheel, extras used e.g. `[dense]`/`[deepseek]`)
- Model backend: (provider/model name per `models.yaml`, or local hosting setup)

## Relevant Output
Paste the failing config, run manifest, CSV rows, or console output/traceback below.

```
Paste error messages, logs, or CSV rows here
```

## Additional Context
Any other context that might be helpful (e.g., rubric/config changes, extension code in `extensions/`, `--set` overrides used).

## Checklist
- [ ] I've searched existing issues for duplicates
- [ ] I've included the full error message/traceback
- [ ] I can reproduce this consistently
- [ ] I've provided the config/template/command needed to reproduce
