# Contributing to abaes-eval-fwk

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this framework.

## Getting Started

1. **Fork and clone** the repository
2. **Create a branch** for your work: `git checkout -b feature/your-feature`
3. **Create a virtualenv and install dev dependencies**:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install -e '.[dev,dense]'
   ```
   (`dense` is needed by the integration tests and the playground's semantic and hybrid runs.)
4. **Install pre-commit hooks**: `pip install pre-commit mypy`, then
   `pre-commit install --install-hooks` (the local `mypy` and `pytest` hooks use the tools on your
   PATH)

## Development Workflow

### Before Committing

Pre-commit hooks run automatically on every commit and will:
- Lint and format code with Ruff (100-char line length)
- Scan for security issues with Bandit
- Type-check with mypy and run the unit tests
- Validate file formatting (trailing whitespace, YAML/JSON syntax, merge conflicts, private keys)

If a hook fails, review the changes/output and commit again.

### Code Style

- **Python 3.11–3.14** required (CI runs the unit tests on all four and the integration tests on
  3.12, the recommended local version)
- Ruff for linting and formatting (100-character line length)
- Bandit and mypy run via pre-commit
- No comments unless the WHY is non-obvious

### Testing

```bash
# Run the unit tests (the pre-commit hook runs these)
pytest

# Run a specific test
pytest tests/test_file.py::test_function

# Run the integration tests: CLI runs and the offline playground suite (several minutes; needs
# the dense extra; CI runs them for PRs into main, pushes to main, and releases)
pytest -m integration

# Smoke-test the CLI against the bundled playground (no model calls)
abaes-eval-fwk init playground
abaes-eval-fwk validate playground/benchmark.yaml --check-data
abaes-eval-fwk run playground/benchmark.yaml
```

### Local Development Commands

```bash
python -m pip install -e '.[dev,dense]'   # Install framework + dev and dense dependencies
ruff check .                        # Lint
ruff format --check .               # Check formatting (ruff format . to apply)
pytest                              # Unit tests
pytest -m integration               # Integration tests
python -m build                     # Build sdist + wheel
pre-commit run --all-files          # Run every hook (ruff, bandit, mypy, unit tests, ...)
```

### Git Workflow

1. **Create a branch** from `main`: `git checkout -b feature/your-feature`
2. **Make commits** with clear messages (conventional commits preferred)
3. **Push to your fork**: `git push origin feature/your-feature`
4. **Open a PR** against `main`

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):
```
type(scope): short description

Optional longer explanation if needed.
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat(retrieval): add hybrid lexical/semantic selector`
- `fix(metrics): correct QWK normalization edge case`
- `docs(readme): update install instructions`
- `test(cli): add coverage for compare subcommand`

## Pull Request Process

1. **Ensure all checks pass**:
   - Pre-commit hooks (run locally)
   - GitHub Actions CI (lint, format, unit tests, and build on Python 3.11–3.14; integration tests and wheel verification on 3.12)
   - Code review approval

2. **Write a clear PR description**:
   - What does this PR do?
   - Why is this change needed?
   - How was it tested?

3. **Link related issues**: `Fixes #123`

4. **Keep PRs focused**: one feature or fix per PR

## Code Review

- Be respectful and constructive
- Ask questions if unclear
- Suggest improvements, don't demand
- Acknowledge good work

## Reporting Issues

Use GitHub Issues for bug reports and feature requests (see the [issue templates](ISSUE_TEMPLATE/)):

**For bugs:**
- Describe what you expected vs. what happened
- Include the config/template, dataset(s), and command used
- Provide Python version and OS
- Attach error logs/tracebacks if helpful

**For features:**
- Explain the use case (which RQ, dataset, or metric comparison it affects)
- Describe your proposed solution
- Discuss alternatives if relevant

## Questions?

- Check existing issues and PRs
- Read the README and [docs/](../docs/)
- Start a discussion for major changes

Thank you for contributing!
