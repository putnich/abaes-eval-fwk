# Packaging and releases

What the wheel and source distribution contain, how versions are set, and how a release is
published.

## Install options

| From | Command | Use when |
|---|---|---|
| A checkout | `pip install -e '.[dev,dense]'` | Developing the framework |
| A GitHub Release | `pip install https://github.com/<owner>/<repo>/releases/download/vX.Y.Z/abaes_eval_fwk-X.Y.Z-py3-none-any.whl` | Using a fixed version |
| A branch or commit | `pip install 'git+https://github.com/<owner>/<repo>.git@main'` | Trying unreleased changes |
| A local build | `python -m build`, then `pip install dist/abaes_eval_fwk-*.whl` | Testing what a release would contain |

Optional extras: `dense` (sentence-transformers, for `semantic`/`hybrid` retrieval), `deepseek`
(the DeepSeek client), and `dev` (pytest, build, ruff). Python 3.11–3.14 is supported.

## What the wheel contains

- The `abaes_eval_fwk` package and the `abaes-eval-fwk` command.
- Packaged defaults and the `playground` and `study` templates (`config/config_files/`), so `init`
  works without a checkout.
- The synthetic playground dataset and rubric, and the six study rubrics (`resources/`).
- The playground's example `extensions/`.
- The `docs/` guides, under `abaes_eval_fwk/reader_docs/` (architecture chapters in
  `architecture/`). Locate them with
  `python -c "import abaes_eval_fwk, pathlib; print(pathlib.Path(abaes_eval_fwk.__file__).parent / 'reader_docs')"`.

It does not contain student datasets, results, credentials, model weights, or tests.

## What the source distribution adds

The sdist also carries `tests/`, `LICENSE`, `README.md`, and `pyproject.toml`, so the test suite
can run from it.

## Versions

The version comes from git tags via setuptools-scm; there is no version number to edit by hand.

| Checkout state | Built version |
|---|---|
| Exactly at tag `v0.2.0`, clean tree | `0.2.0` |
| Commits after a tag, or uncommitted changes | e.g. `0.2.1.dev3+g1a2b3c4` |
| No tags or no git history | falls back to `0.0.0`-based dev versions |

The version is written to `src/abaes_eval_fwk/_version.py` (gitignored) when the package is
built or installed, and is reported by `abaes-eval-fwk --version`. If importing the package fails
with `No module named 'abaes_eval_fwk._version'`, run `pip install -e .` again.

## Releasing

Releases are published as GitHub Releases, not to PyPI.

1. On a clean, up-to-date `main` where CI passes, pick the next semantic version.
2. Tag and check that the version resolves:
   ```bash
   git tag v0.2.0
   python -m build
   ls dist/          # abaes_eval_fwk-0.2.0-py3-none-any.whl and .tar.gz
   ```
3. Push the tag:
   ```bash
   git push origin v0.2.0
   ```
4. The CD workflow (`.github/workflows/cd.yml`) runs the unit and integration tests, then builds
   the sdist and wheel and attaches both to a GitHub Release with generated notes. If a test
   fails, nothing is published. It can also be run by hand from the Actions tab for an existing
   tag.

The workflow refuses to overwrite an existing release. To redo a broken one, delete it first.
Never move a published tag; release the next patch version instead.

## Continuous integration

Tests come in two tiers. Unit tests run with a plain `pytest` and in the pre-commit hook. Tests
that run the framework (CLI runs and the offline playground suite) are marked `integration`, are
excluded by default, and run with `pytest -m integration`.

`.github/workflows/ci.yml` runs on every push and pull request, on Python 3.11, 3.12, 3.13, and
3.14: `ruff check`, `ruff format --check`, the unit tests, and `python -m build`. For pull
requests into `main`, pushes to `main`, and manual runs (`workflow_dispatch`), it also installs
the `dense` extra (CPU-only torch), runs the integration tests, and installs the built wheel
outside the checkout to run `init` and the playground benchmark. `.github/workflows/cd.yml`
installs the same extras and runs both tiers before it builds a release.
