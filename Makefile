PYTHON ?= python
PIP ?= pip

.PHONY: help install precommit lint format format-check typecheck test test-integration coverage build clean playground study

help:
	@echo "install        Install the framework + dev dependencies"
	@echo "precommit      Install pre-commit hooks"
	@echo "lint           Run ruff check"
	@echo "format         Apply ruff format"
	@echo "format-check   Check ruff format without writing"
	@echo "typecheck      Run mypy (requires: pip install mypy)"
	@echo "test           Run unit tests"
	@echo "test-integration Run integration tests (they run the framework)"
	@echo "coverage       Run pytest under coverage"
	@echo "build          Build sdist + wheel"
	@echo "clean          Remove build/test artifacts"
	@echo "playground     Init, validate, and run the bundled demo playground (no model calls)"
	@echo "study          Scaffold a research study project (edit its config before running)"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -e '.[dev]'

precommit:
	pre-commit install --install-hooks

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy src/ --ignore-missing-imports --explicit-package-bases

test:
	$(PYTHON) -m pytest

test-integration:
	$(PYTHON) -m pytest -m integration

coverage:
	$(PIP) install coverage
	coverage run -m pytest
	coverage report -m

build:
	$(PYTHON) -m build

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache

playground:
	rm -rf playground
	abaes-eval-fwk init playground
	abaes-eval-fwk validate playground/benchmark.yaml --check-data
	abaes-eval-fwk run playground/benchmark.yaml

study:
	rm -rf study
	abaes-eval-fwk init study --template study
	@echo "Edit study/config/models.yaml and study/config/datasets.yaml, then:"
	@echo "  abaes-eval-fwk validate study/benchmark.yaml --check-data"
	@echo "  abaes-eval-fwk run study/benchmark.yaml --experiment <name>"
