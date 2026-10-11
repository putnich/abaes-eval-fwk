#!/usr/bin/env python3
"""Clean-room reader gate: install from a wheel into a fresh venv and exercise the reader path.

Each step runs in a temporary directory outside the checkout with no PYTHONPATH, so a pass
proves the installed package is self-sufficient. Exits non-zero at the first failed step.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import venv
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class GateFailure(RuntimeError):
    """A gate step did not hold; the message says which reader capability is missing."""


@dataclass
class Step:
    """One gate step: a name, what it proves, and the output of the command that ran it."""

    name: str
    proves: str
    command: list[str] = field(default_factory=list)
    returncode: int | None = None
    output: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Runner:
    """Run gate steps, record each one's real output, and stop at the first failure."""

    def __init__(self, cwd: Path, log: Path, keep_going: bool) -> None:
        self.cwd = cwd
        self.log = log
        self.keep_going = keep_going
        self.steps: list[Step] = []
        self.failures: list[Step] = []

    def run(
        self,
        name: str,
        proves: str,
        command: list[str],
        *,
        env: dict | None = None,
        cwd: Path | None = None,
    ) -> Step:
        """Run one command, capture combined output, and raise unless it succeeded."""
        step = Step(name=name, proves=proves, command=command)
        step.output = self._announce(step)
        merged = {**os.environ, **(env or {})}
        merged.pop("PYTHONPATH", None)
        started = time.monotonic()
        result = subprocess.run(
            command,
            cwd=cwd or self.cwd,
            env=merged,
            capture_output=True,
            text=True,
            check=False,
        )
        step.returncode = result.returncode
        step.output += (
            f"  cwd: {cwd or self.cwd}\n  command: {' '.join(command)}\n"
            f"  exit: {result.returncode}  ({time.monotonic() - started:.1f}s)\n"
            f"--- output ---\n{result.stdout}{result.stderr}"
        )
        self.steps.append(step)
        self._echo(step)
        if not step.ok:
            self.failures.append(step)
            if not self.keep_going:
                raise GateFailure(f"step '{name}' failed:\n{step.output}")
        return step

    def check(self, name: str, proves: str, condition: bool, detail: str) -> Step:
        """Record a pass/fail assertion that needs no subprocess."""
        step = Step(
            name=name, proves=proves, command=["<assertion>"], returncode=0 if condition else 1
        )
        step.output = (
            self._announce(step)
            + f"  assertion: {detail}\n  result: {'PASS' if condition else 'FAIL'}\n"
        )
        self.steps.append(step)
        self._echo(step)
        if not condition:
            self.failures.append(step)
            if not self.keep_going:
                raise GateFailure(f"step '{name}' failed: {detail}")
        return step

    def _announce(self, step: Step) -> str:
        return f"\n=== {step.name} ===\n  proves: {step.proves}\n"

    def _echo(self, step: Step) -> None:
        marker = "PASS" if step.ok else "FAIL"
        line = f"[{marker}] {step.name} — {step.proves}\n"
        sys.stdout.write(line)
        sys.stdout.flush()
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(f"[{marker}] {step.name}\n{step.output}\n")

    def report(self) -> int:
        """Print a pass/fail summary and return the process exit code."""
        passed = sum(1 for step in self.steps if step.ok)
        sys.stdout.write(f"\n{'=' * 72}\n")
        sys.stdout.write(f"clean-clone gate: {passed}/{len(self.steps)} steps passed\n")
        for step in self.steps:
            sys.stdout.write(f"  {'PASS' if step.ok else 'FAIL'}  {step.name}\n")
        if self.failures:
            sys.stdout.write(f"\nFAILED ({len(self.failures)}):\n")
            for step in self.failures:
                sys.stdout.write(f"  - {step.name}: {step.proves}\n")
            sys.stdout.write(f"\nFull transcript: {self.log}\n")
            return 1
        sys.stdout.write(f"\nAll steps passed. Full transcript: {self.log}\n")
        return 0


# Directories a reader never receives: local environments, caches, and generated run output.
_NOT_DISTRIBUTED = (
    ".venv",
    "__pycache__",
    "*.egg-info",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "runs",
)


def _in_git_tree() -> bool:
    """Whether this checkout sits inside a git work tree, which may be rooted above it."""
    if shutil.which("git") is None:
        return False
    probe = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0 and probe.stdout.strip() == "true"


def export_checkout(destination: Path, *, from_head: bool = False) -> list[str]:
    """Copy the framework to `destination` and return the paths present in the git index.

    The working tree is copied by default so the gate tests the code as it stands, including
    uncommitted work. `from_head` exports the committed tree instead, which is the stricter
    question: would a reader who clones today get a package that works?
    """
    if from_head and _in_git_tree():
        archive = subprocess.run(
            ["git", "-C", str(REPO), "archive", "--format=tar", "HEAD", "."],
            capture_output=True,
            check=True,
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(destination, filter="data")
    else:
        shutil.copytree(
            REPO,
            destination,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*_NOT_DISTRIBUTED),
        )
    if not _in_git_tree():
        return []
    listing = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [name for name in listing.split("\0") if name]


def build_wheel(runner: Runner, checkout: Path, outdir: Path) -> Path:
    """Build a wheel with the `build` package and return its path."""
    outdir.mkdir(parents=True, exist_ok=True)
    runner.run(
        "build wheel",
        "the published artifact builds from a clean checkout with the standard PEP 517 frontend",
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(outdir), str(checkout)],
        env={"PIP_DISABLE_PIP_VERSION_CHECK": "1"},
    )
    wheels = sorted(outdir.glob("*.whl"))
    if not wheels:
        raise GateFailure(f"no wheel produced in {outdir}")
    return wheels[-1]


def make_venv(path: Path) -> Path:
    """Create a fresh virtualenv and return its interpreter."""
    venv.EnvBuilder(with_pip=True, clear=True, symlinks=os.name != "nt").create(path)
    binary = "Scripts" if os.name == "nt" else "bin"
    interpreter = path / binary / ("python.exe" if os.name == "nt" else "python")
    if not interpreter.exists():
        raise GateFailure(f"fresh virtualenv has no interpreter at {interpreter}")
    return interpreter


def reader_selector_source() -> str:
    """A reader-authored retrieval selector: the calibration examples closest in length to the essay."""
    return '''"""A design dimension a paper reader adds: examples ranked by essay-length similarity."""

from __future__ import annotations


def by_length(*, rubric, calibration_pool, student_text, criterion):
    """Return the calibration examples whose length is closest to the submission's, nearest first."""
    target = len(student_text.split())
    return sorted(
        calibration_pool,
        key=lambda item: (abs(len(item.submission.text.split()) - target), item.essay_id),
    )[:3]
'''


def standalone_probe_source() -> str:
    """A reader script that calls the selector with the framework's keyword contract, nothing installed."""
    return '''"""Call the reader's selector the way the framework does, with no framework import."""

from types import SimpleNamespace

from length_selectors import by_length


def item(essay_id, text):
    return SimpleNamespace(essay_id=essay_id, submission=SimpleNamespace(text=text))


pool = [
    item("short", "Too short."),
    item("near", "One two three four five six seven."),
    item("long", " ".join(["word"] * 40)),
]
criterion = SimpleNamespace(name="holistic")
ranked = [
    example.essay_id
    for example in by_length(
        rubric=None,
        calibration_pool=pool,
        student_text="One two three four five six seven eight.",
        criterion=criterion,
    )
]
assert ranked and ranked[0] == "near", f"expected the length-closest first, got {ranked}"
print("reader selector ranked correctly:", ranked)
'''


# Directory and file names under src/ that a clone legitimately does not receive.
_NOT_SOURCE_DIRS = ("__pycache__",)
_NOT_SOURCE_GLOBS = ("*.egg-info", "*.pyc", ".DS_Store")


def _is_generated(path: Path) -> bool:
    """Whether a path is a cache or build artifact rather than source a clone should receive."""
    return any(part in _NOT_SOURCE_DIRS for part in path.parts) or any(
        fnmatch(part, pattern) for part in path.parts for pattern in _NOT_SOURCE_GLOBS
    )


def _source_files_missing_from_head() -> list[str]:
    """Source files present on disk that a clone of HEAD would not deliver.

    Compared against the tracked set directly rather than via `ls-files --others`, so a file
    hidden by a .gitignore rule is caught alongside a merely untracked one.
    """
    if not _in_git_tree():
        return []
    tracked = set(_tracked_files("src"))
    on_disk = {
        str(path.relative_to(REPO))
        for path in REPO.joinpath("src").rglob("*")
        if path.is_file() and not _is_generated(path)
    }
    return sorted(on_disk - tracked)


def _tracked_files(pathspec: str) -> list[str]:
    """Files git tracks under `pathspec`, as paths relative to the framework root.

    `git -C REPO ls-files` reports paths relative to REPO, which is the framework directory.
    """
    listing = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z", "--", pathspec],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [name for name in listing.split("\0") if name]


def _completed_manifests(runs: Path) -> int:
    """How many run manifests under `runs` record a completed experiment."""
    if not runs.exists():
        return 0
    return sum(
        1
        for path in runs.rglob("manifest.json")
        if json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Keep the clean-room tree at this path instead of a temporary directory",
    )
    parser.add_argument(
        "--log", type=Path, help="Write the full transcript here instead of <workdir>/gate.log"
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Run every step and report all failures instead of stopping at the first",
    )
    parser.add_argument(
        "--from-head",
        action="store_true",
        help=(
            "Export the committed tree instead of the working tree, which additionally asserts "
            "that no file under src/ is missing from git"
        ),
    )
    parser.add_argument(
        "--reuse-venv",
        type=Path,
        help="Install into this existing venv instead of creating a fresh one (faster reruns)",
    )
    args = parser.parse_args(argv)

    root = args.workdir
    temporary = root is None
    if temporary:
        root = Path(tempfile.mkdtemp(prefix="mas-clean-clone-")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    log = args.log or root / "gate.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"clean-clone gate\nrepo: {REPO}\nworkdir: {root}\n", encoding="utf-8")

    runner = Runner(root, log, args.keep_going)
    try:
        checkout = root / "checkout"
        if checkout.exists():
            shutil.rmtree(checkout)
        checkout.mkdir()
        export_checkout(checkout, from_head=args.from_head)
        runner.check(
            "checkout exported",
            "the gate starts from the framework copied outside the source tree",
            (checkout / "pyproject.toml").is_file() and not (checkout / ".venv").exists(),
            f"pyproject.toml present={(checkout / 'pyproject.toml').is_file()}, "
            f"source .venv leaked={(checkout / '.venv').exists()}",
        )
        if args.from_head:
            missing = _source_files_missing_from_head()
            runner.check(
                "committed tree is self-sufficient",
                "no source file the package imports is missing from git",
                not missing,
                f"source files on disk that a clone of HEAD would not receive: {missing}",
            )

        dist = root / "dist"
        if dist.exists():
            shutil.rmtree(dist)
        wheel = build_wheel(runner, checkout, dist)

        if args.reuse_venv is not None:
            interpreter = (
                args.reuse_venv
                / ("Scripts" if os.name == "nt" else "bin")
                / ("python.exe" if os.name == "nt" else "python")
            )
            venv_dir = args.reuse_venv
        else:
            venv_dir = root / "venv"
            if venv_dir.exists():
                shutil.rmtree(venv_dir)
            interpreter = make_venv(venv_dir)

        install = ["--quiet"]
        if args.reuse_venv is not None:
            # A reused venv may already hold this exact version, which pip would treat as a no-op.
            install += ["--force-reinstall", "--no-deps"]
        runner.run(
            "install wheel",
            "a reader installs the package with one command and gets a working CLI",
            [str(interpreter), "-m", "pip", "install", *install, str(wheel)],
            env={"PIP_DISABLE_PIP_VERSION_CHECK": "1"},
        )
        binary_dir = venv_dir / ("Scripts" if os.name == "nt" else "bin")
        suffix = ".exe" if os.name == "nt" else ""
        cli = binary_dir / f"abaes-eval-fwk{suffix}"
        runner.check(
            "console script installed",
            "the published entry point `abaes-eval-fwk` exists in the fresh venv",
            cli.is_file(),
            f"expected {cli}",
        )

        reader = root / "reader"
        if reader.exists():
            shutil.rmtree(reader)
        reader.mkdir()
        origin = runner.run(
            "import comes from the venv",
            "no PYTHONPATH or source-tree borrowing is needed to import the package",
            [
                str(interpreter),
                "-c",
                "import abaes_eval_fwk, pathlib, sys; "
                "p = pathlib.Path(abaes_eval_fwk.__file__).resolve(); "
                "print(p); "
                "sys.exit(0 if p.is_relative_to(pathlib.Path(sys.prefix).resolve()) else 1)",
            ],
        )
        runner.check(
            "import path is outside the checkout",
            "the imported package is the installed wheel, not the source tree",
            str(venv_dir.resolve()) in origin.output,
            origin.output.strip().splitlines()[-1] if origin.output.strip() else "no import output",
        )

        runner.run(
            "cli init",
            "a reader creates a runnable benchmark project with one command",
            [str(cli), "init", "project"],
            cwd=reader,
        )
        project = reader / "project"
        runner.check(
            "init produced a project",
            "`init project` writes a benchmark.yaml, a dataset, and a rubric",
            (project / "benchmark.yaml").is_file() and (project / "demo.csv").is_file(),
            f"benchmark.yaml={(project / 'benchmark.yaml').is_file()}, "
            f"demo.csv={(project / 'demo.csv').is_file()}",
        )

        runner.run(
            "cli validate",
            "a reader checks the configuration and local data without contacting a model",
            [str(cli), "validate", "project/benchmark.yaml", "--check-data"],
            cwd=reader,
        )

        runner.run(
            "cli run",
            "a reader runs an experiment end to end offline, against the bundled demo model",
            [str(cli), "run", "project/benchmark.yaml", "--experiment", "rq1-random"],
            cwd=reader,
        )
        complete = _completed_manifests(project / "runs")
        results = list((project / "runs").rglob("*_run_results.csv"))
        runner.check(
            "run produced scores",
            "the run writes a results CSV and a manifest whose status is complete",
            bool(results) and complete > 0,
            f"results CSVs={len(results)}, complete manifests={complete}",
        )

        # ── The reader adds a design dimension of their own ────────────────────────────
        # The selector sits beside the configuration in extensions/, the same place a reader
        # would put it, and is named as `module:function` in an experiment.
        extensions = project / "extensions"
        (extensions / "length_selectors.py").write_text(reader_selector_source(), encoding="utf-8")
        probe = reader / "probe.py"
        probe.write_text(standalone_probe_source(), encoding="utf-8")

        runner.run(
            "extension file runs standalone",
            "a reader's own file imports nothing from the framework and executes with nothing installed",
            [str(interpreter), str(probe)],
            env={"PYTHONPATH": str(extensions)},
        )

        with (project / "config" / "experiments.yaml").open("a", encoding="utf-8") as handle:
            handle.write(
                "- name: reader-length\n  rq: RQ1\n  model: a\n  architecture: holistic\n"
                "  retrieval: extensions.length_selectors:by_length\n"
            )

        # No PYTHONPATH here on purpose: the framework must find a reader's module beside their
        # own configuration, which is the claim the reader path makes.
        runner.run(
            "extension named in a config validates",
            "a reader names their own selector in a run configuration and the CLI accepts it",
            [str(cli), "validate", "project/benchmark.yaml"],
            cwd=reader,
        )

        runner.run(
            "extension runs in a real experiment",
            "a reader's own design dimension is used by a real offline experiment",
            [str(cli), "run", "project/benchmark.yaml", "--experiment", "reader-length"],
            cwd=reader,
        )

        return runner.report()
    except GateFailure as exc:
        sys.stdout.write(f"\nGATE FAILED\n{exc}\n")
        return 1
    finally:
        if temporary and not args.keep_going:
            sys.stdout.write(f"\nClean-room tree kept for inspection: {root}\n")


if __name__ == "__main__":
    raise SystemExit(main())
