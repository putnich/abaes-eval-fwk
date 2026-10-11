"""Pin the reader-facing packaging surface: the demo, the extension example, and the docs.

The static tests assert what setuptools is told to collect, which is the configuration a wheel
build actually reads. `test_wheel_contains_reader_surface` builds a real wheel to confirm that
configuration still produces those files, but only when MAS_TEST_BUILD_WHEEL=1 is set.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

DEMO_FILES = ("demo.csv", "demo-rubric.yaml", "benchmark.yaml")


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def _source_files(directory: str) -> set[str]:
    """Relative POSIX paths of the tracked files under a top-level directory, sans caches."""
    found = set()
    for path in (ROOT / directory).rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            found.add(path.relative_to(ROOT / directory).as_posix())
    return found


def test_package_dir_grafts_docs():
    """docs/ lives outside src/, so it is grafted under the installed package."""
    package_dir = _pyproject()["tool"]["setuptools"]["package-dir"]
    assert package_dir["abaes_eval_fwk.reader_docs"] == "docs"


def test_reader_docs_ship_is_an_allowlist_of_documents_that_exist():
    """Reader docs are named one by one, so a typo fails here rather than silently shipping nothing."""
    patterns = _pyproject()["tool"]["setuptools"]["package-data"]["abaes_eval_fwk.reader_docs"]
    top_level = {name for name in _source_files("docs") if "/" not in name}
    assert set(patterns) - top_level == set()


def test_the_build_audit_trail_is_not_shipped_to_readers():
    """Review and workflow-builder records are the build's history, not reader documentation."""
    patterns = _pyproject()["tool"]["setuptools"]["package-data"]["abaes_eval_fwk.reader_docs"]
    assert not [p for p in patterns if set(p) & set("*?[]")], (
        "a wildcard would ship every process record added to docs/ later"
    )
    shipped = {name for name in _source_files("docs") if "/" not in name and name in patterns}
    process = {
        name
        for name in _source_files("docs")
        if "/" not in name and (name.startswith("wp0-") or "workflow-builder" in name)
    }
    assert shipped & process == set()


def test_architecture_chapters_ship_as_an_allowlist_of_documents_that_exist():
    """The architecture chapters are their own grafted package, named one by one like the top level."""
    setuptools = _pyproject()["tool"]["setuptools"]
    assert (
        setuptools["package-dir"]["abaes_eval_fwk.reader_docs.architecture"] == "docs/architecture"
    )
    patterns = setuptools["package-data"]["abaes_eval_fwk.reader_docs.architecture"]
    assert not [p for p in patterns if set(p) & set("*?[]")]
    assert set(patterns) == _source_files("docs/architecture")


def test_manifest_and_package_data_agree_on_which_docs_ship():
    """`include-package-data` defaults true, so MANIFEST.in reaches the wheel whether or not package-data does."""
    package_data = _pyproject()["tool"]["setuptools"]["package-data"]
    from_package_data = {f"docs/{name}" for name in package_data["abaes_eval_fwk.reader_docs"]} | {
        f"docs/architecture/{name}"
        for name in package_data["abaes_eval_fwk.reader_docs.architecture"]
    }
    from_manifest = {
        line.split()[-1]
        for line in (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
        if line.startswith("include docs/")
    }
    assert from_manifest == from_package_data
    assert not [
        line
        for line in (ROOT / "MANIFEST.in").read_text().splitlines()
        if line.startswith(("recursive-include docs", "graft docs"))
    ], "a recursive docs rule would re-add every process record regardless of package-data"


def test_declared_packages_match_the_source_tree():
    """The explicit package list replaces `find`, so it must track every code directory under src/.

    `abaes_eval_fwk/data` holds private local datasets. They are deliberately not distributed, so
    that tree is not a package and is excluded from enumeration rather than declared.
    """
    private_data = ROOT / "src" / "abaes_eval_fwk" / "data"
    declared = set(_pyproject()["tool"]["setuptools"]["packages"])
    on_disk = {
        path.relative_to(ROOT / "src").as_posix().replace("/", ".")
        for path in (ROOT / "src" / "abaes_eval_fwk").rglob("*")
        if path.is_dir()
        and "__pycache__" not in path.parts
        and not path.is_relative_to(private_data)
    }
    on_disk.add("abaes_eval_fwk")
    assert on_disk - declared == set()


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("MAS_TEST_BUILD_WHEEL") != "1",
    reason="set MAS_TEST_BUILD_WHEEL=1 to build a wheel and check its contents",
)
def test_wheel_contains_reader_surface(tmp_path):
    """Confirm the declared configuration yields the reader files in a real wheel archive."""
    source = tmp_path / "src-tree"
    shutil.copytree(
        ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".venv",
            ".git",
            "build",
            "dist",
            "__pycache__",
            "*.egg-info",
            ".pytest_cache",
            ".ruff_cache",
            "demo/runs",
            "runs",
        ),
    )
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(tmp_path / "out"),
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    wheels = list((tmp_path / "out").glob("*.whl"))
    assert len(wheels) == 1, wheels
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
    for name in ("configuration.md", "models.md", "datasets.md", "architecture/README.md"):
        assert f"abaes_eval_fwk/reader_docs/{name}" in names, name
    assert not [name for name in names if "__pycache__" in name or name.endswith(".pyc")]
