"""Cover the clean-clone reader gate: the script's own contract by default, the full clean-room
run only when `MAS_CLEAN_CLONE_GATE=1`, because building and installing a wheel takes minutes."""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
DRIVER = SCRIPTS / "verify_clean_clone.py"
WRAPPER = SCRIPTS / "verify_clean_clone.sh"
SLOW = os.environ.get("MAS_CLEAN_CLONE_GATE") == "1"


def _source() -> str:
    return DRIVER.read_text(encoding="utf-8")


def _driver():
    """Import the gate driver by path, so the test does not depend on sys.path or installation."""
    spec = importlib.util.spec_from_file_location("verify_clean_clone_driver", DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_gate_has_a_runnable_entry_point() -> None:
    """A reader or CI job must be able to launch the gate unattended."""
    assert DRIVER.is_file() and WRAPPER.is_file()
    assert WRAPPER.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")
    for path in (DRIVER, WRAPPER):
        assert os.access(path, os.X_OK), f"{path} is not executable"


def test_the_gate_runs_unattended_and_is_re_runnable() -> None:
    """The gate must take no input, prompt for nothing, and survive being run twice."""
    source = _source()
    tree = ast.parse(source)
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    for name in ("main", "export_checkout", "build_wheel", "make_venv"):
        assert name in functions, f"the gate no longer defines {name}"
    # Nothing may block on input, or the gate hangs instead of failing.
    for forbidden in ("input(", "getpass", "sys.stdin.read"):
        assert forbidden not in source, f"the gate would block on {forbidden}"
    # Reruns clear the directories they reuse, so a second run cannot pass on stale artifacts.
    for reused in ("checkout", "dist", "reader"):
        assert f"shutil.rmtree({reused}" in source or f"rmtree({reused}" in source, (
            f"the gate does not clear {reused}, so a rerun could reuse stale state"
        )


def test_the_gate_never_borrows_the_source_tree() -> None:
    """A pass must mean the installed wheel is self-sufficient, not that PYTHONPATH helped."""
    source = _source()
    # Every subprocess drops any inherited PYTHONPATH before running a gate step.
    assert 'merged.pop("PYTHONPATH", None)' in source
    runner = ast.get_source_segment(
        source,
        next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ClassDef) and node.name == "Runner"
        ),
    )
    assert '"PYTHONPATH"' in runner


def test_the_gate_covers_the_whole_reader_journey() -> None:
    """Each capability the gate exists to prove must be a named step."""
    source = _source()
    for step in (
        "build wheel",
        "install wheel",
        "console script installed",
        "cli init",
        "cli validate",
        "cli run",
        "extension file runs standalone",
        "extension named in a config validates",
        "extension runs in a real experiment",
    ):
        assert f'"{step}"' in source, f"the gate lost its '{step}' step"


def test_the_gate_uses_a_keyless_offline_model() -> None:
    """The gate must never need an API key, a GPU, or a network model call."""
    packaged = (
        Path(__file__).resolve().parents[1]
        / "src/abaes_eval_fwk/config/config_files/playground/models.yaml"
    ).read_text(encoding="utf-8")
    assert "adapter_factory: abaes_eval_fwk.playground:create_model" in packaged
    for forbidden in ("OPENAI_API_KEY", "localhost:11434", "base_url:"):
        assert forbidden not in _source(), f"the gate would require {forbidden}"
        assert forbidden not in packaged, f"the offline project would require {forbidden}"


def test_reusing_a_venv_cannot_pass_on_a_stale_install() -> None:
    """pip treats a same-version wheel as a no-op, so a reused venv must be forced to reinstall."""
    source = _source()
    assert '"--force-reinstall"' in source, (
        "installing into a reused venv can silently skip, letting a stale package pass the gate"
    )
    tree = ast.parse(source)
    assert "args.reuse_venv is not None" in source, (
        "the reinstall is not conditioned on --reuse-venv"
    )
    assert any(isinstance(node, ast.Name) and node.id == "install" for node in ast.walk(tree))


def test_the_missing_from_head_check_sees_gitignored_source_files() -> None:
    """The check must compare src/ against the tracked set, not just list untracked files."""
    source = _source()
    # `ls-files --others` alone hides any file matched by a .gitignore rule.
    commands = "\n".join(
        ast.unparse(node) for node in ast.walk(ast.parse(source)) if isinstance(node, ast.List)
    )
    assert "--others" not in commands, (
        "the self-sufficiency check relies on --others, which skips gitignored source files"
    )
    driver = _driver()
    assert hasattr(driver, "_source_files_missing_from_head")
    assert hasattr(driver, "_tracked_files")


def test_the_extension_the_gate_writes_imports_nothing_from_the_framework() -> None:
    """A reader's selector is a plain module:function, so it must run without importing the package."""
    driver = _driver()
    for source in (driver.reader_selector_source(), driver.standalone_probe_source()):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("abaes_eval_fwk"), (
                    f"the generated reader code imports {node.module}"
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("abaes_eval_fwk"), (
                        f"the generated reader code imports {alias.name}"
                    )


def test_the_gate_drives_the_packaged_project_not_a_checkout_tree() -> None:
    """Projects come from `init`; the gate must not expect a demo/ directory in the checkout."""
    source = _source()
    assert '"project"' in source
    assert 'checkout / "demo"' not in source and "project/benchmark.yaml" in source


@pytest.mark.skipif(not SLOW, reason="set MAS_CLEAN_CLONE_GATE=1 to build and install a wheel")
def test_the_full_clean_room_run(tmp_path: Path) -> None:
    """Install from a wheel outside the checkout and drive the whole reader path."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(SCRIPTS.parent / "src"), environment.get("PYTHONPATH", "")) if path
    )
    result = subprocess.run(
        [sys.executable, str(DRIVER), "--workdir", str(tmp_path / "clean"), "--keep-going"],
        capture_output=True,
        text=True,
        timeout=3600,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean-clone gate: " in result.stdout
    assert "FAIL" not in result.stdout
