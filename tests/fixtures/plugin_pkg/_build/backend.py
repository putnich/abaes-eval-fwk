"""A stdlib-only PEP 517 backend, so installing this fixture needs no package index. A
researcher's own extension builds with setuptools, as `examples/extension_package` does; this
one exists only so the registry's end-to-end install test runs hermetically and offline."""

from __future__ import annotations

import base64
import hashlib
import tarfile
import tomllib
import zipfile
from pathlib import Path
from typing import Any, Final

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
TAG: Final[str] = "py3-none-any"
DISTRIBUTION_STEM: Final[str] = "aes_bm25_plugin"


def get_requires_for_build_wheel(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    return []


def get_requires_for_build_sdist(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    return []


def _project() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project: dict[str, Any] = tomllib.load(handle)["project"]
    return project


def _metadata(project: dict[str, Any], readme: str) -> str:
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {project['name']}",
        f"Version: {project['version']}",
        f"Summary: {project['description']}",
        f"Requires-Python: {project['requires-python']}",
    ]
    lines += [f"Requires-Dist: {item}" for item in project.get("dependencies", ())]
    return "\n".join(lines) + "\n\n" + readme


def _entry_points(project: dict[str, Any]) -> str:
    groups = project.get("entry-points", {})
    return "".join(
        f"[{group}]\n" + "".join(f"{name} = {value}\n" for name, value in sorted(members.items()))
        for group, members in sorted(groups.items())
    )


def _digest(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Write a pure-Python wheel of the package under `src/` and return its filename."""
    project = _project()
    version = project["version"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    dist_info = f"{DISTRIBUTION_STEM}-{version}.dist-info"
    filename = f"{DISTRIBUTION_STEM}-{version}-{TAG}.whl"
    members: list[tuple[str, bytes]] = [
        (item.relative_to(ROOT / "src").as_posix(), item.read_bytes())
        for item in sorted((ROOT / "src").rglob("*.py"))
    ]
    members += [
        (f"{dist_info}/METADATA", _metadata(project, readme).encode("utf-8")),
        (
            f"{dist_info}/WHEEL",
            f"Wheel-Version: 1.0\nGenerator: plugin_pkg\n"
            f"Root-Is-Purelib: true\nTag: {TAG}\n".encode(),
        ),
        (f"{dist_info}/entry_points.txt", _entry_points(project).encode("utf-8")),
    ]
    record = "".join(f"{name},{_digest(data)},{len(data)}\n" for name, data in members)
    members.append((f"{dist_info}/RECORD", f"{record}{dist_info}/RECORD,,\n".encode("utf-8")))
    with zipfile.ZipFile(Path(wheel_directory) / filename, "w", zipfile.ZIP_DEFLATED) as wheel:
        for name, data in members:
            wheel.writestr(name, data)
    return filename


def build_sdist(sdist_directory: str, config_settings: dict[str, Any] | None = None) -> str:
    """Write a source distribution of the whole package directory and return its filename."""
    project = _project()
    stem = f"{DISTRIBUTION_STEM}-{project['version']}"
    filename = f"{stem}.tar.gz"
    included = ["pyproject.toml", "README.md", "src", "_build"]
    with tarfile.open(Path(sdist_directory) / filename, "w:gz") as archive:
        for name in included:
            archive.add(ROOT / name, arcname=f"{stem}/{name}")
    return filename
