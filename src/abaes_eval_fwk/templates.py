"""Create editable benchmark projects without bundling student datasets or credentials."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from abaes_eval_fwk.utils.resources import config_resource

# The synthetic dataset and its rubric are packaged data, not configuration, so they live under
# resources/playground/ beside the study rubrics. Both are installed at the project root:
# datasets.yaml names the data relative to benchmark.yaml.
PLAYGROUND_DATA = ("demo.csv", "demo-rubric.yaml")
# A reader's `module:function` extension is imported from the configuration's own directory, so
# the example package ships with the playground components and is installed at the project root.
PLAYGROUND_TREES = ("extensions",)


def playground_resource(name: str):
    """Locate a packaged playground data file in source checkouts or installed wheels."""
    return files("abaes_eval_fwk").joinpath("resources", "playground", name)


def _copy_tree(source, destination: Path, skip: frozenset[str] = frozenset()) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for resource in source.iterdir():
        if resource.name in skip:
            continue
        target = destination / resource.name
        if resource.is_dir():
            _copy_tree(resource, target, skip)
        else:
            target.write_bytes(resource.read_bytes())


def initialize(directory: Path, template: str) -> Path:
    """Create a new project, refusing to overwrite existing files."""
    aliases = {"demo": "playground", "simulation": "playground", "full": "study"}
    template = aliases.get(template, template)
    if template not in {"playground", "study"}:
        raise ValueError(f"Unknown configuration template: {template}")
    root = Path(directory).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Project directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if template == "playground":
        for name in PLAYGROUND_DATA:
            (root / name).write_bytes(playground_resource(name).read_bytes())
        for name in PLAYGROUND_TREES:
            _copy_tree(config_resource(f"playground/{name}"), root / name)
    else:
        rubrics = root / "rubrics"
        rubrics.mkdir()
        for resource in files("abaes_eval_fwk").joinpath("resources/rubrics").iterdir():
            if resource.name.endswith(".yaml"):
                (rubrics / resource.name).write_bytes(resource.read_bytes())
    config_dir = root / "config"
    _copy_tree(config_resource(template), config_dir, frozenset(PLAYGROUND_TREES))
    target = root / "benchmark.yaml"
    target.write_bytes((config_dir / "benchmark.yaml").read_bytes())
    (config_dir / "benchmark.yaml").unlink()
    return target
