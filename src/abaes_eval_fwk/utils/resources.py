"""Read packaged YAML configuration definitions."""

from importlib.resources import files

import yaml


def config_resource(name: str):
    """Locate a checked-in configuration file in source checkouts or installed wheels."""
    return files("abaes_eval_fwk").joinpath("config", "config_files", name)


def rubric_resource(name: str):
    """Locate a packaged rubric file in source checkouts or installed wheels."""
    return files("abaes_eval_fwk").joinpath("resources", name)


def load_defaults() -> dict:
    """Load and merge all component defaults from the defaults/ folder."""
    defaults_dir = config_resource("defaults")
    result: dict = {}
    # Load all YAML files from defaults/ in alphabetical order for consistency.
    for resource in sorted(defaults_dir.iterdir()):
        if resource.name.endswith(".yaml"):
            config_dict = yaml.safe_load(resource.read_text(encoding="utf-8")) or {}
            result.update(config_dict)
    return result
