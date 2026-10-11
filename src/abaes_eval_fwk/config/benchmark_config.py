"""The validated, fully-merged top-level benchmark configuration document."""

from __future__ import annotations

from dataclasses import dataclass, field

from abaes_eval_fwk.config.calibration_config import CalibrationConfig
from abaes_eval_fwk.config.metrics_config import MetricsConfig
from abaes_eval_fwk.config.prompts_config import PromptsConfig
from abaes_eval_fwk.config.runtime_config import RuntimeConfig
from abaes_eval_fwk.config.sampling_config import SamplingConfig


@dataclass(frozen=True, kw_only=True)
class BenchmarkConfig:
    """Typed root of a loaded configuration.

    A section is typed here only if nothing ever merges into it again: `calibration`,
    `metrics`, and `runtime` have no per-experiment override anywhere, and `sampling`
    holds the project-level default (re-merged with an experiment override, then
    re-typed, by `sampling_config` in `configuration.py`).

    `retrieval`, `models`, `model_defaults`, `datasets`, and `experiments` stay raw
    mappings/lists because each needs something only known per-experiment merged in
    before it can become its real domain object: `retrieval`'s YAML `strategy` string
    isn't expanded into `examples_selection`/`selector` until after an experiment's own
    `strategy` override is merged against it; `models` needs `model_defaults` plus the
    experiment's `reasoning`; `datasets` needs `output_dir`- and seed-derived paths; and
    `experiments` is a `@name`-reference graph, not an independently resolvable list.
    `retrieval_config`, `model_config`, and `dataset_configs` in `configuration.py` do
    that resolution.

    `config_path` and `base_dir` are injected by the loader, not user-supplied YAML.
    """

    schema_version: int = 1
    data_root: str = ""
    output_dir: str = ""
    subdirs: dict = field(default_factory=dict)
    datasets: list = field(default_factory=list)
    # No default: SamplingConfig/CalibrationConfig/MetricsConfig/RuntimeConfig no longer
    # support zero-arg construction (see those modules), since every one of their fields
    # is already defaulted in config_files/defaults/*.yaml. load_config always supplies
    # a resolved instance explicitly, so this stays required rather than growing a second,
    # unreachable set of Python-side defaults. kw_only=True (above) lets these required
    # fields sit among the defaulted ones below without reordering the class.
    sampling: SamplingConfig
    calibration: CalibrationConfig
    retrieval: dict = field(default_factory=dict)
    metrics: MetricsConfig
    # Resolved grading templates. Typed here for the same reason as `metrics`: nothing merges
    # into prompts again after load, and no experiment overrides them.
    prompts: PromptsConfig
    strategies: dict = field(default_factory=dict)
    runtime: RuntimeConfig
    models: dict = field(default_factory=dict)
    model_defaults: dict = field(default_factory=dict)
    experiments: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    config_path: str = ""
    base_dir: str = ""
