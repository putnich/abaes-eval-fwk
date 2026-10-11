"""Load and validate a portable benchmark configuration without contacting a model."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import asdict, fields
from pathlib import Path

import yaml

from abaes_eval_fwk.config.benchmark_config import BenchmarkConfig
from abaes_eval_fwk.config.calibration_config import CalibrationConfig
from abaes_eval_fwk.config.experiment_spec import (
    CONTROL_DIMENSIONS,
    ExperimentSpec,
    experiment_from_dict,
)
from abaes_eval_fwk.config.llm_config import LLMConfig
from abaes_eval_fwk.config.metrics_config import MetricsConfig
from abaes_eval_fwk.config.prompts_config import (
    SLOTS,
    PromptsConfig,
    default_prompts_config,
    read_template,
)
from abaes_eval_fwk.config.retrieval_config import ExamplesSelection, RetrievalConfig
from abaes_eval_fwk.config.runtime_config import RuntimeConfig
from abaes_eval_fwk.config.sampling_config import SamplingConfig
from abaes_eval_fwk.utils.extensions import split_reference
from abaes_eval_fwk.utils.resources import load_defaults
from abaes_eval_fwk.utils.validation import keys as _keys

ROOT_KEYS = {
    "schema_version",
    "data_root",
    "output_dir",
    "subdirs",
    "datasets",
    "sampling",
    "calibration",
    "retrieval",
    "metrics",
    "prompts",
    "strategies",
    "runtime",
    "models",
    "model_defaults",
    "experiments",
    "metadata",
    "components",
}
# The built-in selection policies. Any other strategy must be a reader's `module:function`.
BUILTIN_RETRIEVAL_STRATEGIES = {"zero_shot", "random", "lexical", "semantic", "hybrid"}
COMPONENT_KEYS = {
    "paths",
    "datasets",
    "sampling",
    "calibration",
    "retrieval",
    "metrics",
    "prompts",
    "runtime",
    "models",
    "experiments",
    "metadata",
}


def merge(base: dict, update: dict) -> dict:
    """Recursively merge mappings while preserving explicit null and false values."""
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else copy.deepcopy(value)
        )
    return result


def _check_json_types(value, path: str) -> None:
    import math

    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings; quote YAML 'on'/'off' keys")
            _check_json_types(item, f"{path}.{key}")
    elif isinstance(value, list):
        for item in value:
            _check_json_types(item, path)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must contain finite numbers")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"{path} must contain only JSON-compatible values")


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> BenchmarkConfig:
    """Read YAML with paths relative to its directory and optional dotted assignments."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        source = yaml.safe_load(stream) or {}
    _keys(source, ROOT_KEYS, "configuration")
    component_paths = source.pop("components", {})
    _keys(component_paths, COMPONENT_KEYS, "components")
    config = load_defaults()
    for name, reference in component_paths.items():
        references = reference if isinstance(reference, list) else [reference]
        if not references or not all(isinstance(item, str) and item for item in references):
            raise ValueError(f"components.{name} must contain YAML paths")
        for item in references:
            component_path = (path.parent / item).resolve()
            if not component_path.is_file():
                raise ValueError(f"Component file not found: {component_path}")
            with component_path.open(encoding="utf-8") as stream:
                component = yaml.safe_load(stream) or {}
            if name in component:
                # Allow reference sections like 'strategies' alongside the main config key
                allowed = {name, "strategies"}
                _keys(component, allowed, f"component {name}")
            else:
                _keys(component, ROOT_KEYS - {"components"}, f"component {name}")
            if name == "experiments" and len(references) > 1:
                config["experiments"] = [
                    *config.get("experiments", []),
                    *component.get("experiments", []),
                ]
            else:
                config = merge(config, component)
    config = merge(config, source)
    for assignment in overrides:
        key, sep, text = assignment.partition("=")
        if not sep:
            raise ValueError(f"Override must have dotted.path=value syntax: {assignment}")
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(f"Override traverses a non-mapping: {key}")
        target[parts[-1]] = yaml.safe_load(text)
    _keys(config, ROOT_KEYS, "configuration")
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    config = merge(load_defaults(), config)
    for name in ("data_root", "output_dir"):
        config[name] = str((path.parent / config[name]).resolve())
    # Resolve subdirs relative to output_dir
    if "subdirs" in config:
        config["subdirs"] = {
            k: str((Path(config["output_dir"]) / v).resolve()) for k, v in config["subdirs"].items()
        }
    _check_json_types(config.get("metadata", {}), "metadata")
    _keys(config["sampling"], {f.name for f in fields(SamplingConfig)}, "sampling")
    _keys(config["runtime"], {f.name for f in fields(RuntimeConfig)}, "runtime")
    benchmark = BenchmarkConfig(
        schema_version=config["schema_version"],
        data_root=config["data_root"],
        output_dir=config["output_dir"],
        subdirs=config.get("subdirs", {}),
        datasets=config.get("datasets", []),
        sampling=SamplingConfig(**config["sampling"]),
        calibration=calibration_config(config.get("calibration", {})),
        retrieval=config.get("retrieval", {}),
        metrics=metrics_config(config.get("metrics", {})),
        prompts=prompts_config(config.get("prompts", {}), path.parent),
        strategies=config.get("strategies", {}),
        runtime=RuntimeConfig(**config["runtime"]),
        models=config.get("models", {}),
        model_defaults=config.get("model_defaults", {}),
        experiments=config.get("experiments", []),
        metadata=config.get("metadata", {}),
        config_path=str(path),
        base_dir=str(path.parent),
    )
    validate_config(benchmark)
    return benchmark


def config_from_dict(data: dict) -> BenchmarkConfig:
    """Rebuild a `BenchmarkConfig` from a plain dict, such as one read back from JSON."""
    data = dict(data)
    data["sampling"] = SamplingConfig(**data["sampling"])
    data["calibration"] = CalibrationConfig(**data["calibration"])
    data["metrics"] = metrics_config(data["metrics"])
    data["prompts"] = PromptsConfig(**data["prompts"])
    data["runtime"] = RuntimeConfig(**data["runtime"])
    return BenchmarkConfig(**data)


def retrieval_config(
    config: BenchmarkConfig, experiment: ExperimentSpec | None = None
) -> RetrievalConfig:
    spec = experiment.retrieval if experiment is not None else {}
    spec = {"strategy": spec} if isinstance(spec, str) else dict(spec)
    spec = merge(config.retrieval, spec)
    spec.setdefault("seed", config.sampling.seed)
    strategy = spec.pop("strategy", "hybrid")
    if strategy not in BUILTIN_RETRIEVAL_STRATEGIES:
        # A reader's `module:function`; the callable supplies the ranking, so only shared knobs apply.
        spec["selector"] = strategy
        _keys(spec, {f.name for f in fields(RetrievalConfig)}, "retrieval")
        return RetrievalConfig(**spec)
    spec["examples_selection"] = (
        ExamplesSelection.ZERO_SHOT if strategy == "zero_shot" else ExamplesSelection(strategy)
    )
    _keys(spec, {f.name for f in fields(RetrievalConfig)}, "retrieval")
    return RetrievalConfig(**spec)


def model_config(config: BenchmarkConfig, experiment: ExperimentSpec) -> LLMConfig:
    model = experiment.model
    if model not in config.models:
        raise ValueError(f"Unknown model profile {model!r} in {experiment.name}")
    spec = merge(config.model_defaults, config.models[model])
    _keys(spec, {f.name for f in fields(LLMConfig)} - {"reasoning"}, f"model {model}")
    spec.setdefault("model_shorthand", model)
    spec["reasoning"] = experiment.reasoning or ""
    model = LLMConfig(**spec)
    if not model.model_name.strip():
        raise ValueError("Each model profile requires model_name")
    if model.provider == "openai_compatible" and not model.base_url and not model.adapter_factory:
        raise ValueError("openai_compatible profiles require base_url")
    return model


def dataset_configs(config: BenchmarkConfig, experiment: ExperimentSpec | None = None) -> list:
    """Resolve built-in datasets or custom normalized CSV/rubric pairs."""
    from abaes_eval_fwk.config.dataset_config import dataset_from_dict

    sampling = sampling_config(config, experiment)
    defaults = {
        "data_root": config.data_root,
        "calibration_root": str(Path(config.output_dir) / "calibration"),
        "cache_root": str(Path(config.output_dir) / "cache"),
        "seed": sampling.seed,
        "calibration_per_score": config.calibration.max_per_score,
    }
    result = [
        dataset_from_dict(item, base_dir=config.base_dir, defaults=defaults)
        for item in config.datasets
    ]
    if len({item.label for item in result}) != len(result):
        raise ValueError("Dataset names must be unique")
    if experiment and experiment.datasets:
        labels = experiment.datasets
        missing = set(labels) - {item.label for item in result}
        if missing:
            raise ValueError(f"Unknown experiment datasets: {sorted(missing)}")
        result = [item for item in result if item.label in labels]
    return result


def sampling_config(
    config: BenchmarkConfig, experiment: ExperimentSpec | None = None
) -> SamplingConfig:
    """Resolve project sampling settings with optional experiment-local overrides."""
    if experiment is None or not experiment.sampling:
        return config.sampling
    resolved = merge(asdict(config.sampling), experiment.sampling)
    _keys(resolved, {f.name for f in fields(SamplingConfig)}, "sampling")
    return SamplingConfig(**resolved)


def metrics_config(spec: dict) -> MetricsConfig:
    """Validate and resolve a metrics configuration section."""
    from abaes_eval_fwk.config.metrics_config import (
        ComparisonConfig,
        MetricsCompute,
        SignificanceThreshold,
        WilcoxonConfig,
    )

    if not isinstance(spec, dict):
        raise ValueError("metrics must be a mapping")
    allowed_keys = {"compute", "significance_thresholds", "wilcoxon", "comparison", "custom"}
    _keys(spec, allowed_keys, "metrics")
    compute_spec = spec.get("compute", {})
    if not isinstance(compute_spec, dict):
        raise ValueError("metrics.compute must be a mapping")
    _keys(
        compute_spec,
        {"nmae", "nrmse", "pearson", "spearman", "qwk", "mae", "rmse"},
        "metrics.compute",
    )
    for key in compute_spec:
        if not isinstance(compute_spec[key], bool):
            raise ValueError(f"metrics.compute.{key} must be boolean")
    compute = MetricsCompute(**compute_spec)
    thresholds_spec = spec.get("significance_thresholds")
    if thresholds_spec is not None:
        if not isinstance(thresholds_spec, list):
            raise ValueError("metrics.significance_thresholds must be a list")
        thresholds = []
        for item in thresholds_spec:
            if not isinstance(item, dict) or "p_value" not in item:
                raise ValueError("significance_thresholds items must have p_value")
            p_val = item["p_value"]
            if not isinstance(p_val, (int, float)) or not 0 <= p_val <= 1:
                raise ValueError(f"p_value must be between 0 and 1, got {p_val}")
            thresholds.append(SignificanceThreshold(float(p_val)))
    else:
        thresholds = None
    wilcoxon_spec = spec.get("wilcoxon", {})
    if not isinstance(wilcoxon_spec, dict):
        raise ValueError("metrics.wilcoxon must be a mapping")
    _keys(wilcoxon_spec, {"alternative", "alpha", "zero_method", "method"}, "metrics.wilcoxon")
    if "alternative" in wilcoxon_spec:
        alt = wilcoxon_spec["alternative"]
        if alt not in {"two-sided", "less", "greater"}:
            raise ValueError("wilcoxon.alternative must be two-sided, less, or greater")
    for key in ("alpha",):
        if key in wilcoxon_spec:
            val = wilcoxon_spec[key]
            if not isinstance(val, (int, float)) or not 0 < val < 1:
                raise ValueError(f"wilcoxon.{key} must be between 0 and 1")
    wilcoxon = WilcoxonConfig(**wilcoxon_spec)
    comparison_spec = spec.get("comparison", {})
    if not isinstance(comparison_spec, dict):
        raise ValueError("metrics.comparison must be a mapping")
    _keys(comparison_spec, {"allow_partial_overlap", "alpha"}, "metrics.comparison")
    for key in ("allow_partial_overlap",):
        if key in comparison_spec:
            if not isinstance(comparison_spec[key], bool):
                raise ValueError(f"metrics.comparison.{key} must be boolean")
    for key in ("alpha",):
        if key in comparison_spec:
            val = comparison_spec[key]
            if not isinstance(val, (int, float)) or not 0 < val < 1:
                raise ValueError(f"metrics.comparison.{key} must be between 0 and 1")
    comparison = ComparisonConfig(**comparison_spec)
    return MetricsConfig(
        compute=compute,
        significance_thresholds=thresholds,
        wilcoxon=wilcoxon,
        comparison=comparison,
        custom=custom_metrics(spec.get("custom") or {}),
    )


def custom_metrics(spec: dict) -> dict[str, str]:
    """Validate `metrics.custom`, so a mistyped reference fails at load, not after a run."""
    from abaes_eval_fwk.experiment.eval.metrics.compute_metrics import BUILTIN_METRICS

    if not isinstance(spec, dict):
        raise ValueError("metrics.custom must be a mapping of name to module:function")
    reserved = set(BUILTIN_METRICS) | {"r", "rho"}
    for name, reference in spec.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each metrics.custom key must be a nonempty metric name")
        if name in reserved:
            raise ValueError(
                f"metrics.custom.{name} shadows a built-in metric; choose another name"
            )
        split_reference(reference, f"metrics.custom.{name}")
    return dict(spec)


def prompts_config(spec: dict, base_dir: str | Path | None = None) -> PromptsConfig:
    """Resolve the grading templates, falling back per slot to the packaged default.

    A project overrides one template without restating the other: an absent slot keeps the
    shipped wording rather than becoming empty.
    """
    if not isinstance(spec, dict):
        raise ValueError("prompts must be a mapping of template slot to template")
    _keys(spec, set(SLOTS), "prompts")
    defaults = default_prompts_config()
    resolved = {
        slot: read_template(spec[slot], slot, base_dir) if slot in spec else defaults.for_slot(slot)
        for slot in SLOTS
    }
    return PromptsConfig(**resolved)


def calibration_config(spec: dict) -> CalibrationConfig:
    """Validate and resolve a calibration configuration section."""
    if not isinstance(spec, dict):
        raise ValueError("calibration must be a mapping")
    allowed_keys = {"max_per_score"}
    _keys(spec, allowed_keys, "calibration")
    if "max_per_score" in spec:
        val = spec["max_per_score"]
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            raise ValueError("calibration.max_per_score must be a positive integer")
    return CalibrationConfig(**spec)


def validate_config(config: BenchmarkConfig) -> None:
    """Reject misspelled settings and incompatible experiment combinations early."""
    if not isinstance(config.models, dict) or not config.models:
        raise ValueError("At least one model profile is required")
    if not isinstance(config.model_defaults, dict):
        raise ValueError("model_defaults must be a mapping")
    if not isinstance(config.datasets, list) or not config.datasets:
        raise ValueError("At least one dataset is required")
    if not isinstance(config.experiments, list) or not config.experiments:
        raise ValueError("At least one experiment is required")
    for name in config.models:
        model_config(config, ExperimentSpec(name="profile_validation", rq="RQ1", model=name))
    experiments = select_experiments(config)
    names = set()
    for experiment in experiments:
        if experiment.name in names:
            raise ValueError(
                f"Experiment names must be unique filesystem-safe names: {experiment.name!r}"
            )
        names.add(experiment.name)
        retrieval = retrieval_config(config, experiment)
        if experiment.procedure != "none" and not retrieval.is_zero_shot:
            raise ValueError(
                f"{experiment.name}: procedures consuming prior results require zero_shot retrieval"
            )
        model_config(config, experiment)
        sampling_config(config, experiment)
        dataset_configs(config, experiment)
    dataset_configs(config)
    validate_controls(config, experiments)


def _control_signature(config: BenchmarkConfig, experiment: ExperimentSpec) -> dict:
    """Return effective settings grouped by the dimensions a comparison may vary."""
    profile = copy.deepcopy(config.models[experiment.model])
    return {
        "model": profile,
        "datasets": list(experiment.datasets)
        if experiment.datasets
        else [item["name"] for item in config.datasets],
        "sampling": sampling_config(config, experiment),
        "retrieval": asdict(retrieval_config(config, experiment)),
        "architecture": experiment.architecture,
        "reasoning": experiment.reasoning or "",
        "procedure": (experiment.procedure, experiment.disagreement_threshold),
        "reuse": experiment.reuse,
    }


def validate_controls(
    config: BenchmarkConfig, experiments: list[ExperimentSpec] | None = None
) -> None:
    """Reject declared controlled comparisons with unintended or missing differences."""
    by_name = {
        item.name: item
        for item in (experiments if experiments is not None else select_experiments(config))
    }
    for variant in by_name.values():
        control = variant.control
        if not control:
            continue
        if control.baseline not in by_name:
            raise ValueError(f"{variant.name}: control.baseline '{control.baseline}' not found")
        baseline = by_name[control.baseline]
        if baseline.rq != variant.rq:
            raise ValueError(f"{variant.name}: controlled comparison must use the same RQ")
        before = _control_signature(config, baseline)
        after = _control_signature(config, variant)
        changed = {name for name in CONTROL_DIMENSIONS if before[name] != after[name]}
        if changed != set(control.on):
            raise ValueError(
                f"{variant.name}: control.on={sorted(control.on)}, effective differences={sorted(changed)}"
            )


def select_experiments(
    config: BenchmarkConfig, names: list[str] | None = None
) -> list[ExperimentSpec]:
    """Order selected experiments and their @dependencies, rejecting missing references and cycles."""
    by_name = {item["name"]: experiment_from_dict(item) for item in config.experiments}
    ordered, visiting, done = [], set(), set()

    def visit(name: str) -> None:
        if name not in by_name:
            raise ValueError(f"Unknown experiment reference: {name}")
        if name in visiting:
            raise ValueError(f"Cyclic experiment dependency at {name}")
        if name in done:
            return
        visiting.add(name)
        experiment = by_name[name]
        reuse = experiment.reuse
        for reference in [*reuse.results_paths, *reuse.reviews_paths, reuse.independent_path]:
            if reference.startswith("@"):
                visit(reference[1:])
        visiting.remove(name)
        done.add(name)
        ordered.append(experiment)

    for name in names or by_name:
        visit(name)
    return ordered
