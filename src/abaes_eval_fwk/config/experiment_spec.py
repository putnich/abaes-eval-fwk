"""One declared experiment: architecture, model, retrieval, reuse, and control dimensions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from abaes_eval_fwk.config.reuse_config import ReuseConfig
from abaes_eval_fwk.utils.validation import keys as _keys

REASONING_KEYS = {"", "yes", "no", "no_think"}
# Maps a declared procedure to the internal FinalAgentProcedure name the engine expects.
PROCEDURES = {
    "none": "",
    "disagreement": "adjudication",
    "review_based": "adjudication_with_review",
    "review": "review",
    "pool_average": "pool_average",
}
CONTROL_DIMENSIONS = {
    "model",
    "datasets",
    "sampling",
    "retrieval",
    "architecture",
    "reasoning",
    "procedure",
    "reuse",
}
EXPERIMENT_KEYS = {
    "name",
    "rq",
    "model",
    "architecture",
    "retrieval",
    "reasoning",
    "procedure",
    "reuse",
    "datasets",
    "sampling",
    "disagreement_threshold",
    "control",
}
ARCHITECTURES = {"holistic", "criterion_specialized", "final_agent", "non_grading"}


@dataclass(frozen=True)
class ControlSpec:
    """A declared controlled comparison: the baseline experiment and the dimensions that may vary."""

    baseline: str
    on: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, str) or not self.baseline:
            raise ValueError("control.baseline must name an experiment")
        on = tuple(self.on)
        object.__setattr__(self, "on", on)
        if not on or len(set(on)) != len(on) or not set(on) <= CONTROL_DIMENSIONS:
            raise ValueError(
                f"control.on must contain unique values from {sorted(CONTROL_DIMENSIONS)}"
            )


@dataclass(frozen=True)
class ExperimentSpec:
    """One declared experiment entry, validated independent of the other declared experiments."""

    name: str
    rq: str
    model: str
    architecture: str = "holistic"
    retrieval: str | dict = field(default_factory=dict)
    reasoning: str = ""
    procedure: str = "none"
    reuse: ReuseConfig = field(default_factory=ReuseConfig)
    datasets: tuple[str, ...] | None = None
    sampling: dict = field(default_factory=dict)
    disagreement_threshold: float = 0.2
    control: ControlSpec | None = None

    def __post_init__(self) -> None:
        if self.datasets is not None:
            object.__setattr__(self, "datasets", tuple(self.datasets))
        self.validate_experiment_spec()

    def validate_experiment_spec(self) -> None:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", self.name):
            raise ValueError(
                f"Experiment names must be unique filesystem-safe names: {self.name!r}"
            )
        if self.rq not in {"RQ1", "RQ2", "RQ3", "RQ4"}:
            raise ValueError(f"{self.name}: rq must be RQ1, RQ2, RQ3, or RQ4")
        if (self.reasoning or "") not in REASONING_KEYS:
            raise ValueError(
                f"{self.name}: reasoning must be quoted 'yes', 'no', no_think, or empty"
            )
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"{self.name}: invalid architecture")
        if self.procedure not in PROCEDURES:
            raise ValueError(f"{self.name}: unknown procedure {self.procedure}")
        if (self.procedure in {"disagreement", "review_based", "pool_average"}) != (
            self.architecture == "final_agent"
        ):
            raise ValueError(
                f"{self.name}: adjudication/pool_average requires architecture final_agent"
            )
        if (self.procedure == "review") != (self.architecture == "non_grading"):
            raise ValueError(f"{self.name}: review requires architecture non_grading")
        reuse = self.reuse
        has_reuse = bool(reuse.results_paths or reuse.reviews_paths or reuse.independent_path)
        if self.procedure != "none":
            minimum = 1 if self.procedure == "review" else 2
            if len(reuse.results_paths) < minimum:
                raise ValueError(
                    f"{self.name}: {self.procedure} needs at least {minimum} source result files"
                )
            if self.procedure == "review" and len(reuse.results_paths) != 1:
                raise ValueError(f"{self.name}: review consumes one source file")
        elif has_reuse:
            raise ValueError(f"{self.name}: initial grading cannot reuse prior results")
        if self.procedure in {"disagreement", "review_based"} and not reuse.independent_path:
            raise ValueError(
                f"{self.name}: an independent result file from the final model is required"
            )
        threshold = self.disagreement_threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (float, int))
            or not 0 <= threshold <= 1
        ):
            raise ValueError("disagreement_threshold must be a fraction between 0 and 1")
        if self.datasets is not None and (
            not self.datasets or len(set(self.datasets)) != len(self.datasets)
        ):
            raise ValueError("Experiment datasets must be a list of unique labels")
        if not isinstance(self.retrieval, (str, dict)):
            raise ValueError(f"{self.name}: retrieval must be a string or mapping")
        if not isinstance(self.sampling, dict):
            raise ValueError(f"{self.name}: sampling override must be a mapping")


def experiment_from_dict(spec: dict) -> ExperimentSpec:
    """Build a validated ExperimentSpec from one YAML experiment entry."""
    _keys(spec, EXPERIMENT_KEYS, "experiment")
    name = spec.get("name", "")
    reuse_raw = spec.get("reuse", {})
    _keys(reuse_raw, {"results", "independent", "reviews"}, f"{name}.reuse")
    for key in ("results", "reviews"):
        if key in reuse_raw and (
            not isinstance(reuse_raw[key], list)
            or not all(isinstance(p, str) and p for p in reuse_raw[key])
        ):
            raise ValueError(
                f"{name}.reuse.{key} must be a list of paths or @experiment references"
            )
    reuse = ReuseConfig(
        results_paths=tuple(reuse_raw.get("results", ())),
        reviews_paths=tuple(reuse_raw.get("reviews", ())),
        independent_path=reuse_raw.get("independent", ""),
    )
    control_raw = spec.get("control")
    control = None
    if control_raw is not None:
        _keys(control_raw, {"baseline", "on"}, f"{name}.control")
        control = ControlSpec(
            baseline=control_raw.get("baseline", ""), on=tuple(control_raw.get("on", ()))
        )
    return ExperimentSpec(
        name=name,
        rq=spec.get("rq", ""),
        model=spec.get("model", ""),
        architecture=spec.get("architecture", "holistic"),
        retrieval=spec.get("retrieval", {}),
        reasoning=spec.get("reasoning") or "",
        procedure=spec.get("procedure", "none"),
        reuse=reuse,
        datasets=spec.get("datasets"),
        sampling=spec.get("sampling", {}),
        disagreement_threshold=spec.get("disagreement_threshold", 0.2),
        control=control,
    )
