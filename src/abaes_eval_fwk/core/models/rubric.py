"""Rubric domain model: grading criteria, score ranges, and guidelines for an assignment dataset."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Final


class CriteriaAggregation(str, Enum):
    """Aggregation mode specifying how criterion-level scores combine into a total score."""

    MEAN = "mean"  # criteria share the overall scale (default)
    SUM = "sum"  # criteria are independently scored and summed (e.g., sets 7 and 8)
    WEIGHTED = "weighted"  # criteria share the overall scale with per-criterion weights


@dataclass(frozen=True)
class Scale:
    """Inclusive integer score range [min, max]."""

    min: int
    max: int


@dataclass(frozen=True)
class ScoreLevel:
    """Guideline text associated with a specific score point within a criterion."""

    score: int
    guideline: str


@dataclass(frozen=True)
class Criterion:
    """Rubric dimension containing score level guidelines."""

    name: str
    score_levels: list[ScoreLevel]


# Module-level constant: frozen, reused in hot paths, never mutated.
HOLISTIC_CRITERION = Criterion(name="holistic", score_levels=[])

# Width of the 1-6 scale three of the six datasets use, so NAPD reads as whole score points rather
# than a 0-1 fraction.
NAPD_REFERENCE_WIDTH: Final[int] = 5


@dataclass(frozen=True)
class Rubric:
    """Immutable rubric definition shared across student submissions for a dataset set."""

    rubric_id: str
    assignment_type: str
    source_dependent: bool
    prompt: str
    scale: Scale
    rubric_guidelines: str
    criteria: list[Criterion] = field(default_factory=list)
    is_dimension: bool = False
    criteria_scale: Scale | None = None
    # A built-in mode, or a reader's `module:function` for a formula the three modes cannot
    # express (a median, a trimmed mean, a cap). The reference is kept as the plain string it
    # was written as: it names the formula in every run snapshot that records this rubric.
    criteria_aggregation: CriteriaAggregation | str = CriteriaAggregation.MEAN
    # Per-criterion weights, required by (and only valid for) the WEIGHTED mode.
    criteria_weights: dict[str, float] = field(default_factory=dict)
    source_text: str = ""
    adjudication_rules: str = ""

    def get_criterion(self, name: str) -> Criterion | None:
        """Look up a criterion by name."""
        return next((c for c in self.criteria if c.name == name), None)

    def total_score_range(self) -> Scale:
        """Return the rubric's overall total score range."""
        return self.scale

    def get_criterion_scale(self) -> Scale:
        """Return scale shared by all criteria (explicit `criteria_scale` if set, else global `scale`)."""
        return self.criteria_scale or self.scale

    def holistic_criterion(self) -> Criterion:
        """Return the shared sentinel criterion (`HOLISTIC_CRITERION`) used for holistic evaluation."""
        return HOLISTIC_CRITERION


def combine_criterion_scores(rubric: Rubric, scores: dict[str, int]) -> int:
    """Combine criteria with the rubric's aggregation formula and half-up rounding.

    Handles SUM, WEIGHTED, custom callables, and defaults to MEAN.
    This is the canonical implementation used by both evaluation and final-agent code.
    """
    from abaes_eval_fwk.utils.extensions import is_reference
    from abaes_eval_fwk.utils.rounding import round_half_up

    if not scores:
        raise ValueError("Cannot aggregate an empty criterion pool")
    aggregation = rubric.criteria_aggregation
    values = list(scores.values())
    if is_reference(aggregation):
        return _call_aggregation(aggregation, rubric, scores)
    if aggregation == CriteriaAggregation.SUM:
        return sum(values)
    if aggregation == CriteriaAggregation.WEIGHTED:
        weights = rubric.criteria_weights
        total = sum(weights[name] for name in scores)
        return round_half_up(sum(scores[name] * weights[name] for name in scores) / total)
    return round_half_up(sum(values) / len(values))


def _call_aggregation(reference: str, rubric: Rubric, scores: dict[str, int]) -> int:
    """Invoke a reader's aggregation callable and round its total the way a built-in's is.

    The result is not range-checked here: a total outside the rubric's scale is rejected by the
    same scale check every prediction meets in resolve_metrics, which names the dataset and essay.
    """
    from abaes_eval_fwk.utils.extensions import resolve_callable
    from abaes_eval_fwk.utils.rounding import round_half_up

    label = f"Rubric {rubric.rubric_id!r} criteria_aggregation"
    value = resolve_callable(reference, label)(rubric=rubric, scores=dict(scores))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must return a number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{label} returned a non-finite total")
    return round_half_up(float(value))
