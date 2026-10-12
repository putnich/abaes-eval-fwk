"""Rubric YAML definition parser and loader."""

import hashlib
import math
from functools import cache
from pathlib import Path

import yaml

from abaes_eval_fwk.core.models.rubric import (
    CriteriaAggregation,
    Criterion,
    Rubric,
    Scale,
    ScoreLevel,
)
from abaes_eval_fwk.utils.extensions import is_reference, split_reference


def load_rubric(yaml_path: str, rubric_id: str) -> Rubric:
    """Load and parse a Rubric object from a target YAML definition file.

    The parse is cached on the bytes actually read, not on the path, so editing a rubric in place
    yields the new rules within one process instead of the rules its path was first seen with.
    """
    raw = Path(yaml_path).read_bytes()
    return _parse_rubric(hashlib.sha256(raw).hexdigest(), raw.decode("utf-8"), rubric_id)


def rubric_source_digest(yaml_path: str) -> str:
    """The content digest a rubric parse is keyed on, so provenance names the bytes it parsed."""
    return hashlib.sha256(Path(yaml_path).read_bytes()).hexdigest()


def _parse_aggregation(
    data: dict, rubric_id: str
) -> tuple[CriteriaAggregation | str, dict[str, float]]:
    """Resolve `criteria_aggregation` and the weights the WEIGHTED mode requires.

    A value naming a `module:function` stays a string: the callable supplies the formula, so no
    built-in mode and no weights apply to it.
    """
    declared = data.get("criteria_aggregation", "mean")
    raw_weights = data.get("criteria_weights") or {}
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError(
            f"{rubric_id}: criteria_aggregation must be a mode name or module:function"
        )
    if is_reference(declared):
        split_reference(declared, f"{rubric_id}: criteria_aggregation")
        if raw_weights:
            raise ValueError(
                f"{rubric_id}: criteria_weights do not apply to an aggregation callable, which "
                "receives every criterion score and weights them itself"
            )
        return declared, {}
    mode = CriteriaAggregation(declared)
    if mode != CriteriaAggregation.WEIGHTED:
        if raw_weights:
            raise ValueError(
                f"{rubric_id}: criteria_weights require criteria_aggregation: weighted"
            )
        return mode, {}
    if not isinstance(raw_weights, dict) or not raw_weights:
        raise ValueError(f"{rubric_id}: criteria_aggregation: weighted requires criteria_weights")
    declared_criteria = set((data.get("criteria") or {}).keys())
    weights: dict[str, float] = {}
    for name, value in raw_weights.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{rubric_id}: criteria_weights.{name} must be a number")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{rubric_id}: criteria_weights.{name} must be finite and >= 0")
        weights[str(name)] = value
    if set(weights) != declared_criteria:
        raise ValueError(
            f"{rubric_id}: criteria_weights must name every rubric criterion exactly once; "
            f"expected {sorted(declared_criteria)}, got {sorted(weights)}"
        )
    if sum(weights.values()) <= 0:
        raise ValueError(f"{rubric_id}: criteria_weights must not all be zero")
    return mode, weights


@cache
def _parse_rubric(digest: str, text: str, rubric_id: str) -> Rubric:
    """Parse rubric YAML text; `digest` is its content hash and part of the cache key only."""
    data = yaml.safe_load(text)
    raw_score_ranges = data.get("score_ranges") or {}  # score_ranges.overall is canonical
    if raw_score_ranges:
        score_min, score_max = (int(x) for x in raw_score_ranges["overall"])
    else:
        score_max = int(data["score_max"])
        score_min = int(data["score_min"])

    raw_criteria: dict[str, dict] = data.get("criteria") or {}
    source_text = (data.get("source_text") or "").strip()
    is_dimension = bool(
        data.get("is_dimension", False)
    )  # Generated from holistic rubric or officially authored
    raw_scale = raw_score_ranges.get("per_dimension" if is_dimension else "per_criterion")
    criteria_scale = Scale(min=int(raw_scale[0]), max=int(raw_scale[1])) if raw_scale else None

    rubric_guidelines = (data.get("rubric_guidelines") or "").strip()

    criteria_aggregation, criteria_weights = _parse_aggregation(data, rubric_id)
    criteria: list[Criterion] = [
        Criterion(
            name=k,
            score_levels=[
                ScoreLevel(score=int(lv["score"]), guideline=str(lv["guideline"]))
                for lv in entry["score_levels"]
            ],
        )
        for k, entry in raw_criteria.items()
    ]

    return Rubric(
        rubric_id=rubric_id,
        assignment_type=data["assignment_type"],
        source_dependent=bool(data.get("source_dependent", False)),
        prompt=(data.get("prompt") or "").strip(),
        scale=Scale(min=score_min, max=score_max),
        rubric_guidelines=rubric_guidelines,
        criteria=criteria,
        is_dimension=is_dimension,
        criteria_scale=criteria_scale,
        criteria_aggregation=criteria_aggregation,
        criteria_weights=criteria_weights,
        source_text=source_text,
        adjudication_rules=(data.get("adjudication_rules") or "").strip(),
    )
