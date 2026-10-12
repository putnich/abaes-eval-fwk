"""Aggregate predicted criteria while retaining official overall human scores."""

from __future__ import annotations

import math

from abaes_eval_fwk.core.models.assessment import Dataset, GradingRecord
from abaes_eval_fwk.core.models.rubric import Rubric, combine_criterion_scores
from abaes_eval_fwk.utils.rounding import round_half_up


def aggregate_criteria(rubric: Rubric, records: list[GradingRecord]) -> int:
    """Combine exactly one prediction for each rubric criterion."""
    names = [r.human_assessment.criterion.name for r in records]
    expected = {c.name for c in rubric.criteria}
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError(f"Expected each criterion exactly once: {sorted(expected)}, got {names}")
    scale = rubric.get_criterion_scale()
    scores = [r.llm_assessment.score.raw_score for r in records]
    if any(not math.isfinite(v) or v < scale.min or v > scale.max for v in scores):
        raise ValueError(f"Criterion prediction outside [{scale.min}, {scale.max}]")
    return combine_criterion_scores(
        rubric,
        {
            record.human_assessment.criterion.name: round_half_up(
                record.llm_assessment.score.raw_score
            )
            for record in records
        },
    )


def official_human_score(datasets: list[Dataset], dataset_label: str, essay_id: str) -> int | None:
    """Find the official overall score independently of analytic criterion scores."""
    dataset = next((d for d in datasets if d.dataset_name == dataset_label), None)
    return dataset.human_score(essay_id) if dataset else None


def get_human_aggregate_criteria(
    rubric: Rubric,
    records: list[GradingRecord],
    datasets: list[Dataset],
    dataset_label: str,
    essay_id: str,
) -> int:
    """Return the official overall target for every dataset, including ELLIPSE."""
    official = official_human_score(datasets, dataset_label, essay_id)
    if official is None:
        raise ValueError(
            f"No official overall human score for {dataset_label!r} essay {essay_id!r}"
        )
    return official
