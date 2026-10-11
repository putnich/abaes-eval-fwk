"""Sampling strategies and submission builders for evaluation setups."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import pandas as pd

from abaes_eval_fwk.config.dataset_config import DatasetConfig
from abaes_eval_fwk.core.models.assessment import (
    Dataset,
    HumanAssessment,
    Sample,
    StudentSubmission,
)
from abaes_eval_fwk.core.models.rubric import Rubric
from abaes_eval_fwk.utils.rounding import round_half_up

logger = logging.getLogger(__name__)


def stratified_quantile_rank_sample(
    df: pd.DataFrame,
    n_per_set: int,
    group_col: str = "set_id",
    exclude_ids: frozenset | None = None,
) -> pd.DataFrame:
    """Select score-diverse, quantile rank-spaced essay rows from each dataset group."""
    if isinstance(n_per_set, bool) or not isinstance(n_per_set, int) or n_per_set <= 0:
        raise ValueError("sample_size must be a positive integer")
    if n_per_set < 25:
        logger.warning(
            "n_per_set=%d is below 25 — rank-based sampling is not guaranteed to cover "
            "every score on the scale at this size (empirically verified down to n=24 "
            "across current datasets); consider n_per_set >= 25 for full scale coverage.",
            n_per_set,
        )
    if exclude_ids:
        df = df[~df["essay_id"].isin(exclude_ids)]
    if df.empty:
        raise ValueError("No evaluation essays remain after exclusions")
    frames = []
    for _, group in df.groupby(group_col):
        k = n_per_set
        group_sorted = group.sort_values(["human_score", "essay_id"], kind="stable")
        k = min(k, len(group_sorted))
        if k <= 1:
            # Single pick: take the median-ranked essay rather than an arbitrary end.
            frames.append(group_sorted.iloc[[len(group_sorted) // 2]])
        else:
            # Quantile sampling: evenly spaced ranks ensure min/max inclusion.
            indices = [int(i * (len(group_sorted) - 1) / (k - 1)) for i in range(k)]
            frames.append(group_sorted.iloc[indices])
    result = pd.concat(frames).reset_index(drop=True) if frames else df.iloc[0:0]
    logger.info(
        "Representative sample: %d rows  per_%s=%s",
        len(result),
        group_col,
        dict(sorted(result.groupby(group_col).size().to_dict().items())),
    )
    return result


def per_criterion_assessments(row: Mapping[str, object], rubric: Rubric) -> list[HumanAssessment]:
    """Build atomic HumanAssessment objects for criteria with available human rater scores.

    `row` is a column-keyed mapping, not a namedtuple: a criterion name may legally contain a
    space or lead with an underscore, and `itertuples` renames such fields, which would silently
    turn a present human score into a missing one.
    """
    essay_id, text = str(row["essay_id"]), str(row["essay"])
    human_assessments = []
    for criterion in rubric.criteria:
        value = row.get(criterion.name)
        if value is not None and not pd.isna(value):
            human_assessments.append(
                HumanAssessment(
                    submission=StudentSubmission(essay_id=essay_id, text=text),
                    human_score=round_half_up(value),
                    criterion=criterion,
                )
            )
    return human_assessments


def build_sample(
    sample_df: pd.DataFrame,
    rubric: Rubric,
    is_criterion: bool,
    dataset_config: DatasetConfig | None = None,
) -> Sample:
    """Construct a Sample container of StudentSubmissions (holistic or per-criterion)."""
    holistic = rubric.holistic_criterion()

    if not is_criterion:
        holistic_assessments = [
            HumanAssessment(
                submission=StudentSubmission(essay_id=str(row["essay_id"]), text=str(row["essay"])),
                human_score=round_half_up(row["human_score"]),
                criterion=holistic,
            )
            for row in sample_df.to_dict("records")
        ]
        dataset = Dataset(dataset_config=dataset_config, human_assessments=holistic_assessments)
        return Sample(rubric=rubric, dataset=dataset)

    human_assessments: list[HumanAssessment] = []
    for row in sample_df.to_dict("records"):
        essay_id, text = str(row["essay_id"]), str(row["essay"])
        holistic_score = round_half_up(row["human_score"])
        real_subs = {s.criterion.name: s for s in per_criterion_assessments(row, rubric)}
        for criterion in rubric.criteria:
            human_assessments.append(
                real_subs.get(criterion.name)
                or HumanAssessment(
                    submission=StudentSubmission(essay_id=essay_id, text=text),
                    human_score=holistic_score,
                    criterion=criterion,
                )
            )
    dataset = Dataset(dataset_config=dataset_config, human_assessments=human_assessments)
    return Sample(rubric=rubric, dataset=dataset)
