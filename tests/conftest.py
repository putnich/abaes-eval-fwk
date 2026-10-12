"""Shared rubric and submission fixtures for framework tests."""

import pytest

from abaes_eval_fwk.core.models.assessment import HumanAssessment, StudentSubmission
from abaes_eval_fwk.core.models.rubric import Criterion, Rubric, Scale, ScoreLevel


@pytest.fixture
def criteria():
    return [
        Criterion(
            name="ideas",
            score_levels=[ScoreLevel(0, "Missing ideas"), ScoreLevel(4, "Well developed ideas")],
        ),
        Criterion(
            name="organization",
            score_levels=[ScoreLevel(0, "No structure"), ScoreLevel(4, "Clear structure")],
        ),
    ]


@pytest.fixture
def rubric(criteria):
    return Rubric(
        rubric_id="set_1",
        assignment_type="essay",
        source_dependent=False,
        prompt="Write an essay about a memorable event.",
        scale=Scale(0, 4),
        rubric_guidelines="Score accuracy, evidence, and clarity.",
        criteria=criteria,
    )


@pytest.fixture
def submission(rubric):
    return HumanAssessment(
        submission=StudentSubmission(essay_id="essay-1", text="A concise student essay."),
        human_score=3,
        criterion=rubric.holistic_criterion(),
    )
