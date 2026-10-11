"""Grading-path regressions kept after the plugin layer was removed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from abaes_eval_fwk.core.models.assessment import (
    HumanAssessment,
    StudentSubmission,
    TokenUsage,
)
from abaes_eval_fwk.core.models.rubric import Criterion, Rubric, Scale


def rubric() -> Rubric:
    return Rubric(
        rubric_id="rubric",
        assignment_type="Essay",
        source_dependent=False,
        prompt="Explain your answer",
        rubric_guidelines="Grade the essay",
        scale=Scale(min=0, max=6),
        criteria=[
            Criterion(name="ideas", score_levels=[]),
            Criterion(name="style", score_levels=[]),
        ],
    )


def test_a_checkpoint_record_is_returned_only_for_its_own_submission() -> None:
    """F1: a recovered record whose essay differs is not this request's result, whatever the key."""
    from abaes_eval_fwk.core.grader.grader import _matches_request
    from abaes_eval_fwk.core.models.assessment import (
        GradingRecord,
        LLMAssessment,
        PredictedScore,
    )

    def record_for(assessment: HumanAssessment) -> GradingRecord:
        return GradingRecord(
            human_assessment=assessment,
            llm_assessment=LLMAssessment(
                score=PredictedScore(3.0), justification="x", confidence=0.9
            ),
        )

    wanted = HumanAssessment(
        StudentSubmission("b:c", "first essay"), 1, rubric().holistic_criterion()
    )
    other = HumanAssessment(
        StudentSubmission("c", "second essay"), 4, rubric().holistic_criterion()
    )
    record = record_for(other)
    assert not _matches_request(record, wanted)
    assert _matches_request(record_for(wanted), wanted)


def test_a_criterion_name_that_is_not_an_identifier_keeps_its_human_score(tmp_path: Path) -> None:
    """F6: `sentence fluency` is a valid rubric criterion, so its column must not be read as missing."""
    import csv

    from abaes_eval_fwk.config.dataset_config import DatasetConfig
    from abaes_eval_fwk.experiment.setup.setup_builder import build_setup

    rubric_path = tmp_path / "rubric.yaml"
    rubric_path.write_text(
        yaml.safe_dump(
            {
                "assignment_type": "essay",
                "prompt": "Explain",
                "score_ranges": {"overall": [0, 4]},
                "criteria": {
                    "sentence fluency": {
                        "score_levels": [{"score": s, "guideline": str(s)} for s in range(5)]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "essays.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["essay_id", "set_id", "essay", "human_score", "sentence fluency"]
        )
        writer.writeheader()
        for index in range(8):
            writer.writerow(
                {
                    "essay_id": f"e{index}",
                    "set_id": "s",
                    "essay": f"Essay {index}",
                    "human_score": 4,
                    "sentence fluency": 1,
                }
            )

    config = DatasetConfig(
        dataset_name="named-criteria",
        normalized_csv=csv_path,
        rubric_file=rubric_path,
        calibration_root=tmp_path / "calibration",
        cache_root=tmp_path / "cache",
        criteria_columns=["sentence fluency"],
    )
    setup = build_setup(config, True, sample_size=3)
    criterion_scores = {
        item.human_score for item in setup.sample if item.criterion.name == "sentence fluency"
    }
    calibration_criteria = {
        item.criterion.name for item in setup.calibration_pool if item.criterion.name != "holistic"
    }
    assert criterion_scores == {1}, "the analytic score must survive, not be replaced by holistic"
    assert calibration_criteria == {"sentence fluency"}


def test_a_retry_reports_every_response_it_consumed(tmp_path: Path) -> None:
    """F7: a rejected first attempt was still paid for, so the assessment carries both responses."""
    from unittest.mock import patch

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.runnables import RunnableLambda

    from abaes_eval_fwk.core.agents.holistic_agent import HolisticGradingAgent
    from abaes_eval_fwk.core.models.rubric import Rubric, Scale

    class OutOfRangeThenValid(BaseChatModel):
        """Rejects its own first score, so the second attempt is what finally succeeds."""

        calls: int = 0

        @property
        def _llm_type(self) -> str:
            return "regression"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.calls += 1
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=json.dumps(
                                {
                                    "score": 99 if self.calls == 1 else 3,
                                    "confidence": 0.5,
                                    "justification": "x",
                                }
                            ),
                            usage_metadata={
                                "input_tokens": 10,
                                "output_tokens": 5,
                                "total_tokens": 15,
                            },
                        )
                    )
                ]
            )

        def with_structured_output(self, schema, **kwargs):
            return self | RunnableLambda(
                lambda raw: {
                    "raw": raw,
                    "parsed": schema.model_validate_json(raw.content),
                    "parsing_error": None,
                }
            )

    legacy = Rubric(
        rubric_id="retry",
        assignment_type="essay",
        source_dependent=False,
        prompt="Explain",
        scale=Scale(0, 4),
        rubric_guidelines="Grade it",
    )
    model = OutOfRangeThenValid()
    sample = HumanAssessment(StudentSubmission("retry", "An essay"), 3, legacy.holistic_criterion())
    with patch("abaes_eval_fwk.core.agents.base_agent.agent_backoff", return_value=0):
        result = HolisticGradingAgent().grade(legacy, sample, model, reference_examples=[])
    usage = result.statistics.token_usage
    assert model.calls == 2
    assert result.statistics.retry_error
    assert (usage.prompt_tokens, usage.completion_tokens) == (
        20,
        10,
    ), "both consumed responses must be reported, not only the successful one"


def test_a_first_attempt_success_keeps_its_original_counts(tmp_path: Path) -> None:
    """F7: accounting a call in total must not inflate an assessment that never retried."""
    from abaes_eval_fwk.core.agents.base_agent import TokenUsageCallback
    from abaes_eval_fwk.core.agents.holistic_agent import HolisticGradingAgent

    agent = HolisticGradingAgent()
    agent._begin_call()
    callback = TokenUsageCallback()
    callback.usage = callback.usage + TokenUsage(prompt_tokens=10, completion_tokens=5)
    agent._record_usage(callback)
    agent._record_usage(callback)
    assert agent._call_usage == TokenUsage(prompt_tokens=10, completion_tokens=5), (
        "one response must be counted once even if its callback is folded in twice"
    )


def test_a_calibration_cache_round_trips_a_leading_zero_label(tmp_path: Path) -> None:
    """F9: `001` is a valid dataset name, so reading it back as `1` must not fail the second load."""
    import csv

    from abaes_eval_fwk.config.dataset_config import DatasetConfig
    from abaes_eval_fwk.experiment.setup.setup_builder import build_setup

    rubric_path = tmp_path / "rubric.yaml"
    rubric_path.write_text(
        yaml.safe_dump(
            {
                "assignment_type": "essay",
                "prompt": "Explain",
                "score_ranges": {"overall": [0, 4]},
                "criteria": {},
            }
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "essays.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["essay_id", "set_id", "essay", "human_score"])
        writer.writeheader()
        for index in range(8):
            writer.writerow(
                {
                    "essay_id": f"e{index}",
                    "set_id": "s",
                    "essay": f"Essay {index}",
                    "human_score": index % 5,
                }
            )
    config = DatasetConfig(
        dataset_name="001",
        normalized_csv=csv_path,
        rubric_file=rubric_path,
        calibration_root=tmp_path / "calibration",
        cache_root=tmp_path / "cache",
    )
    first = build_setup(config, False, sample_size=3)
    second = build_setup(config, False, sample_size=3), "the cache must reload, not mismatch"
    assert [a.submission.essay_id for a in first.calibration_pool] == [
        a.submission.essay_id for a in second[0].calibration_pool
    ]


_REVIEW_CSV_HEADER = (
    "dataset,essay_id,criterion,source_model,final_agent_procedure,source_experiment_id,"
    "reviewer_model,source_score,source_justification,rubric_consistency,"
    "rubric_consistency_justification,evidence_grounding,evidence_grounding_justification,"
    "reasoning,is_admissible\n"
)


def _review_row(source_score: int, source_justification: str, *, experiment: str = "exp-a") -> str:
    return (
        f"ds,essay-1,holistic,model-x,adjudication_with_review,{experiment},reviewer-1,"
        f"{source_score},{source_justification},True,,True,,r,True\n"
    )


def test_a_reused_review_is_rejected_when_its_source_assessment_changes(tmp_path: Path) -> None:
    """Issue 03: the review is bound to the score and justification it judged, not just its labels."""
    from abaes_eval_fwk.core.models.experiment import FinalAgentProcedureName
    from abaes_eval_fwk.experiment.setup.reuse_loader import (
        ReviewCache,
        _source_assessment_digest,
        load_reused_reviews,
    )

    path = tmp_path / "reviews.csv"
    path.write_text(_REVIEW_CSV_HEADER + _review_row(3, "original justification"), encoding="utf-8")
    loaded = load_reused_reviews([str(path)], FinalAgentProcedureName.ADJUDICATION_WITH_REVIEW)
    cache = ReviewCache(
        procedure=FinalAgentProcedureName.ADJUDICATION_WITH_REVIEW,
        writer=None,
        _loaded=loaded,
    )

    def lookup(score: int, justification: str) -> dict | None:
        return cache.get(
            "ds",
            "essay-1",
            "holistic",
            "model-x",
            "exp-a",
            "reviewer-1",
            _source_assessment_digest(score, justification),
        )

    assert lookup(3, "original justification") is not None
    assert lookup(3, "edited justification") is None
    assert lookup(4, "original justification") is None


def test_contradictory_duplicate_reviews_fail_before_grading(tmp_path: Path) -> None:
    """Issue 03: two verdicts for one assessment identity are an error, not a file-order choice."""
    from abaes_eval_fwk.core.models.experiment import FinalAgentProcedureName
    from abaes_eval_fwk.experiment.setup.reuse_loader import load_reused_reviews

    path = tmp_path / "reviews.csv"
    path.write_text(
        _REVIEW_CSV_HEADER
        + _review_row(3, "same")
        + _review_row(3, "same").replace("True,,True", "False,,False"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Conflicting duplicate review"):
        load_reused_reviews([str(path)], FinalAgentProcedureName.ADJUDICATION_WITH_REVIEW)


def test_a_review_from_another_source_experiment_is_not_replayed(tmp_path: Path) -> None:
    """Issue 03: an identical assessment labelled with a different source experiment is a cache miss."""
    from abaes_eval_fwk.core.models.experiment import FinalAgentProcedureName
    from abaes_eval_fwk.experiment.setup.reuse_loader import (
        ReviewCache,
        _source_assessment_digest,
        load_reused_reviews,
    )

    path = tmp_path / "reviews.csv"
    path.write_text(
        _REVIEW_CSV_HEADER + _review_row(3, "same", experiment="exp-a"), encoding="utf-8"
    )
    cache = ReviewCache(
        procedure=FinalAgentProcedureName.ADJUDICATION_WITH_REVIEW,
        writer=None,
        _loaded=load_reused_reviews([str(path)], FinalAgentProcedureName.ADJUDICATION_WITH_REVIEW),
    )
    digest = _source_assessment_digest(3, "same")
    assert cache.get("ds", "essay-1", "holistic", "model-x", "exp-b", "reviewer-1", digest) is None
