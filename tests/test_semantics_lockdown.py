"""Pin the scientific constants and formulas that determine published result values.
The extensibility work wraps these functions, and these assertions protect their behavior."""

from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import replace

import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

from abaes_eval_fwk.config.dataset_config import DatasetConfig
from abaes_eval_fwk.core.agents.final_agent import (
    NAPD_REFERENCE_WIDTH,
    AdjudicationFinalHolisticAgent,
    AdjudicationProcedure,
    Reviewer,
    normalized_average_pairwise_disagreement,
)
from abaes_eval_fwk.core.models.assessment import (
    Dataset,
    GradingRecord,
    HumanAssessment,
    LLMAssessment,
    PredictedScore,
    Review,
    StudentSubmission,
    failures_fieldnames,
    results_fieldnames,
    review_fieldnames,
)
from abaes_eval_fwk.core.models.rubric import CriteriaAggregation, Rubric, Scale
from abaes_eval_fwk.experiment.eval.aggregate_criteria import aggregate_criteria
from abaes_eval_fwk.experiment.eval.baseline.build_pool_average_baseline import (
    build_pool_average_baseline,
)
from abaes_eval_fwk.experiment.eval.metrics.compute_metrics import (
    MetricsInput,
    MetricsSet,
    resolve_metrics,
)
from abaes_eval_fwk.experiment.setup.calibration.calibration_generator import (
    _MAX_CALIBRATION_FRACTION,
    _rng_for,
)
from abaes_eval_fwk.experiment.setup.reuse_loader import ReviewCache
from abaes_eval_fwk.experiment.setup.sample.data_sampler import stratified_quantile_rank_sample
from abaes_eval_fwk.utils.rounding import round_half_up


def _record(criterion, raw_score: float, essay_id: str = "essay-1") -> GradingRecord:
    return GradingRecord(
        human_assessment=HumanAssessment(
            submission=StudentSubmission(essay_id=essay_id, text="A student essay."),
            human_score=3,
            criterion=criterion,
        ),
        llm_assessment=LLMAssessment(
            score=PredictedScore(raw_score), justification="Because.", confidence=0.5
        ),
    )


def test_metric_errors_scale_by_ten_over_scale_width() -> None:
    """NMAE/NRMSE rescale each error by 10 over the scale width, so 0-5 and 0-10 agree."""
    errors = [(2, 3), (4, 5)]
    for scale, expected in ((Scale(0, 10), 1.0), (Scale(0, 5), 2.0), (Scale(1, 6), 2.0)):
        group = MetricsSet(
            "s", [MetricsInput(str(i), p, h, None) for i, (p, h) in enumerate(errors)], scale=scale
        )
        resolve_metrics(group, scale)
        assert group.resolved.nmae == pytest.approx(expected)
        assert group.resolved.nrmse == pytest.approx(expected)


def test_qwk_uses_quadratic_weights_over_every_label_in_the_scale() -> None:
    """QWK is computed against the full scale's labels, not just the observed ones."""
    humans = [1, 2, 3, 4, 5, 0, 1, 2]
    preds = [1, 2, 4, 4, 5, 0, 2, 1]
    scale = Scale(0, 5)
    group = MetricsSet(
        "s",
        [MetricsInput(str(i), p, h, None) for i, (p, h) in enumerate(zip(preds, humans))],
        scale=scale,
    )
    resolve_metrics(group, scale)
    assert group.resolved.qwk == pytest.approx(
        cohen_kappa_score(
            humans, preds, weights="quadratic", labels=list(range(scale.min, scale.max + 1))
        )
    )


def test_disagreement_threshold_is_one_point_of_the_napd_reference_width() -> None:
    """The default 1.0 threshold and the 5-point NAPD reference width together give 20% of a scale."""
    assert NAPD_REFERENCE_WIDTH == 5
    assert AdjudicationProcedure._DISAGREEMENT_THRESHOLD_POINTS == 1.0
    agent = AdjudicationFinalHolisticAgent()
    assert agent._threshold_points(Scale(0, 10)) == 2.0
    assert agent._threshold_points(Scale(0, 5)) == 1.0


def test_napd_is_mean_pairwise_gap_rescaled_to_the_reference_width() -> None:
    """NAPD averages every unique pair once, so it reads as whole points on any rubric width."""
    assert normalized_average_pairwise_disagreement([3, 3, 3], Scale(0, 5)) == 0.0
    assert normalized_average_pairwise_disagreement([0, 5], Scale(0, 5)) == 5.0
    assert normalized_average_pairwise_disagreement([0, 6], Scale(0, 6)) == 5.0
    assert normalized_average_pairwise_disagreement([2, 3, 5], Scale(0, 5)) == pytest.approx(
        (1 + 3 + 2) / 3 / 5 * 5
    )


def test_calibration_draw_is_capped_at_a_quarter_and_seeded_per_dataset_and_score() -> None:
    """A rare score level keeps most of its essays for evaluation, and regenerating a pool is stable."""
    assert _MAX_CALIBRATION_FRACTION == 0.25
    expected = int(hashlib.sha256(b"42|ASAP AES Set 1|3").hexdigest()[:12], 16)
    assert _rng_for("ASAP AES Set 1", 3, 42).random() == random.Random(expected).random()
    assert _rng_for("ASAP AES Set 1", 4, 42).random() != _rng_for("ASAP AES Set 1", 3, 42).random()


def test_review_cache_key_is_dataset_essay_criterion_and_source_model() -> None:
    """A cached verdict is only reusable by the same seven facts, in that order."""
    key = "tuple[str, str, str, str, str, str, str]"
    assert ReviewCache.__dataclass_fields__["_loaded"].type == f"dict[{key}, dict]"
    assert ReviewCache.__dataclass_fields__["_written"].type == f"set[{key}]"


def test_review_is_admissible_only_when_both_criteria_pass() -> None:
    """is_admissible is the conjunction of the two stored flags, and Reviewer re-checks them."""
    assert Reviewer.admissibility_criteria == ("rubric_consistency", "evidence_grounding")
    assert Review(rubric_consistency=True, evidence_grounding=True).is_admissible
    assert not Review(rubric_consistency=True, evidence_grounding=False).is_admissible
    assert not Review(rubric_consistency=False, evidence_grounding=True).is_admissible
    reviewer = object.__new__(Reviewer)
    assert reviewer.is_admissible({"rubric_consistency": True, "evidence_grounding": True})
    assert not reviewer.is_admissible({"rubric_consistency": True, "evidence_grounding": False})
    assert not reviewer.is_admissible({"rubric_consistency": True})


def test_csv_headers_are_frozen() -> None:
    """Result, review, and failure columns are published in order; changing one breaks every CSV."""
    assert results_fieldnames() == [
        "experiment_id",
        "dataset",
        "essay_id",
        "essay_text",
        "model",
        "criterion",
        "human_score",
        "pred_score_raw",
        "pred_score_rounded",
        "confidence",
        "justification",
        "reasoning",
        "examples",
        "example_cap",
        "start_time",
        "total_time_seconds",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "retry_error",
        "branch",
        "pool_average_score",
    ]
    assert review_fieldnames(with_identity=True) == [
        "dataset",
        "essay_id",
        "experiment_id",
        "final_agent_procedure",
        "reviewer_model",
        "criterion",
        "source_experiment_id",
        "source_model",
        "rubric_consistency",
        "rubric_consistency_justification",
        "evidence_grounding",
        "evidence_grounding_justification",
        "is_admissible",
        "reasoning",
        "start_time",
        "total_time_seconds",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "retry_error",
    ]
    assert review_fieldnames() == [
        field
        for field in review_fieldnames(with_identity=True)
        if field not in {"experiment_id", "source_experiment_id", "reviewer_model"}
    ]
    assert failures_fieldnames() == ["essay_id", "dataset", "criterion", "human_score", "error"]


@pytest.mark.parametrize(
    ("value", "decimals", "expected"), [(2.5, 0, 3), (-2.5, 0, -3), (3.5, 0, 4), (1.235, 2, 1.24)]
)
def test_round_half_up_breaks_ties_away_from_zero(value: float, decimals: int, expected) -> None:
    """Rounding is half-up on the decimal string, not on the binary float — 2.5 is 3, never 2."""
    assert round_half_up(value, decimals) == expected
    assert PredictedScore(value).rounded_score == round_half_up(value)


def test_criteria_are_rounded_before_they_are_combined(rubric: Rubric) -> None:
    """Aggregation rounds each criterion first, so 1.5 and 2.5 average to 3 rather than 2."""
    records = [_record(c, raw) for c, raw in zip(rubric.criteria, [1.5, 2.5], strict=True)]
    assert aggregate_criteria(rubric, records) == 3


def test_pool_average_rounds_each_source_before_averaging(rubric: Rubric, tmp_path) -> None:
    """The baseline pools rounded source scores, matching how a final agent resolves them."""
    sources = []
    for name, raw in (("a", 1.5), ("b", 2.5)):
        path = tmp_path / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["dataset", "essay_id", "human_score", "pred_score_raw"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "dataset": rubric.rubric_id,
                    "essay_id": "essay-1",
                    "human_score": 3,
                    "pred_score_raw": raw,
                }
            )
        sources.append(path)
    datasets = [
        Dataset(
            dataset_config=DatasetConfig(
                dataset_name=rubric.rubric_id,
                normalized_csv=tmp_path / "normalized.csv",
                rubric_file=tmp_path / "rubric.yaml",
            ),
            human_assessments=[
                HumanAssessment(
                    StudentSubmission("essay-1", "A student essay."), 3, rubric.holistic_criterion()
                )
            ],
        )
    ]
    out = build_pool_average_baseline(sources, [rubric], datasets, tmp_path / "pool.csv")
    assert pd.read_csv(out)["pred_score_rounded"].tolist() == [3]


def test_evaluation_sampling_is_rank_spaced_and_consumes_no_randomness() -> None:
    """Quantile ranks include the extremes and are stable, so a sample_size change is the only lever."""
    frame = pd.DataFrame(
        {
            "set_id": ["a"] * 5,
            "essay_id": [f"e{i}" for i in range(5)],
            "human_score": [1, 2, 3, 4, 5],
            "essay": ["text"] * 5,
        }
    )
    picked = stratified_quantile_rank_sample(frame, 3)
    assert picked["human_score"].tolist() == [1, 3, 5]
    assert (
        picked["essay_id"].tolist()
        == stratified_quantile_rank_sample(frame, 3)["essay_id"].tolist()
    )
    assert stratified_quantile_rank_sample(frame, 5)["human_score"].tolist() == [1, 2, 3, 4, 5]
    assert stratified_quantile_rank_sample(frame, 1)["human_score"].tolist() == [3]
    with pytest.raises(ValueError, match="No evaluation essays"):
        stratified_quantile_rank_sample(frame.iloc[:0], 1)


def test_aggregation_mode_is_sum_only_when_the_rubric_asks_for_it(rubric: Rubric) -> None:
    """Sets 1/4/ELLIPSE average criteria onto one scale; sets 7/8 sum independent traits."""
    records = [_record(c, 1.5) for c in rubric.criteria]
    mean_rubric = replace(rubric, criteria_aggregation=CriteriaAggregation.MEAN)
    sum_rubric = replace(rubric, criteria_aggregation=CriteriaAggregation.SUM)
    assert aggregate_criteria(mean_rubric, records) == 2
    assert aggregate_criteria(sum_rubric, records) == 4
