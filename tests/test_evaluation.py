"""Regression checks for scoring targets, aggregation, and paired comparisons."""

import csv
from types import SimpleNamespace

import pytest

from abaes_eval_fwk.core.models.assessment import Dataset, HumanAssessment, StudentSubmission
from abaes_eval_fwk.core.models.rubric import CriteriaAggregation, Criterion, Rubric, Scale
from abaes_eval_fwk.experiment.eval.baseline.build_pool_average_baseline import (
    build_pool_average_baseline,
)
from abaes_eval_fwk.experiment.eval.metrics.compute_metrics import (
    MetricsInput,
    MetricsSet,
    TotalMetrics,
    process_csv,
    resolve_metrics,
)
from abaes_eval_fwk.experiment.eval.metrics.wilcoxon import comparison_rows, run_wilcoxon


def rubric(label="Custom", *, sum_scores=False):
    return Rubric(
        label,
        "essay",
        False,
        "Write an essay",
        Scale(0, 6) if sum_scores else Scale(1, 5),
        "Guidelines",
        [Criterion("ideas", []), Criterion("style", [])],
        criteria_scale=Scale(0, 3) if sum_scores else None,
        criteria_aggregation=CriteriaAggregation.SUM if sum_scores else CriteriaAggregation.MEAN,
    )


def humans(label, scores):
    return Dataset(
        SimpleNamespace(label=label),
        [
            HumanAssessment(StudentSubmission(str(i), "Essay"), score, Criterion("holistic", []))
            for i, score in enumerate(scores)
        ],
    )


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_id",
                "dataset",
                "essay_id",
                "criterion",
                "human_score",
                "pred_score_raw",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def row(i, criterion, pred, *, human=1, label="Custom", run="test"):
    return dict(
        experiment_id=run,
        dataset=label,
        essay_id=str(i),
        criterion=criterion,
        human_score=human,
        pred_score_raw=pred,
    )


def total(name, ids, preds, scores, scale=Scale(1, 5)):
    group = MetricsSet(
        "Custom",
        [MetricsInput(str(i), p, h, None) for i, p, h in zip(ids, preds, scores, strict=True)],
    )
    resolve_metrics(group, scale)
    return TotalMetrics(name, [group])


def test_ellipse_uses_loaded_rater_target_not_analytic_mean(tmp_path):
    label = "ELLIPSE"
    path = write_rows(
        tmp_path / "scores.csv", [row(0, "ideas", 4, label=label), row(0, "style", 4, label=label)]
    )
    _, metrics = process_csv(path, [rubric(label)], [humans(label, [4])], tmp_path / "out")
    assert metrics.metrics_sets[0].inputs[0].human == 4
    assert metrics.avg_nmae == 0


@pytest.mark.parametrize("prediction", [0.6, 5.4])
@pytest.mark.parametrize("criteria", [["holistic"], ["ideas", "style"]])
def test_metrics_reject_raw_predictions_outside_scale_before_rounding(
    tmp_path, prediction, criteria
):
    path = write_rows(
        tmp_path / "scores.csv", [row(0, criterion, prediction) for criterion in criteria]
    )
    with pytest.raises(ValueError, match="outside"):
        process_csv(path, [rubric()], [humans("Custom", [1])], tmp_path / "out")


@pytest.mark.parametrize("prediction", [0.6, 5.4])
@pytest.mark.parametrize("criteria", [["holistic"], ["ideas", "style"]])
def test_baseline_rejects_raw_predictions_outside_scale_before_rounding(
    tmp_path, prediction, criteria
):
    path = write_rows(
        tmp_path / "scores.csv", [row(0, criterion, prediction) for criterion in criteria]
    )
    with pytest.raises(ValueError, match="Out-of-range"):
        build_pool_average_baseline(
            [path], [rubric()], [humans("Custom", [1])], tmp_path / "baseline.csv"
        )


@pytest.mark.parametrize("criteria", [["holistic"], ["ideas", "style"]])
def test_valid_fractional_predictions_use_half_up_rounding(tmp_path, criteria):
    path = write_rows(
        tmp_path / "scores.csv", [row(0, criterion, 2.5, human=3) for criterion in criteria]
    )
    _, metrics = process_csv(path, [rubric()], [humans("Custom", [3])], tmp_path / "out")
    assert metrics.metrics_sets[0].inputs[0].pred == 3
    baseline = build_pool_average_baseline(
        [path], [rubric()], [humans("Custom", [3])], tmp_path / "baseline.csv"
    )
    _, pooled = process_csv(baseline, [rubric()], [humans("Custom", [3])], tmp_path / "out")
    assert pooled.metrics_sets[0].inputs[0].pred == 3


@pytest.mark.parametrize("prediction", [float("nan"), float("inf"), float("-inf")])
def test_metrics_and_baselines_reject_nonfinite_predictions(tmp_path, prediction):
    path = write_rows(tmp_path / "scores.csv", [row(0, "holistic", prediction)])
    with pytest.raises(ValueError, match="Invalid numeric"):
        process_csv(path, [rubric()], [humans("Custom", [1])], tmp_path / "out")
    with pytest.raises(ValueError, match="Invalid numeric"):
        build_pool_average_baseline(
            [path], [rubric()], [humans("Custom", [1])], tmp_path / "baseline.csv"
        )


def test_missing_and_duplicate_criteria_are_rejected(tmp_path):
    path = write_rows(tmp_path / "scores.csv", [row(0, "ideas", 3)])
    with pytest.raises(ValueError, match="each criterion"):
        process_csv(path, [rubric()], [humans("Custom", [3])], tmp_path / "out")
    write_rows(path, [row(0, "ideas", 3), row(0, "ideas", 3)])
    with pytest.raises(ValueError, match="Duplicate"):
        process_csv(path, [rubric()], [humans("Custom", [3])], tmp_path / "out")


def test_wilcoxon_small_samples_and_all_equal_errors():
    identical = run_wilcoxon([0, 1], [0, 1])
    assert identical.p_value == 1 and identical.effect_size == 0
    improved = run_wilcoxon([1] * 6, [0] * 6)
    assert improved.p_value == pytest.approx(0.03125)
    assert improved.effect_size == 1
    with pytest.raises(ValueError, match="equally sized"):
        run_wilcoxon([1], [1, 2])


def test_pairing_requires_ids_and_human_targets_match():
    first = total("first", [0, 1], [1, 2], [1, 2])
    second = total("second", [1, 2], [2, 3], [2, 3])
    with pytest.raises(ValueError, match="essay IDs differ"):
        comparison_rows(first, second)
    paired = comparison_rows(first, second, allow_partial=True)[0]
    assert paired["N Paired"] == 1
    assert paired["N First Only"] == paired["N Second Only"] == 1
    conflicting = total("second", [0, 1], [1, 2], [1, 3])
    with pytest.raises(ValueError, match="human scores differ"):
        comparison_rows(first, conflicting)


def test_wilcoxon_preserves_equal_rational_error_ties():
    result = run_wilcoxon([10 / 3, 20 / 3, 0], [0, 10 / 3, 10 / 3])
    assert result.effect_size == pytest.approx(1 / 3)


def test_baseline_aggregates_each_source_before_pooling(tmp_path):
    """Baseline C (paper, Final Agent Deliberation): each source's criteria are aggregated into
    an overall score first, and the mean of those overall scores is rounded half-up. Sources
    totalling 0 and 2 give 1; pooling each criterion first would give round(0.5) * 2 = 2."""
    r = rubric(sum_scores=True)
    first = write_rows(tmp_path / "first.csv", [row(0, "ideas", 0), row(0, "style", 0)])
    second = write_rows(tmp_path / "second.csv", [row(0, "ideas", 1), row(0, "style", 1)])
    output = build_pool_average_baseline(
        [first, second], [r], [humans("Custom", [2])], tmp_path / "baseline.csv"
    )
    _, metrics = process_csv(output, [r], [humans("Custom", [2])], tmp_path / "out")
    assert metrics.metrics_sets[0].inputs[0].pred == 1
