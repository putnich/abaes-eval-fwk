"""Adapts config-based evaluation interface to underlying metrics computation functions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from abaes_eval_fwk.config.dataset_config import DatasetConfig
from abaes_eval_fwk.config.metrics_config import MetricsConfig
from abaes_eval_fwk.core.models.assessment import Dataset, HumanAssessment, StudentSubmission
from abaes_eval_fwk.experiment.eval.metrics.compute_metrics import (
    TotalMetrics,
    grading_records,
    process_csv,
)
from abaes_eval_fwk.experiment.eval.metrics.wilcoxon import save_comparison
from abaes_eval_fwk.experiment.setup.dataset.dataset_loader import load_dataset
from abaes_eval_fwk.experiment.setup.rubric.rubric_loader import load_rubric
from abaes_eval_fwk.utils.rounding import round_half_up


def evaluate(
    csv_path: Path,
    dataset_configs: Sequence[DatasetConfig],
    output_dir: Path,
    metrics: MetricsConfig | None = None,
) -> tuple[Path, TotalMetrics]:
    """Evaluate one run against its configured rubrics and official overall human scores.

    `metrics` selects the reported columns and supplies any `metrics.custom` callables; without
    it the five metrics the framework has always reported are saved.
    """
    csv_path, output_dir = Path(csv_path), Path(output_dir)
    needed = set(grading_records(csv_path))
    configured = {config.label: config for config in dataset_configs}
    if missing := needed - configured.keys():
        raise ValueError(f"Dataset configurations missing for: {sorted(missing)}")
    rubrics, datasets = [], []
    for label in sorted(needed):
        config = configured[label]
        rubric = load_rubric(config.rubric_path, label)
        frame = load_dataset(config)
        assessments = [
            HumanAssessment(
                StudentSubmission(str(row.essay_id), str(row.essay)),
                round_half_up(row.human_score),
                rubric.holistic_criterion(),
            )
            for row in frame.itertuples()
        ]
        rubrics.append(rubric)
        datasets.append(Dataset(config, assessments))
    return process_csv(csv_path, rubrics, datasets, output_dir, metrics)


def compare(
    first: TotalMetrics,
    second: TotalMetrics,
    output_path: Path,
    *,
    config: MetricsConfig | None = None,
    allow_partial: bool | None = None,
    alpha: float | None = None,
) -> Path:
    """Compare two evaluated runs with explicit pairing and per-dataset Wilcoxon tests.

    Uses configuration from MetricsConfig if provided, otherwise uses explicit parameters
    or falls back to defaults. Explicit parameters override config values for CLI usage.
    """
    # Resolve configuration precedence: explicit CLI args > MetricsConfig > defaults
    if config is not None:
        comparison_alpha = alpha if alpha is not None else config.comparison.alpha
        comparison_allow_partial = (
            allow_partial if allow_partial is not None else config.comparison.allow_partial
        )
        wilcoxon_alternative = config.wilcoxon.alternative
        wilcoxon_zero_method = config.wilcoxon.zero_method
        wilcoxon_method = config.wilcoxon.method
    else:
        comparison_alpha = alpha if alpha is not None else 0.05
        comparison_allow_partial = allow_partial if allow_partial is not None else False
        wilcoxon_alternative = "two-sided"
        wilcoxon_zero_method = "wilcox"
        wilcoxon_method = "auto"

    return save_comparison(
        first,
        second,
        Path(output_path),
        allow_partial=comparison_allow_partial,
        alpha=comparison_alpha,
        alternative=wilcoxon_alternative,
        zero_method=wilcoxon_zero_method,
        method=wilcoxon_method,
    )


def pool_average(
    source_csvs: Sequence[Path],
    dataset_configs: Sequence[DatasetConfig],
    output_path: Path,
    *,
    experiment_id: str = "pool_average",
) -> Path:
    """Pool matched prior scores against configured rubrics and official human targets."""
    from abaes_eval_fwk.experiment.eval.baseline.build_pool_average_baseline import (
        build_pool_average_baseline,
    )

    needed = set(grading_records(Path(source_csvs[0]))) if source_csvs else set()
    configured = {config.label: config for config in dataset_configs}
    if missing := needed - configured.keys():
        raise ValueError(f"Dataset configurations missing for: {sorted(missing)}")
    rubrics, datasets = [], []
    for label in sorted(needed):
        config = configured[label]
        rubric = load_rubric(config.rubric_path, label)
        frame = load_dataset(config)
        assessments = [
            HumanAssessment(
                StudentSubmission(str(row.essay_id), str(row.essay)),
                round_half_up(row.human_score),
                rubric.holistic_criterion(),
            )
            for row in frame.itertuples()
        ]
        rubrics.append(rubric)
        datasets.append(Dataset(config, assessments))
    return build_pool_average_baseline(
        source_csvs, rubrics, datasets, output_path, experiment_id=experiment_id
    )
