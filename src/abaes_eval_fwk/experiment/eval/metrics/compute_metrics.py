"""Evaluate grading CSVs against rubric scales and official overall human targets."""

from __future__ import annotations

import csv
import itertools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final, NamedTuple

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

from abaes_eval_fwk.config.dataset_config import DatasetNames, get_dataset_config
from abaes_eval_fwk.config.metrics_config import MetricsConfig
from abaes_eval_fwk.core.models.assessment import (
    Dataset,
    GradingRecord,
    HumanAssessment,
    LLMAssessment,
    PredictedScore,
    Statistics,
    StudentSubmission,
    TokenUsage,
)
from abaes_eval_fwk.core.models.rubric import Criterion, Rubric, Scale
from abaes_eval_fwk.experiment.eval.aggregate_criteria import (
    aggregate_criteria,
    official_human_score,
)
from abaes_eval_fwk.experiment.setup.rubric.rubric_loader import load_rubric
from abaes_eval_fwk.utils.extensions import resolve_callable


@cache
def _rubric_for(name: DatasetNames) -> Rubric:
    return load_rubric(get_dataset_config(name).rubric_path, str(name))


def score_range(name: DatasetNames) -> Scale:
    return _rubric_for(name).total_score_range()


def criterion_score_range(name: DatasetNames) -> Scale:
    return _rubric_for(name).get_criterion_scale()


def sig_stars(p: float | None) -> str:
    if p is None:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def normalize(score: float, lo: int, hi: int) -> float:
    """Map a score to the shared 0–10 reference scale."""
    if hi <= lo:
        raise ValueError("Score range must have positive width")
    return (score - lo) / (hi - lo) * 10


def _errors(preds: list[int], humans: list[int]) -> list[int]:
    return [p - h for p, h in zip(preds, humans, strict=True)]


def _width(scale: Scale) -> float:
    """The 0-10 reference factor a normalized error metric scales by."""
    return 10 / (scale.max - scale.min)


def mae(*, preds: list[int], humans: list[int], scale: Scale) -> float:
    """Mean absolute error in the dataset's own score points."""
    return sum(abs(e) for e in _errors(preds, humans)) / len(preds)


def rmse(*, preds: list[int], humans: list[int], scale: Scale) -> float:
    """Root mean squared error in the dataset's own score points."""
    return math.sqrt(sum(e**2 for e in _errors(preds, humans)) / len(preds))


def nmae(*, preds: list[int], humans: list[int], scale: Scale) -> float:
    """Mean absolute error on the shared 0-10 reference scale."""
    return mae(preds=preds, humans=humans, scale=scale) * _width(scale)


def nrmse(*, preds: list[int], humans: list[int], scale: Scale) -> float:
    """Root mean squared error on the shared 0-10 reference scale."""
    return rmse(preds=preds, humans=humans, scale=scale) * _width(scale)


def pearson(*, preds: list[int], humans: list[int], scale: Scale) -> float:
    """Pearson correlation, NaN when either side is constant and it is undefined."""
    if len(preds) < 2 or len(set(preds)) < 2 or len(set(humans)) < 2:
        return math.nan
    return float(pearsonr(preds, humans).statistic)


def spearman(*, preds: list[int], humans: list[int], scale: Scale) -> float:
    """Spearman rank correlation, NaN when either side is constant and it is undefined."""
    if len(preds) < 2 or len(set(preds)) < 2 or len(set(humans)) < 2:
        return math.nan
    return float(spearmanr(preds, humans).statistic)


def qwk(*, preds: list[int], humans: list[int], scale: Scale) -> float:
    """Quadratic weighted kappa over every label in the scale, not just the observed ones."""
    if len(preds) < 2 or len(set(preds + humans)) < 2:
        return math.nan
    return float(
        cohen_kappa_score(
            humans, preds, weights="quadratic", labels=list(range(scale.min, scale.max + 1))
        )
    )


# Every metric the framework computes, called with the paired scores and the dataset's scale. A
# reader's `metrics.custom` callable has exactly this signature, so a contributed metric is not a
# lesser kind of metric than a built-in one.
BUILTIN_METRICS: Final[dict[str, Callable[..., float]]] = {
    "nmae": nmae,
    "nrmse": nrmse,
    "pearson": pearson,
    "spearman": spearman,
    "qwk": qwk,
    "mae": mae,
    "rmse": rmse,
}
# Output column order; also the order `metrics.compute` flags are read in. Custom metrics follow,
# in configured order, so enabling one never moves an existing column.
_OUTPUT_ORDER: Final[tuple[str, ...]] = (
    "nmae",
    "nrmse",
    "pearson",
    "spearman",
    "qwk",
    "mae",
    "rmse",
)
_HEADERS: Final[dict[str, str]] = {
    "nmae": "NMAE",
    "nrmse": "NRMSE",
    "pearson": "Pearson",
    "spearman": "Spearman",
    "qwk": "QWK",
    "mae": "MAE",
    "rmse": "RMSE",
}
# `r` and `rho` name the two correlations throughout the engine and in saved analyses; the
# registry keys match the configuration instead (`pearson`, `spearman`), so both reach the value.
_ALIASES: Final[dict[str, str]] = {"r": "pearson", "rho": "spearman"}


class Metrics(Mapping):
    """One dataset's computed metrics, keyed by metric name.

    A mapping rather than a fixed set of fields: a reader's custom metric is then an ordinary
    entry instead of something this class would have to grow a column for. Attribute access
    (`metrics.nmae`, `metrics.r`) is kept because the engine, the saved CSVs, and the comparison
    report all name metrics that way.
    """

    def __init__(self, values: Mapping[str, float]) -> None:
        self._values = dict(values)

    def __getitem__(self, name: str) -> float:
        return self._values[_ALIASES.get(name, name)]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, name: str) -> float:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__!r} has no metric {name!r}; computed: {sorted(self._values)}"
            ) from None

    def __repr__(self) -> str:
        return f"Metrics({self._values!r})"


class MetricsInput(NamedTuple):
    essay_id: str
    pred: int
    human: int
    completion_tokens: int | None


@dataclass
class MetricsSet:
    set_id: str
    inputs: list[MetricsInput]
    metrics: Metrics | None = None
    avg_completion_tokens: float | None = None
    scale: Scale | None = None

    @property
    def resolved(self) -> Metrics:
        if self.metrics is None:
            raise ValueError(f"{self.set_id} metrics have not been computed")
        return self.metrics


@dataclass
class TotalMetrics:
    experiment_id: str
    metrics_sets: list[MetricsSet]

    def _average(self, field: str) -> float:
        values = [m.resolved[field] for m in self.metrics_sets]
        return sum(values) / len(values) if values else math.nan

    @property
    def avg_nmae(self) -> float:
        return self._average("nmae")

    @property
    def avg_nrmse(self) -> float:
        return self._average("nrmse")

    @property
    def avg_r(self) -> float:
        return self._average("r")

    @property
    def avg_rho(self) -> float:
        return self._average("rho")

    @property
    def avg_qwk(self) -> float:
        return self._average("qwk")

    @property
    def avg_completion_tokens(self) -> float | None:
        values = [
            m.avg_completion_tokens
            for m in self.metrics_sets
            if m.avg_completion_tokens is not None
        ]
        return sum(values) / len(values) if values else None


def _metric_value(name: str, value: object) -> float:
    """Accept a metric's result, allowing NaN for 'undefined here' but nothing unusable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Metric {name!r} must return a number, got {type(value).__name__}")
    value = float(value)
    if math.isinf(value):
        raise ValueError(f"Metric {name!r} returned an infinite value")
    return value


def resolve_metrics(
    metrics_set: MetricsSet, scale: Scale, custom: Mapping[str, str] | None = None
) -> None:
    """Compute every metric for one dataset; undefined correlations and kappa are NaN.

    All built-ins are computed whatever `metrics.compute` says: those flags select what reaches
    the saved table, and computing a mean absolute error that nobody asked to print costs nothing.
    `custom` maps a reader's metric name to the `module:function` that computes it.
    """
    n = len(metrics_set.inputs)
    if not n:
        raise ValueError(f"{metrics_set.set_id} has no scores")
    if scale.max <= scale.min:
        raise ValueError("Score range must have positive width")
    preds = [i.pred for i in metrics_set.inputs]
    humans = [i.human for i in metrics_set.inputs]
    if any(not math.isfinite(v) or v < scale.min or v > scale.max for v in preds + humans):
        raise ValueError(f"{metrics_set.set_id} scores outside [{scale.min}, {scale.max}]")
    if len({i.essay_id for i in metrics_set.inputs}) != n:
        raise ValueError(f"{metrics_set.set_id} contains duplicate essay IDs")
    functions: dict[str, Callable[..., float]] = dict(BUILTIN_METRICS)
    for name, reference in (custom or {}).items():
        functions[name] = resolve_callable(reference, f"metrics.custom.{name}")
    values = {
        name: _metric_value(name, function(preds=preds, humans=humans, scale=scale))
        for name, function in functions.items()
    }
    metrics_set.metrics = Metrics(values)
    metrics_set.scale = scale
    tokens = [i.completion_tokens for i in metrics_set.inputs if i.completion_tokens is not None]
    metrics_set.avg_completion_tokens = sum(tokens) / len(tokens) if tokens else None


def _cell(value: float | None) -> str | float:
    return "" if value is None or not math.isfinite(value) else value


def reported_metrics(config: MetricsConfig | None = None) -> list[str]:
    """The metric names the saved table carries, in column order.

    With no configuration this is the five metrics the framework has always reported, so an
    analysis written before `metrics.compute` was honored keeps its exact columns.
    """
    if config is None:
        return ["nmae", "nrmse", "pearson", "spearman", "qwk"]
    enabled = [name for name in _OUTPUT_ORDER if getattr(config.compute, name)]
    return enabled + list(config.custom)


def save_metric_csv(
    out_dir: Path, total_metrics: TotalMetrics, config: MetricsConfig | None = None
) -> tuple[Path, TotalMetrics]:
    """Save full-precision metrics and an unweighted mean across dataset metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(total_metrics.experiment_id).name}.csv"
    names = reported_metrics(config)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Dataset",
                "N",
                "Score Min",
                "Score Max",
                *[_HEADERS.get(name, name) for name in names],
                "Avg Completion Tokens",
            ]
        )
        for group in total_metrics.metrics_sets:
            m = group.resolved
            if group.scale is None:
                raise ValueError(f"{group.set_id} has no score scale")
            writer.writerow(
                [
                    str(group.set_id),
                    len(group.inputs),
                    group.scale.min,
                    group.scale.max,
                    *[_cell(m[name]) for name in names],
                    _cell(group.avg_completion_tokens),
                ]
            )
        writer.writerow(
            [
                "Average",
                "",
                "",
                "",
                *[_cell(total_metrics._average(name)) for name in names],
                _cell(total_metrics.avg_completion_tokens),
            ]
        )
    return out_path, total_metrics


def experiment_id_of(csv_path: Path) -> str:
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if frame.empty:
        raise ValueError(f"Results CSV is empty: {csv_path}")
    if "experiment_id" not in frame or not frame["experiment_id"].any():
        return csv_path.stem
    values = set(frame["experiment_id"])
    if len(values) != 1:
        raise ValueError("Results CSV must contain exactly one experiment_id")
    return values.pop()


def model_name_of(csv_path: Path) -> str:
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    values = set(frame.get("model", pd.Series([""])))
    if len(values) != 1:
        raise ValueError("Results CSV must contain exactly one model")
    return values.pop()


def grading_records(csv_path: Path) -> dict[str, list[GradingRecord]]:
    """Read both framework results and historical score-only baseline CSVs."""
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required = {"dataset", "essay_id", "human_score"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing results columns: {sorted(missing)}")
    score_column = next(
        (c for c in ("pred_score_raw", "pred_score_rounded", "pred_score") if c in frame), None
    )
    if score_column is None:
        raise ValueError("Results need pred_score_raw, pred_score_rounded, or pred_score")
    records_by_dataset: dict[str, list[GradingRecord]] = {}
    seen: set[tuple[str, str, str]] = set()
    for row in frame.to_dict("records"):
        dataset, essay_id = row["dataset"], row["essay_id"]
        criterion = row.get("criterion") or "holistic"
        key = dataset, essay_id, criterion
        if key in seen:
            raise ValueError(f"Duplicate result row: {key}")
        seen.add(key)
        human, prediction = float(row["human_score"]), float(row[score_column])
        if not math.isfinite(human) or not human.is_integer() or not math.isfinite(prediction):
            raise ValueError(f"Invalid numeric scores for {key}")
        usage = TokenUsage(
            int(row.get("prompt_tokens") or 0), int(row.get("completion_tokens") or 0)
        )
        record = GradingRecord(
            HumanAssessment(
                StudentSubmission(essay_id, row.get("essay_text", "")),
                int(human),
                Criterion(criterion, []),
            ),
            LLMAssessment(
                PredictedScore(prediction),
                row.get("justification", ""),
                float(row["confidence"]) if row.get("confidence") else None,
                reasoning=row.get("reasoning") or None,
                statistics=Statistics(token_usage=usage),
            ),
        )
        records_by_dataset.setdefault(dataset, []).append(record)
    return records_by_dataset


def metrics_input_init(
    csv_path: Path, rubrics: list[Rubric], datasets: list[Dataset]
) -> list[MetricsSet]:
    """Aggregate complete criterion predictions against official overall targets."""
    rubrics_by_label = {str(r.rubric_id): r for r in rubrics}
    sets = []
    for label, records in grading_records(csv_path).items():
        if label not in rubrics_by_label:
            raise ValueError(f"No rubric supplied for {label!r}")
        rubric = rubrics_by_label[label]
        group = MetricsSet(label, [], scale=rubric.total_score_range())
        records.sort(key=lambda r: r.human_assessment.essay_id)
        for essay_id, grouped in itertools.groupby(
            records, key=lambda r: r.human_assessment.essay_id
        ):
            rows = list(grouped)
            holistic = all(r.human_assessment.criterion.name == "holistic" for r in rows)
            official = official_human_score(datasets, label, essay_id)
            if official is None:
                raise ValueError(f"No official overall human score for {label}/{essay_id}")
            if holistic:
                if len(rows) != 1:
                    raise ValueError(f"Expected one holistic score for {label}/{essay_id}")
                raw_prediction = rows[0].llm_assessment.score.raw_score
                scale = rubric.total_score_range()
                if (
                    not math.isfinite(raw_prediction)
                    or not scale.min <= raw_prediction <= scale.max
                ):
                    raise ValueError(
                        f"Holistic prediction outside [{scale.min}, {scale.max}] "
                        f"for {label}/{essay_id}"
                    )
                pred = rows[0].llm_assessment.score.rounded_score
                human = rows[0].human_assessment.human_score
                if official is not None and official != human:
                    raise ValueError(
                        f"Results human score differs from official overall for {label}/{essay_id}"
                    )
            else:
                if official is None:
                    raise ValueError(f"No official overall human score for {label}/{essay_id}")
                pred, human = aggregate_criteria(rubric, rows), official
            tokens = sum(r.llm_assessment.statistics.token_usage.completion_tokens for r in rows)
            group.inputs.append(MetricsInput(essay_id, pred, human, tokens))
        sets.append(group)
    return sets


def process_csv(
    csv_path: Path,
    rubrics: list[Rubric],
    datasets: list[Dataset],
    out_root: Path,
    config: MetricsConfig | None = None,
) -> tuple[Path, TotalMetrics]:
    """Compute metrics for one results CSV using supplied rubrics and human datasets."""
    total = TotalMetrics(
        experiment_id_of(csv_path), metrics_input_init(csv_path, rubrics, datasets)
    )
    custom = config.custom if config else {}
    for group in total.metrics_sets:
        resolve_metrics(group, group.scale, custom)
    return save_metric_csv(out_root / csv_path.stem, total, config)
