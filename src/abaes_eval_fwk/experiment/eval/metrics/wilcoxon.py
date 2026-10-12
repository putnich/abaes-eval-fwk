"""Compare paired per-essay normalized absolute errors without changing their ties."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.stats import rankdata, wilcoxon

from abaes_eval_fwk.experiment.eval.metrics.compute_metrics import (
    TotalMetrics,
    score_range,
    sig_stars,
)


def rank_biserial(diffs: list[float]) -> float:
    """Return signed paired rank-biserial correlation, or zero for identical errors."""
    nonzero = [d for d in diffs if d != 0.0]
    if not nonzero:
        return 0.0
    ranks = rankdata([abs(d) for d in nonzero])
    return float(
        sum(rank if diff > 0 else -rank for rank, diff in zip(ranks, nonzero, strict=True))
        / sum(ranks)
    )


@dataclass
class WilcoxonResult:
    statistic: float | None = None
    p_value: float | None = None
    effect_size: float | None = None
    n_pairs: int = 0
    n_nonzero: int = 0


def run_wilcoxon(
    errors_first: list[float],
    errors_second: list[float],
    alternative: str = "two-sided",
    zero_method: str = "wilcox",
    method: str = "auto",
) -> WilcoxonResult:
    """Run the paired test with zero differences removed and stable numeric ties.

    Uses configured Wilcoxon test parameters: alternative (test direction), zero_method
    (how to handle zero differences), and method (auto selection of exact or approximation).
    """
    if len(errors_first) != len(errors_second) or not errors_first:
        raise ValueError("Wilcoxon requires equally sized, nonempty paired errors")
    if any(not math.isfinite(v) for v in errors_first + errors_second):
        raise ValueError("Wilcoxon errors must be finite")
    diffs = [round(a - b, 12) for a, b in zip(errors_first, errors_second, strict=True)]
    nonzero = sum(d != 0 for d in diffs)
    if not nonzero:
        return WilcoxonResult(0.0, 1.0, 0.0, len(diffs), 0)
    result = wilcoxon(diffs, alternative=alternative, zero_method=zero_method, method=method)
    return WilcoxonResult(
        float(result.statistic), float(result.pvalue), rank_biserial(diffs), len(diffs), nonzero
    )


def comparison_rows(
    first: TotalMetrics,
    second: TotalMetrics,
    *,
    allow_partial: bool = False,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    zero_method: str = "wilcox",
    method: str = "auto",
) -> list[dict]:
    """Pair datasets and essay IDs strictly unless partial overlap is explicitly enabled."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    left = {str(m.set_id): m for m in first.metrics_sets}
    right = {str(m.set_id): m for m in second.metrics_sets}
    if not allow_partial and left.keys() != right.keys():
        raise ValueError(
            "Compared runs have different dataset sets; use allow_partial=True for their intersection"
        )
    rows = []
    for label in sorted(left.keys() & right.keys()):
        a, b = left[label], right[label]
        scale_a = a.scale or score_range(a.set_id)
        scale_b = b.scale or score_range(b.set_id)
        if scale_a != scale_b:
            raise ValueError(f"Compared score scales differ for {label}")
        first_essays, second_essays = (
            {i.essay_id: i for i in a.inputs},
            {i.essay_id: i for i in b.inputs},
        )
        if len(first_essays) != len(a.inputs) or len(second_essays) != len(b.inputs):
            raise ValueError(f"Duplicate essay IDs in {label}")
        if not allow_partial and first_essays.keys() != second_essays.keys():
            raise ValueError(
                f"Compared essay IDs differ for {label}; use allow_partial=True for their intersection"
            )
        common = sorted(first_essays.keys() & second_essays.keys())
        if not common:
            raise ValueError(f"No paired essays for {label}")
        if any(first_essays[k].human != second_essays[k].human for k in common):
            raise ValueError(f"Compared official human scores differ for {label}")
        factor = 10 / (scale_a.max - scale_a.min)
        error_a = [abs(first_essays[k].pred - first_essays[k].human) * factor for k in common]
        error_b = [abs(second_essays[k].pred - second_essays[k].human) * factor for k in common]
        result = run_wilcoxon(
            error_a, error_b, alternative=alternative, zero_method=zero_method, method=method
        )
        mean_a, mean_b = sum(error_a) / len(common), sum(error_b) / len(common)
        winner = ""
        if result.p_value is not None and result.p_value < alpha and mean_a != mean_b:
            winner = first.experiment_id if mean_a < mean_b else second.experiment_id
        rows.append(
            {
                "Dataset": label,
                "First": first.experiment_id,
                "Second": second.experiment_id,
                "N Paired": result.n_pairs,
                "N Nonzero": result.n_nonzero,
                "N First Only": len(first_essays.keys() - second_essays.keys()),
                "N Second Only": len(second_essays.keys() - first_essays.keys()),
                "First NMAE": mean_a,
                "Second NMAE": mean_b,
                "Delta NMAE (Second-First)": mean_b - mean_a,
                "Statistic": result.statistic,
                "p-value": result.p_value,
                "Rank Biserial (First-Second)": result.effect_size,
                "Better System": winner,
                "Significance": sig_stars(result.p_value),
            }
        )
    if not rows:
        raise ValueError("Compared runs have no datasets in common")
    return rows


def save_comparison(
    first: TotalMetrics,
    second: TotalMetrics,
    output_path: Path,
    *,
    allow_partial: bool = False,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    zero_method: str = "wilcox",
    method: str = "auto",
) -> Path:
    """Save full-precision per-dataset tests with sample overlap and signed effect size.

    Uses configured comparison settings: allow_partial (dataset overlap), alpha (significance),
    and Wilcoxon test parameters (alternative, zero_method, method).
    """
    rows = comparison_rows(
        first,
        second,
        allow_partial=allow_partial,
        alpha=alpha,
        alternative=alternative,
        zero_method=zero_method,
        method=method,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def compare_experiments(
    total_metrics_first: TotalMetrics,
    total_metrics_second: TotalMetrics,
    metrics_csv_second: Path,
    out_root: Path,
    alpha: float = 0.05,
    *,
    allow_partial: bool = False,
) -> Path:
    """Save a paired verdict and an annotated copy of the second run's metrics."""
    first_name, second_name = (
        Path(total_metrics_first.experiment_id).name,
        Path(total_metrics_second.experiment_id).name,
    )
    out_dir = out_root / f"{first_name}_to_{second_name}"
    verdict = save_comparison(
        total_metrics_first,
        total_metrics_second,
        out_dir / "verdict.csv",
        alpha=alpha,
        allow_partial=allow_partial,
    )
    rows = pd.read_csv(verdict, keep_default_na=False)
    stars = dict(zip(rows["Dataset"], rows["Significance"], strict=True))
    metrics = pd.read_csv(metrics_csv_second, dtype=str, keep_default_na=False)
    metrics["NMAE"] = [
        f"{value}{stars.get(dataset, '')}"
        for dataset, value in zip(metrics["Dataset"], metrics["NMAE"], strict=True)
    ]
    metrics.to_csv(out_dir / metrics_csv_second.name, index=False)
    return verdict
