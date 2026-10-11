"""Configuration for metrics computation and statistical tests."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetricsCompute:
    """Which metrics to include in output.

    No field has a default here: all seven are already defaulted in
    config/config_files/defaults/metrics.yaml (metrics.compute), which load_config()
    always merges in before construction.

    Every built-in is computed on every run regardless (see resolve_metrics); these flags
    select which ones reach the saved table. `mae` and `rmse` report the same errors as
    `nmae`/`nrmse` in the dataset's own score points rather than on the shared 0-10 scale,
    and default to off so a project's existing metric columns do not move.
    """

    nmae: bool
    nrmse: bool
    pearson: bool
    spearman: bool
    qwk: bool
    mae: bool
    rmse: bool


@dataclass
class SignificanceThreshold:
    """Threshold for a significance star marker."""

    p_value: float


@dataclass
class WilcoxonConfig:
    """Paired Wilcoxon signed-rank test configuration.

    No field has a default here: all four are already defaulted in
    config/config_files/defaults/metrics.yaml (metrics.wilcoxon), which load_config()
    always merges in before construction.
    """

    alternative: str
    alpha: float
    zero_method: str
    method: str


@dataclass
class ComparisonConfig:
    """Configuration for comparing two evaluated runs.

    No field has a default here: both are already defaulted in
    config/config_files/defaults/metrics.yaml (metrics.comparison), which load_config()
    always merges in before construction.
    """

    allow_partial_overlap: bool
    alpha: float


@dataclass
class MetricsConfig:
    """Top-level metrics configuration.

    `compute`, `wilcoxon`, and `comparison` are required: the loader always resolves a
    real instance for each before constructing this (configuration.py's metrics_config
    rejects a null override instead of silently falling back). `significance_thresholds`
    is the one field the loader lets a project explicitly null out to mean "use the
    built-in thresholds", so it keeps its own default here.
    """

    compute: MetricsCompute
    wilcoxon: WilcoxonConfig
    comparison: ComparisonConfig
    significance_thresholds: list[SignificanceThreshold] | None = None
    # Reader-supplied metrics, as name -> `module:function`. Each callable is passed the paired
    # scores and the dataset's scale, exactly like a built-in, and its name becomes its column.
    custom: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.significance_thresholds is None:
            self.significance_thresholds = [
                SignificanceThreshold(0.001),
                SignificanceThreshold(0.01),
                SignificanceThreshold(0.05),
            ]
