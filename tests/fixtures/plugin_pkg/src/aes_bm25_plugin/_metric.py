"""Mean absolute error against the selected human targets, with the coverage it was computed on."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from abaes_eval_fwk.api import (
    Capability,
    ComponentCategory,
    ComponentConfig,
    ComponentContext,
    ComponentDescriptor,
    ComponentError,
    ErrorCode,
    MetricRequest,
    MetricResult,
)

__all__ = [
    "MEAN_ABSOLUTE_ERROR_ID",
    "MeanAbsoluteErrorMetric",
    "MeanAbsoluteErrorMetricConfig",
    "mean_absolute_error_descriptor",
]

MEAN_ABSOLUTE_ERROR_ID: Final[str] = "aes_bm25.metric.mean_absolute_error"
METRIC_VERSION: Final[str] = "1.0"


class MeanAbsoluteErrorMetricConfig(ComponentConfig):
    """Whether a prediction with no reference is dropped, or makes the metric unavailable."""

    require_full_coverage: bool = False


class MeanAbsoluteErrorMetric:
    """Compares every prediction's raw score with the overall reference of the same sample."""

    def __init__(self, config: MeanAbsoluteErrorMetricConfig) -> None:
        self._config = config

    async def compute(self, request: MetricRequest, context: ComponentContext) -> MetricResult:
        """Return the mean absolute error, or state why no value could be computed for it."""
        references = {item.key.as_tuple(): item.overall for item in request.references}
        matched = [
            (prediction, references[prediction.key.as_tuple()])
            for prediction in request.predictions
            if prediction.key.as_tuple() in references
        ]
        settings = {"given": len(request.predictions), "matched": len(matched)}
        given = len(request.predictions)
        if not matched or (self._config.require_full_coverage and len(matched) != given):
            reason = (
                f"none of the {given} predictions has a reference score"
                if not matched
                else f"{given - len(matched)} of {given} predictions have no reference score"
            )
            return MetricResult.unavailable(
                MEAN_ABSOLUTE_ERROR_ID,
                reason,
                scale=request.scale,
                target=request.target,
                metric_version=METRIC_VERSION,
                settings=settings,
            )
        errors = [abs(prediction.raw_score - overall) for prediction, overall in matched]
        return MetricResult(
            metric_id=MEAN_ABSOLUTE_ERROR_ID,
            metric_version=METRIC_VERSION,
            value=sum(errors) / len(errors),
            n_samples=len(matched),
            scale=request.scale,
            target=request.target,
            settings={**settings, "max_absolute_error": max(errors)},
        )


def _build(
    config: ComponentConfig, bindings: Mapping[str, str], context: ComponentContext
) -> MeanAbsoluteErrorMetric:
    """The `ComponentFactory` for this metric, which declares no dependency bindings."""
    if not isinstance(config, MeanAbsoluteErrorMetricConfig):
        raise ComponentError(
            f"metric {MEAN_ABSOLUTE_ERROR_ID} requires a MeanAbsoluteErrorMetricConfig, not "
            f"{type(config).__name__}",
            code=ErrorCode.COMPONENT_INVALID,
            component_id=MEAN_ABSOLUTE_ERROR_ID,
        )
    return MeanAbsoluteErrorMetric(config)


def mean_absolute_error_descriptor() -> ComponentDescriptor:
    """The metric as data: its identity, contracts, configuration model, factory, capabilities."""
    return ComponentDescriptor(
        component_id=MEAN_ABSOLUTE_ERROR_ID,
        category=ComponentCategory.METRIC,
        description="mean absolute error of raw scores against the selected human targets",
        config_model=MeanAbsoluteErrorMetricConfig,
        factory=_build,
        input_contract="MetricRequest",
        output_contract="MetricResult",
        # Arithmetic over the scores a run already recorded is reproducible.
        capabilities=frozenset({Capability.DETERMINISTIC}),
    )
