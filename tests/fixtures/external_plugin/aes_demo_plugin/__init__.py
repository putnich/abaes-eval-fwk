"""A separately installed extension package, importing public contracts only. Nothing here
reaches into `core/`, `config/`, or `experiment/`, which is what makes the plugin usable
without the source checkout; the distribution metadata beside this package is what
`importlib.metadata` discovers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from abaes_eval_fwk.api import (
    Capability,
    ComponentCategory,
    ComponentConfig,
    ComponentContext,
    ComponentDescriptor,
    DatasetBundle,
    DatasetLoadRequest,
    MetricRequest,
    MetricResult,
    PluginDescriptor,
    ReferenceBatch,
    RetrievalRequest,
)
from pydantic import Field

__all__ = [
    "COLLIDING_PLUGIN",
    "PLUGIN",
    "CollidingPlugin",
    "DemoPlugin",
    "LexicalRetriever",
    "LexicalRetrieverConfig",
]

RETRIEVER_ID = "demo.retrieval.lexical"
LOADER_ID = "demo.datasets.toy"
METRIC_ID = "demo.metric.coverage"
# A built-in identifier, claimed by `CollidingPlugin` so the duplicate path is a real collision.
COLLIDING_ID = "aes.datasets.asap_aes_set_1"


class LexicalRetrieverConfig(ComponentConfig):
    """How many pool examples this retriever hands a grader, and whether to sort by score."""

    max_examples: int = Field(default=3, ge=1)
    descending: bool = True


class LexicalRetriever:
    """The first examples of the pool, which is deterministic and needs no index."""

    def __init__(self, config: LexicalRetrieverConfig) -> None:
        self._config = config

    async def retrieve(
        self, request: RetrievalRequest, context: ComponentContext
    ) -> ReferenceBatch:
        """Return the highest-scoring examples of the request's pool, capped by the config."""
        limit = request.max_examples or self._config.max_examples
        pool = sorted(
            request.pool, key=lambda example: example.score, reverse=self._config.descending
        )
        return ReferenceBatch(
            key=request.submission.key,
            strategy_id=RETRIEVER_ID,
            examples=pool[:limit],
        )


class ToyLoaderConfig(ComponentConfig):
    """One option, so a test can show a plugin's own configuration rejecting an unknown key."""

    rows: int = Field(default=2, ge=1)


class ToyDatasetLoader:
    """Returns an empty bundle, which is enough to prove the registration path without any data."""

    def __init__(self, config: ToyLoaderConfig) -> None:
        self._config = config

    async def load(self, request: DatasetLoadRequest, context: ComponentContext) -> DatasetBundle:
        """Return an empty bundle for the requested dataset, with no submissions."""
        return DatasetBundle(dataset_id=request.dataset_id)


class CoverageMetric:
    """Counts the predictions it was given, so a test can see a plugin metric was built."""

    def __init__(self, config: ComponentConfig) -> None:
        self._config = config

    async def compute(self, request: MetricRequest, context: ComponentContext) -> MetricResult:
        """Report how many predictions this metric saw, as a count rather than a quality score."""
        return MetricResult(
            metric_id=METRIC_ID,
            value=float(len(request.predictions)),
            n_samples=len(request.predictions),
            scale=request.scale,
            target=request.target,
        )


def _retriever_factory(
    config: ComponentConfig, bindings: Mapping[str, str], context: ComponentContext
) -> LexicalRetriever:
    return LexicalRetriever(config)  # type: ignore[arg-type]


def _loader_factory(
    config: ComponentConfig, bindings: Mapping[str, str], context: ComponentContext
) -> ToyDatasetLoader:
    return ToyDatasetLoader(config)  # type: ignore[arg-type]


def _metric_factory(
    config: ComponentConfig, bindings: Mapping[str, str], context: ComponentContext
) -> CoverageMetric:
    return CoverageMetric(config)


def component_descriptors() -> tuple[ComponentDescriptor, ...]:
    """The three components this package contributes, described as data."""
    return (
        ComponentDescriptor(
            component_id=RETRIEVER_ID,
            category=ComponentCategory.RETRIEVER,
            description="lexical retriever taking the top-scoring pool examples",
            config_model=LexicalRetrieverConfig,
            factory=_retriever_factory,
            input_contract="RetrievalRequest",
            output_contract="ReferenceBatch",
            capabilities=frozenset({Capability.DETERMINISTIC}),
        ),
        ComponentDescriptor(
            component_id=LOADER_ID,
            category=ComponentCategory.DATASET_LOADER,
            description="toy loader returning an empty bundle",
            config_model=ToyLoaderConfig,
            factory=_loader_factory,
            input_contract="DatasetLoadRequest",
            output_contract="DatasetBundle",
            capabilities=frozenset({Capability.DETERMINISTIC}),
        ),
        ComponentDescriptor(
            component_id=METRIC_ID,
            category=ComponentCategory.METRIC,
            description="coverage metric counting the predictions it was given",
            config_model=ComponentConfig,
            factory=_metric_factory,
            input_contract="MetricRequest",
            output_contract="MetricResult",
        ),
    )


PLUGIN_DESCRIPTOR = PluginDescriptor(
    plugin_id="demo.aes_demo_plugin",
    version="0.2.0",
    api_range=">=1.0,<2.0",
    registrations=(RETRIEVER_ID, LOADER_ID, METRIC_ID),
    requires={"abaes-eval-fwk": ">=0.1"},
    description="demonstration extension package",
)


class DemoPlugin:
    """The provider object the entry point names: its metadata and the descriptors it registers."""

    def plugin_descriptor(self) -> PluginDescriptor:
        return PLUGIN_DESCRIPTOR

    def component_descriptors(self) -> Sequence[ComponentDescriptor]:
        return component_descriptors()


class CollidingPlugin:
    """Claims a built-in's identifier, so a registry that allows it would silently replace one."""

    def plugin_descriptor(self) -> PluginDescriptor:
        return PluginDescriptor(
            plugin_id="demo.colliding",
            version="0.1.0",
            api_range=">=1.0,<2.0",
            registrations=(COLLIDING_ID,),
        )

    def component_descriptors(self) -> Sequence[ComponentDescriptor]:
        return (
            ComponentDescriptor(
                component_id=COLLIDING_ID,
                category=ComponentCategory.DATASET_LOADER,
                description="an external loader claiming a built-in identifier",
                config_model=ToyLoaderConfig,
                factory=_loader_factory,
                input_contract="DatasetLoadRequest",
                output_contract="DatasetBundle",
            ),
        )


PLUGIN = DemoPlugin()
COLLIDING_PLUGIN = CollidingPlugin()
