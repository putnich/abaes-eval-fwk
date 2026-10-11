"""A plugin declaring a public API range this framework does not support, plus a malformed
config. Both live in the same installed distribution so the entry-point tests exercise
rejection rather than a hand-built descriptor, which is the path a real incompatible package
would take."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from abaes_eval_fwk.api import (
    ComponentCategory,
    ComponentConfig,
    ComponentContext,
    ComponentDescriptor,
    PluginDescriptor,
)

__all__ = [
    "INCOMPATIBLE_PLUGIN",
    "INCOMPATIBLE_RETRIEVER_ID",
    "MALFORMED_PLUGIN",
    "MALFORMED_RETRIEVER_ID",
    "IncompatiblePlugin",
    "MalformedPlugin",
]

INCOMPATIBLE_RETRIEVER_ID = "demo.retrieval.future"
MALFORMED_RETRIEVER_ID = "demo.retrieval.malformed"


def _factory(config: ComponentConfig, bindings: Mapping[str, str], context: ComponentContext):
    return object()


def _incompatible_descriptor() -> ComponentDescriptor:
    return ComponentDescriptor(
        component_id=INCOMPATIBLE_RETRIEVER_ID,
        category=ComponentCategory.RETRIEVER,
        description="compiled against a public API this framework does not support",
        config_model=ComponentConfig,
        factory=_factory,
        input_contract="RetrievalRequest",
        output_contract="ReferenceBatch",
    )


def _malformed_descriptor() -> ComponentDescriptor:
    # A retriever declaring a metric's contracts is the mistake a registry must catch, not a caller.
    return ComponentDescriptor(
        component_id=MALFORMED_RETRIEVER_ID,
        category=ComponentCategory.RETRIEVER,
        description="a retriever claiming contracts that are not a retriever's",
        config_model=ComponentConfig,
        factory=_factory,
        input_contract="MetricRequest",
        output_contract="MetricResult",
    )


class IncompatiblePlugin:
    """Declares `>=2.0`, which excludes the running public API 1.0."""

    def plugin_descriptor(self) -> PluginDescriptor:
        return PluginDescriptor(
            plugin_id="demo.incompatible",
            version="1.0.0",
            api_range=">=2.0,<3.0",
            registrations=(INCOMPATIBLE_RETRIEVER_ID,),
        )

    def component_descriptors(self) -> Sequence[ComponentDescriptor]:
        return (_incompatible_descriptor(),)


class MalformedPlugin:
    """Declares a compatible API range, so only its descriptor's contracts are wrong."""

    def plugin_descriptor(self) -> PluginDescriptor:
        return PluginDescriptor(
            plugin_id="demo.malformed",
            version="0.1.0",
            api_range=">=1.0,<2.0",
            registrations=(MALFORMED_RETRIEVER_ID,),
        )

    def component_descriptors(self) -> Sequence[ComponentDescriptor]:
        return (_malformed_descriptor(),)


INCOMPATIBLE_PLUGIN = IncompatiblePlugin()
MALFORMED_PLUGIN = MalformedPlugin()
