"""A separately installed extension registering a BM25 retriever and a mean-absolute-error
metric. Every framework name it uses comes from `abaes_eval_fwk.api`; nothing here
reaches into `contracts`, `registry`, `core`, or `experiment`, so this package works without
the source checkout."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Final

from abaes_eval_fwk.api import (
    PUBLIC_API_VERSION,
    ComponentDescriptor,
    PluginDescriptor,
)

from ._metric import (
    MEAN_ABSOLUTE_ERROR_ID,
    MeanAbsoluteErrorMetric,
    MeanAbsoluteErrorMetricConfig,
    mean_absolute_error_descriptor,
)
from ._retrieval import (
    BM25_RETRIEVER_ID,
    Bm25Retriever,
    Bm25RetrieverConfig,
    bm25_descriptor,
    bm25_scores,
    tokenize,
)

__all__ = [
    "BM25_RETRIEVER_ID",
    "MEAN_ABSOLUTE_ERROR_ID",
    "PLUGIN",
    "PLUGIN_ID",
    "API_RANGE_ENV",
    "Bm25Plugin",
    "Bm25Retriever",
    "Bm25RetrieverConfig",
    "MeanAbsoluteErrorMetric",
    "MeanAbsoluteErrorMetricConfig",
    "__version__",
    "bm25_descriptor",
    "bm25_scores",
    "component_descriptors",
    "mean_absolute_error_descriptor",
    "tokenize",
]

__version__: Final[str] = "1.1.0"
PLUGIN_ID: Final[str] = "aes.aes_bm25_plugin"
API_RANGE_ENV: Final[str] = "AES_BM25_PLUGIN_API_RANGE"
DEFAULT_API_RANGE: Final[str] = (
    f">={PUBLIC_API_VERSION},<{int(PUBLIC_API_VERSION.split('.')[0]) + 1}.0"
)


def component_descriptors() -> tuple[ComponentDescriptor, ...]:
    """The two components this package contributes, described as data."""
    return (bm25_descriptor(), mean_absolute_error_descriptor())


class Bm25Plugin:
    """The provider the entry point names: its own metadata, and the descriptors it registers."""

    def plugin_descriptor(self) -> PluginDescriptor:
        return PluginDescriptor(
            plugin_id=PLUGIN_ID,
            version=__version__,
            # An incompatible range comes from the environment, so one installed distribution can
            # exercise the rejection path without a second package.
            api_range=os.environ.get(API_RANGE_ENV, DEFAULT_API_RANGE),
            registrations=(BM25_RETRIEVER_ID, MEAN_ABSOLUTE_ERROR_ID),
            requires={"abaes-eval-fwk": ">=0.1"},
            description="BM25 few-shot retrieval and mean absolute error, computed offline",
        )

    def component_descriptors(self) -> Sequence[ComponentDescriptor]:
        return component_descriptors()


PLUGIN = Bm25Plugin()
