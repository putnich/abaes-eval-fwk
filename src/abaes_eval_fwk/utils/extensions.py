"""Resolve reader-supplied `module:function` extensions.

Every extension point — retrieval selectors, sampling strategies, LLM adapters, dataset
loaders, custom metrics, criteria aggregation — names its callable the same way, so the
parse, the import, and the "is it actually callable" check live here once rather than in
each loader. `runner.py` fingerprints whatever these references name, which is what lets a
resume notice that a reader changed their own code.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable


def is_reference(value: object) -> bool:
    """Whether a configured value names a callable rather than a built-in policy."""
    return isinstance(value, str) and ":" in value


def split_reference(reference: str, label: str) -> tuple[str, str]:
    """Split `module:function`, rejecting the shapes an import would fail on obscurely."""
    if not isinstance(reference, str):
        raise ValueError(f"{label} must be a 'module:function' string")
    module_name, separator, function_name = reference.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(f"{label} must be 'module:function': {reference!r}")
    return module_name, function_name


def module_of(reference: str, label: str) -> str:
    """The module half of a reference, for fingerprinting its source."""
    return split_reference(reference, label)[0]


def resolve_callable(reference: str, label: str) -> Callable:
    """Import a reference and return the callable it names.

    The module is imported from the configuration's own directory (put on `sys.path` by
    `runner._ensure_extensions_importable`), so a reader's extension needs no install.
    """
    module_name, function_name = split_reference(reference, label)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{label} names module {module_name!r}, which could not be imported: {exc}"
        ) from exc
    try:
        function = getattr(module, function_name)
    except AttributeError as exc:
        raise AttributeError(
            f"{label} names {function_name!r}, which {module_name!r} does not define"
        ) from exc
    if not callable(function):
        raise TypeError(f"{label} is not callable: {reference}")
    return function
