"""Resolve built-in or reader-supplied calibration and evaluation sampling strategies."""

from __future__ import annotations

import importlib
from collections.abc import Callable

import pandas as pd


def resolve_strategy(name: str, builtin: str, implementation: Callable) -> Callable:
    """Resolve a built-in strategy name or a `module:function` callable."""
    if name == builtin:
        return implementation
    module_name, separator, function_name = name.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(f"Sampling strategy must be {builtin!r} or module:function: {name!r}")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"Sampling strategy is not callable: {name}")
    return function


def require_frame(value: object, strategy: str) -> pd.DataFrame:
    """Require a sampling strategy to return a nonempty pandas DataFrame."""
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"Sampling strategy {strategy!r} must return a pandas DataFrame")
    if value.empty:
        raise ValueError(f"Sampling strategy {strategy!r} returned no rows")
    return value.reset_index(drop=True)


def allow_empty_frame(value: object, strategy: str) -> pd.DataFrame:
    """Validate a sampling strategy returns a pandas DataFrame, allowing empty frames.

    Used for calibration pools where zero-shot grading does not require examples.
    """
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"Sampling strategy {strategy!r} must return a pandas DataFrame")
    return value.reset_index(drop=True)
