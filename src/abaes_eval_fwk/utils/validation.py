"""Shared dict-shape validation helpers for YAML-sourced configuration."""

from __future__ import annotations


def keys(value: dict, allowed: set[str], label: str) -> None:
    """Reject a mapping containing any key outside the allowed set."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"Unknown {label} fields: {sorted(extra)}")
