"""Per-run execution limits: concurrency, retries, and failure handling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution limits and failure-handling policy for one run; never overridden per-experiment.

    No field has a default here: all five are already defaulted in
    config/config_files/defaults/runtime.yaml, which load_config() always merges in
    before construction.
    """

    max_workers: int
    max_attempts: int
    compute_metrics: bool
    save_conversations: bool
    context_overflow: str

    def __post_init__(self) -> None:
        self.validate_runtime_config()

    def validate_runtime_config(self) -> None:
        for name in ("max_workers", "max_attempts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        for name in ("compute_metrics", "save_conversations"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.context_overflow not in {"error", "truncate_examples"}:
            raise ValueError("context_overflow must be error or truncate_examples")
