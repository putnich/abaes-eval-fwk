"""Per-run sample size, essay selection, and sampling strategy configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingConfig:
    """Resolved sampling settings for one project or experiment-level override."""

    # No default here: every one of these is already defaulted in
    # config/config_files/defaults/sampling.yaml, which load_config() always merges in
    # before construction, so a Python-side default would just be a second, driftable
    # copy of the same value.
    sample_size: int
    seed: int
    evaluation_strategy: str
    calibration_strategy: str
    essay_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.essay_ids is not None:
            object.__setattr__(self, "essay_ids", tuple(self.essay_ids))
        self.validate_sampling_config()

    def validate_sampling_config(self) -> None:
        for name, minimum in (("sample_size", 1), ("seed", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.essay_ids is not None and (
            not self.essay_ids or len({str(i) for i in self.essay_ids}) != len(self.essay_ids)
        ):
            raise ValueError("essay_ids must be a nonempty list of unique IDs")
        for name in ("evaluation_strategy", "calibration_strategy"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a built-in name or module:function")
