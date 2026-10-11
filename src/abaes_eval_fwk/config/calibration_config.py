"""Calibration pool selection configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CalibrationConfig:
    """Configuration for calibration pool sampling.

    No field has a default here: this field is already defaulted in
    config/config_files/defaults/calibration.yaml, which load_config() always merges
    in before construction.

    Note: calibration strategy is configured via sampling.calibration_strategy, not here.
    """

    max_per_score: int
