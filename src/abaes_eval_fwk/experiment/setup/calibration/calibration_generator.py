"""Generates a score-stratified calibration pool from a dataset's raw human-scored essays. Used by calibration_loader.load_calibration_pool when no pool CSV exists yet at DatasetConfig.calibration_pool_path, mirroring the build-if-missing/else-read pattern used for retrieval indices."""

import hashlib
import logging
import math
import random

import pandas as pd

from abaes_eval_fwk.config.dataset_config import DatasetConfig
from abaes_eval_fwk.core.models.rubric import Rubric

logger = logging.getLogger(__name__)

# A level's calibration draw never exceeds this fraction of its available essays (a 25/75
# calibration/eval split), so a rare level still keeps most of itself for evaluation instead of
# calibration consuming it entirely; plentiful levels are unaffected.
_MAX_CALIBRATION_FRACTION = 0.25


def _rng_for(dataset_label: str, score_level: float, seed: int = 42) -> random.Random:
    """Deterministic per-(dataset, score level) RNG, so regenerating a pool is reproducible."""
    seed = int(
        hashlib.sha256(f"{seed}|{dataset_label}|{score_level}".encode()).hexdigest()[:12], 16
    )
    return random.Random(seed)


def _criterion_source_column(dataset: DatasetConfig, criterion_name: str) -> str | None:
    """Return the raw dataframe column holding one criterion's ground-truth value, if any."""
    if criterion_name in dataset.criteria_columns:
        return criterion_name
    return None


def generate_calibration_pool(
    dataset: DatasetConfig, df_full: pd.DataFrame, rubric: Rubric
) -> pd.DataFrame:
    """Build a score-stratified calibration pool with a configurable cap per human-score level."""
    target_n = dataset.calibration_per_score

    picked_frames = []
    for score_level in sorted(df_full["human_score"].dropna().unique()):
        group = df_full[df_full["human_score"] == score_level].sort_values(
            "essay_id", kind="stable"
        )
        n = min(target_n, math.floor(len(group) * _MAX_CALIBRATION_FRACTION))
        if n == 0 and len(group) >= 2:
            n = 1
        rng = _rng_for(dataset.label, score_level, dataset.seed)
        picked_idx = rng.sample(list(group.index), n)
        picked_frames.append(df_full.loc[picked_idx])

    picked = (
        (pd.concat(picked_frames) if picked_frames else df_full.iloc[:0])
        .sort_values(["human_score", "essay_id"], kind="stable")
        .reset_index(drop=True)
    )

    pool_df = pd.DataFrame(
        {
            "rubric_id": dataset.label,
            "essay_id": picked["essay_id"].astype(str),
            "human_score": picked["human_score"],
        }
    )
    for criterion in rubric.criteria:
        source_col = criterion.name if criterion.name in df_full.columns else None
        pool_df[criterion.name] = (
            picked[source_col].to_numpy() if source_col and source_col in picked.columns else pd.NA
        )

    logger.info(
        "Generated calibration pool for '%s': %d essays across %d score levels (target %d/level)",
        dataset.label,
        len(pool_df),
        pool_df["human_score"].nunique(),
        target_n,
    )
    return pool_df
