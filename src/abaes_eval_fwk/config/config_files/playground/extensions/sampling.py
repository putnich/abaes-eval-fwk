"""Reader-supplied evaluation and calibration sampling, named as `extensions.sampling:<function>`."""

from __future__ import annotations

import pandas as pd


def lowest_scores_first(
    df: pd.DataFrame, n_per_set: int, group_col: str = "set_id", exclude_ids=None
):
    """Take the lowest-scored essays in each set, so the split differs from quantile_rank."""
    frame = df if not exclude_ids else df[~df["essay_id"].isin(exclude_ids)]
    ordered = frame.sort_values(["human_score", "essay_id"], kind="stable")
    return ordered.groupby(group_col, group_keys=False).head(n_per_set).reset_index(drop=True)


def one_per_score(dataset, df_full: pd.DataFrame, rubric) -> pd.DataFrame:
    """Take the first essay at each human score as the calibration pool."""
    picked = (
        df_full.sort_values(["human_score", "essay_id"], kind="stable")
        .groupby("human_score", group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )
    pool = pd.DataFrame(
        {
            "rubric_id": dataset.label,
            "essay_id": picked["essay_id"].astype(str),
            "human_score": picked["human_score"],
        }
    )
    for criterion in rubric.criteria:
        pool[criterion.name] = (
            picked[criterion.name].to_numpy() if criterion.name in picked.columns else pd.NA
        )
    return pool
