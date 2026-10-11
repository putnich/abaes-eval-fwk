"""A reader-supplied dataset loader, named as `datasets[].loader: extensions.datasets:<function>`.

A loader replaces the file-reading step for a corpus no built-in family reads. It is handed the
`DatasetConfig` and the configured path (or None when the dataset configures no file at all), and
returns a pandas DataFrame carrying at least `essay_id`, `set_id`, `essay`, and `human_score`,
plus one column per rubric criterion it has analytic scores for. Everything after that is
unchanged: the frame goes through the same validation a packaged dataset's does, against this
dataset's own rubric, so nothing enters a run less checked than ASAP or ELLIPSE would be.
"""

from __future__ import annotations

import pandas as pd


def transposed_csv(*, dataset, data_path):
    """Read a CSV whose score column is named `grade` and whose text column is named `response`.

    Stands in for the usual reason to write a loader: the corpus is a CSV, but not in the shape
    `csv:` expects, and normalizing it by hand before every run is the step worth deleting.
    """
    if data_path is None:
        raise ValueError(f"{dataset.label}: this loader needs a csv path")
    frame = pd.read_csv(data_path, keep_default_na=False, na_values=[""], dtype={"id": "string"})
    return pd.DataFrame(
        {
            "essay_id": frame["id"].astype(str),
            "set_id": dataset.subset_id or dataset.slug,
            "essay": frame["response"].astype(str),
            "human_score": pd.to_numeric(frame["grade"]),
        }
    )


def single_set_demo(*, dataset, data_path):
    """Load the playground's own demo.csv and keep one prompt set, as a loader rather than `csv:`.

    Useful as a first extension to run: the frame it returns is the one the built-in custom-CSV
    path would have produced, so a run with this loader and a run with `csv:` agree.
    """
    if data_path is None:
        raise ValueError(f"{dataset.label}: this loader needs a csv path")
    frame = pd.read_csv(
        data_path,
        keep_default_na=False,
        na_values=[""],
        dtype={"essay_id": "string", "set_id": "string"},
    )
    wanted = dataset.subset_id or str(frame["set_id"].iloc[0])
    return frame[frame["set_id"].astype(str) == wanted].reset_index(drop=True)
