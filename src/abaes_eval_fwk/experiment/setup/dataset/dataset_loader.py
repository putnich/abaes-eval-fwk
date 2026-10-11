"""Load and validate built-in essay datasets or researcher-supplied normalized CSVs into Dataset objects."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from abaes_eval_fwk.config.dataset_config import (
    DatasetConfig,
    DatasetFamily,
    DatasetNames,
    get_dataset_config,
)
from abaes_eval_fwk.core.models.assessment import Dataset, HumanAssessment, StudentSubmission
from abaes_eval_fwk.experiment.setup.rubric.rubric_loader import load_rubric
from abaes_eval_fwk.utils.extensions import resolve_callable
from abaes_eval_fwk.utils.rounding import round_half_up

logger = logging.getLogger(__name__)


def _read_raw(path: str, sep: str = ",", encoding: str = "utf-8") -> pd.DataFrame:
    """Read raw files without retaining stale frames after files change."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Dataset file not found at '{path}'")
    return pd.read_csv(
        path,
        sep=sep,
        encoding=encoding,
        keep_default_na=False,
        na_values=[""],
        dtype={"essay_id": "string", "text_id_kaggle": "string", "set_id": "string"},
    )


def _require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def _load_asap_aes(dataset: DatasetConfig, data_path: str | None = None) -> pd.DataFrame:
    """Load one ASAP AES prompt set using the first rater's holistic and available trait scores."""
    df = _read_raw(data_path or dataset.raw_data_path, sep="\t", encoding="latin-1")
    df.columns = [c.lower() for c in df.columns]
    _require_columns(df, dataset.raw_columns, dataset.label)
    df = df.rename(columns=dataset.column_map)
    df = df[df["set_id"].astype(str) == str(dataset.set)].copy()
    return df[
        dataset.base_columns + [c for c in dataset.criteria_columns if c in df.columns]
    ].reset_index(drop=True)


def _load_ellipse(
    dataset: DatasetConfig,
    data_path: str | None = None,
    raw_ratings_path: str | None = None,
) -> pd.DataFrame:
    """Join ELLIPSE text with the first rater's official Overall and dimension scores."""
    final_path = data_path or dataset.raw_data_path
    raw_path = raw_ratings_path or str(Path(dataset.dataset_folder_path) / dataset.raw_filenames[1])
    final = _read_raw(final_path)
    raw = _read_raw(raw_path)
    _require_columns(final, ["text_id_kaggle", "full_text"], "ELLIPSE essay file")
    _require_columns(raw, ["text_id_kaggle"] + dataset.raw_columns, "ELLIPSE rater file")
    raw = raw[["text_id_kaggle"] + dataset.raw_columns]
    relevant = raw[raw["text_id_kaggle"].isin(final["text_id_kaggle"])]
    if relevant["text_id_kaggle"].duplicated().any():
        raise ValueError("ELLIPSE contains duplicate raw rater rows for an essay ID")
    df = final[["text_id_kaggle", "full_text"]].merge(
        relevant, on="text_id_kaggle", how="left", validate="many_to_one"
    )
    df = df.rename(columns=dataset.column_map)
    df["set_id"] = "ell"
    return df[dataset.base_columns + dataset.criteria_columns].reset_index(drop=True)


def _load_asap_2_0(dataset: DatasetConfig, data_path: str | None = None) -> pd.DataFrame:
    """Load and normalize one ASAP 2.0 prompt set."""
    df = _read_raw(data_path or dataset.raw_data_path)
    _require_columns(df, ["prompt_name", "essay_id", "full_text", "score"], dataset.label)
    df = df[df["prompt_name"] == dataset.prompt_name].copy()
    df["set_id"] = f"asap2_{dataset.set}"
    df = df.rename(columns={"full_text": "essay", "score": "human_score"})
    return df[dataset.base_columns].reset_index(drop=True)


def _validate_scores(
    df: pd.DataFrame, column: str, minimum: float, maximum: float, *, optional: bool = False
) -> None:
    raw = df[column]
    values = pd.to_numeric(raw, errors="coerce")
    invalid = ~np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)) | ~values.between(
        minimum, maximum
    )
    if optional:
        invalid &= raw.notna()
    if invalid.any():
        ids = df.loc[invalid, "essay_id"].astype(str).head(5).tolist()
        raise ValueError(
            f"{column} must contain finite scores in [{minimum}, {maximum}]; invalid essay IDs: {ids}"
        )
    df[column] = values


def _validate_dataset(df: pd.DataFrame, dataset: DatasetConfig) -> pd.DataFrame:
    """Reject empty subsets, invalid identifiers or text, and invalid human scores before sampling."""
    _require_columns(df, dataset.base_columns, dataset.label)
    if df.empty:
        raise ValueError(f"{dataset.label}: requested dataset subset is empty")
    df = df.copy()
    for column in ("essay_id", "set_id"):
        missing = df[column].isna() | df[column].astype(str).str.strip().eq("")
        if missing.any():
            raise ValueError(f"{dataset.label}: {column} contains missing or empty IDs")
        df[column] = df[column].astype(str)
    if df["essay_id"].duplicated().any():
        ids = df.loc[df["essay_id"].duplicated(keep=False), "essay_id"].head(5).tolist()
        raise ValueError(f"{dataset.label}: duplicate essay IDs: {ids}")
    if (df["essay"].isna() | df["essay"].astype(str).str.strip().eq("")).any():
        raise ValueError(f"{dataset.label}: essay text cannot be missing or empty")
    df["essay"] = df["essay"].astype(str)
    rubric = load_rubric(dataset.rubric_path, dataset.label)
    _validate_scores(df, "human_score", rubric.scale.min, rubric.scale.max)
    criterion_names = {c.name for c in rubric.criteria}
    unknown = set(dataset.criteria_columns) - criterion_names
    if unknown:
        raise ValueError(
            f"Configured criterion columns are absent from the rubric: {sorted(unknown)}"
        )
    _require_columns(df, dataset.criteria_columns, dataset.label)
    scale = rubric.get_criterion_scale()
    for criterion in rubric.criteria:
        if criterion.name in df.columns:
            _validate_scores(df, criterion.name, scale.min, scale.max, optional=True)
    return df.reset_index(drop=True)


def _load_with_reader_loader(dataset: DatasetConfig, data_path: str | None = None) -> pd.DataFrame:
    """Call a reader's loader and hold its frame to the contract a built-in family's frame meets.

    The callable receives `dataset` and `data_path` as keyword arguments and returns a pandas
    DataFrame carrying at least essay_id, set_id, essay, and human_score. It replaces the reading
    and normalizing step; every check in `_validate_dataset` still runs on what it returns, so a
    reader's corpus cannot enter a run less validated than a packaged one.
    """
    label = f"Dataset {dataset.label!r} loader"
    frame = resolve_callable(dataset.loader, label)(
        dataset=dataset, data_path=data_path or dataset.raw_data_path or None
    )
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} must return a pandas DataFrame, got {type(frame).__name__}")
    if frame.empty:
        raise ValueError(f"{label} returned no rows")
    frame = frame.copy()
    _require_columns(frame, dataset.base_columns, dataset.label)
    if dataset.subset_id is not None:
        frame = frame[frame["set_id"].astype(str) == dataset.subset_id].copy()
    return frame


def load_dataset(dataset: DatasetConfig, data_path: str | None = None) -> pd.DataFrame:
    """Load a supported dataset and validate all rows against its rubric without dropping invalid scores."""
    if dataset.loader:
        df = _load_with_reader_loader(dataset, data_path)
    elif dataset.family == DatasetFamily.CUSTOM:
        df = _read_raw(data_path or dataset.raw_data_path)
        _require_columns(df, dataset.base_columns, dataset.label)
        if dataset.subset_id is not None:
            df = df[df["set_id"].astype(str) == dataset.subset_id].copy()
    elif dataset.family == DatasetFamily.ELLIPSE:
        df = _load_ellipse(dataset, data_path)
    elif dataset.family == DatasetFamily.ASAP_2_0:
        df = _load_asap_2_0(dataset, data_path)
    elif dataset.family == DatasetFamily.ASAP_AES:
        df = _load_asap_aes(dataset, data_path)
    else:
        raise ValueError(f"Unsupported dataset family: {dataset.family}")
    result = _validate_dataset(df, dataset)
    logger.info("Loaded %d valid essays for '%s'", len(result), dataset.label)
    return result


class DatasetLoader:
    """Loads Dataset objects (config + submissions) for one or all registered DatasetNames."""

    @staticmethod
    def load(name: DatasetConfig | DatasetNames, **paths) -> Dataset:
        """Load and return the Dataset (config + submissions) for one DatasetName."""
        config = name if isinstance(name, DatasetConfig) else get_dataset_config(name, **paths)
        rubric = load_rubric(config.rubric_path, config.label)
        holistic = rubric.holistic_criterion()
        df = load_dataset(config)
        human_assessments = [
            HumanAssessment(
                submission=StudentSubmission(essay_id=str(row.essay_id), text=str(row.essay)),
                human_score=round_half_up(row.human_score),
                criterion=holistic,
            )
            for row in df.itertuples()
        ]
        return Dataset(dataset_config=config, human_assessments=human_assessments)

    @staticmethod
    def load_all(family: DatasetFamily | None = None, **paths) -> list[Dataset]:
        """Load and return the Dataset for every DatasetName, optionally filtered to one family."""
        names = [
            n for n in DatasetNames if family is None or get_dataset_config(n).family == family
        ]
        return [DatasetLoader.load(n, **paths) for n in names]
