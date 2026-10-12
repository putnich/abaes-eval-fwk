"""Persist deterministic score-stratified calibration pools keyed by input contents and split settings."""

import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from abaes_eval_fwk.config.dataset_config import DatasetConfig
from abaes_eval_fwk.core.models.assessment import HumanAssessment, StudentSubmission
from abaes_eval_fwk.core.models.experiment import CalibrationPool
from abaes_eval_fwk.core.models.rubric import Rubric
from abaes_eval_fwk.experiment.setup.calibration.calibration_generator import (
    generate_calibration_pool,
)
from abaes_eval_fwk.experiment.setup.sample.data_sampler import per_criterion_assessments
from abaes_eval_fwk.experiment.setup.sample.strategies import allow_empty_frame, resolve_strategy
from abaes_eval_fwk.utils.rounding import round_half_up


def _pool_fingerprint(
    df_full: pd.DataFrame, dataset: DatasetConfig, calibration_strategy: str
) -> str:
    digest = hashlib.sha256()
    digest.update(df_full.sort_values("essay_id", kind="stable").to_csv(index=False).encode())
    digest.update(Path(dataset.rubric_path).read_bytes())
    digest.update(
        json.dumps(
            [
                "split-v2",
                dataset.label,
                dataset.seed,
                dataset.calibration_per_score,
                calibration_strategy,
            ]
        ).encode()
    )
    return digest.hexdigest()[:24]


# Columns that name a row rather than measure one, so they must survive the CSV round trip as the
# exact text written: pandas would otherwise infer "001" as 1 and read a literal "NA" as missing.
_IDENTITY_COLUMNS = ("rubric_id", "essay_id")


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Put a cached or freshly generated pool in one comparable shape.

    Identity columns stay verbatim text and score columns stay numeric, so a comparison judges the
    split's content rather than the round trip's type inference.
    """
    normalized = frame.copy()
    for column in normalized.columns:
        if column in _IDENTITY_COLUMNS:
            normalized[column] = normalized[column].astype("string")
        else:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def _read_calibration_csv(
    path: str,
    df_full: pd.DataFrame,
    dataset: DatasetConfig,
    rubric: Rubric,
    calibration_strategy: str,
) -> pd.DataFrame:
    """Validate an existing split or write its deterministic replacement under a content-specific filename."""
    base = Path(path)
    destination = base.with_name(
        f"{base.stem}.{_pool_fingerprint(df_full, dataset, calibration_strategy)}.csv"
    )
    generator = resolve_strategy(calibration_strategy, "stratified", generate_calibration_pool)
    expected = allow_empty_frame(generator(dataset, df_full, rubric), calibration_strategy)
    if destination.exists():
        pool = pd.read_csv(
            destination,
            dtype=dict.fromkeys(_IDENTITY_COLUMNS, "string"),
            keep_default_na=False,
            na_values=[""],
        )
        try:
            pd.testing.assert_frame_equal(
                _normalize(pool).fillna("").astype(str),
                _normalize(expected).fillna("").astype(str),
                check_dtype=False,
            )
        except AssertionError as exc:
            raise ValueError(
                f"Calibration cache does not match its deterministic split: {destination}; remove it to rebuild"
            ) from exc
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=destination.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            expected.to_csv(handle, index=False)
        temporary.replace(destination)
        pool = expected
    # Merged as text on both sides, so a leading-zero or "NA" essay id still joins its own row.
    return _normalize(pool).merge(
        df_full[["essay_id", "essay"]].assign(
            essay_id=lambda frame: frame["essay_id"].astype("string")
        ),
        on="essay_id",
        how="left",
        validate="one_to_one",
    )


def load_calibration_pool(
    dataset: DatasetConfig,
    df_full: pd.DataFrame,
    rubric: Rubric,
    calibration_strategy: str = "stratified",
) -> CalibrationPool:
    """Build calibration assessments from a validated, deterministic split."""
    pool_df = _read_calibration_csv(
        dataset.calibration_pool_path, df_full, dataset, rubric, calibration_strategy
    )
    human_assessments: list[HumanAssessment] = []
    for row in pool_df.to_dict("records"):
        human_assessments.append(
            HumanAssessment(
                submission=StudentSubmission(essay_id=str(row["essay_id"]), text=str(row["essay"])),
                human_score=round_half_up(row["human_score"]),
                criterion=rubric.holistic_criterion(),
            )
        )
        human_assessments.extend(per_criterion_assessments(row, rubric))
    return CalibrationPool(human_assessments=human_assessments, dataset=dataset)
