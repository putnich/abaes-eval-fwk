"""Builds ExperimentalSetup objects: sampled essays, rubric, and calibration pool per dataset."""

from __future__ import annotations

import dataclasses
import logging

from abaes_eval_fwk.config.dataset_config import DatasetConfig
from abaes_eval_fwk.config.reuse_config import ReuseConfig
from abaes_eval_fwk.core.models.experiment import (
    CalibrationPool,
    Experiment,
    ExperimentalSetup,
    ExperimentType,
    FinalAgentProcedureName,
)
from abaes_eval_fwk.experiment.setup.calibration.calibration_loader import (
    load_calibration_pool,
)
from abaes_eval_fwk.experiment.setup.dataset.dataset_loader import load_dataset
from abaes_eval_fwk.experiment.setup.reuse_loader import read_rows, source_is_criterion
from abaes_eval_fwk.experiment.setup.rubric.rubric_loader import load_rubric
from abaes_eval_fwk.experiment.setup.sample.data_sampler import (
    build_sample,
    stratified_quantile_rank_sample,
)
from abaes_eval_fwk.experiment.setup.sample.strategies import require_frame, resolve_strategy

logger = logging.getLogger(__name__)


def build_experiment(
    experiment: Experiment,
    datasets: list[DatasetConfig],
    sample_size: int,
    evaluation_strategy: str,
    calibration_strategy: str,
    essay_ids: list[str] | None = None,
    reuse: ReuseConfig | None = None,
) -> Experiment:
    """Populate an Experiment configuration with ExperimentalSetup instances for each dataset."""
    procedure = experiment.final_agent_procedure.name
    reuse = reuse or ReuseConfig()
    is_criterion = experiment.experiment_type == ExperimentType.CRITERION_SPECIALIZED
    if procedure == FinalAgentProcedureName.REVIEW:
        reuse.require(experiment.name, procedure.value)
    setups = [
        build_setup(
            dataset,
            is_criterion,
            sample_size,
            essay_ids=essay_ids,
            reuse=reuse if experiment.experiment_type.reuses_results else None,
            procedure=procedure,
            evaluation_strategy=evaluation_strategy,
            calibration_strategy=calibration_strategy,
        )
        for dataset in datasets
    ]
    return dataclasses.replace(experiment, setups=setups)


def build_setup(
    dataset: DatasetConfig,
    is_criterion: bool,
    sample_size: int = 200,
    *,
    essay_ids: list[str] | None = None,
    reuse: ReuseConfig | None = None,
    procedure: FinalAgentProcedureName = FinalAgentProcedureName.NONE,
    evaluation_strategy: str = "quantile_rank",
    calibration_strategy: str = "stratified",
) -> ExperimentalSetup:
    """Build a single ExperimentalSetup containing rubric, sample submissions, and calibration pool. `reuse` is set only for a run working from prior results, which supply its sample instead of sampling."""
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    if essay_ids is not None:
        essay_ids = [str(value) for value in essay_ids]
        if not essay_ids or len(essay_ids) != len(set(essay_ids)):
            raise ValueError("essay_ids must be a nonempty list of unique IDs")
    rubric = load_rubric(dataset.rubric_path, dataset.label)
    if reuse is not None:
        if procedure == FinalAgentProcedureName.REVIEW:
            is_criterion = source_is_criterion(reuse.results_paths[0], rubric)
        reused = list(read_rows(reuse.results_paths[0], rubric.rubric_id))
        if essay_ids is not None and (missing := set(essay_ids) - set(reused)):
            raise ValueError(
                f"Requested essay IDs are absent from reused results: {sorted(missing)}"
            )
        essay_ids = essay_ids if essay_ids is not None else reused
        if not essay_ids:
            raise ValueError(f"Reused results contain no essays for {dataset.label}")
        logger.info(
            "%s: sample taken from prior results — %d essays", dataset.label, len(essay_ids)
        )
    df_full = load_dataset(dataset)
    if reuse is not None:
        # A reuse run retrieves no examples, and excluding the pool would drop prior results.
        calibration_pool, sample_df = CalibrationPool(), df_full
    else:
        calibration_pool = load_calibration_pool(dataset, df_full, rubric, calibration_strategy)
        excluded_ids = {s.essay_id for s in calibration_pool}
        sample_df = df_full[~df_full["essay_id"].isin(excluded_ids)].reset_index(drop=True)
        logger.info("%s calibration essays excluded: %d", dataset.label, len(excluded_ids))

    if essay_ids is not None:
        missing = set(essay_ids) - set(sample_df["essay_id"])
        if missing:
            raise ValueError(
                f"Requested essay IDs are missing from evaluation data or belong to calibration: {sorted(missing)}"
            )
        sample_df = sample_df[sample_df["essay_id"].isin(essay_ids)].reset_index(drop=True)
    else:
        sampler = resolve_strategy(
            evaluation_strategy, "quantile_rank", stratified_quantile_rank_sample
        )
        sample_df = require_frame(sampler(sample_df, n_per_set=sample_size), evaluation_strategy)
    logger.info("%s dataset: %d essays", dataset.label, len(sample_df))

    return ExperimentalSetup(
        sample=build_sample(sample_df, rubric, is_criterion, dataset_config=dataset),
        calibration_pool=calibration_pool,
    )
