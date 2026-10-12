"""Compatibility helpers for evaluating registered datasets from result files."""

from pathlib import Path

from abaes_eval_fwk.config.dataset_config import DatasetNames, get_dataset_config
from abaes_eval_fwk.core.models.assessment import Dataset, HumanAssessment, StudentSubmission
from abaes_eval_fwk.experiment.eval.metrics.compute_metrics import grading_records
from abaes_eval_fwk.experiment.setup.dataset.dataset_loader import load_dataset
from abaes_eval_fwk.experiment.setup.rubric.rubric_loader import load_rubric
from abaes_eval_fwk.utils.rounding import round_half_up


def load_rubrics_and_datasets(csv_path: Path) -> tuple[list, list[Dataset]]:
    """Load only builtin datasets present in a historical results CSV."""
    rubrics, datasets = [], []
    for label, records in grading_records(csv_path).items():
        config = get_dataset_config(DatasetNames(label))
        rubric = load_rubric(config.rubric_path, label)
        needed = {r.human_assessment.essay_id for r in records}
        frame = load_dataset(config)
        humans = [
            HumanAssessment(
                StudentSubmission(str(row.essay_id), str(row.essay)),
                round_half_up(row.human_score),
                rubric.holistic_criterion(),
            )
            for row in frame.itertuples()
            if str(row.essay_id) in needed
        ]
        rubrics.append(rubric)
        datasets.append(Dataset(config, humans))
    return rubrics, datasets
