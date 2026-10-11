"""Incremental CSV result writer with thread-safe file handling."""

import csv
import logging
import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, NamedTuple, Self

from abaes_eval_fwk.config.reuse_config import ReuseConfig
from abaes_eval_fwk.core.models.assessment import (
    FAILURES_FILENAME_SUFFIX,
    RESULTS_FILENAME_SUFFIX,
    RESULTS_TIMESTAMP_FORMAT,
    REVIEWS_FILENAME_SUFFIX,
    FailureField,
    GradingRecord,
    HumanAssessment,
    ResultsRow,
    Review,
    ReviewField,
    ReviewRecord,
    ReviewRow,
    csv_flag,
    failures_fieldnames,
    results_fieldnames,
    review_fieldnames,
)
from abaes_eval_fwk.core.models.experiment import Experiment, FinalAgentProcedureName
from abaes_eval_fwk.core.models.rubric import Rubric
from abaes_eval_fwk.utils.prompt_formatters import format_examples

logger = logging.getLogger(__name__)


def _row_for(
    grading_record: GradingRecord, experiment: Experiment, rubric: Rubric, reuse: ReuseConfig
) -> ResultsRow:
    """Format a GradingRecord object into a flat CSV row."""
    result = grading_record.llm_assessment
    row = ResultsRow(
        experiment_id=experiment.name,
        dataset=rubric.rubric_id,
        essay_id=grading_record.human_assessment.essay_id,
        essay_text=grading_record.human_assessment.submission.text,
        model=experiment.llm.model_name,
        criterion=grading_record.human_assessment.criterion.name,
        human_score=grading_record.human_assessment.human_score,
        pred_score_raw=result.score.raw_score,
        pred_score_rounded=result.score.rounded_score,
        confidence=result.confidence,
        justification=result.justification,
        reasoning=result.reasoning or "",
        examples=format_examples(grading_record.reference_examples),
        example_cap=result.statistics.example_cap,
        start_time=result.statistics.start_time,
        total_time_seconds=result.statistics.total_time_seconds,
        prompt_tokens=result.statistics.token_usage.prompt_tokens,
        completion_tokens=result.statistics.token_usage.completion_tokens,
        total_tokens=result.statistics.token_usage.total_tokens,
        retry_error=result.statistics.retry_error,
        branch=result.branch,
        pool_average_score=result.pool_average_score,
    )
    _check_columns(row, results_fieldnames(), "_row_for/results_fieldnames")
    return row


def _review_row_for(
    record: ReviewRecord, experiment: Experiment, rubric: Rubric, reuse: ReuseConfig
) -> ReviewRow:
    """Format one ReviewRecord as a reviews CSV row: the verdict, plus the columns naming the
    assessment it was passed."""
    review = record.review
    row = ReviewRow(
        dataset=rubric.rubric_id,
        essay_id=record.human_assessment.essay_id,
        final_agent_procedure=experiment.final_agent_procedure.name.value,
        rubric_consistency=csv_flag(review.rubric_consistency),
        rubric_consistency_justification=review.rubric_consistency_justification,
        evidence_grounding=csv_flag(review.evidence_grounding),
        evidence_grounding_justification=review.evidence_grounding_justification,
        reasoning=review.reasoning,
        is_admissible=csv_flag(review.is_admissible),
        experiment_id=experiment.name,
        criterion=record.human_assessment.criterion.name,
        source_model=reuse.source_models[0] if reuse.source_models else "",
        source_experiment_id=reuse.source_experiments[0] if reuse.source_experiments else "",
        reviewer_model=experiment.llm.model_name,
        start_time=record.statistics.start_time,
        total_time_seconds=record.statistics.total_time_seconds,
        prompt_tokens=record.statistics.token_usage.prompt_tokens,
        completion_tokens=record.statistics.token_usage.completion_tokens,
        total_tokens=record.statistics.token_usage.total_tokens,
        retry_error=record.statistics.retry_error,
    )
    _check_columns(row, review_fieldnames(with_identity=True), "_review_row_for/review_fieldnames")
    return row


def _check_columns(row: Mapping[str, object], expected: list[str], where: str) -> None:
    """Guard against a row and its CSV header drifting apart."""
    if set(row) != set(expected):
        raise RuntimeError(f"{where} drifted: {set(row) ^ set(expected)}")


class _RowFormat(NamedTuple):
    """How one run's records become CSV rows: the file they are written to, its columns, and the
    builder for one row. A run's format is fixed by its procedure, so it is chosen once and
    `append` never has to inspect the record it is handed."""

    suffix: str
    fieldnames: list[str]
    build: Callable[..., ResultsRow | ReviewRow]


_GRADING_ROWS = _RowFormat(RESULTS_FILENAME_SUFFIX, results_fieldnames(), _row_for)
# A Review run decides no score, so its reviews CSV is its whole output — there is no results CSV.
_REVIEW_ROWS = _RowFormat(
    REVIEWS_FILENAME_SUFFIX, review_fieldnames(with_identity=True), _review_row_for
)


class ResultsWriter:
    """Thread-safe context manager that appends GradingRecord rows incrementally to CSV files."""

    def __init__(
        self,
        experiment: Experiment,
        results_dir: str,
        run_id: str = "",
        reuse: ReuseConfig | None = None,
    ) -> None:
        """Initialize ResultsWriter and create CSV header on disk."""
        self.is_review = experiment.final_agent_procedure.name == FinalAgentProcedureName.REVIEW
        self._rows = _REVIEW_ROWS if self.is_review else _GRADING_ROWS
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        self.ts = run_id if run_id else datetime.now(UTC).strftime(RESULTS_TIMESTAMP_FORMAT)
        stem = str(Path(results_dir) / f"{experiment.name}_{self.ts}")
        self.failures_csv_path = stem + FAILURES_FILENAME_SUFFIX
        self.reviews_csv_path = stem + REVIEWS_FILENAME_SUFFIX
        self.csv_path = stem + self._rows.suffix
        self.experiment = experiment
        self.reuse = reuse or ReuseConfig()
        self._lock = threading.Lock()
        self._n = 0
        self._n_failures = 0
        self._n_reviews = 0
        self._file = open(self.csv_path, "w", newline="", encoding="utf-8")  # noqa: SIM115
        self._writer = csv.DictWriter(self._file, fieldnames=self._rows.fieldnames)
        self._writer.writeheader()
        self._failures_file: IO[str] | None = None
        self._failures_writer: csv.DictWriter | None = None
        self._reviews_file: IO[str] | None = None
        self._reviews_writer: csv.DictWriter | None = None

    def append(self, record: GradingRecord | ReviewRecord, rubric: Rubric) -> None:
        """Thread-safely write one grading or review record to the CSV file."""
        row = self._rows.build(record, self.experiment, rubric, self.reuse)
        with self._lock:
            self._writer.writerow(row)
            self._file.flush()
            self._n += 1

    def append_failure(self, human_assessment: HumanAssessment, rubric: Rubric, error: str) -> None:
        """Thread-safely write one failed-grading record to the failures CSV file."""
        row = {
            FailureField.ESSAY_ID: human_assessment.submission.essay_id,
            FailureField.DATASET: rubric.rubric_id,
            FailureField.CRITERION: human_assessment.criterion.name,
            FailureField.HUMAN_SCORE: human_assessment.human_score,
            FailureField.ERROR: error,
        }
        with self._lock:
            if self._failures_writer is None:
                self._failures_file = open(  # noqa: SIM115
                    self.failures_csv_path, "w", newline="", encoding="utf-8"
                )
                self._failures_writer = csv.DictWriter(
                    self._failures_file, fieldnames=failures_fieldnames()
                )
                self._failures_writer.writeheader()
            self._failures_writer.writerow(row)
            if self._failures_file is not None:
                self._failures_file.flush()
            self._n_failures += 1

    @property
    def has_failures(self) -> bool:
        """True if at least one failure was recorded (i.e. the failures CSV was created)."""
        return self._n_failures > 0

    def append_review(
        self,
        procedure: FinalAgentProcedureName,
        dataset: str,
        essay_id: str,
        criterion: str,
        model: str,
        review: dict,
    ) -> None:
        """Thread-safely persist one review-turn judgment for a later run of the SAME procedure to
        replay (see ReviewCache). The file opens lazily, so procedures that never review leave none.
        """
        if self.is_review:
            raise RuntimeError("append_review would truncate a Review run's results CSV")
        verdict = Review(**review)
        row = {
            ReviewField.DATASET: dataset,
            ReviewField.ESSAY_ID: essay_id,
            ReviewField.FINAL_AGENT_PROCEDURE: procedure.value,
            ReviewField.CRITERION: criterion,
            ReviewField.SOURCE_MODEL: model,
            ReviewField.RUBRIC_CONSISTENCY: csv_flag(verdict.rubric_consistency),
            ReviewField.RUBRIC_CONSISTENCY_JUSTIFICATION: review.get(
                "rubric_consistency_justification", ""
            ),
            ReviewField.EVIDENCE_GROUNDING: csv_flag(verdict.evidence_grounding),
            ReviewField.EVIDENCE_GROUNDING_JUSTIFICATION: review.get(
                "evidence_grounding_justification", ""
            ),
            ReviewField.REASONING: review.get("reasoning", ""),
            ReviewField.IS_ADMISSIBLE: csv_flag(verdict.is_admissible),
        }
        with self._lock:
            if self._reviews_writer is None:
                existing = Path(self.reviews_csv_path).exists()
                self._reviews_file = open(  # noqa: SIM115
                    self.reviews_csv_path, "a", newline="", encoding="utf-8"
                )
                self._reviews_writer = csv.DictWriter(
                    self._reviews_file, fieldnames=review_fieldnames()
                )
                if not existing:
                    self._reviews_writer.writeheader()
            self._reviews_writer.writerow(row)
            if self._reviews_file is not None:
                self._reviews_file.flush()
            self._n_reviews += 1

    @property
    def has_reviews(self) -> bool:
        """True if at least one review was recorded (i.e. the reviews CSV was created)."""
        return self._n_reviews > 0

    def close(self, finalize: bool = True) -> None:
        """Close file handle(s)."""
        self._file.close()
        kind = "reviews" if self.is_review else "results"
        if finalize:
            logger.info("  [%s] %d records → %s", kind, self._n, Path(self.csv_path).name)
        else:
            logger.warning(
                "  [%s] run did not complete — %d record(s) written to %s",
                kind,
                self._n,
                Path(self.csv_path).name,
            )
        if self._failures_file is not None:
            self._failures_file.close()
            logger.info(
                "  [failures] %d records → %s", self._n_failures, Path(self.failures_csv_path).name
            )
        if self._reviews_file is not None:
            self._reviews_file.close()
            logger.info(
                "  [reviews] %d records → %s", self._n_reviews, Path(self.reviews_csv_path).name
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(finalize=exc_type is None)
