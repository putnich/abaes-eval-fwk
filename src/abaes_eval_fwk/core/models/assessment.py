"""Domain objects and scoring contracts exchanged across agents, orchestrators, and savers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import cached_property
from typing import TypedDict

from abaes_eval_fwk.config.dataset_config import DatasetConfig
from abaes_eval_fwk.core.models.rubric import Criterion, Rubric
from abaes_eval_fwk.utils.rounding import round_half_up


@dataclass(frozen=True)
class TokenUsage:
    """Accumulated token counts for one LLM call or pipeline run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total token count (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


@dataclass
class Statistics:
    """Timing, token-use, and error metadata recorded for an agent execution."""

    token_usage: TokenUsage = field(default_factory=TokenUsage)
    retry_error: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    example_cap: float | None = (
        None  # fraction each example's length was capped to; <1.0 means capping kicked in
    )

    @property
    def total_time_seconds(self) -> float | None:
        """Total elapsed execution time in seconds, or None if timestamps are unset."""
        if self.start_time is None or self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()


@dataclass(frozen=True)
class PredictedScore:
    """Numeric predicted score output by an agent."""

    raw_score: float

    @property
    def rounded_score(self) -> int:
        """Round score to nearest integer using half-up rounding (e.g. 3.5 -> 4)."""
        return round_half_up(self.raw_score)


@dataclass(frozen=True)
class LLMAssessment:
    """One agent call's assessment of one submission — the model's own score and reasoning. Deliberately holds no reference back to what was assessed, so code building prompts from an LLMAssessment has no path to the ground-truth score."""

    score: PredictedScore
    justification: str
    confidence: float | None
    reasoning: str | None = None
    branch: str = ""  # final-agent verdict branch (agreement/disagreement, positive/mixed/negative)
    pool_average_score: str = ""  # final-agent pool's naive average score, regardless of branch
    statistics: Statistics = field(default_factory=Statistics)


class ResultsField(str, Enum):
    """Fixed columns in every results CSV row."""

    EXPERIMENT_ID = "experiment_id"
    DATASET = "dataset"
    ESSAY_ID = "essay_id"
    ESSAY_TEXT = "essay_text"
    MODEL = "model"
    CRITERION = "criterion"
    HUMAN_SCORE = "human_score"
    PRED_SCORE_RAW = "pred_score_raw"
    PRED_SCORE_ROUNDED = "pred_score_rounded"
    CONFIDENCE = "confidence"
    JUSTIFICATION = "justification"
    REASONING = "reasoning"
    EXAMPLES = "examples"
    EXAMPLE_CAP = "example_cap"
    START_TIME = "start_time"
    TOTAL_TIME_SECONDS = "total_time_seconds"
    PROMPT_TOKENS = "prompt_tokens"
    COMPLETION_TOKENS = "completion_tokens"
    TOTAL_TOKENS = "total_tokens"
    RETRY_ERROR = "retry_error"
    BRANCH = "branch"
    POOL_AVERAGE_SCORE = "pool_average_score"


class ResultsRow(TypedDict):
    """One results CSV row. Keys mirror `ResultsField`; `results_fieldnames` fixes the order."""

    experiment_id: str
    dataset: str
    essay_id: str
    essay_text: str
    model: str
    criterion: str
    human_score: int
    pred_score_raw: float
    pred_score_rounded: int
    confidence: float | None
    justification: str
    reasoning: str
    examples: str
    example_cap: float | None
    start_time: datetime | None
    total_time_seconds: float | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    retry_error: str | None
    branch: str
    pool_average_score: str


def results_fieldnames() -> list[str]:
    """Return results CSV columns."""
    return list(ResultsField)


class FailureField(str, Enum):
    """Fixed columns in the failures CSV (one row per essay that raised during grading)."""

    ESSAY_ID = "essay_id"
    DATASET = "dataset"
    CRITERION = "criterion"
    HUMAN_SCORE = "human_score"
    ERROR = "error"


def failures_fieldnames() -> list[str]:
    """Return failures CSV columns."""
    return list(FailureField)


class ReviewField(str, Enum):
    """Fixed columns in the reviews CSV (one row per FinalAgent review-turn judgment, so a later run can reuse it instead of re-invoking the LLM). FINAL_AGENT_PROCEDURE scopes each row to the procedure that produced it, since reviews from one procedure must never satisfy another's lookup."""

    DATASET = "dataset"
    ESSAY_ID = "essay_id"
    FINAL_AGENT_PROCEDURE = "final_agent_procedure"
    CRITERION = "criterion"
    SOURCE_MODEL = "source_model"
    RUBRIC_CONSISTENCY = "rubric_consistency"
    RUBRIC_CONSISTENCY_JUSTIFICATION = "rubric_consistency_justification"
    EVIDENCE_GROUNDING = "evidence_grounding"
    EVIDENCE_GROUNDING_JUSTIFICATION = "evidence_grounding_justification"
    REASONING = "reasoning"
    IS_ADMISSIBLE = "is_admissible"
    START_TIME = "start_time"
    TOTAL_TIME_SECONDS = "total_time_seconds"
    PROMPT_TOKENS = "prompt_tokens"
    COMPLETION_TOKENS = "completion_tokens"
    TOTAL_TOKENS = "total_tokens"
    RETRY_ERROR = "retry_error"


class ReviewRow(TypedDict):
    """One reviews CSV row as a Review run writes it — the verdict plus the identity columns naming the run it judged. A run that only caches reviews leaves the identity columns out."""

    dataset: str
    essay_id: str
    experiment_id: str
    final_agent_procedure: str
    reviewer_model: str
    criterion: str
    source_experiment_id: str
    source_model: str
    rubric_consistency: int
    rubric_consistency_justification: str
    evidence_grounding: int
    evidence_grounding_justification: str
    is_admissible: int
    reasoning: str
    start_time: datetime | None
    total_time_seconds: float | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    retry_error: str | None


def csv_flag(value: bool) -> int:
    """Write a boolean CSV column as 1 or 0, the one spelling every framework CSV uses."""
    return 1 if value else 0


def parse_csv_flag(value: str | None, column: str) -> bool:
    """Read a boolean CSV column written as 1 or 0; any other spelling is malformed."""
    if value == "1":
        return True
    if value == "0":
        return False
    raise ValueError(f"{column} must be 1 or 0, got {value!r}")


REVIEWER_MODEL_FIELD = "reviewer_model"
SOURCE_EXPERIMENT_ID_FIELD = "source_experiment_id"

# Written only by a Review run, whose reviews CSV is its whole output and so has to name the run
# each verdict was passed. `source_model`/`source_experiment_id` (base columns) name its source.
REVIEW_IDENTITY_FIELDS = (
    ResultsField.EXPERIMENT_ID,
    SOURCE_EXPERIMENT_ID_FIELD,
    REVIEWER_MODEL_FIELD,
)


def review_fieldnames(with_identity: bool = False) -> list[str]:
    """Return reviews CSV columns in output order; a Review run adds the identity columns naming itself and the run it judged."""
    ordered = [
        ReviewField.DATASET,
        ReviewField.ESSAY_ID,
        ResultsField.EXPERIMENT_ID,
        ReviewField.FINAL_AGENT_PROCEDURE,
        REVIEWER_MODEL_FIELD,
        ReviewField.CRITERION,
        SOURCE_EXPERIMENT_ID_FIELD,
        ReviewField.SOURCE_MODEL,
        ReviewField.RUBRIC_CONSISTENCY,
        ReviewField.RUBRIC_CONSISTENCY_JUSTIFICATION,
        ReviewField.EVIDENCE_GROUNDING,
        ReviewField.EVIDENCE_GROUNDING_JUSTIFICATION,
        ReviewField.IS_ADMISSIBLE,
        ReviewField.REASONING,
        ReviewField.START_TIME,
        ReviewField.TOTAL_TIME_SECONDS,
        ReviewField.PROMPT_TOKENS,
        ReviewField.COMPLETION_TOKENS,
        ReviewField.TOTAL_TOKENS,
        ReviewField.RETRY_ERROR,
    ]
    if with_identity:
        return list(ordered)
    return [f for f in ordered if f not in REVIEW_IDENTITY_FIELDS]


RESULTS_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"  # Default timestamp format for results filenames
RESULTS_FILENAME_SUFFIX = "_results.csv"
FAILURES_FILENAME_SUFFIX = "_failures.csv"
REVIEWS_FILENAME_SUFFIX = "_reviews.csv"


@dataclass(frozen=True)
class StudentSubmission:
    """A student essay: the artifact being graded, and nothing about how it was scored."""

    essay_id: str
    text: str


@dataclass(frozen=True)
class HumanAssessment:
    """A human rater's assessment of one submission against one criterion. This is the grain of the source CSVs and of the retrieval pool alike: one row per (essay x criterion), with holistic datasets using the rubric's single "holistic" criterion for a uniform shape."""

    submission: StudentSubmission
    human_score: int
    criterion: Criterion

    @property
    def essay_id(self) -> str:
        """The assessed essay's id — by far the most-read field of the wrapped submission."""
        return self.submission.essay_id

    def as_example(self) -> str:
        """Format submission as a few-shot example block for LLM prompts."""
        return f'--- Score {self.human_score} ---\n"""\n{self.submission.text}\n"""'


@dataclass(frozen=True)
class Dataset:
    """A dataset config's loaded submissions, each with its human score for one criterion."""

    dataset_config: DatasetConfig | None = None
    human_assessments: list[HumanAssessment] = field(default_factory=list)

    @property
    def dataset_name(self) -> str | None:
        """Human-readable label (e.g. 'ASAP AES Set 1'), from the source DatasetConfig."""
        return self.dataset_config.label if self.dataset_config else None

    @cached_property
    def _scores_by_essay(self) -> dict[str, int]:
        return {a.submission.essay_id: a.human_score for a in self.human_assessments}

    def human_score(self, essay_id: str) -> int | None:
        """This dataset's officially recorded human score for one essay, or None if absent."""
        return self._scores_by_essay.get(essay_id)

    def __len__(self) -> int:
        return len(self.human_assessments)

    def __iter__(self):
        return iter(self.human_assessments)


@dataclass(frozen=True)
class Assignment:
    """A rubric-scored writing prompt: its rubric and dataset of submissions."""

    rubric: Rubric
    dataset: Dataset = field(default_factory=Dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        return iter(self.dataset)


class Sample(Assignment):
    """An Assignment's submissions sampled for grading (no results yet)."""


@dataclass(frozen=True)
class Review:
    """One admissibility verdict on a prior assessment: whether its justification supports its score under the rubric, and whether its claims are grounded in the submission. Each `*_justification` says what failed, and is empty when its criterion passed."""

    rubric_consistency: bool = False
    rubric_consistency_justification: str = ""
    evidence_grounding: bool = False
    evidence_grounding_justification: str = ""
    reasoning: str = ""  # native <think> content, when the model exposes it
    is_admissible: bool = False  # derived in __post_init__; never passed by a caller

    def __post_init__(self) -> None:
        """Store the verdict rather than re-deriving it, so it travels with the review."""
        object.__setattr__(
            self, "is_admissible", self.rubric_consistency and self.evidence_grounding
        )


# One shape for every record's run log, so the two kinds line up column-wise in the output. A
# record leaves empty whatever its own operation never produced.
_LOG_SUMMARY = (
    "[{label}] score={score}  human_score={human_score}  error={error}  "
    "confidence={confidence}  branch={branch}  tokens={tokens}"
)
# A review decides no score, so its line carries only the fields it actually produced.
_REVIEW_LOG_SUMMARY = (
    "[{label}] human_score={human_score}  admissible={admissible}  tokens={tokens}"
)
_REVIEW_LOG_DETAIL = (
    "    time={time}s  retry_error={retry_error}\n    justification={justification}"
)
_LOG_DETAIL = (
    "    time={time}s  retry_error={retry_error}\n"
    "    justification={justification}\n"
    "    reasoning={reasoning}"
)


@dataclass(frozen=True)
class ReviewRecord:
    """The complete record of one review operation: the verdict, and which prior assessment it was passed. A review decides no score, so there is no model score here to join against the human's — `label` is the reviewed assessment's cache key, naming its criterion and source model."""

    human_assessment: HumanAssessment
    review: Review
    label: str
    statistics: Statistics = field(default_factory=Statistics)

    @property
    def log_summary(self) -> str:
        """This record's run-log line, omitting the scoring fields no review decides."""
        return _REVIEW_LOG_SUMMARY.format(
            label=self.label,
            human_score=self.human_assessment.human_score,
            admissible=self.review.is_admissible,
            tokens=self.statistics.token_usage.total_tokens,
        )

    @property
    def log_detail(self) -> str:
        """This record's verbose run-log detail; the review's own justifications stand in for the assessment text a grading record would carry."""
        return _REVIEW_LOG_DETAIL.format(
            time=f"{self.statistics.total_time_seconds or 0.0:.2f}",
            retry_error=self.statistics.retry_error or "-",
            justification=" ".join(
                j
                for j in (
                    self.review.rubric_consistency_justification,
                    self.review.evidence_grounding_justification,
                )
                if j
            ),
        )


@dataclass(frozen=True)
class GradingRecord:
    """The complete record of one grading operation: what was graded and how it scored by both
    raters, plus the retrieval context the agent saw. Not itself an assessment — it is the join
    of the human's and the model's, which is what makes `error` computable."""

    human_assessment: HumanAssessment
    llm_assessment: LLMAssessment
    reference_examples: list[HumanAssessment] = field(default_factory=list)
    # Pre-formatted reference-examples text, set when this record is loaded back from a prior
    # run's CSV instead of freshly graded — `reference_examples` doesn't survive that round trip.
    examples_text: str = ""

    @property
    def error(self) -> int:
        """Absolute difference between predicted and human score."""
        return abs(self.llm_assessment.score.rounded_score - self.human_assessment.human_score)

    @property
    def log_summary(self) -> str:
        """This record's run-log line; branch stays empty outside a final-agent run."""
        return _LOG_SUMMARY.format(
            label=self.human_assessment.criterion.name,
            score=self.llm_assessment.score.rounded_score,
            human_score=self.human_assessment.human_score,
            error=self.error,
            confidence=self.llm_assessment.confidence,
            branch=self.llm_assessment.branch,
            tokens=self.llm_assessment.statistics.token_usage.total_tokens,
        )

    @property
    def log_detail(self) -> str:
        """This record's verbose run-log detail: timing, retry, and the agent's own text."""
        return _LOG_DETAIL.format(
            time=f"{self.llm_assessment.statistics.total_time_seconds or 0.0:.2f}",
            retry_error=self.llm_assessment.statistics.retry_error or "-",
            justification=self.llm_assessment.justification,
            reasoning=self.llm_assessment.reasoning or "-",
        )


@dataclass(frozen=True)
class GradedAssignment(Assignment):
    """An Assignment whose submissions have each produced a GradingRecord."""

    records: list[GradingRecord] = field(default_factory=list)
