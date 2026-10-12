"""Experiment variant configurations, retrieval settings, and setup abstractions."""

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum

from abaes_eval_fwk.config.dataset_config import DatasetConfig
from abaes_eval_fwk.config.llm_config import LLMConfig
from abaes_eval_fwk.config.prompts_config import PromptsConfig, default_prompts_config
from abaes_eval_fwk.config.retrieval_config import RetrievalConfig
from abaes_eval_fwk.core.models.assessment import HumanAssessment, Sample
from abaes_eval_fwk.core.models.rubric import Rubric


class ResearchQuestion(str, Enum):
    """Which research question (RQ1-RQ4) an experiment variant targets."""

    RQ1 = "RQ1"  # Example retrieval: zero-shot vs semantic/hybrid/lexical/random similarity
    RQ2 = "RQ2"  # Criterion specialization: per-criterion specialists vs holistic
    RQ3 = "RQ3"  # Reasoning (Qwen): model's native thinking mode on/off
    RQ4 = "RQ4"  # Final-agent procedures: Adjudication vs Adjudication with Review


class ExperimentType(str, Enum):
    """Supported grading architecture modes (RQ1-RQ4)."""

    HOLISTIC = "holistic"
    CRITERION_SPECIALIZED = "criterion_specialized"
    FINAL_AGENT = "final_agent"
    NON_GRADING = "non_grading"  # consumes prior results but decides no score of its own

    @property
    def is_final_agent(self) -> bool:
        """True if the experiment type is FINAL_AGENT adjudication."""
        return self == ExperimentType.FINAL_AGENT

    @property
    def is_non_grading(self) -> bool:
        """True if the experiment type produces no score of its own."""
        return self == ExperimentType.NON_GRADING

    @property
    def reuses_results(self) -> bool:
        """True if the experiment type works from prior results rather than grading from scratch."""
        return self.is_final_agent or self.is_non_grading


class FinalAgentProcedureName(str, Enum):
    """Which procedure the FinalAgent runs over prior results."""

    ADJUDICATION = "adjudication"  # Compare the models' scores, take the independent assessment on disagreement
    ADJUDICATION_WITH_REVIEW = "adjudication_with_review"  # Review every model for admissibility, then adjudicate on the outcome
    REVIEW = "review"  # Non-grading: review one model's assessments and stop
    NONE = ""  # default for experiments that grade from scratch


@dataclass(frozen=True)
class FinalAgentProcedure:
    """Configuration for the final agent's adjudication procedure."""

    name: FinalAgentProcedureName

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", FinalAgentProcedureName(self.name))

    @classmethod
    def none(cls) -> "FinalAgentProcedure":
        return cls(name=FinalAgentProcedureName.NONE)


@dataclass(frozen=True)
class CalibrationPool:
    """Few-shot calibration submissions and their source dataset."""

    human_assessments: list[HumanAssessment] = field(default_factory=list)
    dataset: DatasetConfig | None = None
    # Retrieval-index location for pools with no DatasetConfig to derive one from — a
    # user-uploaded pool has no registered DatasetName, and an empty path makes the index
    # loader try to create a directory named "".
    index_path: str = ""

    @property
    def calibration_index_path(self) -> str:
        return self.index_path or (self.dataset.calibration_index_path if self.dataset else "")

    def __bool__(self) -> bool:
        return bool(self.human_assessments)

    def __len__(self) -> int:
        return len(self.human_assessments)

    def __iter__(self) -> Iterator[HumanAssessment]:
        return iter(self.human_assessments)


@dataclass(frozen=True)
class ExperimentalSetup:
    """A rubric's evaluation sample and calibration pool."""

    sample: Sample
    calibration_pool: CalibrationPool = field(default_factory=CalibrationPool)

    @property
    def rubric(self) -> Rubric:
        return self.sample.rubric


@dataclass(frozen=True)
class Experiment:
    """Fully specifies one experiment variant for comparison studies."""

    rq: ResearchQuestion
    experiment_type: ExperimentType
    llm: LLMConfig  # RQ3: override reasoning mode per variant; every experiment names its model
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    prompts: PromptsConfig = field(default_factory=default_prompts_config)
    final_agent_procedure: FinalAgentProcedure = field(default_factory=FinalAgentProcedure.none)
    setups: list[ExperimentalSetup] = field(default_factory=list)
    description: str = ""

    @property
    def name(self) -> str:
        """Identifier used as the CSV experiment_id / output filename prefix."""
        segments = [
            self.rq.value,
            self.experiment_type.value,
            # A reader's selector replaces the built-in policy name; without this a custom
            # retriever and a `random` one would produce the same experiment_id.
            self.retrieval.selector or self.retrieval.examples_selection.value,
            self.final_agent_procedure.name.value,
            self.llm.model_shorthand,
            self.llm.reasoning.value,
        ]
        raw = "_".join(s for s in segments if s)
        return self._sanitize_id(raw)

    @staticmethod
    def _sanitize_id(raw: str) -> str:
        """Sanitize ID to filesystem-safe length, using hash suffix for truncation."""
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
        return (
            safe
            if len(safe) <= 160
            else safe[:140] + "_" + hashlib.sha256(raw.encode()).hexdigest()[:16]
        )
