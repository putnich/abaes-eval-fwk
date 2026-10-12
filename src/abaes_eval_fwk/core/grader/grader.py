"""LangGraph state machine for grading one submission, with optional SQLite checkpointing."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, TypedDict

from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from abaes_eval_fwk.config.retrieval_config import RetrievalConfig
from abaes_eval_fwk.config.reuse_config import ReuseConfig
from abaes_eval_fwk.core.agents import HolisticGradingAgent, SpecialistAgent
from abaes_eval_fwk.core.agents.final_agent import FinalAgentVariantFactory, ReviewFinalAgent
from abaes_eval_fwk.core.models.assessment import (
    GradingRecord,
    HumanAssessment,
    LLMAssessment,
    PredictedScore,
    Review,
    ReviewRecord,
    Statistics,
    StudentSubmission,
    TokenUsage,
)
from abaes_eval_fwk.core.models.experiment import (
    CalibrationPool,
    Experiment,
    ExperimentType,
    FinalAgentProcedureName,
)
from abaes_eval_fwk.core.models.rubric import (
    CriteriaAggregation,
    Criterion,
    Rubric,
    Scale,
    ScoreLevel,
)
from abaes_eval_fwk.core.retriever.retriever import Retriever
from abaes_eval_fwk.experiment.setup.reuse_loader import ReviewCache

# Bump when the thread-id encoding changes: ids from an older scheme then address no checkpoint,
# so a saved run is regraded instead of resumed against rules it was not written under.
CHECKPOINT_KEY_VERSION = "ckpt-v2"

# Every domain type reachable from GradingState. LangGraph refuses to deserialize types a
# checkpoint doesn't declare, so the ones our own state carries are allowlisted here.
CHECKPOINTED_TYPES = (
    CriteriaAggregation,
    Criterion,
    GradingRecord,
    HumanAssessment,
    LLMAssessment,
    PredictedScore,
    Review,
    ReviewRecord,
    Rubric,
    Scale,
    ScoreLevel,
    Statistics,
    StudentSubmission,
    TokenUsage,
)


class GradingState(TypedDict, total=False):
    """The per-essay graph state — and, with a checkpointer attached, exactly what gets serialized into the checkpoint DB on every graph step. Deliberately minimal: anything constant for the whole run lives on `_RunContext` instead, since holding the `Experiment` here re-serialized every dataset's sample and calibration pool into every essay's row and cost one 60-essay run a 663 MB checkpoint."""

    rubric: Rubric
    human_assessment: HumanAssessment
    grading_records: list[list[GradingRecord]]
    # The final agent's own prior assessment of this essay, standing in for the independent
    # reassessment every adjudication branch that discards the candidate scores calls for.
    independent_records: list[GradingRecord]
    is_criterion: bool
    grading_record: GradingRecord
    # A Review run decides no score, so its node fills this slot instead of `grading_record`.
    review_record: ReviewRecord


def _outcome(values: dict) -> GradingRecord | ReviewRecord | None:
    """The graph's single outcome, from whichever slot this experiment's node fills."""
    return values.get("grading_record") or values.get("review_record")


def checkpoint_thread_id(rubric_id: str, essay_id: str, criterion_name: str) -> str:
    """The LangGraph thread id for one (rubric, essay, criterion), encoded so no two distinct
    tuples can collide.
    """
    identity = json.dumps(
        [rubric_id, essay_id, criterion_name], ensure_ascii=False, separators=(",", ":")
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"{CHECKPOINT_KEY_VERSION}:{digest}"


def _matches_request(
    record: GradingRecord | ReviewRecord, human_assessment: HumanAssessment
) -> bool:
    """Whether a recovered record is this submission's, not merely stored under a key like it."""
    cached = record.human_assessment
    return (
        cached.submission.essay_id == human_assessment.submission.essay_id
        and cached.criterion.name == human_assessment.criterion.name
        and cached.submission.text == human_assessment.submission.text
    )


@dataclass
class _RunContext:
    """Run-invariant inputs the nodes need, held outside `GradingState` so they are never
    checkpointed. `llm`/`review_cache` are set by `Grader.grade` (identical on every call within
    a run); `calibration_pools` is keyed by rubric_id since it varies per dataset."""

    experiment: Experiment
    reuse: ReuseConfig = field(default_factory=ReuseConfig)
    llm: Runnable | None = None
    review_cache: ReviewCache | None = None
    calibration_pools: dict[str, CalibrationPool] = field(default_factory=dict)
    run_id: str = "run"
    artifact_dir: Path = field(default_factory=lambda: Path("."))


# ── Nodes ─────────────────────────────────────────────────────────────────────


def _get_reference_examples(
    rubric: Rubric,
    calibration_pool: CalibrationPool | None,
    retrieval: RetrievalConfig,
    human_assessment: HumanAssessment,
    criterion: Criterion,
) -> list[HumanAssessment]:
    """Retrieve reference examples per the experiment's retrieval strategy or named selector."""
    return Retriever.select_reference_examples(
        rubric,
        calibration_pool,
        human_assessment.submission.text,
        retrieval,
        criterion,
    )


# ── Graph builder ─────────────────────────────────────────────────────────────


def build_graph(ctx: _RunContext, checkpointer: BaseCheckpointSaver | None = None):
    """Compile the workflow for one experiment type."""
    experiment = ctx.experiment

    def node_holistic(state: GradingState) -> GradingState:
        """HOLISTIC: grade the submission with a single LLM call."""
        rubric = state["rubric"]
        human_assessment = state["human_assessment"]
        reference_examples = _get_reference_examples(
            rubric,
            ctx.calibration_pools.get(rubric.rubric_id),
            experiment.retrieval,
            human_assessment,
            rubric.holistic_criterion(),
        )
        llm_assessment = HolisticGradingAgent(experiment.prompts.holistic).grade(
            rubric,
            human_assessment,
            ctx.llm,
            reference_examples=reference_examples,
            rq=experiment.rq,
            reasoning=experiment.llm.reasoning,
        )
        grading_record = GradingRecord(
            human_assessment=human_assessment,
            llm_assessment=llm_assessment,
            reference_examples=reference_examples,
        )
        return {"grading_record": grading_record}

    def node_grade_specialist(state: GradingState) -> GradingState:
        """Grade one submission with its criterion-specific specialist."""
        rubric = state["rubric"]
        human_assessment = state["human_assessment"]
        specialist = SpecialistAgent(
            human_assessment.criterion, experiment.prompts.criterion_specialized
        )
        reference_examples = _get_reference_examples(
            rubric,
            ctx.calibration_pools.get(rubric.rubric_id),
            experiment.retrieval,
            human_assessment,
            human_assessment.criterion,
        )
        llm_assessment = specialist.grade(
            rubric,
            human_assessment,
            ctx.llm,
            reference_examples=reference_examples,
            rq=experiment.rq,
            reasoning=experiment.llm.reasoning,
        )
        grading_record = GradingRecord(
            human_assessment=human_assessment,
            llm_assessment=llm_assessment,
            reference_examples=reference_examples,
        )
        return {"grading_record": grading_record}

    def node_final_agent(state: GradingState) -> GradingState:
        """Adjudicate caller-supplied results into one holistic assessment."""
        results = state.get("grading_records")
        if not results:
            raise ValueError(
                f"Experiment '{experiment.name}': experiment_type=final_agent "
                "adjudicates over already-graded `results` — none were provided in the initial state."
            )
        rubric = state["rubric"]
        human_assessment = state["human_assessment"]
        is_criterion = state.get("is_criterion", False)
        agent = (
            FinalAgentVariantFactory(is_criterion)
            .produce(experiment.final_agent_procedure)
            .produce()
        )
        agent.source_labels = ctx.reuse.source_models
        llm_assessment = agent.grade(
            rubric,
            human_assessment,
            ctx.llm,
            results=results,
            independent=state.get("independent_records") or [],
            review_cache=ctx.review_cache,
            rq=experiment.rq,
            reasoning=experiment.llm.reasoning,
        )
        grading_record = GradingRecord(
            human_assessment=human_assessment,
            llm_assessment=llm_assessment,
        )
        return {"grading_record": grading_record}

    def node_review(state: GradingState) -> GradingState:
        """REVIEW: judge one prior assessment for admissibility, deciding no score."""
        results = state.get("grading_records")
        if not results:
            raise ValueError(
                f"Experiment '{experiment.name}': the review procedure reviews already-graded "
                "`results` — none were provided in the initial state."
            )
        human_assessment = state["human_assessment"]
        agent = ReviewFinalAgent(human_assessment.criterion)
        agent.source_labels = ctx.reuse.source_models
        review_record = agent.review(
            state["rubric"],
            human_assessment,
            ctx.llm,
            results=results,
            review_cache=ctx.review_cache,
            rq=experiment.rq,
            reasoning=experiment.llm.reasoning,
        )
        return {"review_record": review_record}

    builder = StateGraph(GradingState)

    if experiment.experiment_type == ExperimentType.HOLISTIC:
        builder.add_node("holistic", node_holistic)
        builder.add_edge(START, "holistic")
        builder.add_edge("holistic", END)

    elif experiment.experiment_type == ExperimentType.CRITERION_SPECIALIZED:
        builder.add_node("grade_specialist", node_grade_specialist)
        builder.add_edge(START, "grade_specialist")
        builder.add_edge("grade_specialist", END)

    elif experiment.experiment_type == ExperimentType.FINAL_AGENT:
        builder.add_node("final_agent", node_final_agent)
        builder.add_edge(START, "final_agent")
        builder.add_edge("final_agent", END)

    elif (
        experiment.experiment_type == ExperimentType.NON_GRADING
        and experiment.final_agent_procedure.name == FinalAgentProcedureName.REVIEW
    ):
        builder.add_node("review", node_review)
        builder.add_edge(START, "review")
        builder.add_edge("review", END)

    else:
        raise ValueError(f"Unhandled experiment_type: {experiment.experiment_type}")

    return builder.compile(checkpointer=checkpointer)


# ── Grader ────────────────────────────────────────────────────────────────────

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

# Stay under macOS/Linux's 255-byte filename-component limit, leaving room for ".sqlite".
_MAX_SLUG_LEN = 200


def checkpoint_path_for(
    experiment_name: str,
    model_name: str,
    source_models: tuple[str, ...] = (),
    source_experiments: tuple[str, ...] = (),
    independent_experiment: str = "",
    base: str | None = None,
) -> str:
    """Checkpoint DB path for one (variant, model, sources) tuple. An entry is looked up by (rubric_id, essay_id, criterion) alone, so any two runs sharing a DB would return each other's records — source_experiments is included alongside source_models so two experiments reusing the same source model (e.g."""
    if base is None:
        raise ValueError("base path is required")
    parts = [
        experiment_name,
        model_name,
        *source_models,
        *source_experiments,
        *((independent_experiment,) if independent_experiment else ()),
    ]
    slug = _UNSAFE_FILENAME_CHARS.sub("_", "__".join(parts))
    if len(slug) > _MAX_SLUG_LEN:
        # Too many/verbose reuse sources to fit a filename — identify the run by its two
        # required names plus a hash of every part, so distinct source/independent
        # combinations still never collide.
        digest = hashlib.sha256("__".join(parts).encode()).hexdigest()[:16]
        prefix = _UNSAFE_FILENAME_CHARS.sub("_", f"{experiment_name}__{model_name}")[:100]
        slug = f"{prefix}__{digest}"
    return str(Path(base).with_name(f"{slug}{Path(base).suffix}"))


def clear_checkpoint(checkpoint_path: str) -> None:
    """Delete a grader checkpoint DB and its WAL/SHM sidecars, if present."""
    for suffix in ("", "-shm", "-wal"):
        path = Path(f"{checkpoint_path}{suffix}")
        if path.exists():
            path.unlink()


class Grader:
    """Run an experiment's compiled workflow for individual submissions."""

    def __init__(
        self,
        experiment: Experiment,
        checkpoint_path: str | None = None,
        reuse: ReuseConfig | None = None,
        run_id: str = "run",
        artifact_dir: Path | None = None,
    ) -> None:
        self.experiment = experiment
        self.reuse = reuse or ReuseConfig()
        self._ctx = _RunContext(
            experiment=experiment,
            reuse=self.reuse,
            run_id=run_id,
            artifact_dir=artifact_dir or Path("."),
        )
        self._conn: sqlite3.Connection | None = None
        self._checkpointer: SqliteSaver | None = None
        if checkpoint_path is not None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                checkpoint_path, check_same_thread=False
            )  # For concurrent workers
            self._checkpointer = SqliteSaver(
                self._conn,
                serde=JsonPlusSerializer(
                    pickle_fallback=True, allowed_msgpack_modules=CHECKPOINTED_TYPES
                ),
            )
        self._graph = build_graph(self._ctx, checkpointer=self._checkpointer)

    def close(self) -> None:
        """Close the checkpoint DB connection, flushing its WAL. Idempotent."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def grade(
        self,
        rubric: Rubric,
        human_assessment: HumanAssessment,
        llm: Runnable,
        calibration_pool: CalibrationPool | None = None,
        reuse_records: list[list[GradingRecord]] | None = None,
        independent_records: list[GradingRecord] | None = None,
        is_criterion: bool = False,
        review_cache: ReviewCache | None = None,
    ) -> GradingRecord | ReviewRecord:
        """Grade one submission and return its record — a ReviewRecord on a Review run, which decides no score."""
        # Run-invariant inputs go on the context, not into the checkpointed state. Workers all
        # pass the same `llm`/`review_cache`, and a rubric_id's pool is always the same object.
        self._ctx.llm = llm
        self._ctx.review_cache = review_cache
        if calibration_pool is not None:
            self._ctx.calibration_pools[rubric.rubric_id] = calibration_pool

        initial: GradingState = {
            "rubric": rubric,
            "human_assessment": human_assessment,
            "grading_records": reuse_records or [],
            "independent_records": independent_records or [],
            "is_criterion": is_criterion,
        }

        if self._checkpointer is None:
            state = self._graph.invoke(initial)
            return _outcome(state)  # type: ignore[return-value]

        config = {
            "configurable": {
                "thread_id": checkpoint_thread_id(
                    rubric.rubric_id,
                    human_assessment.essay_id,
                    human_assessment.criterion.name,
                ),
            }
        }
        existing = self._graph.get_state(config)
        cached = _outcome(existing.values)
        # A key match alone is not identity: a record recovered for another submission is discarded
        # and regraded rather than returned as this essay's result.
        if not existing.next and cached and _matches_request(cached, human_assessment):
            return cached
        state = self._graph.invoke(None if existing.next else initial, config=config)
        return _outcome(state)  # type: ignore[return-value]
