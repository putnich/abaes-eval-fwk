"""Final agents combine a conversation structure with a review or scoring procedure. `BaseFinalAgent` declares both axes' stub methods so a class supplying only one axis stays abstract until combined with the other."""

import itertools
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from abaes_eval_fwk.config.llm_config import Reasoning
from abaes_eval_fwk.core.agents.base_agent import TokenUsageCallback, TruncatedResponseError
from abaes_eval_fwk.core.agents.prompts.final_agent.criterion.adjudication_prompts import (
    ADJUDICATION_CRITERION_TEMPLATES,
)
from abaes_eval_fwk.core.agents.prompts.final_agent.criterion.adjudication_with_review_prompts import (
    ADJUDICATION_WITH_REVIEW_CRITERION_TEMPLATES,
)
from abaes_eval_fwk.core.agents.prompts.final_agent.holistic.adjudication_prompts import (
    ADJUDICATION_HOLISTIC_TEMPLATES,
)
from abaes_eval_fwk.core.agents.prompts.final_agent.holistic.adjudication_with_review_prompts import (
    ADJUDICATION_WITH_REVIEW_HOLISTIC_TEMPLATES,
)
from abaes_eval_fwk.core.agents.prompts.final_agent.review_prompts import (
    FINAL_PROMPT_REVIEW_ITEM_SYSTEM_TEMPLATE,
    FINAL_PROMPT_REVIEW_ITEM_USER_TEMPLATE,
)
from abaes_eval_fwk.core.agents.schemas.results import GradingNoScoreResponse, ReviewResult
from abaes_eval_fwk.core.llm.llm_loader import bind_tools, structured_output
from abaes_eval_fwk.core.models.assessment import (
    GradingRecord,
    HumanAssessment,
    LLMAssessment,
    Review,
    ReviewRecord,
    Statistics,
)
from abaes_eval_fwk.core.models.experiment import (
    FinalAgentProcedure,
    FinalAgentProcedureName,
    ResearchQuestion,
)
from abaes_eval_fwk.core.models.rubric import (
    NAPD_REFERENCE_WIDTH,
    Criterion,
    Rubric,
    Scale,
    combine_criterion_scores,
)
from abaes_eval_fwk.experiment.result.result_parsers import parse_grading_result
from abaes_eval_fwk.experiment.setup.reuse_loader import ReviewCache, _source_assessment_digest
from abaes_eval_fwk.utils.prompt_formatters import format_criteria
from abaes_eval_fwk.utils.rounding import round_half_up

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

SourceAssessments = list[GradingRecord]


# How a holistic item is named to the LLM; its criterion name keys the cache instead.
HOLISTIC_LABEL = "holistic, the submission as a whole"


class _ItemResult(TypedDict, total=False):
    """One item's parsed turn result: schema fields plus the branch metadata `_process_item` adds."""

    score: int
    confidence: float | None
    justification: str
    reasoning: str
    branch: str
    pool_average_score: str


def review_label(criterion: str, model: str) -> str:
    """Name one reviewed assessment for the run log, matching the `{criterion}: {branch}` shape
    every other turn header uses."""
    return f"{criterion}: review — {model}"


class _GradingItem(NamedTuple):
    """A criterion or whole-submission unit for adjudication."""

    label: str  # criterion name; also the review-cache key
    prompt_label: str  # how the item is named to the LLM
    scale: Scale  # this item's own score range
    guidelines: str  # rubric guidance scoped to this item
    assessments: list[LLMAssessment]
    template_inputs: dict  # values available to this item's turn templates
    # The final agent's own prior assessment of this item, reused as the independent assessment
    # every reassessing branch calls for. None only for a procedure that never reassesses.
    independent: LLMAssessment | None = None


class Turn(NamedTuple):
    """Formatted [system, user] text for one LLM turn."""

    system: str
    user: str


class Verdict(NamedTuple):
    """The procedure-specific outcome that selects an item's branch turn, carrying any score the
    procedure could already resolve without an LLM call."""

    branch: str
    resolved_score: int | None = None
    winner_label: str | None = None
    # Index into `_GradingItem.assessments` of the assessment this branch adopts, when the branch
    # adopts one wholesale (see `BaseFinalAgent._adopt_branches`). Left None for a branch that
    # instead adopts the item's independent assessment, since that one isn't part of the pool.
    winner_idx: int | None = None
    # Indices into `_GradingItem.assessments` of the assessments a review found admissible.
    admissible_idx: tuple[int, ...] = ()


class _SharedTemplates(NamedTuple):
    """Prompt templates supplied by a structure agent."""

    intro_system: str | None = None
    intro_user: str | None = None
    intro_fields: tuple[str, ...] = ()
    final_system: str | None = None
    final_user: str | None = None

    @classmethod
    def from_bundle(cls, bundle: Any) -> "_SharedTemplates":
        """Build from a `prompts.final_agent._bundles` instance — every bundle type carries
        these same five fields alongside its own procedure-specific branch templates."""
        return cls(
            intro_system=bundle.intro_system,
            intro_user=bundle.intro_user,
            intro_fields=bundle.intro_fields,
            final_system=bundle.final_system,
            final_user=bundle.final_user,
        )


@dataclass
class _AttemptState:
    """Threads BaseFinalAgent's fixed intro -> items -> final-turn -> parse skeleton through the LCEL
    pipeline in `BaseFinalAgent._attempt`; each stage returns a new state derived from the last."""

    inputs: dict
    rubric: Rubric
    conversation: list = field(default_factory=list)
    cb: TokenUsageCallback = field(default_factory=TokenUsageCallback)
    scores: dict[str, int] = field(default_factory=dict)
    # Every item's own parsed turn result, keyed by label — CriterionStructure's final synthesis
    # turn shows each criterion's score and justification back to the model from this.
    item_results: dict[str, _ItemResult] = field(default_factory=dict)
    # The last-processed item's own parsed turn result — HolisticStructure (single item) reuses
    # it directly as its final result rather than running (or re-parsing) a further turn.
    last_item_result: _ItemResult = field(default_factory=dict)  # type: ignore[assignment]
    result: dict = field(default_factory=dict)


class Reviewer:
    """Runs and caches one procedure's review turn per source assessment. Each review is generated in its own conversation so no assessment's verdict can anchor another's, which is also what makes a cached review row a faithful record of how it was produced."""

    system_template: str = FINAL_PROMPT_REVIEW_ITEM_SYSTEM_TEMPLATE
    user_template: str = FINAL_PROMPT_REVIEW_ITEM_USER_TEMPLATE
    result_schema: type[BaseModel] = ReviewResult
    admissibility_criteria: tuple[str, ...] = ("rubric_consistency", "evidence_grounding")

    def __init__(
        self,
        run_turn: Callable[[list, TokenUsageCallback, str, str, type[BaseModel], str], dict],
        cache: ReviewCache | None,
        rubric_id: str,
        essay_id: str,
        assignment_prompt: str,
        student_work: str,
        source_experiment_id: str = "",
        reviewer_model: str = "",
        require_cache: bool = False,
    ) -> None:
        self._run_turn = run_turn
        self._cache = cache
        self._rubric_id = rubric_id
        self._essay_id = essay_id
        self._assignment_prompt = assignment_prompt
        self._student_work = student_work
        self._source_experiment_id = source_experiment_id
        self._reviewer_model = reviewer_model
        # True for a procedure that only ever replays reviews (AdjudicationWithReviewProcedure) —
        # a cache miss there means the reused reviews don't cover this run, not that a fresh
        # review should be generated. False for the standalone Review agent, whose whole job is
        # generating reviews that don't already have a cached hit.
        self._require_cache = require_cache

    def review(
        self,
        conversation: list,
        cb: TokenUsageCallback,
        criterion: str,
        model: str,
        prompt_label: str,
        score: int,
        justification: str,
        guidelines: str,
        scale: Scale,
    ) -> dict:
        """Run this review turn, or replay a cached one; either way append it to `conversation`. (`criterion`, `model`) key the cache and name the turn in the log; only `prompt_label` reaches the LLM."""
        review_text = PromptTemplate.from_template(self.user_template).format(
            item_label=prompt_label,
            score=score,
            justification=justification,
            guidelines=guidelines,
            score_min=scale.min,
            score_max=scale.max,
            assignment_prompt=self._assignment_prompt,
            student_work=self._student_work,
        )

        # Compute digest of the source assessment to bind the cached review to its inputs
        source_digest = _source_assessment_digest(score, justification)
        cached = (
            None
            if self._cache is None
            else self._cache.get(
                dataset=self._rubric_id,
                essay_id=self._essay_id,
                criterion=criterion,
                source_model=model,
                source_experiment_id=self._source_experiment_id,
                reviewer_model=self._reviewer_model,
                source_assessment_digest=source_digest,
            )
        )
        if cached is not None:
            conversation.append(SystemMessage(content=self.system_template))
            conversation.append(HumanMessage(content=review_text))
            # Only the schema's own fields, so a replayed turn reads exactly like a generated one.
            verdict = {k: v for k, v in cached.items() if k in self.result_schema.model_fields}
            conversation.append(AIMessage(content=json.dumps(verdict)))
            review = cached
        elif self._require_cache:
            raise ValueError(
                f"No cached review for (rubric={self._rubric_id!r}, essay={self._essay_id!r}, "
                f"criterion={criterion!r}, model={model!r}). adjudication_with_review only "
                "replays reviews — pass --reuse-reviews-paths covering every (essay, criterion, "
                "model) in this run, or generate them first with --deliberation-mode review."
            )
        else:
            # Generate against `turn` alone, then hand the finished exchange to `conversation` — the
            # branch turn still reads every verdict, but no review is generated in sight of another.
            turn: list = []
            review = self._run_turn(
                turn,
                cb,
                self.system_template,
                review_text,
                self.result_schema,
                review_label(criterion, model),
            )
            conversation.extend(turn)
        # Stored on either path, so this run's reviews CSV records every review it used.
        if self._cache is not None:
            self._cache.store(
                dataset=self._rubric_id,
                essay_id=self._essay_id,
                criterion=criterion,
                source_model=model,
                source_experiment_id=self._source_experiment_id,
                reviewer_model=self._reviewer_model,
                source_assessment_digest=source_digest,
                review=review,
            )
        return review

    def is_admissible(self, review: dict) -> bool:
        """Return whether a review satisfies every criterion this Reviewer checks."""
        return all(bool(review.get(c)) for c in self.admissibility_criteria)


def normalized_average_pairwise_disagreement(scores: list[int], scale: Scale) -> float:
    """Mean absolute difference over every unique pair of scores, normalized by the item's scale width and rescaled to `NAPD_REFERENCE_WIDTH`. One threshold then holds across rubrics of different widths and across any number of assessments."""
    pairs = list(itertools.combinations(scores, 2))
    mean_difference = sum(abs(x - y) for x, y in pairs) / len(pairs)
    return mean_difference / (scale.max - scale.min) * NAPD_REFERENCE_WIDTH


def independent_assessment(item: _GradingItem) -> LLMAssessment:
    """The final agent's own prior assessment of this item, or raise — a branch that discards the source scores has nothing else to fall back on."""
    if item.independent is None:
        raise ValueError(
            f"'{item.label}': this branch reassesses, but no independent assessment was reused. "
            "Pass the final agent's own prior results with --reuse-independent-path."
        )
    return item.independent


def independent_score(item: _GradingItem) -> int:
    """The score of the final agent's own prior assessment of this item."""
    return independent_assessment(item).score.rounded_score


def pool_average_score(item: _GradingItem) -> int:
    """The pool's naive average score, regardless of branch — the comparison baseline a
    disagreement verdict overrides with the independent assessment instead."""
    scores = [a.score.rounded_score for a in item.assessments]
    return round_half_up(sum(scores) / len(scores))


def _examples_block(own_record: GradingRecord | None) -> str:
    """The reference-examples text the reused independent assessment was made with, formatted for the disagreement template — empty when there is none."""
    examples_text = own_record.examples_text if own_record else ""
    return f"\n\nReference examples:\n\n{examples_text}" if examples_text else ""


class BaseFinalAgent(BaseAgent, ABC):
    """Base for agents that adjudicate existing grading results."""

    _tool: StructuredTool | None = None
    _reviewer: Reviewer | None = None
    _rq: ResearchQuestion | None = None
    _reasoning: Reasoning = Reasoning.NONE

    source_labels: tuple[str, ...] = ()

    # Branches whose outcome is a source assessment adopted whole — resolved by
    # `_adopt_assessment` without an LLM call. Supplied by the procedure axis.
    _adopt_branches: tuple[str, ...] = ()

    def __init__(self, templates: _SharedTemplates, prompt_template: str | None = None) -> None:
        super().__init__(prompt_template)
        self._templates = templates

    def _make_chain(
        self,
        llm: BaseChatModel,
        rq: ResearchQuestion | None,
        reasoning: Reasoning,
    ) -> Runnable:
        """Return the (possibly tool-bound) LLM directly — `_attempt` builds messages itself."""
        self._rq = rq
        self._reasoning = reasoning
        if self._tool is None:
            return llm
        return bind_tools(llm, self._tool)

    def _build_inputs(  # type: ignore[override]
        self,
        rubric: Rubric,
        human_assessment: HumanAssessment,
        results: list[SourceAssessments],
        independent: SourceAssessments | None = None,
        review_cache: ReviewCache | None = None,
    ) -> dict:
        """Build adjudication inputs from `results`/`independent`/`review_cache` — the kwargs
        BaseAgent.grade() was called with for this agent, in place of fresh grading's
        `reference_examples`."""
        self._tool = None
        self._reviewer = self._make_reviewer(review_cache, rubric, human_assessment)
        return self._build_inputs_and_tools(
            results, rubric, human_assessment.submission.text, independent or []
        )

    def _run_turn(
        self,
        conversation: list,
        cb: TokenUsageCallback,
        system_text: str,
        human_text: str,
        schema: type[BaseModel],
        turn_name: str = "Turn",
    ) -> dict:
        """Append a turn, invoke the structured-output chain for `schema` (bound fresh per call since different turns need different schemas), and return the result."""
        conversation.append(SystemMessage(content=system_text))
        conversation.append(HumanMessage(content=human_text))
        config = self._invoke_config(cb, turn_name=turn_name)
        # self._chain is a BaseChatModel, or bind_tools()'s _ChatModelBinding proxying
        # unknown attributes to it — either way with_structured_output resolves at runtime,
        # but neither is expressed by the public Runnable type bind_tools() returns.
        structured_chain = structured_output(self._chain, schema)
        try:
            structured = structured_chain.invoke(conversation, config=config)
        finally:
            # This turn's tokens are this grading call's cost whatever the turn's outcome, so a
            # rejected parse does not erase them.
            self._record_usage(cb)
        parsed = structured["parsed"]
        if parsed is None:
            if structured["raw"].response_metadata.get("finish_reason") == "length":
                raise TruncatedResponseError(
                    "structured-output generation was cut off (finish_reason=length); "
                    "likely hit max_tokens"
                )
            raise ValueError(f"structured output parsing failed: {structured['parsing_error']}")
        result = parsed.model_dump()
        ai_message = AIMessage(content=json.dumps(result))
        conversation.append(ai_message)
        if self._conversation_logger is not None:
            # on_llm_end already rendered this turn's answer in its own format, so the
            # conversation log shouldn't also dump it raw the next time this list is resent.
            self._conversation_logger.mark_logged(ai_message)
        # Added after the message so the turn's replayed text stays the verdict alone.
        reasoning = structured["raw"].additional_kwargs.get("reasoning_content")
        if reasoning:
            result["reasoning"] = reasoning
        return result

    def _build_common_inputs(self, rubric: Rubric, student_work: str) -> dict:  # type: ignore[override]
        """Build prompt inputs shared by final-agent procedures."""
        scale = rubric.total_score_range()
        return {
            "assignment_prompt": rubric.prompt,
            "student_work": student_work,
            "score_min": scale.min,
            "score_max": scale.max,
        }

    @staticmethod
    def _render_turn(system_template: str, user_template: str, **fmt) -> Turn:
        """Format a branch's [system, user] templates with the same input values."""
        return Turn(
            PromptTemplate.from_template(system_template).format(**fmt),
            PromptTemplate.from_template(user_template).format(**fmt),
        )

    def _process_item(
        self, conversation: list, cb: TokenUsageCallback, item: _GradingItem
    ) -> tuple[str, int, _ItemResult]:
        """Run an item's procedure-selected turn and resolve its final score."""
        verdict = self._verdict(conversation, cb, item)
        turn = self._build_turn(item, item.template_inputs, verdict)
        if verdict.branch in self._adopt_branches:
            item_result = self._adopt_assessment(conversation, turn, item, verdict)
        else:
            item_result = self._run_turn(
                conversation,
                cb,
                turn.system,
                turn.user,
                self._turn_schema(verdict),
                f"{item.label}: {verdict.branch}",
            )
        item_result["branch"] = verdict.branch
        item_result["pool_average_score"] = str(pool_average_score(item))
        typed_result = cast(_ItemResult, item_result)
        return item.label, self._resolve_score(item, verdict, typed_result), typed_result

    def _adopt_assessment(
        self, conversation: list, turn: Turn, item: _GradingItem, verdict: Verdict
    ) -> dict:
        """Take an assessment's text over verbatim, with no LLM call, since the procedure has already decided that assessment *is* the outcome — a pool source when `winner_idx` picks one, the item's independent assessment otherwise. The turn is still appended to `conversation` so a criterion skipped here still reaches the final synthesis turn."""
        source = (
            item.assessments[verdict.winner_idx]
            if verdict.winner_idx is not None
            else independent_assessment(item)
        )
        result = {
            "confidence": source.confidence,
            "justification": source.justification,
        }
        system_message = SystemMessage(content=turn.system)
        human_message = HumanMessage(content=turn.user)
        ai_message = AIMessage(content=json.dumps(result))
        if self._conversation_logger is not None:
            self._conversation_logger.log_adopted_turn(
                f"{item.label}: {verdict.branch}",
                [system_message, human_message],
                {"score": source.score.rounded_score, **result},
            )
            self._conversation_logger.mark_logged(ai_message)
        conversation.append(system_message)
        conversation.append(human_message)
        conversation.append(ai_message)
        return result

    # ── intro turn — same scaffolding under either structure ──────────────────────────────────

    def _build_intro(self, inputs: dict) -> list:
        """The [system, human] intro messages, or [] when this procedure has no intro turn."""
        if self._templates.intro_system is None or self._templates.intro_user is None:
            return []
        intro_text = PromptTemplate.from_template(self._templates.intro_user).format(
            **{field: inputs[field] for field in self._templates.intro_fields}
        )
        return [
            SystemMessage(content=self._templates.intro_system),
            HumanMessage(content=intro_text),
        ]

    def _build_inputs_and_tools(
        self,
        results: list[SourceAssessments],
        rubric: Rubric,
        student_work: str,
        independent: SourceAssessments | None = None,
    ) -> dict:
        """Build shared prompt inputs and the structure-specific adjudication items."""
        common_inputs = self._build_common_inputs(rubric, student_work)
        inputs = {
            **common_inputs,
            "rubric_guidelines": rubric.rubric_guidelines,
            "rubric_document": format_criteria(rubric),
        }
        inputs["items"] = self._build_grading_items(
            results, rubric, common_inputs, independent or []
        )
        return inputs

    # ── the fixed attempt skeleton, as an LCEL pipeline of stages every concrete agent shares ──

    def _start_conversation(self, state: _AttemptState) -> _AttemptState:
        return replace(state, conversation=self._build_intro(state.inputs), cb=TokenUsageCallback())

    def _process_items(self, state: _AttemptState) -> _AttemptState:
        results = [
            self._process_item(state.conversation, state.cb, item) for item in state.inputs["items"]
        ]
        scores = {label: score for label, score, _ in results}
        item_results = {label: item_result for label, _, item_result in results}
        last_item_result = results[-1][2] if results else {}
        return replace(
            state, scores=scores, item_results=item_results, last_item_result=last_item_result
        )

    def _parse_result(self, state: _AttemptState) -> LLMAssessment:  # type: ignore[override]
        scale = state.rubric.total_score_range()
        return parse_grading_result(state.result, scale.min, scale.max, state.cb.usage)

    def _attempt(self, inputs: dict, rubric: Rubric) -> LLMAssessment:
        """Run this agent's four fixed stages — intro, items, structure-specific final turn,
        parse — as one LCEL pipeline, so every concrete agent shares the same shape and only
        `_final_turn` differs by structure."""
        pipeline = (
            RunnableLambda(self._start_conversation)
            | RunnableLambda(self._process_items)
            | RunnableLambda(self._final_turn)
            | RunnableLambda(self._parse_result)
        )
        return pipeline.invoke(_AttemptState(inputs=inputs, rubric=rubric))

    # ── structure-axis stubs (implemented by CriterionStructure / HolisticStructure) ──────────

    # Part of `_attempt`'s item pipeline, so only a structure that runs it implements this — an
    # agent that overrides `_attempt` outright (ReviewFinalAgent) has no synthesis to run.
    def _final_turn(self, state: _AttemptState) -> _AttemptState:
        """Run this structure's synthesis turn (if any) and set the item-derived total score."""
        raise NotImplementedError

    @abstractmethod
    def _build_grading_items(
        self,
        results: list[SourceAssessments],
        rubric: Rubric,
        common_inputs: dict,
        independent: SourceAssessments,
    ) -> list[_GradingItem]:
        """Split prior results into the items this structure adjudicates."""
        raise NotImplementedError

    # ── procedure-axis stubs (implemented by Adjudication/AdjudicationWithReviewProcedure) ────

    # Called from `_build_inputs`, so only a procedure reached through it implements this — an
    # agent that builds its own Reviewer (ReviewFinalAgent) has no cache to be handed.
    def _make_reviewer(
        self, review_cache: ReviewCache | None, rubric: Rubric, human_assessment: HumanAssessment
    ) -> Reviewer | None:
        """Build this procedure's Reviewer, or None for a procedure that never reviews."""
        raise NotImplementedError

    def _verdict(self, conversation: list, cb: TokenUsageCallback, item: _GradingItem) -> Verdict:
        """Return the procedure-specific verdict that selects this item's branch."""
        raise NotImplementedError

    def _build_turn(self, item: _GradingItem, item_inputs: dict, verdict: Verdict) -> Turn:
        """Build this item's [system, user] LLM turn for the verdict's branch."""
        raise NotImplementedError

    def _turn_schema(self, verdict: Verdict) -> type[BaseModel]:
        """Return the structured-output schema for the verdict's branch turn."""
        raise NotImplementedError

    def _resolve_score(self, item: _GradingItem, verdict: Verdict, item_result: _ItemResult) -> int:
        """Resolve this item's final score from its verdict and LLM result."""
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Structure axis — the shape of the conversation
# ─────────────────────────────────────────────────────────────────────────────────────────────


class CriterionStructure(BaseFinalAgent):
    """Adjudicate each criterion in a shared conversation, then synthesize the results."""

    def _build_grading_items(
        self,
        results: list[SourceAssessments],
        rubric: Rubric,
        common_inputs: dict,
        independent: SourceAssessments,
    ) -> list[_GradingItem]:
        """Build one item per criterion. `load_reused_results` already validates that every
        essay's rows match rubric.criteria and sorts them into that order, so sources and the
        independent assessment alike are safe to consume positionally here."""
        scale = rubric.get_criterion_scale()
        own = independent or [None] * len(rubric.criteria)  # type: ignore[list-item]
        items = []
        for criterion, own_record, *sources in zip(rubric.criteria, own, *results, strict=True):
            guidelines = "\n".join(
                f"Score {lv.score}: {lv.guideline}" for lv in criterion.score_levels
            )
            items.append(
                _GradingItem(
                    label=criterion.name,
                    prompt_label=criterion.name.replace("_", " "),
                    scale=scale,
                    guidelines=guidelines,
                    assessments=[a.llm_assessment for a in sources],
                    template_inputs={
                        "criterion_name": criterion.name.replace("_", " "),
                        "rubric_guidelines": guidelines,
                        "score_min": scale.min,
                        "score_max": scale.max,
                        "examples_block": _examples_block(own_record),
                    },
                    independent=own_record.llm_assessment if own_record else None,
                )
            )
        return items

    def _final_turn(self, state: _AttemptState) -> _AttemptState:
        assert self._templates.final_system is not None
        assert self._templates.final_user is not None
        # Restate each criterion's resolved score alongside its justification: a disagreement
        # criterion's score never otherwise appears in the conversation (its item turn withholds
        # it to keep the from-scratch grading framing intact), so without this the synthesis
        # would only ever see scores for the criteria that happened to agree.
        criteria_summary = "\n\n".join(
            f"Criterion: {label} (score {score})\n"
            f"Justification: {state.item_results[label]['justification']}"
            for label, score in state.scores.items()
        )
        overall_score = combine_criterion_scores(state.rubric, state.scores)
        final_user = PromptTemplate.from_template(self._templates.final_user).format(
            criteria_summary=criteria_summary, overall_score=overall_score
        )
        result = self._run_turn(
            state.conversation,
            state.cb,
            self._templates.final_system,
            final_user,
            GradingNoScoreResponse,
            "Final synthesis",
        )
        result["score"] = overall_score
        result["branch"] = ", ".join(
            f"{label}={state.item_results[label]['branch']}" for label in state.scores
        )
        result["pool_average_score"] = ", ".join(
            f"{label}={state.item_results[label]['pool_average_score']}" for label in state.scores
        )
        return replace(state, result=result)


class HolisticStructure(BaseFinalAgent):
    """Adjudicate the submission as one item, without a synthesis turn."""

    def _build_grading_items(
        self,
        results: list[SourceAssessments],
        rubric: Rubric,
        common_inputs: dict,
        independent: SourceAssessments,
    ) -> list[_GradingItem]:
        scale = rubric.total_score_range()
        own_record = independent[0] if independent else None
        return [
            _GradingItem(
                label=rubric.holistic_criterion().name,
                prompt_label=HOLISTIC_LABEL,
                scale=scale,
                guidelines=rubric.rubric_guidelines,
                assessments=[source[0].llm_assessment for source in results],
                template_inputs={
                    "rubric_guidelines": rubric.rubric_guidelines,
                    "score_min": scale.min,
                    "score_max": scale.max,
                    "assignment_prompt": common_inputs["assignment_prompt"],
                    "student_work": common_inputs["student_work"],
                    "examples_block": _examples_block(own_record),
                },
                independent=own_record.llm_assessment if own_record else None,
            )
        ]

    def _final_turn(self, state: _AttemptState) -> _AttemptState:
        # _process_item's own turn is the last one; its already-parsed result is this agent's
        # result, so reuse it directly rather than running (or re-parsing) a further turn.
        ((_, score),) = state.scores.items()
        result = dict(state.last_item_result)
        result["score"] = score
        return replace(state, result=result)


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Procedure axis — what happens within one item
# ─────────────────────────────────────────────────────────────────────────────────────────────


class AdjudicationProcedure(BaseFinalAgent):
    """Adjudicate any number of source assessments for one item, on the agreement among their scores."""

    _agreement_system: str
    _agreement_user: str
    _disagreement_system: str
    _disagreement_user: str

    _adopt_branches = ("disagreement",)

    # Scores agree while their NAPD stays within one score point — 20% of `NAPD_REFERENCE_WIDTH`,
    # the agreement level Jonsson & Svingby (2007) report across rubric studies. Fixed in code
    # rather than per-experiment, since it defines what counts as substantial disagreement.
    _DISAGREEMENT_THRESHOLD_POINTS = 1.0

    def _make_reviewer(
        self, review_cache: ReviewCache | None, rubric: Rubric, human_assessment: HumanAssessment
    ) -> None:
        """Adjudication resolves agreement deterministically and never reviews an assessment."""
        return

    def _verdict(self, conversation: list, cb: TokenUsageCallback, item: _GradingItem) -> Verdict:
        """Compute agreement from the known scores without an LLM call. Either branch already
        knows its score: agreement averages the sources, disagreement takes the independent one."""
        scores = [a.score.rounded_score for a in item.assessments]
        napd = normalized_average_pairwise_disagreement(scores, item.scale)
        if napd > self._DISAGREEMENT_THRESHOLD_POINTS:
            return Verdict("disagreement", resolved_score=independent_score(item))
        return Verdict("agreement", resolved_score=round_half_up(sum(scores) / len(scores)))

    def _threshold_points(self, scale: Scale) -> float:
        """The agreement threshold in this item's own score points, so the prompt can state it on the scale the assessments were given on."""
        return self._DISAGREEMENT_THRESHOLD_POINTS / NAPD_REFERENCE_WIDTH * (scale.max - scale.min)

    def _build_turn(self, item: _GradingItem, item_inputs: dict, verdict: Verdict) -> Turn:
        disagreement_threshold = f"{self._threshold_points(item.scale):.1f}"
        if verdict.branch == "agreement":
            # The score is already resolved, so only the justification is asked for; showing
            # each source's own score alongside it lets it connect both rather than reading the
            # justifications in isolation.
            assessments = "\n\n".join(
                f'Assessment {i} (score {a.score.rounded_score}): """{a.justification}"""'
                for i, a in enumerate(item.assessments, start=1)
            )
            return self._render_turn(
                self._agreement_system,
                self._agreement_user,
                **item_inputs,
                assessments=assessments,
                resolved_score=verdict.resolved_score,
                disagreement_threshold=disagreement_threshold,
            )
        # Disagreement rejects the sources outright and asks for an independent grading from
        # scratch, worded like the primary grading prompt — this turn is never actually sent,
        # though: `_adopt_assessment` intercepts it and substitutes the reused independent
        # assessment as the answer, so the transcript reads as if that grading had happened here.
        return self._render_turn(
            self._disagreement_system,
            self._disagreement_user,
            **item_inputs,
            disagreement_threshold=disagreement_threshold,
        )

    def _turn_schema(self, verdict: Verdict) -> type[BaseModel]:
        """Neither branch decides a score, so neither turn is asked for one."""
        return GradingNoScoreResponse

    def _resolve_score(self, item: _GradingItem, verdict: Verdict, item_result: _ItemResult) -> int:
        assert verdict.resolved_score is not None
        return verdict.resolved_score


class AdjudicationWithReviewProcedure(BaseFinalAgent):
    """Adjudicate any number of source assessments for one item, on the review of each."""

    _positive_system: str
    _positive_user: str
    _mixed_system: str
    _mixed_user: str
    _negative_system: str
    _negative_user: str

    # A negative review rejects every source outright, same as AdjudicationProcedure's
    # disagreement branch — so it's adopted the same way: from-scratch framing, no LLM call.
    _adopt_branches = ("negative",)

    def _make_reviewer(
        self, review_cache: ReviewCache | None, rubric: Rubric, human_assessment: HumanAssessment
    ) -> Reviewer:
        return Reviewer(
            self._run_turn,
            review_cache,
            rubric_id=rubric.rubric_id,
            essay_id=human_assessment.essay_id,
            assignment_prompt=rubric.prompt,
            student_work=human_assessment.submission.text,
            source_experiment_id="",
            reviewer_model="",
            require_cache=False,
        )

    @staticmethod
    def _average(scores: list[float]) -> int:
        """Average a pool's scores onto the nearest rubric score level."""
        return round_half_up(sum(scores) / len(scores))

    def _verdict(self, conversation: list, cb: TokenUsageCallback, item: _GradingItem) -> Verdict:
        """Review every source assessment and select the branch its outcome implies. Each branch's score is settled here: the pool alone when every assessment is admissible, the pool plus the independent assessment when some are, the independent assessment alone when none are."""
        assert self._reviewer is not None
        admissible = tuple(
            i
            for i, (source, llm_assessment) in enumerate(
                zip(self.source_labels, item.assessments, strict=True)
            )
            if self._reviewer.is_admissible(
                self._reviewer.review(
                    conversation,
                    cb,
                    criterion=item.label,
                    model=source,
                    prompt_label=item.prompt_label,
                    score=llm_assessment.score.rounded_score,
                    justification=llm_assessment.justification,
                    guidelines=item.guidelines,
                    scale=item.scale,
                )
            )
        )
        if not admissible:
            return Verdict("negative", resolved_score=independent_score(item))
        retained = [item.assessments[i].score.rounded_score for i in admissible]
        if len(admissible) < len(item.assessments):
            return Verdict(
                "mixed",
                resolved_score=self._average([*retained, independent_score(item)]),
                admissible_idx=admissible,
            )
        return Verdict(
            "positive", resolved_score=self._average(retained), admissible_idx=admissible
        )

    def _build_turn(self, item: _GradingItem, item_inputs: dict, verdict: Verdict) -> Turn:
        if verdict.branch == "negative":
            # Framed as a from-scratch grading turn, like AdjudicationProcedure's disagreement
            # branch — and never actually sent either, since `_adopt_assessment` intercepts it and
            # substitutes the reused independent assessment as the answer.
            return self._render_turn(self._negative_system, self._negative_user, **item_inputs)
        system, user = {
            "positive": (self._positive_system, self._positive_user),
            "mixed": (self._mixed_system, self._mixed_user),
        }[verdict.branch]
        # Justification, but no score: every branch's score is already averaged into
        # `resolved_score`, so the individual scores would only compete with the one the turn
        # has to justify. Retained assessments keep their position in the pool, so each lines up
        # with its review above.
        retained = [
            f'Assessment {i + 1}: """{item.assessments[i].justification}"""'
            for i in verdict.admissible_idx
        ]
        if verdict.branch == "positive":
            block = "\n\n".join(retained)
        else:  # mixed
            own = independent_assessment(item)
            block = "\n\n".join([*retained, f'Your own: """{own.justification}"""'])
        return self._render_turn(
            system, user, **item_inputs, assessments=block, resolved_score=verdict.resolved_score
        )

    def _turn_schema(self, verdict: Verdict) -> type[BaseModel]:
        """No branch decides a score, so no turn is asked for one."""
        return GradingNoScoreResponse

    def _resolve_score(self, item: _GradingItem, verdict: Verdict, item_result: _ItemResult) -> int:
        assert verdict.resolved_score is not None
        return verdict.resolved_score


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Review — its own experiment type, so it implements both axes itself rather than pairing them
# ─────────────────────────────────────────────────────────────────────────────────────────────


class ReviewFinalAgent(BaseFinalAgent):
    """Review one prior assessment for admissibility and stop. One job is one assessment, so a criterion-specialized source yields one of these per criterion."""

    def __init__(self, criterion: Criterion) -> None:
        super().__init__(_SharedTemplates())
        self._criterion = criterion

    def review(
        self,
        rubric: Rubric,
        human_assessment: HumanAssessment,
        llm: BaseChatModel,
        results: list[SourceAssessments],
        review_cache: ReviewCache | None = None,
        rq: ResearchQuestion | None = None,
        reasoning: Reasoning = Reasoning.NONE,
    ) -> ReviewRecord:
        """Review this job's one source assessment. Mirrors `BaseAgent.grade()`'s setup so the retry, length-recovery, and conversation-logging machinery still applies."""
        self._chain = self._make_chain(llm, rq, reasoning)
        self._conversation_logger = self._make_conversation_logger(rubric, human_assessment)
        self._human_assessment = human_assessment
        self._reviewer = Reviewer(
            self._run_turn,
            review_cache,
            rubric_id=rubric.rubric_id,
            essay_id=human_assessment.essay_id,
            assignment_prompt=rubric.prompt,
            student_work=human_assessment.submission.text,
            source_experiment_id="",
            reviewer_model="",
        )
        inputs = self._build_inputs_and_tools(results, rubric, human_assessment.submission.text)
        return self._run_chain(inputs, rubric)  # type: ignore[return-value]

    def _attempt(self, inputs: dict, rubric: Rubric) -> ReviewRecord:  # type: ignore[override]
        """The whole attempt is the one review turn — no items loop, no synthesis, no parse."""
        assert self._reviewer is not None
        (item,) = inputs["items"]
        (assessment,) = item.assessments
        model = self.source_labels[0]
        cb = TokenUsageCallback()
        review = self._reviewer.review(
            [],
            cb,
            criterion=item.label,
            model=model,
            prompt_label=item.prompt_label,
            score=assessment.score.rounded_score,
            justification=assessment.justification,
            guidelines=item.guidelines,
            scale=item.scale,
        )
        return ReviewRecord(
            human_assessment=self._human_assessment,
            review=Review(**review),
            label=review_label(item.label, model),
            statistics=Statistics(token_usage=cb.usage),
        )

    def _build_grading_items(
        self,
        results: list[SourceAssessments],
        rubric: Rubric,
        common_inputs: dict,
        independent: SourceAssessments,
    ) -> list[_GradingItem]:
        """Build the single item for this job's criterion, from the one source model's results. A Review decides no score, so it never reassesses and takes no independent assessment."""
        (sources,) = results  # Review takes exactly one --reuse-path
        record = next(
            r for r in sources if r.human_assessment.criterion.name == self._criterion.name
        )
        is_holistic = self._criterion.name == rubric.holistic_criterion().name
        return [
            _GradingItem(
                label=self._criterion.name,
                prompt_label=(
                    HOLISTIC_LABEL if is_holistic else self._criterion.name.replace("_", " ")
                ),
                scale=rubric.total_score_range() if is_holistic else rubric.get_criterion_scale(),
                guidelines=(
                    rubric.rubric_guidelines
                    if is_holistic
                    else "\n".join(
                        f"Score {lv.score}: {lv.guideline}" for lv in self._criterion.score_levels
                    )
                ),
                assessments=[record.llm_assessment],
                template_inputs={},
            )
        ]


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Concrete agents — one structure x one procedure each
# ─────────────────────────────────────────────────────────────────────────────────────────────


class AdjudicationFinalCriterionAgent(CriterionStructure, AdjudicationProcedure):
    """Adjudicate two criterion-specialized grading results, one criterion at a time."""

    _agreement_system = ADJUDICATION_CRITERION_TEMPLATES.agreement_system
    _agreement_user = ADJUDICATION_CRITERION_TEMPLATES.agreement_user
    _disagreement_system = ADJUDICATION_CRITERION_TEMPLATES.disagreement_system
    _disagreement_user = ADJUDICATION_CRITERION_TEMPLATES.disagreement_user

    def __init__(self) -> None:
        super().__init__(_SharedTemplates.from_bundle(ADJUDICATION_CRITERION_TEMPLATES))


class AdjudicationFinalHolisticAgent(HolisticStructure, AdjudicationProcedure):
    """Adjudicate two holistic grading results in one LLM call."""

    _agreement_system = ADJUDICATION_HOLISTIC_TEMPLATES.agreement_system
    _agreement_user = ADJUDICATION_HOLISTIC_TEMPLATES.agreement_user
    _disagreement_system = ADJUDICATION_HOLISTIC_TEMPLATES.disagreement_system
    _disagreement_user = ADJUDICATION_HOLISTIC_TEMPLATES.disagreement_user

    def __init__(self) -> None:
        super().__init__(_SharedTemplates.from_bundle(ADJUDICATION_HOLISTIC_TEMPLATES))


class AdjudicationWithReviewFinalCriterionAgent(
    CriterionStructure, AdjudicationWithReviewProcedure
):
    """Adjudicate criterion-specialized assessments one criterion at a time, on their review."""

    _positive_system = ADJUDICATION_WITH_REVIEW_CRITERION_TEMPLATES.positive_system
    _positive_user = ADJUDICATION_WITH_REVIEW_CRITERION_TEMPLATES.positive_user
    _mixed_system = ADJUDICATION_WITH_REVIEW_CRITERION_TEMPLATES.mixed_system
    _mixed_user = ADJUDICATION_WITH_REVIEW_CRITERION_TEMPLATES.mixed_user
    _negative_system = ADJUDICATION_WITH_REVIEW_CRITERION_TEMPLATES.negative_system
    _negative_user = ADJUDICATION_WITH_REVIEW_CRITERION_TEMPLATES.negative_user

    def __init__(self) -> None:
        super().__init__(_SharedTemplates.from_bundle(ADJUDICATION_WITH_REVIEW_CRITERION_TEMPLATES))


class AdjudicationWithReviewFinalHolisticAgent(HolisticStructure, AdjudicationWithReviewProcedure):
    """Adjudicate holistic assessments using the full admissibility check."""

    _positive_system = ADJUDICATION_WITH_REVIEW_HOLISTIC_TEMPLATES.positive_system
    _positive_user = ADJUDICATION_WITH_REVIEW_HOLISTIC_TEMPLATES.positive_user
    _mixed_system = ADJUDICATION_WITH_REVIEW_HOLISTIC_TEMPLATES.mixed_system
    _mixed_user = ADJUDICATION_WITH_REVIEW_HOLISTIC_TEMPLATES.mixed_user
    _negative_system = ADJUDICATION_WITH_REVIEW_HOLISTIC_TEMPLATES.negative_system
    _negative_user = ADJUDICATION_WITH_REVIEW_HOLISTIC_TEMPLATES.negative_user

    def __init__(self) -> None:
        super().__init__(_SharedTemplates.from_bundle(ADJUDICATION_WITH_REVIEW_HOLISTIC_TEMPLATES))


class FinalAgentFactory:
    """Base factory for constructing a BaseFinalAgent from a procedure."""

    def __init__(self, procedure: FinalAgentProcedure) -> None:
        self.procedure = procedure

    def produce(self) -> BaseFinalAgent:
        """Construct a BaseFinalAgent matching the procedure."""
        raise NotImplementedError


class CriterionFinalAgentFactory(FinalAgentFactory):
    """BaseFinalAgent factory for criterion-specialized results."""

    def produce(self) -> BaseFinalAgent:
        if self.procedure.name == FinalAgentProcedureName.ADJUDICATION:
            return AdjudicationFinalCriterionAgent()
        if self.procedure.name == FinalAgentProcedureName.ADJUDICATION_WITH_REVIEW:
            return AdjudicationWithReviewFinalCriterionAgent()
        raise ValueError(f"No criterion final agent for procedure '{self.procedure.name.value}'")


class HolisticFinalAgentFactory(FinalAgentFactory):
    """BaseFinalAgent factory for holistic results."""

    def produce(self) -> BaseFinalAgent:
        if self.procedure.name == FinalAgentProcedureName.ADJUDICATION:
            return AdjudicationFinalHolisticAgent()
        if self.procedure.name == FinalAgentProcedureName.ADJUDICATION_WITH_REVIEW:
            return AdjudicationWithReviewFinalHolisticAgent()
        raise ValueError(f"No holistic final agent for procedure '{self.procedure.name.value}'")


class FinalAgentVariantFactory:
    """Factory that selects the appropriate FinalAgentFactory type by input shape."""

    def __init__(self, is_criterion: bool) -> None:
        self.is_criterion = is_criterion

    def produce(self, procedure: FinalAgentProcedure) -> FinalAgentFactory:
        """Return a FinalAgentFactory for the given procedure."""
        if self.is_criterion:
            return CriterionFinalAgentFactory(procedure)
        return HolisticFinalAgentFactory(procedure)
