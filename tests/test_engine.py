"""Regression coverage retained from the original grading engine and final-agent branches."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ValidationError

from abaes_eval_fwk.config.llm_config import LLMConfig
from abaes_eval_fwk.config.retrieval_config import ExamplesSelection, RetrievalConfig
from abaes_eval_fwk.core.agents.base_agent import BaseAgent, TokenUsageCallback
from abaes_eval_fwk.core.agents.final_agent import (
    AdjudicationFinalCriterionAgent,
    AdjudicationFinalHolisticAgent,
    AdjudicationProcedure,
    AdjudicationWithReviewFinalCriterionAgent,
    AdjudicationWithReviewFinalHolisticAgent,
    AdjudicationWithReviewProcedure,
    BaseFinalAgent,
    CriterionStructure,
    HolisticStructure,
    Verdict,
    independent_assessment,
)
from abaes_eval_fwk.core.agents.prompts.final_agent.holistic.adjudication_prompts import (
    FINAL_PROMPT_ADJUDICATION_HOLISTIC_SYSTEM_PROCEDURE_AGREEMENT_TEMPLATE,
    FINAL_PROMPT_ADJUDICATION_HOLISTIC_SYSTEM_PROCEDURE_DISAGREEMENT_TEMPLATE,
)
from abaes_eval_fwk.core.agents.schemas.results import (
    AdjudicationWithReviewDecision,
    GradingResponse,
)
from abaes_eval_fwk.core.models.assessment import (
    GradingRecord,
    HumanAssessment,
    LLMAssessment,
    PredictedScore,
    Statistics,
    StudentSubmission,
    TokenUsage,
)
from abaes_eval_fwk.core.models.experiment import (
    Experiment,
    ExperimentType,
    FinalAgentProcedure,
    FinalAgentProcedureName,
    ResearchQuestion,
)
from abaes_eval_fwk.core.models.rubric import Criterion, Rubric, Scale
from abaes_eval_fwk.experiment.result.result_parsers import parse_grading_result
from abaes_eval_fwk.experiment.setup.reuse_loader import ReviewCache
from abaes_eval_fwk.utils.prompt_formatters import (
    format_criteria,
    format_examples,
    truncate_examples_block,
)


class _StructuredInvoker(Runnable):
    """Returned by `_FakeSequentialChatModel.with_structured_output(schema, include_raw=True)`. Pops the next canned dict, builds `schema(**payload)`, and wraps it in the `{"raw", "parsed", "parsing_error"}` shape real structured output returns, delegating call bookkeeping back onto the fake model itself so `llm.calls == N` assertions keep working unchanged."""

    def __init__(self, fake: "_FakeSequentialChatModel", schema: type[BaseModel]) -> None:
        self._fake = fake
        self._schema = schema

    def invoke(self, input, config=None, **kwargs):
        self._fake.calls += 1
        self._fake.call_lengths.append(len(input))
        self._fake.inputs.append(list(input))
        payload = self._fake._responses.pop(0)
        return {
            "raw": AIMessage(content=json.dumps(payload)),
            "parsed": self._schema(**payload),
            "parsing_error": None,
        }


class _FakeSequentialChatModel(Runnable):
    """Fake chat model returning one pre-canned structured response per turn, in order. Each canned response is a plain dict of field values matching whatever schema that turn requests (see `_StructuredInvoker`), not raw JSON-string content."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls = 0
        self.call_lengths: list[int] = []
        self.inputs: list[list] = []

    def with_structured_output(self, schema, include_raw=True, **kwargs):
        return _StructuredInvoker(self, schema)

    def invoke(self, input, config=None, **kwargs):
        raise NotImplementedError("FinalAgent turns go through with_structured_output now")


_SOURCE_LABELS = ("qwen3:8b", "gemma3:27b")


def _assessment(criterion: Criterion, score: float, justification: str = "j") -> GradingRecord:
    return GradingRecord(
        human_assessment=HumanAssessment(
            submission=StudentSubmission(essay_id="essay-1", text="essay text"),
            human_score=0,
            criterion=criterion,
        ),
        llm_assessment=LLMAssessment(
            score=PredictedScore(score),
            justification=justification,
            confidence=0.8,
            statistics=Statistics(token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1)),
        ),
    )


_MATRIX = [
    (AdjudicationFinalCriterionAgent, CriterionStructure, AdjudicationProcedure),
    (AdjudicationFinalHolisticAgent, HolisticStructure, AdjudicationProcedure),
    (
        AdjudicationWithReviewFinalCriterionAgent,
        CriterionStructure,
        AdjudicationWithReviewProcedure,
    ),
    (AdjudicationWithReviewFinalHolisticAgent, HolisticStructure, AdjudicationWithReviewProcedure),
]
_STRUCTURE_METHODS = ("_final_turn", "_build_grading_items")
_PROCEDURE_METHODS = ("_verdict", "_build_turn", "_turn_schema", "_resolve_score", "_make_reviewer")


def _defining_class(cls: type, name: str) -> type:
    """The class in `cls`'s MRO that actually provides `name`."""
    return next((c for c in cls.__mro__ if name in c.__dict__))


@pytest.mark.parametrize(("agent_cls", "structure", "procedure"), _MATRIX)
def test_each_axis_method_resolves_to_its_own_axis(agent_cls, structure, procedure):
    """The invariant the two-axis design rests on: neither axis may declare even a stub for the other axis's methods, since a stub would sit earlier in the MRO than the real implementation and silently shadow it. Only BaseFinalAgent may hold stubs — being last, it shadows nothing — so resolving to BaseFinalAgent means the real implementation was lost."""
    for name in _STRUCTURE_METHODS:
        owner = _defining_class(agent_cls, name)
        assert owner is not BaseFinalAgent, f"{agent_cls.__name__}.{name} fell through to the stub"
        assert issubclass(structure, owner), f"{name} resolved outside the structure axis: {owner}"
    for name in _PROCEDURE_METHODS:
        owner = _defining_class(agent_cls, name)
        assert owner is not BaseFinalAgent, f"{agent_cls.__name__}.{name} fell through to the stub"
        assert issubclass(procedure, owner), f"{name} resolved outside the procedure axis: {owner}"


def _widen_scale(rubric: Rubric, max_score: int) -> Rubric:
    """Return `rubric` with its scale widened to `max_score`. The conftest fixture's scale (0-4) floors `0.2 * max` to 0 — real AES/ASAP2/ELLIPSE rubrics range 3-12, so tests need a wider scale to exercise a non-trivial disagreement threshold."""
    return Rubric(**{**rubric.__dict__, "scale": Scale(0, max_score)})


def _adjudication_holistic_branch(rubric, score_a, score_b, own_score=2):
    """Drive one holistic adjudication item as far as its branch turn, without any LLM: the verdict is deterministic Python, and _build_turn only formats templates."""
    holistic = rubric.holistic_criterion()
    agent = AdjudicationFinalHolisticAgent()
    agent.source_labels = _SOURCE_LABELS
    inputs = agent._build_inputs_and_tools(
        [[_assessment(holistic, score_a)], [_assessment(holistic, score_b)]],
        rubric,
        "work",
        [_assessment(holistic, own_score, "own j")],
    )
    (item,) = inputs["items"]
    verdict = agent._verdict([], TokenUsageCallback(), item)
    turn = agent._build_turn(item, item.template_inputs, verdict)

    def expected(template: str) -> str:
        """Templates escape their JSON schema braces as {{...}}, so the expected text has to go through the same formatting pass the agent's own turn does."""
        return PromptTemplate.from_template(template).format(
            **item.template_inputs,
            resolved_score=verdict.resolved_score,
            disagreement_threshold=f"{agent._threshold_points(item.scale):.1f}",
        )

    return (agent, verdict, turn.system, expected)


_JUSTIFICATION_ONLY = {"reasoning": "r", "confidence": 0.9, "justification": "j"}


def test_adjudication_holistic_resolves_agreement_in_python(rubric):
    """Agreement is decided by the procedure axis's _verdict — no LLM call — and _build_turn then selects the matching standing instruction for the single branch turn."""
    wide = _widen_scale(rubric, 5)
    agent, verdict, system_text, expected = _adjudication_holistic_branch(
        wide, score_a=3, score_b=4
    )
    assert agent._tool is None
    assert verdict == Verdict("agreement", resolved_score=4)
    assert system_text == expected(
        FINAL_PROMPT_ADJUDICATION_HOLISTIC_SYSTEM_PROCEDURE_AGREEMENT_TEMPLATE
    )


def test_adjudication_holistic_resolves_disagreement_in_python(rubric):
    """Disagreement discards both source scores for the reused independent assessment's, so this branch too knows its score before any LLM call."""
    wide = _widen_scale(rubric, 5)
    _, verdict, system_text, expected = _adjudication_holistic_branch(
        wide, score_a=1, score_b=4, own_score=2
    )
    assert verdict == Verdict("disagreement", resolved_score=2)
    assert system_text == expected(
        FINAL_PROMPT_ADJUDICATION_HOLISTIC_SYSTEM_PROCEDURE_DISAGREEMENT_TEMPLATE
    )


def test_adjudication_holistic_disagreement_threshold_scales_with_rubric_range(rubric):
    """NAPD normalizes by the item's own scale before the fixed 1-point threshold applies, so the same 2-point gap that disagrees on a narrow rubric can agree on a wider one (AES set 7's 0-12 scale)."""
    wide = _widen_scale(rubric, 12)
    _, verdict, _, _ = _adjudication_holistic_branch(wide, score_a=2, score_b=4)
    assert verdict == Verdict("agreement", resolved_score=3)
    _, verdict, _, _ = _adjudication_holistic_branch(wide, score_a=1, score_b=4, own_score=2)
    assert verdict == Verdict("disagreement", resolved_score=2)


def _holistic_independent(rubric, score, justification="own j"):
    """The one-record independent pool a holistic final agent reuses."""
    return [_assessment(rubric.holistic_criterion(), score, justification)]


def test_adjudication_holistic_grade_averages_sources_on_agreement(rubric, submission):
    wide = _widen_scale(rubric, 5)
    holistic = wide.holistic_criterion()
    agent = AdjudicationFinalHolisticAgent()
    agent.source_labels = _SOURCE_LABELS
    llm = _FakeSequentialChatModel([{**_JUSTIFICATION_ONLY, "justification": "combined view"}])
    result = agent.grade(
        wide,
        submission,
        llm,
        results=[[_assessment(holistic, 3)], [_assessment(holistic, 4)]],
        independent=_holistic_independent(wide, 1),
    )
    assert result.score.rounded_score == 4
    assert result.justification == "combined view"
    assert llm.calls == 1


def test_adjudication_holistic_grade_adopts_independent_assessment_on_disagreement(
    rubric, submission
):
    """Disagreement discards the sources entirely for the final agent's own prior independent assessment: its score and justification are adopted verbatim, with no LLM call at all — holistic has no further turn to make one in either."""
    wide = _widen_scale(rubric, 5)
    holistic = wide.holistic_criterion()
    agent = AdjudicationFinalHolisticAgent()
    agent.source_labels = _SOURCE_LABELS
    llm = _FakeSequentialChatModel([])
    result = agent.grade(
        wide,
        submission,
        llm,
        results=[[_assessment(holistic, 1)], [_assessment(holistic, 4)]],
        independent=_holistic_independent(wide, 3, "own j"),
    )
    assert result.score.rounded_score == 3
    assert result.justification == "own j"
    assert llm.calls == 0


def test_adjudication_criterion_build_inputs_returns_per_item_data(rubric):
    """No tool, no batching: _build_inputs_and_tools just packages one _GradingItem per criterion, each carrying both sources' (score, justification) in results[0]/results[1] order plus the matching independent record — agreement resolution and templating happen per item."""
    agent = AdjudicationFinalCriterionAgent()
    agent.source_labels = _SOURCE_LABELS
    a = [_assessment(rubric.criteria[0], 3, "a-ideas"), _assessment(rubric.criteria[1], 4, "a-org")]
    b = [_assessment(rubric.criteria[0], 3, "b-ideas"), _assessment(rubric.criteria[1], 2, "b-org")]
    own = [
        _assessment(rubric.criteria[0], 1, "own-ideas"),
        _assessment(rubric.criteria[1], 2, "own-org"),
    ]
    inputs = agent._build_inputs_and_tools([a, b], rubric, "work", own)
    assert agent._tool is None
    assert [item.label for item in inputs["items"]] == ["ideas", "organization"]
    assert [(r.score.rounded_score, r.justification) for r in inputs["items"][1].assessments] == [
        (4, "a-org"),
        (2, "b-org"),
    ]
    assert inputs["items"][1].independent.justification == "own-org"


def test_adjudication_criterion_grade_full_flow(rubric, submission):
    """One turn per agreeing criterion (ideas agrees -> LLM-synthesized justification over the locked average); a disagreeing criterion (organization) is adopted from the reused independent assessment with no LLM call at all. The final turn still runs once to synthesize one overall justification."""
    wide = _widen_scale(rubric, 5)
    agent = AdjudicationFinalCriterionAgent()
    agent.source_labels = _SOURCE_LABELS
    a = [_assessment(wide.criteria[0], 3, "a-ideas"), _assessment(wide.criteria[1], 4, "a-org")]
    b = [_assessment(wide.criteria[0], 3, "b-ideas"), _assessment(wide.criteria[1], 2, "b-org")]
    own = [
        _assessment(wide.criteria[0], 0, "own-ideas"),
        _assessment(wide.criteria[1], 3, "own-org"),
    ]
    llm = _FakeSequentialChatModel(
        [
            {**_JUSTIFICATION_ONLY, "justification": "ideas agreed"},
            {**_JUSTIFICATION_ONLY, "justification": "overall summary"},
        ]
    )
    result = agent.grade(wide, submission, llm, results=[a, b], independent=own)
    assert llm.calls == 2
    assert result.score.rounded_score == 3
    assert result.justification == "overall summary"
    assert llm.call_lengths == [4, 10]


def test_adjudication_criterion_grade_all_agree(rubric, submission):
    """When every criterion agrees, no criterion ever reaches for the independent assessment and each per-criterion turn only synthesizes a justification."""
    agent = AdjudicationFinalCriterionAgent()
    agent.source_labels = _SOURCE_LABELS
    a = [_assessment(rubric.criteria[0], 3, "a-ideas"), _assessment(rubric.criteria[1], 4, "a-org")]
    b = [_assessment(rubric.criteria[0], 3, "b-ideas"), _assessment(rubric.criteria[1], 4, "b-org")]
    llm = _FakeSequentialChatModel(
        [
            {**_JUSTIFICATION_ONLY, "justification": "ideas agreed"},
            {**_JUSTIFICATION_ONLY, "justification": "organization agreed"},
            {**_JUSTIFICATION_ONLY, "justification": "overall summary"},
        ]
    )
    result = agent.grade(rubric, submission, llm, results=[a, b])
    assert llm.calls == 3
    assert result.score.rounded_score == 4


def test_adjudication_raises_when_the_independent_assessment_is_missing(rubric):
    """A reassessing branch has no score of its own to fall back on, so a run started without --reuse-independent-path must fail loudly rather than silently averaging the sources."""
    holistic = rubric.holistic_criterion()
    agent = AdjudicationFinalHolisticAgent()
    agent.source_labels = _SOURCE_LABELS
    inputs = agent._build_inputs_and_tools(
        [[_assessment(holistic, 1)], [_assessment(holistic, 4)]], rubric, "work"
    )
    (item,) = inputs["items"]
    with pytest.raises(ValueError, match="reuse-independent-path"):
        independent_assessment(item)


_ADMISSIBLE_REVIEW = {
    "reasoning": "r",
    "rubric_consistency": True,
    "rubric_consistency_justification": "",
    "evidence_grounding": True,
    "evidence_grounding_justification": "",
}
_REJECTED_REVIEW = {
    "reasoning": "r",
    "rubric_consistency": False,
    "rubric_consistency_justification": "rc fail",
    "evidence_grounding": False,
    "evidence_grounding_justification": "eg fail",
}


class _ReplayByReviewIdentity(ReviewCache):
    """Replays a review by its four identity fields. Binding to the source experiment, reviewer,
    and assessment digest is covered directly in test_grading_regressions.py; these adjudication
    tests exercise what happens once a verdict is in hand."""

    def get(self, dataset, essay_id, criterion, source_model, **_binding):
        return self._loaded.get((dataset, essay_id, criterion, source_model))


def _review_cache(rubric, reviews, criteria=("holistic",), source_labels=_SOURCE_LABELS):
    """Build a fully populated review cache in item-then-source traversal order."""
    keys = [
        (rubric.rubric_id, "essay-1", criterion, source)
        for criterion in criteria
        for source in source_labels
    ]
    assert len(keys) == len(reviews)
    return _ReplayByReviewIdentity(
        procedure=FinalAgentProcedureName.ADJUDICATION_WITH_REVIEW,
        writer=_RecordingWriter(),
        _loaded=dict(zip(keys, reviews, strict=True)),
    )


def test_adjudication_with_review_holistic_positive(rubric, submission):
    """Every assessment admissible: the pool's scores are averaged in Python and the turn only synthesizes a justification from their claims."""
    holistic = rubric.holistic_criterion()
    agent = AdjudicationWithReviewFinalHolisticAgent()
    agent.source_labels = _SOURCE_LABELS
    llm = _FakeSequentialChatModel([{**_JUSTIFICATION_ONLY, "justification": "unified"}])
    result = agent.grade(
        rubric,
        submission,
        llm,
        results=[[_assessment(holistic, 4, "a j")], [_assessment(holistic, 3, "b j")]],
        independent=_holistic_independent(rubric, 0),
        review_cache=_review_cache(rubric, [_ADMISSIBLE_REVIEW, _ADMISSIBLE_REVIEW]),
    )
    assert llm.calls == 1
    assert result.score.rounded_score == 4
    assert result.justification == "unified"
    assert "own j" not in _all_prompt_text_of(llm.inputs[-1][-2:])


def test_adjudication_with_review_holistic_mixed(rubric, submission):
    """Mixed review: the inadmissible assessment is dropped, the reused independent score joins the retained one, and the enlarged pool is averaged in Python."""
    holistic = rubric.holistic_criterion()
    agent = AdjudicationWithReviewFinalHolisticAgent()
    agent.source_labels = _SOURCE_LABELS
    llm = _FakeSequentialChatModel([{**_JUSTIFICATION_ONLY, "justification": "retained plus own"}])
    result = agent.grade(
        rubric,
        submission,
        llm,
        results=[[_assessment(holistic, 3, "a j")], [_assessment(holistic, 1, "b j")]],
        independent=_holistic_independent(rubric, 2),
        review_cache=_review_cache(rubric, [_ADMISSIBLE_REVIEW, _REJECTED_REVIEW]),
    )
    assert llm.calls == 1
    assert result.score.rounded_score == 3
    assert result.justification == "retained plus own"
    branch_prompt = _all_prompt_text_of(llm.inputs[-1][-2:])
    assert "a j" in branch_prompt
    assert "own j" in branch_prompt
    assert "b j" not in branch_prompt


def test_adjudication_with_review_holistic_negative(rubric, submission):
    """No assessment admissible: the reused independent assessment is the decision, and the one turn only justifies it."""
    holistic = rubric.holistic_criterion()
    agent = AdjudicationWithReviewFinalHolisticAgent()
    agent.source_labels = _SOURCE_LABELS
    llm = _FakeSequentialChatModel([])
    result = agent.grade(
        rubric,
        submission,
        llm,
        results=[[_assessment(holistic, 3, "a j")], [_assessment(holistic, 1, "b j")]],
        independent=_holistic_independent(rubric, 2),
        review_cache=_review_cache(rubric, [_REJECTED_REVIEW, _REJECTED_REVIEW]),
    )
    assert llm.calls == 0
    assert result.score.rounded_score == 2
    assert result.justification == "own j"


def test_adjudication_with_review_holistic_pools_more_than_two_sources(rubric, submission):
    """Three source assessments, one rejected: the branch is still mixed, and every remaining score plus the reused independent one is averaged — nothing here assumes a pair."""
    holistic = rubric.holistic_criterion()
    agent = AdjudicationWithReviewFinalHolisticAgent()
    agent.source_labels = (*_SOURCE_LABELS, "llama3.1:8b")
    llm = _FakeSequentialChatModel([{**_JUSTIFICATION_ONLY, "justification": "pooled"}])
    result = agent.grade(
        rubric,
        submission,
        llm,
        results=[
            [_assessment(holistic, 3, "a j")],
            [_assessment(holistic, 0, "b j")],
            [_assessment(holistic, 2, "c j")],
        ],
        independent=_holistic_independent(rubric, 4),
        review_cache=_review_cache(
            rubric,
            [_ADMISSIBLE_REVIEW, _REJECTED_REVIEW, _ADMISSIBLE_REVIEW],
            source_labels=agent.source_labels,
        ),
    )
    assert llm.calls == 1
    assert result.score.rounded_score == 3


def test_adjudication_with_review_criterion_grade_full_flow(rubric, submission):
    """One criterion lands in the positive branch, the other in mixed — mirrors AdjudicationWithReviewFinalHolisticAgent's branch resolution, just per criterion."""
    agent = AdjudicationWithReviewFinalCriterionAgent()
    agent.source_labels = _SOURCE_LABELS
    a = [_assessment(rubric.criteria[0], 3, "a-ideas"), _assessment(rubric.criteria[1], 4, "a-org")]
    b = [_assessment(rubric.criteria[0], 3, "b-ideas"), _assessment(rubric.criteria[1], 2, "b-org")]
    own = [
        _assessment(rubric.criteria[0], 0, "own-ideas"),
        _assessment(rubric.criteria[1], 4, "own-org"),
    ]
    llm = _FakeSequentialChatModel(
        [
            {**_JUSTIFICATION_ONLY, "justification": "ideas unified"},
            {**_JUSTIFICATION_ONLY, "justification": "organization pooled"},
            {**_JUSTIFICATION_ONLY, "justification": "overall summary"},
        ]
    )
    result = agent.grade(
        rubric,
        submission,
        llm,
        results=[a, b],
        independent=own,
        review_cache=_review_cache(
            rubric,
            [_ADMISSIBLE_REVIEW, _ADMISSIBLE_REVIEW, _ADMISSIBLE_REVIEW, _REJECTED_REVIEW],
            criteria=("ideas", "organization"),
        ),
    )
    assert llm.calls == 3
    assert result.score.rounded_score == 4
    assert result.justification == "overall summary"


def test_adjudication_with_review_requires_evidence_grounding(rubric, submission):
    """Admissibility is a conjunction: an assessment that is rubric-consistent but not grounded in the submission still fails, so both sides are inadmissible and the independent assessment wins."""
    holistic = rubric.holistic_criterion()
    agent = AdjudicationWithReviewFinalHolisticAgent()
    agent.source_labels = _SOURCE_LABELS
    consistent_but_ungrounded = {
        "reasoning": "r",
        "rubric_consistency": True,
        "rubric_consistency_justification": "",
        "evidence_grounding": False,
        "evidence_grounding_justification": "eg fail",
    }
    llm = _FakeSequentialChatModel([])
    result = agent.grade(
        rubric,
        submission,
        llm,
        results=[[_assessment(holistic, 4, "a j")], [_assessment(holistic, 2, "b j")]],
        independent=_holistic_independent(rubric, 1),
        review_cache=_review_cache(rubric, [consistent_but_ungrounded, consistent_but_ungrounded]),
    )
    assert llm.calls == 0
    assert result.score.rounded_score == 1
    assert result.justification == "own j"


def _all_prompt_text_of(turn: list) -> str:
    """One turn's system/human messages as one blob."""
    return "\n".join((m.content for m in turn if not isinstance(m, AIMessage)))


def _all_prompt_text(llm) -> str:
    """Every system/human message the agent sent, across all turns, as one blob."""
    return "\n".join((_all_prompt_text_of(turn) for turn in llm.inputs))


def test_criterion_structure_never_shows_the_whole_rubric(rubric, submission):
    """A criterion agent sees one criterion's guidelines per turn and never the rubric as a whole — so a specialist's criterion can't be judged against a neighbouring criterion's levels."""
    agent = AdjudicationWithReviewFinalCriterionAgent()
    agent.source_labels = _SOURCE_LABELS
    a = [_assessment(rubric.criteria[0], 3, "a-ideas"), _assessment(rubric.criteria[1], 3, "a-org")]
    b = [_assessment(rubric.criteria[0], 3, "b-ideas"), _assessment(rubric.criteria[1], 3, "b-org")]
    admissible = {
        "reasoning": "r",
        "rubric_consistency": True,
        "rubric_consistency_justification": "",
        "evidence_grounding": True,
        "evidence_grounding_justification": "",
    }
    unified = {"reasoning": "r", "confidence": 0.9, "justification": "u"}
    llm = _FakeSequentialChatModel([unified, unified, unified])
    agent.grade(
        rubric,
        submission,
        llm,
        results=[a, b],
        review_cache=_review_cache(rubric, [admissible] * 4, criteria=("ideas", "organization")),
    )
    prompts = _all_prompt_text(llm)
    assert not any(
        (
            "Missing ideas" in m.content and "No structure" in m.content
            for turn in llm.inputs
            for m in turn
            if not isinstance(m, AIMessage)
        )
    )
    assert "Missing ideas" in prompts
    assert "No structure" in prompts


class _RecordingWriter:
    """Stands in for ResultsWriter — ReviewCache only ever calls append_review on it."""

    def __init__(self) -> None:
        self.rows: list[tuple] = []

    def append_review(self, procedure, dataset, essay_id, criterion, model, review) -> None:
        self.rows.append((procedure, dataset, essay_id, criterion, model, review))


_INADMISSIBLE_REVIEW = {
    "rubric_consistency": False,
    "rubric_consistency_justification": "rc fail",
    "evidence_grounding": False,
    "evidence_grounding_justification": "eg fail",
}


def test_cached_reviews_are_readable_in_branch_conversation(rubric, submission):
    """Replayed reviews appear as JSON assistant messages in the branch context."""
    holistic = rubric.holistic_criterion()
    agent = AdjudicationWithReviewFinalHolisticAgent()
    agent.source_labels = _SOURCE_LABELS
    llm = _FakeSequentialChatModel([_JUSTIFICATION_ONLY])
    agent.grade(
        rubric,
        submission,
        llm,
        results=[[_assessment(holistic, 3, "a j")], [_assessment(holistic, 2, "b j")]],
        independent=_holistic_independent(rubric, 2),
        review_cache=_review_cache(rubric, [_ADMISSIBLE_REVIEW, _INADMISSIBLE_REVIEW]),
    )
    reviews = [json.loads(m.content) for m in llm.inputs[0] if isinstance(m, AIMessage)]
    expected = [
        {key: value for key, value in review.items() if key != "reasoning"}
        for review in (_ADMISSIBLE_REVIEW, _INADMISSIBLE_REVIEW)
    ]
    assert reviews == expected


def test_adjudication_with_review_generates_missing_reviews(rubric, submission):
    """An absent cached review is generated in its own conversation before synthesis."""
    agent = AdjudicationWithReviewFinalHolisticAgent()
    agent.source_labels = _SOURCE_LABELS
    cache = _review_cache(rubric, [_ADMISSIBLE_REVIEW], source_labels=_SOURCE_LABELS[:1])
    llm = _FakeSequentialChatModel(
        [
            _ADMISSIBLE_REVIEW,
            {
                "confidence": 0.5,
                "justification": "Both assessments are supported.",
            },
        ]
    )
    result = agent.grade(
        rubric,
        submission,
        llm,
        results=[
            [_assessment(rubric.holistic_criterion(), 3)],
            [_assessment(rubric.holistic_criterion(), 2)],
        ],
        review_cache=cache,
    )
    assert result.score.rounded_score == 3
    assert llm.calls == 2


class _FakeAgent(BaseAgent):
    """Return parsed results directly for invocation and retry regression tests."""

    def _parse_result(self, result, usage, rubric):
        return result


def _agent_with_chain(response: dict) -> _FakeAgent:
    """A _FakeAgent whose chain.invoke(...) returns a canned structured-output response, regardless of inputs — enough to drive _attempt() without any real LLM call."""
    agent = _FakeAgent.__new__(_FakeAgent)
    agent._chain = type("_Chain", (), {"invoke": staticmethod(lambda inputs, config: response)})()
    return agent


def test_confidence_over_one_is_normalized_as_a_percentage():
    """Regression test: llama3.1 8B answered confidence=90 (meaning 90%) consistently, not as a transient slip, so a bare ge=0.0/le=1.0 constraint with no normalization cost the whole essay's results after MAX_ATTEMPTS was exhausted. Applies to every schema with a confidence field, not just GradingResponse."""
    assert GradingResponse(score=2, confidence=90, justification="j").confidence == 0.9
    assert (
        AdjudicationWithReviewDecision(
            decision="select_a", score=2, confidence=90, justification="j"
        ).confidence
        == 0.9
    )


def test_confidence_still_rejects_values_a_percentage_reading_cant_fix():
    """500/100 = 5, still out of range — a real error, not a percentage misunderstanding, and must still fail loudly rather than being silently clamped into looking valid."""
    with pytest.raises(ValidationError):
        GradingResponse(score=2, confidence=500, justification="j")


def _retrying_agent(failures: int) -> tuple[_FakeAgent, list[str]]:
    """A _FakeAgent whose `_attempt` raises `failures` times, then returns a real LLMAssessment. Also records the turn_name `_invoke_config` would hand the conversation logger each try."""
    agent = _FakeAgent.__new__(_FakeAgent)
    agent._conversation_logger = None
    agent._attempt_number = 0
    seen: list[str] = []
    attempts = [0]

    def fake_attempt(inputs, rubric):
        seen.append(agent._invoke_config(turn_name="SpecialistAgent")["metadata"]["turn_name"])
        if attempts[0] < failures:
            attempts[0] += 1
            raise ValueError("boom")
        return LLMAssessment(
            score=PredictedScore(3.0), justification="j", confidence=0.9, statistics=Statistics()
        )

    agent._attempt = fake_attempt
    return (agent, seen)


def test_invoke_config_labels_retries_in_the_turn_name():
    """A retry re-invokes the turn from scratch, so without a distinguishing label the conversation log reads like a duplication bug rather than a retry. _attempt_with_retry sets _attempt_number before each try; _invoke_config must fold that into the turn_name it hands the logger."""
    agent, seen = _retrying_agent(failures=2)
    result = agent._attempt_with_retry({}, None)
    assert result.score.raw_score == 3.0
    assert seen == ["SpecialistAgent", "SpecialistAgent (retry 1)", "SpecialistAgent (retry 2)"]


def test_recovered_failure_is_recorded_on_the_result():
    """A failure a later attempt recovers from must survive into the results CSV — otherwise the only trace that a row needed 3 tries to score is a log line nobody reads at analysis time."""
    agent, _ = _retrying_agent(failures=2)
    result = agent._attempt_with_retry({}, None)
    assert result.statistics.retry_error == "attempt 2/3: ValueError: boom"


def test_first_try_success_leaves_retry_error_blank():
    """The common case must stay empty, so a non-blank retry_error always means something."""
    agent, _ = _retrying_agent(failures=0)
    assert agent._attempt_with_retry({}, None).statistics.retry_error is None


def test_attempt_captures_native_reasoning(rubric):
    """GradingResponse has no reasoning field of its own — ChatDeepSeek's native <think> capture (see ChatDeepSeek._create_chat_result) is the only source for result["reasoning"], so _attempt must add it onto the parsed dict rather than dropping it."""
    parsed = GradingResponse(score=3, confidence=0.8, justification="j")
    raw = AIMessage(content="", additional_kwargs={"reasoning_content": "genuine <think> trace"})
    agent = _agent_with_chain({"raw": raw, "parsed": parsed, "parsing_error": None})
    result = agent._attempt({}, rubric)
    assert result["reasoning"] == "genuine <think> trace"


def test_attempt_leaves_reasoning_absent_when_native_absent(rubric):
    """Plain ChatOpenAI never populates reasoning_content (see llm_loader.py) — _attempt must not crash or invent a value, just leave reasoning unset like GradingResponse itself does."""
    parsed = GradingResponse(score=3, confidence=0.8, justification="j")
    raw = AIMessage(content="")
    agent = _agent_with_chain({"raw": raw, "parsed": parsed, "parsing_error": None})
    result = agent._attempt({}, rubric)
    assert "reasoning" not in result


def test_statistics_and_predicted_score_derive_values():
    start = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    statistics = Statistics(
        token_usage=TokenUsage(prompt_tokens=12, completion_tokens=8),
        start_time=start,
        end_time=start + timedelta(seconds=1.5),
    )
    assert statistics.token_usage.total_tokens == 20
    assert statistics.total_time_seconds == 1.5
    assert PredictedScore(2.5).rounded_score == 3


def test_token_usage_adds():
    total = TokenUsage(prompt_tokens=1, completion_tokens=2) + TokenUsage(
        prompt_tokens=3, completion_tokens=4
    )
    assert total == TokenUsage(prompt_tokens=4, completion_tokens=6)


def test_parse_grading_result_preserves_optional_fields():
    usage = TokenUsage(prompt_tokens=3, completion_tokens=4)
    result = parse_grading_result(
        {"score": "3.5", "justification": "Evidence supports it.", "confidence": 0.8},
        score_min=0,
        score_max=4,
        usage=usage,
    )
    assert result.score.raw_score == 3.5
    assert result.score.rounded_score == 4
    assert result.justification == "Evidence supports it."
    assert result.confidence == 0.8
    assert result.statistics.token_usage == usage


def test_parse_grading_result_rejects_scores_outside_rubric_range():
    with pytest.raises(ValueError, match="out of range"):
        parse_grading_result({"score": 5}, 0, 4, TokenUsage())


def test_prompt_formatters_render_examples_and_criteria(rubric, submission):
    examples = format_examples([submission])
    criteria = format_criteria(rubric)
    assert '--- Score 3 ---\n"""\nA concise student essay.\n"""' in examples
    assert examples.startswith("--- Score 3 ---\n")
    assert "ideas (scale 0–4):" in criteria
    assert "Score 4: Clear structure" in criteria
    assert format_examples([]) == ""


def test_student_submission_as_example(rubric):
    submission = HumanAssessment(
        submission=StudentSubmission(essay_id="essay-2", text="First line.\nSecond line."),
        human_score=2,
        criterion=rubric.holistic_criterion(),
    )
    assert submission.as_example() == '--- Score 2 ---\n"""\nFirst line.\nSecond line.\n"""'


def test_truncate_examples_block_keeps_closing_fence(rubric, submission):
    '''Regression test: truncation must never drop an example's closing quote fence — the word count that drives truncation only spans the essay text between the fences, not the fences themselves (a prior bug truncated the trailing `"""` away along with the last words).'''
    other = HumanAssessment(
        submission=StudentSubmission(
            essay_id="essay-3", text="one two three four five six seven eight"
        ),
        human_score=1,
        criterion=rubric.holistic_criterion(),
    )
    block = format_examples([submission, other])
    truncated = truncate_examples_block(block, 0.5)
    assert truncated.count('"""') == block.count('"""')
    assert "[...]" in truncated
    assert "--- Score 3 ---" in truncated and "--- Score 1 ---" in truncated


def test_truncate_examples_block_noop_above_full_fraction(rubric, submission):
    block = format_examples([submission])
    assert truncate_examples_block(block, 1.0) == block
    assert truncate_examples_block("", 0.5) == ""


def test_experiment_configuration_validates_and_builds_name():
    experiment = Experiment(
        rq=ResearchQuestion.RQ1,
        experiment_type=ExperimentType.HOLISTIC,
        llm=LLMConfig(
            provider="openai_compatible",
            max_tokens=4096,
            timeout=120.0,
            streaming=False,
            tool_choice="required",
            structured_output_method="function_calling",
            model_name="test-model",
        ),
        retrieval=RetrievalConfig(
            n_ex_per_score=1, max_examples=6, examples_selection=ExamplesSelection.SEMANTIC
        ),
        final_agent_procedure=FinalAgentProcedure.none(),
    )
    assert experiment.name == "RQ1_holistic_semantic_test-model"
    with pytest.raises(ValueError, match="n_ex_per_score"):
        RetrievalConfig(n_ex_per_score=-1, max_examples=6)
