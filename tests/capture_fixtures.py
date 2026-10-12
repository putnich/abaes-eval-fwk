"""Regenerate deterministic fixtures that pin this checkout's behavior.
Run from framework/ with `.venv/bin/python tests/capture_fixtures.py`."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from langchain_core.prompts import PromptTemplate

from abaes_eval_fwk.core.agents.base_agent import ALLOW_CONTEXT_RECOVERY, TokenUsageCallback
from abaes_eval_fwk.core.agents.final_agent import (
    AdjudicationFinalCriterionAgent,
    AdjudicationFinalHolisticAgent,
    AdjudicationWithReviewFinalCriterionAgent,
    AdjudicationWithReviewFinalHolisticAgent,
    Reviewer,
    Turn,
    Verdict,
    normalized_average_pairwise_disagreement,
)
from abaes_eval_fwk.core.agents.holistic_agent import HolisticGradingAgent
from abaes_eval_fwk.core.agents.specialized_agent import SpecialistAgent
from abaes_eval_fwk.core.models.assessment import (
    GradingRecord,
    HumanAssessment,
    LLMAssessment,
    PredictedScore,
    Statistics,
    StudentSubmission,
    TokenUsage,
)
from abaes_eval_fwk.core.models.rubric import (
    CriteriaAggregation,
    Criterion,
    Rubric,
    Scale,
    ScoreLevel,
)
from abaes_eval_fwk.templates import initialize
from abaes_eval_fwk.utils.prompt_formatters import truncate_examples_block

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

# The fixture rubric is 0-12 so the fixed 1.0-point disagreement threshold lands at 2.4 points and
# a two-source pool can disagree, the way AES sets 7 and 8 do.
SCALE = Scale(0, 12)
RUBRIC_ID = "fixture_set_1"
STUDENT_WORK = "A concise student essay about the school science fair."
EXAMPLE_WORK = "We built a volcano from baking soda and vinegar."
SOURCE_LABELS = ("qwen3:8b", "gemma3:27b", "llama3.1:8b")

# Names the holistic item's single criterion carries, so a case can say which structure it drives.
HOLISTIC = ""

ADJUDICATION_AGENTS = {
    "AdjudicationFinalHolisticAgent": AdjudicationFinalHolisticAgent,
    "AdjudicationFinalCriterionAgent": AdjudicationFinalCriterionAgent,
    "AdjudicationWithReviewFinalHolisticAgent": AdjudicationWithReviewFinalHolisticAgent,
    "AdjudicationWithReviewFinalCriterionAgent": AdjudicationWithReviewFinalCriterionAgent,
}


class ScaleSpec(TypedDict):
    """A `Scale` as `asdict` serializes it."""

    min: int
    max: int


class ScoreLevelSpec(TypedDict):
    """A `ScoreLevel` as `asdict` serializes it."""

    score: int
    guideline: str


class CriterionSpec(TypedDict):
    """A `Criterion` as `asdict` serializes it."""

    name: str
    score_levels: list[ScoreLevelSpec]


class RubricSpec(TypedDict):
    """A `Rubric` as `asdict` serializes it, so a fixture row can rebuild the rubric it used."""

    rubric_id: str
    assignment_type: str
    source_dependent: bool
    prompt: str
    scale: ScaleSpec
    rubric_guidelines: str
    criteria: list[CriterionSpec]
    is_dimension: bool
    criteria_scale: ScaleSpec | None
    criteria_aggregation: str
    source_text: str
    adjudication_rules: str


class GradingCase(TypedDict):
    """A primary grading prompt case, rendered with one human-scored reference example."""

    kind: Literal["grading"]
    criterion: str
    example_scores: list[int]


class AdjudicationCase(TypedDict):
    """A final-agent case with positional source and independent criterion scores.
    The verdict is computed from scores or supplied to exercise a review branch without an LLM."""

    kind: Literal["adjudication"]
    agent: str
    criterion: str
    sources: list[list[float]]
    independent: list[float]
    source_labels: list[str]
    verdict: Literal["computed", "given"]
    branch: str
    admissible_idx: list[int]
    resolved_score: int


class ReviewCase(TypedDict):
    """A Reviewer's own turn over one source assessment of one item."""

    kind: Literal["review"]
    agent: str
    criterion: str
    sources: list[list[float]]
    independent: list[float]
    source_labels: list[str]


PromptCase = GradingCase | AdjudicationCase | ReviewCase


class PromptRecord(TypedDict):
    """One captured agent turn: the case that produced it, its rubric, and the rendered text."""

    name: str
    rubric: RubricSpec
    case: PromptCase
    branch: str
    system: str
    user: str


class BranchRecord(TypedDict):
    """One final-agent branch decision, re-derivable from the fields recorded here."""

    agent: str
    scores: list[int]
    scale: ScaleSpec
    source_labels: list[str]
    independent_score: int
    admissible_pattern: list[list[bool]] | None
    expected_branch: str
    expected_resolved_score: int
    expected_threshold_points: NotRequired[float]
    napd: float


class SplitRecord(TypedDict):
    """One prepared dataset's split identity."""

    dataset: str
    evaluation_ids: list[str]
    calibration_ids: list[str]
    evaluation_sha256: str
    calibration_sha256: str


class SamplingRecord(TypedDict):
    """The sampling configuration `runner.prepare` records alongside the splits."""

    sample_size: int
    seed: int
    calibration_per_score: int
    essay_ids: list[str] | None


class SplitFile(TypedDict):
    """tests/fixtures/splits.json — the demo's prepared split identity."""

    sampling: SamplingRecord
    datasets: list[SplitRecord]


def fixture_rubric(scale: Scale = SCALE) -> Rubric:
    """The deterministic two-criterion rubric every prompt and branch fixture is rendered against."""
    return Rubric(
        rubric_id=RUBRIC_ID,
        assignment_type="essay",
        source_dependent=False,
        prompt="Write an essay about a memorable event.",
        scale=scale,
        rubric_guidelines="Score accuracy, evidence, and clarity.",
        criteria=[
            Criterion(
                name="ideas",
                score_levels=[
                    ScoreLevel(0, "Missing ideas"),
                    ScoreLevel(scale.max, "Well developed ideas"),
                ],
            ),
            Criterion(
                name="organization",
                score_levels=[
                    ScoreLevel(0, "No structure"),
                    ScoreLevel(scale.max, "Clear structure"),
                ],
            ),
        ],
    )


def rubric_from_spec(spec: RubricSpec) -> Rubric:
    """Rebuild the rubric a fixture row was rendered against, from that row's own recorded spec."""
    return Rubric(
        rubric_id=spec["rubric_id"],
        assignment_type=spec["assignment_type"],
        source_dependent=spec["source_dependent"],
        prompt=spec["prompt"],
        scale=Scale(**spec["scale"]),
        rubric_guidelines=spec["rubric_guidelines"],
        criteria=[
            Criterion(
                name=criterion["name"],
                score_levels=[
                    ScoreLevel(level["score"], level["guideline"])
                    for level in criterion["score_levels"]
                ],
            )
            for criterion in spec["criteria"]
        ],
        is_dimension=spec["is_dimension"],
        criteria_scale=Scale(**spec["criteria_scale"]) if spec["criteria_scale"] else None,
        criteria_aggregation=CriteriaAggregation(spec["criteria_aggregation"]),
        source_text=spec["source_text"],
        adjudication_rules=spec["adjudication_rules"],
    )


def _case_criteria(rubric: Rubric, criterion: str) -> list[Criterion]:
    """The criteria one case supplies: the holistic item's single one, or the rubric's own."""
    return [rubric.holistic_criterion()] if criterion == HOLISTIC else list(rubric.criteria)


def _record(criterion: Criterion, score: float, label: str) -> GradingRecord:
    """One prior assessment of one criterion, the shape final agents pool."""
    return GradingRecord(
        human_assessment=HumanAssessment(
            submission=StudentSubmission(essay_id="fixture-essay-1", text=STUDENT_WORK),
            human_score=6,
            criterion=criterion,
        ),
        llm_assessment=LLMAssessment(
            score=PredictedScore(score),
            justification=f"{label} {criterion.name} assessment",
            confidence=0.8,
            statistics=Statistics(token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1)),
        ),
    )


def _pool(
    rubric: Rubric, case: AdjudicationCase
) -> tuple[list[list[GradingRecord]], list[GradingRecord]]:
    """The source pool and independent assessment `_build_inputs_and_tools` expects for a case."""
    criteria = _case_criteria(rubric, case["criterion"])
    sources = [
        [
            _record(criterion, score, f"source {source_index + 1}")
            for criterion, score in zip(criteria, source, strict=True)
        ]
        for source_index, source in enumerate(case["sources"])
    ]
    independent = [
        _record(criterion, score, "your own")
        for criterion, score in zip(criteria, case["independent"], strict=True)
    ]
    return sources, independent


def _build_items(rubric: Rubric, case: AdjudicationCase) -> list:
    """Build this case's grading items through the real structure axis."""
    agent = ADJUDICATION_AGENTS[case["agent"]]()
    agent.source_labels = tuple(case["source_labels"])
    sources, independent = _pool(rubric, case)
    return list(agent._build_inputs_and_tools(sources, rubric, STUDENT_WORK, independent)["items"])


def _item(rubric: Rubric, case: AdjudicationCase):
    """The one grading item this case names, matched by the label the structure axis gave it."""
    wanted = case["criterion"] or rubric.holistic_criterion().name
    (item,) = [item for item in _build_items(rubric, case) if item.label == wanted]
    return item


def _agent_for(case: AdjudicationCase):
    """The case's final agent, with its source labels set as the procedure axis expects."""
    agent = ADJUDICATION_AGENTS[case["agent"]]()
    agent.source_labels = tuple(case["source_labels"])
    return agent


def _adjudication_case(
    agent: str, criterion: str, *, width: int, **overrides: object
) -> AdjudicationCase:
    """An adjudication case with the two-source, one-independent defaults the cases share.
    `width` is how many scores each source carries: one for a holistic case, one per criterion
    for a criterion case, which is what `CriterionStructure` zips against `rubric.criteria`."""
    case: AdjudicationCase = {
        "kind": "adjudication",
        "agent": agent,
        "criterion": criterion,
        "sources": [[7.0] * width, [7.0] * width],
        "independent": [5.0] * width,
        "source_labels": list(SOURCE_LABELS[:2]),
        "verdict": "computed",
        "branch": "",
        "admissible_idx": [],
        "resolved_score": 0,
    }
    case.update(overrides)  # type: ignore[typeddict-item]
    return case


def _render_adjudication(rubric: Rubric, case: AdjudicationCase) -> tuple[Turn, str]:
    """Render one final-agent branch turn and report the branch the verdict resolved to. The
    review-based branches pick themselves through an LLM, so those cases supply the `Verdict`
    outright while the score-based ones run the procedure's own `_verdict`."""
    agent = _agent_for(case)
    item = _item(rubric, case)
    verdict = (
        agent._verdict([], TokenUsageCallback(), item)
        if case["verdict"] == "computed"
        else Verdict(
            case["branch"],
            resolved_score=case["resolved_score"],
            admissible_idx=tuple(case["admissible_idx"]),
        )
    )
    return agent._build_turn(item, item.template_inputs, verdict), verdict.branch


def _render_grading(rubric: Rubric, case: GradingCase) -> str:
    """Render a primary grading prompt with the input keys `_run_chain` itself adds. With
    `ALLOW_CONTEXT_RECOVERY` off `_run_chain` makes one attempt, so the fixtures hold the
    untruncated examples block, the comparison step that block earns, and no brevity
    instruction."""
    if ALLOW_CONTEXT_RECOVERY:
        raise RuntimeError("Prompt fixtures pin _run_chain's first attempt, not context recovery")
    criterion = rubric.get_criterion(case["criterion"]) or rubric.holistic_criterion()
    agent = HolisticGradingAgent() if case["criterion"] == HOLISTIC else SpecialistAgent(criterion)
    examples = [
        HumanAssessment(
            submission=StudentSubmission(essay_id=f"fixture-example-{index}", text=EXAMPLE_WORK),
            human_score=score,
            criterion=criterion,
        )
        for index, score in enumerate(case["example_scores"])
    ]
    assessment = HumanAssessment(
        submission=StudentSubmission(essay_id="fixture-essay-1", text=STUDENT_WORK),
        human_score=6,
        criterion=criterion,
    )
    raw = agent._build_inputs(rubric, assessment, examples)
    examples_text = truncate_examples_block(raw["examples"], 1.0)
    return PromptTemplate.from_template(agent._prompt_template).format(
        **raw,
        examples_block=f"\nReference examples:\n\n{examples_text}" if examples_text else "",
        examples_comparison_step=(
            " Also compare it against the reference examples provided to calibrate your judgment."
            if examples_text
            else ""
        ),
        brevity_instruction="",
    )


class _TurnCapture:
    """Stands in for a Reviewer's `run_turn`, keeping the turn text `Reviewer.review` formatted."""

    def __init__(self) -> None:
        self.turns: list[Turn] = []

    def __call__(
        self,
        conversation: list,
        cb: TokenUsageCallback,
        system_text: str,
        human_text: str,
        schema: type,
        turn_name: str,
    ) -> dict:
        self.turns.append(Turn(system=system_text, user=human_text))
        return {
            "reasoning": "r",
            "rubric_consistency": True,
            "rubric_consistency_justification": "",
            "evidence_grounding": True,
            "evidence_grounding_justification": "",
        }


def _render_review(rubric: Rubric, case: ReviewCase) -> Turn:
    """Drive a real `Reviewer.review` over one source assessment and return the turn it sent."""
    item = _item(
        rubric,
        _adjudication_case(
            case["agent"],
            case["criterion"],
            width=len(case["independent"]),
            sources=case["sources"],
            independent=case["independent"],
            source_labels=case["source_labels"],
        ),
    )
    capture = _TurnCapture()
    reviewer = Reviewer(
        capture,
        None,
        rubric_id=rubric.rubric_id,
        essay_id="fixture-essay-1",
        assignment_prompt=rubric.prompt,
        student_work=STUDENT_WORK,
    )
    assessment = item.assessments[0]
    reviewer.review(
        [],
        TokenUsageCallback(),
        criterion=item.label,
        model=case["source_labels"][0],
        prompt_label=item.prompt_label,
        score=assessment.score.rounded_score,
        justification=assessment.justification,
        guidelines=item.guidelines,
        scale=item.scale,
    )
    return capture.turns[0]


def render_prompt_record(name: str, case: PromptCase, rubric: Rubric) -> PromptRecord:
    """Render one named case against `rubric`, through the same code path the agent itself uses."""
    spec: RubricSpec = asdict(rubric)  # type: ignore[assignment]
    if case["kind"] == "grading":
        turn, branch = Turn(system="", user=_render_grading(rubric, case)), ""
    elif case["kind"] == "adjudication":
        turn, branch = _render_adjudication(rubric, case)
    else:
        turn, branch = _render_review(rubric, case), ""
    return PromptRecord(
        name=name, rubric=spec, case=case, branch=branch, system=turn.system, user=turn.user
    )


def _prompt_cases() -> list[tuple[str, PromptCase]]:
    """Every prompt case, in fixture order, with the name its record is filed under."""
    cases: list[tuple[str, PromptCase]] = [
        ("grading:holistic", {"kind": "grading", "criterion": HOLISTIC, "example_scores": [6, 9]}),
        ("grading:ideas", {"kind": "grading", "criterion": "ideas", "example_scores": [5, 11]}),
        (
            "grading:organization",
            {"kind": "grading", "criterion": "organization", "example_scores": [4]},
        ),
    ]
    cases.append(
        (
            "adjudication:holistic:agreement",
            _adjudication_case("AdjudicationFinalHolisticAgent", HOLISTIC, width=1),
        )
    )
    cases.append(
        (
            "adjudication:holistic:disagreement",
            _adjudication_case(
                "AdjudicationFinalHolisticAgent", HOLISTIC, width=1, sources=[[1.0], [10.0]]
            ),
        )
    )
    # The two sources agree on ideas and disagree on organization, so the criterion agent's one
    # fixture pair covers both of its branches.
    for criterion in ("ideas", "organization"):
        cases.append(
            (
                f"adjudication:criterion:{criterion}",
                _adjudication_case(
                    "AdjudicationFinalCriterionAgent",
                    criterion,
                    width=2,
                    sources=[[7.0, 7.0], [7.0, 1.0]],
                ),
            )
        )
    for branch, admissible_idx, resolved_score, sources, independent in (
        ("positive", [0, 1], 8, [[7.0], [8.0]], [4.0]),
        ("mixed", [0], 6, [[7.0], [8.0]], [4.0]),
        ("negative", [], 4, [[7.0], [1.0]], [4.0]),
    ):
        cases.append(
            (
                f"adjudication-with-review:holistic:{branch}",
                _adjudication_case(
                    "AdjudicationWithReviewFinalHolisticAgent",
                    HOLISTIC,
                    width=1,
                    sources=sources,
                    independent=independent,
                    verdict="given",
                    branch=branch,
                    admissible_idx=admissible_idx,
                    resolved_score=resolved_score,
                ),
            )
        )
    for branch, admissible_idx, resolved_score in (
        ("positive", [0, 1], 4),
        ("mixed", [0], 6),
        ("negative", [], 5),
    ):
        cases.append(
            (
                f"adjudication-with-review:criterion:ideas:{branch}",
                _adjudication_case(
                    "AdjudicationWithReviewFinalCriterionAgent",
                    "ideas",
                    width=2,
                    sources=[[7.0, 7.0], [7.0, 1.0]],
                    independent=[5.0, 5.0],
                    verdict="given",
                    branch=branch,
                    admissible_idx=admissible_idx,
                    resolved_score=resolved_score,
                ),
            )
        )
    for agent, criterion in (
        ("AdjudicationWithReviewFinalHolisticAgent", HOLISTIC),
        ("AdjudicationWithReviewFinalCriterionAgent", "ideas"),
    ):
        cases.append(
            (
                f"review:{criterion or 'holistic'}",
                {
                    "kind": "review",
                    "agent": agent,
                    "criterion": criterion,
                    "sources": [[7.0, 7.0], [7.0, 1.0]] if criterion else [[7.0], [8.0]],
                    "independent": [5.0, 5.0] if criterion else [4.0],
                    "source_labels": list(SOURCE_LABELS[:2]),
                },
            )
        )
    return cases


def prompt_records(rubric: Rubric | None = None) -> list[PromptRecord]:
    """Every captured prompt turn, each carrying the rubric spec that reproduces it."""
    records = [
        render_prompt_record(name, case, rubric or fixture_rubric())
        for name, case in _prompt_cases()
    ]
    if len({record["name"] for record in records}) != len(records):
        raise RuntimeError("Prompt fixture names must be unique")
    return records


class _PatternReviewer:
    """Duck-types `Reviewer` for `_verdict`: admissibility comes from a recorded pattern keyed by
    (criterion, source model), so the real procedure still picks the branch and the score."""

    def __init__(self, pattern: dict[tuple[str, str], tuple[bool, bool]]) -> None:
        self._pattern = pattern

    def is_admissible(self, review: dict) -> bool:
        return all(bool(review.get(name)) for name in Reviewer.admissibility_criteria)

    def review(
        self,
        conversation: list,
        cb: TokenUsageCallback,
        *,
        criterion: str,
        model: str,
        prompt_label: str,
        score: int,
        justification: str,
        guidelines: str,
        scale: Scale,
    ) -> dict:
        consistent, grounded = self._pattern[(criterion, model)]
        return {
            "rubric_consistency": consistent,
            "rubric_consistency_justification": "" if consistent else "off-level",
            "evidence_grounding": grounded,
            "evidence_grounding_justification": "" if grounded else "unsupported",
        }


def _branch_case(
    agent_name: str,
    rubric: Rubric,
    scores: list[int],
    independent: int,
    admissible: list[tuple[bool, bool]] | None = None,
) -> BranchRecord:
    """Drive one item's real `_verdict` and record the branch and score it produced."""
    source_labels = list(SOURCE_LABELS[: len(scores)])
    case = _adjudication_case(
        agent_name,
        HOLISTIC,
        width=1,
        sources=[[float(score)] for score in scores],
        independent=[float(independent)],
        source_labels=source_labels,
    )
    agent = _agent_for(case)
    item = _item(rubric, case)
    if admissible is not None:
        agent._reviewer = _PatternReviewer(
            {
                (item.label, label): flags
                for label, flags in zip(source_labels, admissible, strict=True)
            }
        )
    verdict = agent._verdict([], TokenUsageCallback(), item)
    record: BranchRecord = {
        "agent": agent_name,
        "scores": scores,
        "scale": asdict(item.scale),
        "source_labels": source_labels,
        "independent_score": independent,
        "admissible_pattern": [list(flags) for flags in admissible] if admissible else None,
        "expected_branch": verdict.branch,
        "expected_resolved_score": verdict.resolved_score,  # type: ignore[typeddict-item]
        "napd": normalized_average_pairwise_disagreement(scores, item.scale),
    }
    # Only the score-based procedure has a threshold to state in its prompt.
    if hasattr(agent, "_threshold_points"):
        record["expected_threshold_points"] = agent._threshold_points(item.scale)
    return record


def branch_records() -> list[BranchRecord]:
    """Every branch decision, driven through the real procedures rather than restated by hand."""
    records: list[BranchRecord] = []
    for scale in (Scale(0, 5), Scale(0, 12)):
        rubric = fixture_rubric(scale)
        for scores in ([3, 3], [2, 3], [1, 4], [2, 2, 3], [1, 2, 5]):
            records.append(
                _branch_case("AdjudicationFinalHolisticAgent", rubric, scores, scores[0] + 2)
            )
    # On 0-12 the threshold is 2.4 points, which no integer gap can equal, so the 0-5 rows' exact
    # threshold case is the only one that lands on it and these bracket the boundary.
    wide = fixture_rubric(Scale(0, 12))
    for scores in ([6, 6], [6, 8], [6, 9], [5, 5, 6], [4, 4, 8]):
        records.append(_branch_case("AdjudicationFinalHolisticAgent", wide, scores, scores[0] + 1))
    for admissible in (
        ((True, True), (True, True)),  # every source retained: positive
        ((True, True), (False, False)),  # one retained: mixed
        ((True, False), (False, False)),  # none retained: negative
    ):
        records.append(
            _branch_case(
                "AdjudicationWithReviewFinalHolisticAgent",
                fixture_rubric(),
                [7, 1],
                5,
                list(admissible),
            )
        )
    return records


def demo_benchmark(root: Path) -> Path:
    """Initialize the offline playground template in `root`, as `init` does."""
    return initialize(root / "demo", "playground")


def split_record(root: Path) -> SplitFile:
    """Prepare the demo's splits with `runner.prepare` and cross-check `runner._snapshots`."""
    from abaes_eval_fwk import runner
    from abaes_eval_fwk.configuration import dataset_configs, load_config
    from abaes_eval_fwk.experiment.setup.setup_builder import build_setup

    config = load_config(demo_benchmark(root))
    prepared = json.loads(runner.prepare(config).read_text(encoding="utf-8"))
    essay_ids = (
        [str(value) for value in config.sampling.essay_ids] if config.sampling.essay_ids else None
    )
    snapshots = runner._snapshots(
        [
            build_setup(dataset, False, config.sampling.sample_size, essay_ids=essay_ids)
            for dataset in dataset_configs(config)
        ]
    )
    if prepared["datasets"] != snapshots:
        raise RuntimeError("runner.prepare and runner._snapshots disagree on the demo splits")
    return SplitFile(sampling=prepared["sampling"], datasets=list(snapshots))


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def capture_prompts() -> list[PromptRecord]:
    """Write tests/fixtures/prompts.jsonl."""
    records = prompt_records()
    _write_jsonl(FIXTURES / "prompts.jsonl", list(records))
    return records


def capture_branches() -> list[BranchRecord]:
    """Write tests/fixtures/branches.jsonl."""
    records = branch_records()
    _write_jsonl(FIXTURES / "branches.jsonl", list(records))
    return records


def capture_splits() -> SplitFile:
    """Write tests/fixtures/splits.json from a freshly prepared demo project."""
    with tempfile.TemporaryDirectory() as directory:
        record = split_record(Path(directory))
    _write_json(FIXTURES / "splits.json", dict(record))
    return record


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    prompts = capture_prompts()
    branches = capture_branches()
    splits = capture_splits()
    print(
        f"wrote {len(prompts)} prompts, {len(branches)} branches, and "
        f"{len(splits['datasets'])} splits to {FIXTURES}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
