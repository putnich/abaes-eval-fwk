"""Protect deterministic prompt, branch, and split fixtures against behavioral regressions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import capture_fixtures as capture  # noqa: E402

from abaes_eval_fwk.core.agents.base_agent import TokenUsageCallback  # noqa: E402
from abaes_eval_fwk.core.agents.final_agent import (  # noqa: E402
    normalized_average_pairwise_disagreement,
)
from abaes_eval_fwk.core.models.rubric import Scale  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def _jsonl(name: str) -> list[dict]:
    """The recorded rows of one fixture file."""
    return [json.loads(line) for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines()]


def _json(name: str) -> dict:
    """One recorded fixture file."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _branch_verdict(record: dict):
    """The real `_verdict` result for a recorded branch row, reviewer stub included when needed."""
    case = capture._adjudication_case(
        record["agent"],
        capture.HOLISTIC,
        width=1,
        sources=[[float(score)] for score in record["scores"]],
        independent=[float(record["independent_score"])],
        source_labels=record["source_labels"],
    )
    agent = capture._agent_for(case)
    item = capture._item(capture.fixture_rubric(Scale(**record["scale"])), case)
    if record["admissible_pattern"]:
        agent._reviewer = capture._PatternReviewer(
            {
                (item.label, label): tuple(flags)
                for label, flags in zip(
                    record["source_labels"], record["admissible_pattern"], strict=True
                )
            }
        )
    return agent, item, agent._verdict([], TokenUsageCallback(), item)


@pytest.mark.parametrize("record", _jsonl("prompts.jsonl"), ids=lambda record: record["name"])
def test_prompt_fixture_matches_a_fresh_rendering(record: dict) -> None:
    """Every captured turn is rebuilt from its own recorded rubric and case, then compared."""
    fresh = capture.render_prompt_record(
        record["name"], record["case"], capture.rubric_from_spec(record["rubric"])
    )
    assert fresh["rubric"] == record["rubric"]
    assert fresh["branch"] == record["branch"]
    assert fresh["system"] == record["system"]
    assert fresh["user"] == record["user"]


def test_prompt_fixtures_cover_every_agent_and_branch() -> None:
    """Both procedure axes, both structure axes, their branches, and the review turn are pinned."""
    records = _jsonl("prompts.jsonl")
    assert len({record["name"] for record in records}) == len(records)
    assert {record["case"]["kind"] for record in records} == {"grading", "adjudication", "review"}
    assert {record["branch"] for record in records if record["case"]["kind"] == "adjudication"} == {
        "agreement",
        "disagreement",
        "positive",
        "mixed",
        "negative",
    }
    agents = {record["case"]["agent"] for record in records if record["case"]["kind"] != "grading"}
    assert agents == set(capture.ADJUDICATION_AGENTS)


@pytest.mark.parametrize(
    "record", _jsonl("branches.jsonl"), ids=lambda record: str(record["scores"])
)
def test_branch_fixture_matches_a_fresh_verdict(record: dict) -> None:
    """Each branch row's decision, score, threshold, and NAPD are re-derived from its pool."""
    agent, item, verdict = _branch_verdict(record)
    assert verdict.branch == record["expected_branch"]
    assert verdict.resolved_score == record["expected_resolved_score"]
    assert normalized_average_pairwise_disagreement(
        record["scores"], Scale(**record["scale"])
    ) == pytest.approx(record["napd"])
    if "expected_threshold_points" in record:
        assert agent._threshold_points(item.scale) == record["expected_threshold_points"]


def test_branch_fixtures_pin_both_procedures_and_straddle_the_threshold() -> None:
    """Every pickable branch appears, and the score-based one has a row at its threshold."""
    records = _jsonl("branches.jsonl")
    score_based = [record for record in records if "expected_threshold_points" in record]
    review_based = [record for record in records if "expected_threshold_points" not in record]
    assert {record["expected_branch"] for record in score_based} == {"agreement", "disagreement"}
    assert {record["expected_branch"] for record in review_based} == {
        "positive",
        "mixed",
        "negative",
    }
    assert {record["scale"]["max"] for record in records} == {5, 12}
    assert {len(record["scores"]) for record in records} == {2, 3}
    exact = [
        record for record in score_based if record["napd"] == record["expected_threshold_points"]
    ]
    assert exact, "a row must sit exactly on the disagreement threshold"


def test_split_fixture_matches_a_freshly_prepared_demo(tmp_path: Path) -> None:
    """The recorded essay ids and hashes come from running prepare on a demo project built now."""
    assert _json("splits.json") == capture.split_record(tmp_path)
