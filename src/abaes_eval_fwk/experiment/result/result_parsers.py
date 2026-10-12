"""Validates and converts extracted LLM JSON responses into LLMAssessment domain objects."""

import logging

from abaes_eval_fwk.core.models.assessment import (
    LLMAssessment,
    PredictedScore,
    Statistics,
    TokenUsage,
)

logger = logging.getLogger(__name__)


def parse_grading_result(
    result: dict,
    score_min: float,
    score_max: float,
    usage: TokenUsage,
) -> LLMAssessment:
    """Validate numeric score boundaries and parse LLM dictionary into a LLMAssessment object."""
    score = float(result["score"])
    if not (score_min <= score <= score_max):
        raise ValueError(f"score {score} out of range [{score_min}, {score_max}]")
    logger.debug("  parsed score=%.1f", score)
    return LLMAssessment(
        score=PredictedScore(raw_score=score),
        justification=result.get("justification", ""),
        confidence=result.get("confidence"),
        reasoning=result.get("reasoning"),
        branch=result.get("branch", ""),
        pool_average_score=result.get("pool_average_score", ""),
        statistics=Statistics(token_usage=usage),
    )
