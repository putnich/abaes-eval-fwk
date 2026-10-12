"""Grades an essay holistically across all rubric dimensions in a single LLM pass."""

import logging

from abaes_eval_fwk.core.agents.prompts.holistic_grading_prompts import HOLISTIC_GRADING_PROMPT
from abaes_eval_fwk.core.models.assessment import HumanAssessment, LLMAssessment
from abaes_eval_fwk.core.models.rubric import Rubric
from abaes_eval_fwk.experiment.result.result_parsers import parse_grading_result

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class HolisticGradingAgent(BaseAgent):
    """Grades an essay holistically in one LLM call, producing a score on the global rubric scale."""

    def __init__(self, prompt_template: str | None = None) -> None:
        super().__init__(prompt_template or HOLISTIC_GRADING_PROMPT)

    def _build_inputs(  # type: ignore[override]
        self,
        rubric: Rubric,
        human_assessment: HumanAssessment,
        reference_examples: list[HumanAssessment],
    ) -> dict:
        """Build prompt input key-value pairs for holistic grading."""
        scale = rubric.total_score_range()
        return {
            **self._build_common_inputs(rubric, human_assessment, reference_examples),
            "rubric_guidelines": rubric.rubric_guidelines,
            "score_min": scale.min,
            "score_max": scale.max,
        }

    def _parse_result(
        self,
        result: dict,
        usage,
        rubric: Rubric,
    ) -> LLMAssessment:
        """Parse raw LLM response dict into a standardized LLMAssessment instance."""
        scale = rubric.total_score_range()
        return parse_grading_result(result, scale.min, scale.max, usage)
