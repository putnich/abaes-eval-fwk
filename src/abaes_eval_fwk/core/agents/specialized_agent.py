"""Agent bound to a single rubric criterion, for parallel specialist scoring."""

import logging

from abaes_eval_fwk.core.agents.prompts.criterion_specialized_grading_prompts import (
    CRITERION_SPECIALIZED_GRADING_PROMPT,
)
from abaes_eval_fwk.core.models.assessment import HumanAssessment, LLMAssessment
from abaes_eval_fwk.core.models.rubric import Criterion, Rubric
from abaes_eval_fwk.experiment.result.result_parsers import parse_grading_result

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SpecialistAgent(BaseAgent):
    """Specialist agent focused on evaluating a single rubric criterion."""

    def __init__(self, criterion: Criterion, prompt_template: str | None = None):
        super().__init__(prompt_template or CRITERION_SPECIALIZED_GRADING_PROMPT)
        self.criterion = criterion

    def _build_inputs(  # type: ignore[override]
        self,
        rubric: Rubric,
        human_assessment: HumanAssessment,
        reference_examples: list[HumanAssessment],
    ) -> dict:
        """Build prompt inputs targeting the specialist's assigned criterion."""
        scale = rubric.get_criterion_scale()
        criterion_guidelines = "\n".join(
            f"Score {lv.score}: {lv.guideline}" for lv in self.criterion.score_levels
        )
        return {
            **self._build_common_inputs(rubric, human_assessment, reference_examples),
            "criterion_label": self.criterion.name.replace("_", " "),
            "criterion_guidelines": criterion_guidelines,
            "score_min": scale.min,
            "score_max": scale.max,
        }

    def _parse_result(
        self,
        result: dict,
        usage,
        rubric: Rubric,
    ) -> LLMAssessment:
        """Parse LLM output into a LLMAssessment structured for this criterion scale."""
        scale = rubric.get_criterion_scale()
        return parse_grading_result(result, scale.min, scale.max, usage)
