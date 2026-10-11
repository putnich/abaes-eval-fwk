"""Structured-output response schemas for LLM grading calls — the response half of each agent's
LLM contract, mirroring how core/agents/prompts/ holds the request (template) half.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

_CONFIDENCE_DESCRIPTION = (
    "Confidence in this assessment, as a float from 0.0 to 1.0 (not a percentage). "
    "Above 0.75 = high confidence: clearly supported by the rubric and specific evidence "
    "from the submission, with little or no ambiguity. "
    "0.5-0.75 = moderate confidence: generally supported by the rubric and evidence, but "
    "some ambiguity or uncertainty remains. "
    "Below 0.5 = low confidence: limited or mixed support from the rubric and evidence, "
    "or the submission is borderline between score levels."
)


class _NormalizesConfidence:
    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, v):
        if isinstance(v, (int, float)) and v > 1:
            return v / 100
        return v


_JUSTIFICATION_DESCRIPTION = (
    "Justification of the grading decision. Explain why the assigned score is supported by "
    "connecting specific evidence from the submission to the rubric's score-level guidelines."
)

_RUBRIC_CONSISTENCY_DESCRIPTION = (
    "Whether the justification supports the assigned score and aligns with the rubric "
    "guidelines at that score level."
)

_EVIDENCE_GROUNDING_DESCRIPTION = (
    "Whether the justification's claims are accurate and supported by evidence actually "
    "present in the student submission."
)

_RUBRIC_CONSISTENCY_JUSTIFICATION_DESCRIPTION = (
    "Why rubric consistency failed: name the specific mismatch between the justification and "
    "the rubric guidelines at the assigned score level. Leave empty when rubric_consistency is "
    "true."
)

_EVIDENCE_GROUNDING_JUSTIFICATION_DESCRIPTION = (
    "Why evidence grounding failed: name the specific claim in the justification that is "
    "inaccurate or unsupported by the submission. Leave empty when evidence_grounding is true."
)


class GradingResponse(_NormalizesConfidence, BaseModel):
    """Grading calls whose score is being decided now by the FinalAgent."""

    score: float = Field(
        description="The final score, within the official range given in the prompt."
    )
    confidence: float = Field(ge=0.0, le=1.0, description=_CONFIDENCE_DESCRIPTION)
    justification: str = Field(description=_JUSTIFICATION_DESCRIPTION)


class GradingNoScoreResponse(_NormalizesConfidence, BaseModel):
    """A turn whose score is already resolved programmatically — text only."""

    confidence: float = Field(ge=0.0, le=1.0, description=_CONFIDENCE_DESCRIPTION)
    justification: str = Field(description=_JUSTIFICATION_DESCRIPTION)


class ReviewResult(BaseModel):
    """A review turn's verdict on one prior assessment (Adjudication with Review)."""

    rubric_consistency: bool = Field(description=_RUBRIC_CONSISTENCY_DESCRIPTION)
    rubric_consistency_justification: str = Field(
        description=_RUBRIC_CONSISTENCY_JUSTIFICATION_DESCRIPTION
    )
    evidence_grounding: bool = Field(description=_EVIDENCE_GROUNDING_DESCRIPTION)
    evidence_grounding_justification: str = Field(
        description=_EVIDENCE_GROUNDING_JUSTIFICATION_DESCRIPTION
    )


class AdjudicationWithReviewDecision(_NormalizesConfidence, BaseModel):
    """Adopt one of two assessments' scores, or average both."""

    decision: Literal["select_a", "select_b", "equally_strong"] = Field(
        description="Which assessment is better supported, or whether they're equally strong."
    )
    confidence: float = Field(ge=0.0, le=1.0, description=_CONFIDENCE_DESCRIPTION)
    justification: str = Field(description=_JUSTIFICATION_DESCRIPTION)
