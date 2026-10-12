"""Content-aware synthetic responses for the bundled playground project, never model benchmark
results."""

from __future__ import annotations

import math
import re

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from abaes_eval_fwk.utils.rounding import round_half_up

_SUBMISSION_RE = re.compile(r'Student submission to grade:\s*"""\s*(.*?)\s*"""', re.DOTALL)

# Calibrated against the bundled demo.csv's shortest ("score 0") and longest ("score 4") essays,
# traceability suffix included, so word count alone recovers each essay's intended score.
_WORD_COUNT_FLOOR = 13
_WORD_COUNT_CEIL = 39


def _message_text(messages) -> str:
    """The full rendered prompt text, whichever shape the chain hands `with_structured_output`:
    a plain string normally, or a [Human, AI] pair when reasoning is `no_think`."""
    if isinstance(messages, str):
        return messages
    return "\n".join(getattr(message, "content", "") for message in messages)


def _continuous_score(messages) -> float:
    """A 0-4 estimate of essay quality from the submission's own length — real content signal in
    the bundled demo essays, standing in for language understanding a real model would apply.
    Retrieved examples and rubric guidelines precede this marker in the prompt, so they are not
    mistaken for the submission itself."""
    match = _SUBMISSION_RE.search(_message_text(messages))
    if match is None:
        return 2.0
    word_count = len(match.group(1).split())
    fraction = (word_count - _WORD_COUNT_FLOOR) / (_WORD_COUNT_CEIL - _WORD_COUNT_FLOOR)
    return max(0.0, min(4.0, fraction * 4))


class DemoModel:
    """Score from the submission's own content through the real grading and adjudication
    pipeline, so disagreement and review procedures resolve something real instead of a constant
    gap. "a" grades harshly, "b" leniently, and "final" lands on the accurate value they disagree
    around — the independent assessment disagreement/review procedures fall back to."""

    _SCORERS = {
        "demo-a": math.floor,
        "demo-b": math.ceil,
        "demo-final": round_half_up,
    }

    def __init__(self, config):
        self._score = self._SCORERS[config.model_name]

    def invoke(self, messages, **kwargs):
        raise RuntimeError("The synthetic adapter requires a structured output schema")

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def respond(messages):
            if "rubric_consistency" in schema.model_fields:
                values = {
                    "rubric_consistency": True,
                    "evidence_grounding": True,
                    "rubric_consistency_justification": "",
                    "evidence_grounding_justification": "",
                }
            else:
                values = {
                    "confidence": 0.5,
                    "justification": "Synthetic playground response; this is not a model assessment.",
                }
                if "score" in schema.model_fields:
                    values["score"] = self._score(_continuous_score(messages))
            parsed = schema(**values)
            raw = AIMessage(
                content=parsed.model_dump_json(),
                usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(respond)


def create_model(config):
    """Create the offline adapter for the bundled playground project."""
    return DemoModel(config)
