"""Prompt formatting helpers for rendering reference examples and rubric guidelines."""

from __future__ import annotations

import re

from abaes_eval_fwk.core.models.assessment import HumanAssessment
from abaes_eval_fwk.core.models.rubric import Rubric

_SCORE_HEADER_RE = re.compile(r"(--- Score \d+ ---)")


def format_examples(reference_examples: list[HumanAssessment] | None) -> str:
    """Render reference examples as a few-shot text block for prompts."""
    if not reference_examples:
        return ""
    blocks = [ex.as_example() for ex in reference_examples]
    return "\n\n".join(blocks)


def truncate_examples_block(examples_text: str, fraction: float) -> str:
    """Truncate each formatted example to a fraction of its words."""
    if not examples_text or fraction >= 1.0:
        return examples_text
    parts = _SCORE_HEADER_RE.split(examples_text)
    result = [parts[0]]
    i = 1
    while i + 1 < len(parts):
        header = parts[i]
        body = parts[i + 1].strip()
        result.append(header)
        if body.startswith('"""') and body.endswith('"""'):
            words = body[3:-3].strip().split()
            target = max(1, int(len(words) * fraction))
            truncated = " ".join(words[:target]) + (" [...]" if target < len(words) else "")
            result.append(f'\n"""\n{truncated}\n"""\n')
        else:
            result.append(parts[i + 1])
        i += 2
    return "\n".join(result)


def format_criteria(rubric: Rubric) -> str:
    """Format rubric criteria and score-level guidelines into a structured prompt block."""
    lines = []
    for c in rubric.criteria:
        scale = rubric.get_criterion_scale()
        header = f"{c.name} (scale {scale.min}–{scale.max}):"
        guidelines = "\n".join(f"Score {lv.score}: {lv.guideline}" for lv in c.score_levels)
        lines.append(f"{header}\n{guidelines}")
    return "\n\n".join(lines)
