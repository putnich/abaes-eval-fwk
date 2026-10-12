"""A reader-supplied retrieval selector, named as `extensions.selectors:<function>`."""

from __future__ import annotations


def lowest_score_first(*, rubric, calibration_pool, student_text, criterion):
    """Return the criterion's calibration examples ordered by human score, lowest first."""
    name = criterion.name
    matching = [item for item in calibration_pool if item.criterion.name == name]
    if not matching:
        matching = [item for item in calibration_pool if item.criterion.name == "holistic"]
    return sorted(matching, key=lambda item: (item.human_score, item.essay_id))[:3]
