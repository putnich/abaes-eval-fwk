"""A reader-supplied criteria aggregation, named in a rubric as
`criteria_aggregation: extensions.aggregation:<function>`.

The callable receives the rubric and that essay's criterion scores, keyed by criterion name, and
returns the overall score. It is only needed for a formula the three built-in modes cannot state:
`mean` and `sum` need no code, and `weighted` takes per-criterion weights in the rubric YAML:

    criteria_aggregation: weighted
    criteria_weights:
      ideas: 0.5
      organization: 0.3
      conventions: 0.2

The returned total is rounded half-up exactly as a built-in mode's is.
"""

from __future__ import annotations

import statistics


def median(*, rubric, scores):
    """The median criterion score, which one extreme criterion cannot drag."""
    return statistics.median(scores.values())


def lowest_two_mean(*, rubric, scores):
    """The mean of the two weakest criteria, for a rubric that grades on what is missing."""
    ordered = sorted(scores.values())
    weakest = ordered[:2] or ordered
    return sum(weakest) / len(weakest)


def capped_mean(*, rubric, scores):
    """The mean, but no higher than one point above the weakest criterion.

    A worked example of a rule the declarative modes cannot express: the cap depends on the
    spread between criteria, not on fixed weights.
    """
    values = list(scores.values())
    return min(sum(values) / len(values), min(values) + 1)
