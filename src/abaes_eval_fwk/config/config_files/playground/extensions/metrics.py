"""Reader-supplied metrics, named as `metrics.custom.<name>: extensions.metrics:<function>`.

A metric is handed the paired rounded scores and the dataset's scale as keyword arguments, and
returns a number — the same signature the built-in metrics have. Return NaN for "undefined on
this sample" rather than raising, the way the correlations do when a side is constant; the saved
table prints an empty cell for it. The configured name becomes the column header.
"""

from __future__ import annotations

import math


def exact_agreement(*, preds, humans, scale):
    """The fraction of essays scored exactly as the human rater scored them."""
    matches = sum(1 for p, h in zip(preds, humans, strict=True) if p == h)
    return matches / len(preds)


def adjacent_agreement(*, preds, humans, scale):
    """The fraction scored within one point of the human rater — the usual AES reporting band."""
    within = sum(1 for p, h in zip(preds, humans, strict=True) if abs(p - h) <= 1)
    return within / len(preds)


def mean_signed_error(*, preds, humans, scale):
    """Mean signed error, so a model that reads consistently high is distinguishable from noise."""
    return sum(p - h for p, h in zip(preds, humans, strict=True)) / len(preds)


def score_range_coverage(*, preds, humans, scale):
    """The share of the scale's score points the model ever actually used.

    NaN when the scale is a single point, which is the one case where coverage says nothing.
    """
    points = scale.max - scale.min + 1
    if points <= 1:
        return math.nan
    return len(set(preds)) / points
