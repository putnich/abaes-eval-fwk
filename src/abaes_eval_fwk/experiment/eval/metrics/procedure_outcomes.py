"""Count final-agent procedure outcomes and review cost from the files a deliberation run reads.

Disagreement-based adjudication (DA) outcomes are recomputed from the pool's own source results
with the rule the procedure applies at run time, so they never depend on a recorded branch column.
Review-based adjudication (RA) outcomes and review cost come from reviews CSVs, one row per
reviewed source assessment, as a Review run writes them.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from abaes_eval_fwk.core.agents.final_agent import normalized_average_pairwise_disagreement
from abaes_eval_fwk.core.models.assessment import ReviewField, parse_csv_flag
from abaes_eval_fwk.core.models.rubric import NAPD_REFERENCE_WIDTH, Rubric
from abaes_eval_fwk.experiment.eval.baseline.build_pool_average_baseline import (
    matched_source_records,
)

DA_OUTCOMES = ("agreement", "disagreement")
RA_OUTCOMES = ("positive", "mixed", "negative")
# Pooled over every item, as the share of all pools a procedure resolved one way; not a dataset
# average, since criterion-specialized datasets contribute different numbers of items.
ALL_ITEMS = "All items"

Item = tuple[str, str, str]  # (dataset, essay_id, criterion)


class ItemOutcome(NamedTuple):
    outcome: str
    disagreement: float | None = None  # DA: normalized mean pairwise distance, a 0–1 fraction
    admissible: int | None = None  # RA: admissible reviews for the item
    reviewed: int | None = None  # RA: reviews for the item, one per source


def disagreement_outcomes(
    source_csvs: Sequence[Path], rubrics: Mapping[str, Rubric], threshold: float = 0.2
) -> dict[Item, ItemOutcome]:
    """Classify every pooled item as DA would: disagreement when the normalized mean pairwise
    distance between the rounded source scores exceeds `threshold`, agreement otherwise."""
    if isinstance(threshold, bool) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a fraction between 0 and 1")
    sources = matched_source_records(source_csvs)
    if len(sources) < 2:
        raise ValueError("Disagreement needs at least two pool sources")
    # The same conversion `run` applies to an experiment's `disagreement_threshold`.
    threshold_points = threshold * NAPD_REFERENCE_WIDTH
    outcomes = {}
    for key in sorted(sources[0]):
        label, _, criterion = key
        if label not in rubrics:
            raise ValueError(f"No rubric supplied for {label!r}")
        rubric = rubrics[label]
        scale = (
            rubric.total_score_range() if criterion == "holistic" else rubric.get_criterion_scale()
        )
        scores = [source[key].llm_assessment.score.rounded_score for source in sources]
        napd = normalized_average_pairwise_disagreement(scores, scale)
        outcomes[key] = ItemOutcome(
            "disagreement" if napd > threshold_points else "agreement",
            disagreement=napd / NAPD_REFERENCE_WIDTH,
        )
    return outcomes


_REVIEW_COLUMNS = (
    ReviewField.DATASET,
    ReviewField.ESSAY_ID,
    ReviewField.CRITERION,
    ReviewField.SOURCE_MODEL,
    ReviewField.IS_ADMISSIBLE,
    ReviewField.COMPLETION_TOKENS,
)


def read_reviews(review_csvs: Sequence[Path]) -> pd.DataFrame:
    """Read reviews CSVs into one frame with one row per (item, source model)."""
    if not review_csvs:
        raise ValueError("At least one reviews CSV is required")
    frames = []
    for path in review_csvs:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if missing := {c.value for c in _REVIEW_COLUMNS} - set(frame.columns):
            raise ValueError(f"{path}: missing reviews columns {sorted(missing)}")
        frames.append(frame[[c.value for c in _REVIEW_COLUMNS]])
    reviews = pd.concat(frames, ignore_index=True)
    key = ["dataset", "essay_id", "criterion", "source_model"]
    if reviews.duplicated(key).any():
        first = reviews[reviews.duplicated(key)].iloc[0]
        raise ValueError(f"Duplicate review for {tuple(first[key])}")
    reviews["is_admissible"] = [parse_csv_flag(v, "is_admissible") for v in reviews.is_admissible]
    return reviews


def review_outcomes(reviews: pd.DataFrame) -> dict[Item, ItemOutcome]:
    """Classify every reviewed item as RA would: positive when every source assessment is
    admissible, negative when none is, mixed otherwise."""
    models = set(reviews.source_model)
    outcomes = {}
    for (label, essay_id, criterion), group in reviews.groupby(
        ["dataset", "essay_id", "criterion"], sort=True
    ):
        if set(group.source_model) != models:
            raise ValueError(
                f"{label}/{essay_id}/{criterion} lacks reviews for "
                f"{sorted(models - set(group.source_model))}"
            )
        admissible = int(group.is_admissible.sum())
        outcome = (
            "positive" if admissible == len(group) else "negative" if admissible == 0 else "mixed"
        )
        outcomes[(label, essay_id, criterion)] = ItemOutcome(
            outcome, admissible=admissible, reviewed=len(group)
        )
    return outcomes


def has_review_tokens(reviews: pd.DataFrame) -> bool:
    """Whether the reviews record their cost. A Review run records it for every review; reviews
    an adjudication run caches for replay record none."""
    missing = reviews.completion_tokens == ""
    if missing.any() and not missing.all():
        raise ValueError("completion_tokens is recorded for some reviews but not others")
    return not missing.any()


def review_tokens(reviews: pd.DataFrame) -> pd.DataFrame:
    """Average review completion tokens per essay for each reviewed source model: an essay's
    reviews are summed across its criteria, then averaged per dataset and across datasets with
    equal weight, as `metrics` averages grading tokens."""
    if not has_review_tokens(reviews):
        raise ValueError("Reviews lack completion_tokens; a Review run's reviews CSV records them")
    frame = reviews.assign(completion_tokens=reviews.completion_tokens.astype(int))
    per_essay = frame.groupby(["source_model", "dataset", "essay_id"]).completion_tokens.sum()
    by_dataset = per_essay.groupby(["source_model", "dataset"]).agg(["size", "mean"])
    rows = []
    for model, group in by_dataset.groupby(level="source_model", sort=True):
        for (_, label), values in group.iterrows():
            rows.append(
                {
                    "Source Model": model,
                    "Dataset": label,
                    "Essays": int(values["size"]),
                    "Avg Completion Tokens": values["mean"],
                }
            )
        rows.append(
            {
                "Source Model": model,
                "Dataset": "Average",
                "Essays": int(group["size"].sum()),
                "Avg Completion Tokens": group["mean"].mean(),
            }
        )
    return pd.DataFrame(rows)


def summarize(procedure: str, outcomes: Mapping[Item, ItemOutcome]) -> list[dict]:
    """Count each outcome per dataset and over all items, with its share of the items."""
    names = DA_OUTCOMES if procedure == "DA" else RA_OUTCOMES
    labels = sorted({label for label, _, _ in outcomes})
    rows = []
    for label in [*labels, ALL_ITEMS]:
        selected = [o.outcome for (d, _, _), o in outcomes.items() if label in (d, ALL_ITEMS)]
        for name in names:
            count = selected.count(name)
            rows.append(
                {
                    "Procedure": procedure,
                    "Dataset": label,
                    "Items": len(selected),
                    "Outcome": name,
                    "Count": count,
                    "Percent": 100 * count / len(selected),
                }
            )
    return rows


def save_procedure_outcomes(
    output_dir: Path,
    *,
    disagreement: Mapping[Item, ItemOutcome] | None = None,
    reviews: pd.DataFrame | None = None,
) -> list[Path]:
    """Write the outcome summary, the per-item outcomes, and the review cost when the reviews
    record it."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_procedure = {}
    if disagreement is not None:
        by_procedure["DA"] = disagreement
    if reviews is not None:
        by_procedure["RA"] = review_outcomes(reviews)
    if not by_procedure:
        raise ValueError("Supply pool sources, reviews, or both")
    summary_path = output_dir / "outcomes.csv"
    pd.DataFrame(
        [row for name, items in by_procedure.items() for row in summarize(name, items)]
    ).to_csv(summary_path, index=False)
    items_path = output_dir / "outcome_items.csv"
    with items_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Procedure",
                "Dataset",
                "Essay ID",
                "Criterion",
                "Outcome",
                "Disagreement",
                "Admissible",
                "Reviewed",
            ]
        )
        for name, items in by_procedure.items():
            for (label, essay_id, criterion), item in sorted(items.items()):
                writer.writerow(
                    [name, label, essay_id, criterion, item.outcome]
                    + ["" if v is None else v for v in item[1:]]
                )
    paths = [summary_path, items_path]
    if reviews is not None and has_review_tokens(reviews):
        tokens_path = output_dir / "review_tokens.csv"
        review_tokens(reviews).to_csv(tokens_path, index=False)
        paths.append(tokens_path)
    return paths
