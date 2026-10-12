"""Build the RQ4 baseline: each source's overall score, averaged across sources and rounded."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path

from abaes_eval_fwk.core.models.assessment import Dataset, GradingRecord, results_fieldnames
from abaes_eval_fwk.core.models.rubric import Rubric, combine_criterion_scores
from abaes_eval_fwk.experiment.eval.aggregate_criteria import official_human_score
from abaes_eval_fwk.experiment.eval.metrics.compute_metrics import grading_records
from abaes_eval_fwk.utils.rounding import round_half_up


def matched_source_records(
    source_csvs: Sequence[Path],
) -> list[dict[tuple[str, str, str], GradingRecord]]:
    """Read each pool source keyed by (dataset, essay, criterion), requiring identical keys, so
    every pooled item has exactly one assessment from every source."""
    if not source_csvs:
        raise ValueError("A pool requires at least one source results CSV")
    sources = []
    for path in source_csvs:
        records = grading_records(Path(path))
        sources.append(
            {
                (label, r.human_assessment.essay_id, r.human_assessment.criterion.name): r
                for label, group in records.items()
                for r in group
            }
        )
    keys = sources[0].keys()
    if any(source.keys() != keys for source in sources[1:]):
        raise ValueError("Pool sources must contain identical dataset, essay and criterion keys")
    return sources


def build_pool_average_baseline(
    source_csvs: Sequence[Path],
    rubrics: Sequence[Rubric],
    datasets: list[Dataset],
    output_path: Path,
    *,
    experiment_id: str = "pool_average",
) -> Path:
    """Require matched source roles, aggregate each source's criteria, then pool the totals."""
    sources = matched_source_records(source_csvs)
    keys = sources[0].keys()
    rubrics_by_label = {str(r.rubric_id): r for r in rubrics}
    # (dataset, essay) -> one {criterion: rounded score} mapping per source, in source order.
    per_source: dict[tuple[str, str], list[dict[str, int]]] = {}
    for label, essay_id, criterion in keys:
        raw_scores = [
            s[(label, essay_id, criterion)].llm_assessment.score.raw_score for s in sources
        ]
        if (
            len(
                {
                    source[(label, essay_id, criterion)].human_assessment.human_score
                    for source in sources
                }
            )
            != 1
        ):
            raise ValueError(f"Source human scores differ for {label}/{essay_id}/{criterion}")
        rubric = rubrics_by_label[label]
        scale = (
            rubric.total_score_range() if criterion == "holistic" else rubric.get_criterion_scale()
        )
        if any(
            not math.isfinite(score) or score < scale.min or score > scale.max
            for score in raw_scores
        ):
            raise ValueError(f"Out-of-range source score for {label}/{essay_id}/{criterion}")
        pools = per_source.setdefault((label, essay_id), [{} for _ in sources])
        for pool, score in zip(pools, raw_scores, strict=True):
            pool[criterion] = round_half_up(score)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results_fieldnames())
        writer.writeheader()
        for (label, essay_id), pools in sorted(per_source.items()):
            rubric = rubrics_by_label[label]
            human = official_human_score(datasets, label, essay_id)
            if human is None:
                raise ValueError(f"No official overall human score for {label}/{essay_id}")
            # Each source's own overall score first: a holistic source has one, and a criterion
            # source combines its criteria with the rubric's formula. The baseline is the
            # half-up rounded mean of those overall scores, so pooling never mixes criteria
            # across sources before aggregation.
            if set(pools[0]) == {"holistic"}:
                overall = [pool["holistic"] for pool in pools]
            else:
                if set(pools[0]) != {c.name for c in rubric.criteria}:
                    raise ValueError(f"Incomplete criterion pool for {label}/{essay_id}")
                overall = [combine_criterion_scores(rubric, pool) for pool in pools]
            pred = round_half_up(sum(overall) / len(overall))
            writer.writerow(
                {
                    "experiment_id": experiment_id,
                    "dataset": label,
                    "essay_id": essay_id,
                    "criterion": "holistic",
                    "human_score": human,
                    "pred_score_raw": pred,
                    "pred_score_rounded": pred,
                    "model": "pool_average",
                }
            )
    return output_path
