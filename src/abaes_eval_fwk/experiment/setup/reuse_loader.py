"""Parses prior CSV outputs back into GradingRecord objects for FinalAgent adjudication."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from abaes_eval_fwk.config.reuse_config import ReuseConfig
from abaes_eval_fwk.core.models.assessment import (
    GradingRecord,
    HumanAssessment,
    LLMAssessment,
    PredictedScore,
    ResultsField,
    ReviewField,
    Statistics,
    StudentSubmission,
    TokenUsage,
    parse_csv_flag,
)
from abaes_eval_fwk.core.models.experiment import (
    Experiment,
    ExperimentalSetup,
    FinalAgentProcedureName,
)
from abaes_eval_fwk.core.models.rubric import Rubric
from abaes_eval_fwk.experiment.result.results_saver import ResultsWriter

logger = logging.getLogger(__name__)


def read_rows(path: str, rubric_id: str) -> dict[str, list[dict]]:
    """Group one rubric's prior CSV rows by essay ID."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Reuse CSV not found: '{path}'")

    by_essay: dict[str, list[dict]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get(ResultsField.DATASET) != rubric_id:
                continue
            essay_id = row.get(ResultsField.ESSAY_ID, "")
            if essay_id:
                by_essay[essay_id].append(row)
    return by_essay


def _first_row(path: str) -> dict:
    """The first data row of a prior results CSV, for the run-wide values every row repeats."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Reuse CSV not found: '{path}'")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Reuse CSV is empty: {path}")
    for column in (ResultsField.MODEL, ResultsField.EXPERIMENT_ID):
        values = {row.get(column, "") for row in rows}
        if len(values) != 1 or not next(iter(values)):
            raise ValueError(f"Reuse CSV must identify exactly one {column.value}: {path}")
    return rows[0]


def source_model(path: str) -> str:
    """The model that produced a prior results CSV. Final agents label their sources by this rather than by --reuse-paths position, so a review's cache keys follow the model."""
    return _first_row(path).get(ResultsField.MODEL, "")


def source_experiment(path: str) -> str:
    """The experiment_id that produced a prior results CSV, naming the variant a review judged."""
    return _first_row(path).get(ResultsField.EXPERIMENT_ID, "")


def source_is_criterion(path: str, rubric: Rubric) -> bool:
    """True when a prior results CSV holds per-criterion rows rather than one holistic row per essay. A non-grading Review run needs this before its sample is built, so it fans out one job per source row."""
    holistic = rubric.holistic_criterion().name
    return any(
        rows[0][ResultsField.CRITERION] != holistic
        for rows in read_rows(path, rubric.rubric_id).values()
    )


def _grading_record_from_row(row: dict, rubric: Rubric) -> GradingRecord:
    """Convert one saved CSV row dictionary into a GradingRecord domain instance."""
    confidence = row.get(ResultsField.CONFIDENCE) or ""
    name = row.get(ResultsField.CRITERION, "holistic")
    criterion = rubric.holistic_criterion() if name == "holistic" else rubric.get_criterion(name)
    if criterion is None:
        raise ValueError(f"Unknown reused criterion: {name}")
    score = float(row[ResultsField.PRED_SCORE_RAW])
    scale = rubric.total_score_range() if name == "holistic" else rubric.get_criterion_scale()
    if not math.isfinite(score) or not scale.min <= score <= scale.max:
        raise ValueError(f"Reused score outside the rubric scale: {score}")
    submission = HumanAssessment(
        submission=StudentSubmission(
            essay_id=row.get(ResultsField.ESSAY_ID, ""), text=row.get(ResultsField.ESSAY_TEXT, "")
        ),
        human_score=int(row[ResultsField.HUMAN_SCORE]),
        criterion=criterion,
    )
    llm_assessment = LLMAssessment(
        score=PredictedScore(raw_score=float(row[ResultsField.PRED_SCORE_RAW])),
        justification=row.get(ResultsField.JUSTIFICATION, ""),
        confidence=float(confidence) if confidence else None,
        reasoning=row.get(ResultsField.REASONING) or None,
        statistics=Statistics(
            token_usage=TokenUsage(
                prompt_tokens=int(row.get(ResultsField.PROMPT_TOKENS) or 0),
                completion_tokens=int(row.get(ResultsField.COMPLETION_TOKENS) or 0),
            )
        ),
    )
    return GradingRecord(
        human_assessment=submission,
        llm_assessment=llm_assessment,
        examples_text=row.get(ResultsField.EXAMPLES) or "",
    )


def load_reused_results(
    path: str,
    rubric: Rubric,
    essay_ids: Iterable[str] | None = None,
) -> tuple[bool, dict[str, list[GradingRecord]]]:
    """Load prior results from a CSV file for a set of essays, as Assessments. Holistic CSVs yield one GradingRecord per essay; criterion-specialized CSVs yield every specialist's GradingRecord per essay, since the FinalAgent decides per-criterion outcomes itself rather than being handed a pre-combined score."""
    by_essay = read_rows(path, rubric.rubric_id)
    if essay_ids is not None:
        wanted = {str(e) for e in essay_ids}
        by_essay = {eid: rows for eid, rows in by_essay.items() if eid in wanted}
    if not by_essay:
        raise ValueError(f"'{path}' shares no essays with rubric '{rubric.rubric_id}'.")

    holistic_name = rubric.holistic_criterion().name
    is_criterion = any(
        rows[0][ResultsField.CRITERION] != holistic_name for rows in by_essay.values()
    )

    results = {
        eid: [_grading_record_from_row(row, rubric) for row in rows]
        for eid, rows in by_essay.items()
    }
    if not is_criterion and any(len(records) != 1 for records in results.values()):
        raise ValueError("Holistic reuse requires exactly one result per essay")
    if is_criterion:
        criterion_order = {c.name: i for i, c in enumerate(rubric.criteria)}
        for eid, grading_records in results.items():
            names = [a.human_assessment.criterion.name for a in grading_records]
            if sorted(names) != sorted(criterion_order):
                raise ValueError(
                    f"'{path}' essay '{eid}': criteria {names} don't match rubric "
                    f"'{rubric.rubric_id}' criteria {list(criterion_order)}"
                )
            grading_records.sort(key=lambda a: criterion_order[a.human_assessment.criterion.name])

    logger.info(
        "Reuse from '%s': %d essays (%s), rubric='%s'",
        Path(path).name,
        len(results),
        "criterion-specialized, per-criterion" if is_criterion else "holistic",
        rubric.rubric_id,
    )
    return is_criterion, results


def load_final_agent_reuse(
    paths: list[str],
    rubric: Rubric,
    essay_ids: Iterable[str] | None = None,
) -> tuple[bool, dict[str, list[list[GradingRecord]]]]:
    """Load and align prior results from multiple prior run CSV files for FinalAgent adjudication."""
    if not paths:
        raise ValueError("load_final_agent_reuse: at least one path is required.")

    loaded = [load_reused_results(p, rubric, essay_ids) for p in paths]
    shapes = {is_criterion for is_criterion, _ in loaded}
    if len(shapes) > 1:
        raise ValueError(
            "Reuse paths disagree on shape (holistic vs criterion-specialized): "
            f"{list(zip(paths, (s for s, _ in loaded)))}"
        )
    is_criterion = shapes.pop()

    expected_ids = set(loaded[0][1])
    if any(set(results) != expected_ids for _, results in loaded[1:]):
        raise ValueError("Candidate result files must contain the same requested essay IDs")
    common_ids = expected_ids
    if not common_ids:
        raise ValueError(f"No common essay_ids across reuse paths: {paths}")

    aligned = {eid: [results[eid] for _, results in loaded] for eid in common_ids}
    logger.info(
        "Final-agent reuse ready: %d common essays across %d path(s) (%s)",
        len(aligned),
        len(paths),
        "criterion-specialized" if is_criterion else "holistic",
    )
    return is_criterion, aligned


@dataclass(frozen=True)
class SetupReuse:
    """One setup's prior results, aligned per essay, with the shape they were loaded in. `independent_by_essay` is the final agent's own prior run, empty for a procedure that never reassesses."""

    setup: ExperimentalSetup
    is_criterion: bool
    by_essay: dict[str, list[list[GradingRecord]]]
    independent_by_essay: dict[str, list[GradingRecord]] = field(default_factory=dict)

    def results_for(self, essay_id: str) -> list[list[GradingRecord]]:
        """Return this essay's prior results, or raise — a missing essay means misaligned reuse."""
        results = self.by_essay.get(essay_id)
        if results is None:
            raise ValueError(
                f"No reused results for essay '{essay_id}' (rubric '{self.setup.rubric.rubric_id}')"
            )
        return results

    def independent_for(self, essay_id: str) -> list[GradingRecord] | None:
        """Return this essay's independent assessment, or raise once one was loaded but this essay is missing from it — a branch that reassesses has no other source of a score."""
        if not self.independent_by_essay:
            return None
        records = self.independent_by_essay.get(essay_id)
        if records is None:
            raise ValueError(
                f"No independent assessment for essay '{essay_id}' "
                f"(rubric '{self.setup.rubric.rubric_id}')"
            )
        return records


class ReuseLookup:
    """Prior results for every setup in one run, looked up per (setup, essay). Empty for experiments that grade from scratch, which then answer "no reuse" without needing any --reuse-paths."""

    def __init__(self, loaded: list[SetupReuse] | None = None) -> None:
        self._by_setup = {id(r.setup): r for r in loaded or []}

    def __call__(
        self, setup: ExperimentalSetup, essay_id: str
    ) -> tuple[bool, list | None, list | None]:
        reuse = self._by_setup.get(id(setup))
        if reuse is None:
            return False, None, None
        return reuse.is_criterion, reuse.results_for(essay_id), reuse.independent_for(essay_id)


def build_reuse_by_setup(experiment: Experiment, reuse: ReuseConfig) -> ReuseLookup:
    """Load each setup's final-agent reuse data once, and return a lookup over it."""
    if not experiment.experiment_type.reuses_results:
        return ReuseLookup()

    procedure = experiment.final_agent_procedure.name
    reassesses = procedure != FinalAgentProcedureName.REVIEW
    reuse.require(experiment.name, procedure.value, needs_independent=reassesses)

    loaded = []
    for setup in experiment.setups:
        essay_ids = [s.essay_id for s in setup.sample]
        is_criterion, by_essay = load_final_agent_reuse(
            list(reuse.results_paths), setup.rubric, essay_ids=essay_ids
        )
        independent_by_essay: dict[str, list[GradingRecord]] = {}
        if reassesses:
            independent_is_criterion, independent_by_essay = load_reused_results(
                reuse.independent_path, setup.rubric, essay_ids=essay_ids
            )
            if independent_is_criterion != is_criterion:
                raise ValueError(
                    f"--reuse-independent-path '{reuse.independent_path}' is "
                    f"{'criterion-specialized' if independent_is_criterion else 'holistic'}, but "
                    "the candidate pool is not — both must share the adjudicated shape."
                )
        if set(by_essay) != set(essay_ids):
            raise ValueError("Candidate results are missing requested essay IDs")
        if reassesses and set(independent_by_essay) != set(essay_ids):
            raise ValueError("Independent results are missing requested essay IDs")
        current_text = {item.essay_id: item.submission.text for item in setup.sample}
        for groups in by_essay.values():
            for group in groups:
                for record in group:
                    if (
                        record.human_assessment.submission.text
                        != current_text[record.human_assessment.essay_id]
                    ):
                        raise ValueError("Candidate essay text differs from the configured dataset")
        for records in independent_by_essay.values():
            for record in records:
                if (
                    record.human_assessment.submission.text
                    != current_text[record.human_assessment.essay_id]
                ):
                    raise ValueError("Independent essay text differs from the configured dataset")
        loaded.append(
            SetupReuse(
                setup=setup,
                is_criterion=is_criterion,
                by_essay=by_essay,
                independent_by_essay=independent_by_essay,
            )
        )
    return ReuseLookup(loaded)


def load_reused_reviews(
    paths: list[str], procedure: FinalAgentProcedureName
) -> dict[tuple[str, str, str, str, str, str, str], dict]:
    """Load prior FinalAgent review-turn judgments from one or more _reviews.csv files, keyed by
    (dataset, essay_id, criterion, source_model, source_experiment_id, reviewer_model, source_assessment_digest).

    Only rows from this procedure or from a Review run replay. The expanded key prevents reviews from one
    variant/reviewer/assessment from being wrongly used for another.
    """
    merged: dict[tuple[str, str, str, str, str, str, str], dict] = {}
    for path in paths:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Reuse reviews CSV not found: '{path}'")
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                # A Review run's turn is this procedure's review turn verbatim, so its rows replay.
                if row.get(ReviewField.FINAL_AGENT_PROCEDURE) not in (
                    procedure.value,
                    FinalAgentProcedureName.REVIEW.value,
                ):
                    continue
                source_experiment_id = row.get("source_experiment_id", "")
                reviewer_model = row.get("reviewer_model", "")
                # For backwards compatibility with reviews CSVs that lack these columns,
                # use empty string as default. This means old CSVs can be loaded but won't
                # be perfectly keyed (both old and new runs might collide on incomplete keys).
                source_score = (
                    int(row.get("source_score", 0)) if row.get("source_score", "").isdigit() else 0
                )
                source_justification = row.get("source_justification", "")
                source_assessment_digest = _source_assessment_digest(
                    source_score, source_justification
                )

                key = (
                    row[ReviewField.DATASET],
                    row[ReviewField.ESSAY_ID],
                    row[ReviewField.CRITERION],
                    row[ReviewField.SOURCE_MODEL],
                    source_experiment_id,
                    reviewer_model,
                    source_assessment_digest,
                )
                if key in merged:
                    # Conflicting duplicates: same exact key from different rows or paths
                    raise ValueError(
                        f"Conflicting duplicate review for {key}: "
                        f"Reviews from the same source experiment, reviewer, and assessment "
                        f"should not appear in multiple reuse paths or as duplicates in one path."
                    )
                rubric_consistency = parse_csv_flag(
                    row[ReviewField.RUBRIC_CONSISTENCY], ReviewField.RUBRIC_CONSISTENCY.value
                )
                evidence_grounding = parse_csv_flag(
                    row[ReviewField.EVIDENCE_GROUNDING], ReviewField.EVIDENCE_GROUNDING.value
                )
                merged[key] = {
                    "rubric_consistency": rubric_consistency,
                    "rubric_consistency_justification": row.get(
                        ReviewField.RUBRIC_CONSISTENCY_JUSTIFICATION
                    )
                    or "",
                    "evidence_grounding": evidence_grounding,
                    "evidence_grounding_justification": row.get(
                        ReviewField.EVIDENCE_GROUNDING_JUSTIFICATION
                    )
                    or "",
                    "reasoning": row.get(ReviewField.REASONING) or "",
                    # Absent from reviews CSVs written before this column existed; Review
                    # re-derives it from the two flags either way.
                    "is_admissible": (
                        parse_csv_flag(
                            row[ReviewField.IS_ADMISSIBLE], ReviewField.IS_ADMISSIBLE.value
                        )
                        if ReviewField.IS_ADMISSIBLE in row
                        else rubric_consistency and evidence_grounding
                    ),
                }

    if not merged:
        raise ValueError(
            f"No reviews for procedure '{procedure.value}' found across: {paths}. "
            "A reviews CSV from a different --deliberation-mode never satisfies this lookup."
        )
    logger.info(
        "Review reuse ready: %d cached reviews for procedure '%s' across %d path(s)",
        len(merged),
        procedure.value,
        len(paths),
    )
    return merged


def _source_assessment_digest(score: int, justification: str) -> str:
    """Compute a short digest of a source assessment to detect changes in the reviewed content."""
    content = json.dumps({"score": score, "justification": justification}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ReviewCache:
    """Per-procedure cache for FinalAgent review-turn judgments: looks up prior results loaded from --reuse-reviews-paths, and persists every review — reused or freshly computed — to this run's own reviews CSV via `writer`. This way a same-run retry also hits the cache, and future runs can reuse this run's reviews in turn.

    Cache key now includes source_experiment_id, reviewer_model, and source_assessment_digest to bind reviews to their actual inputs. Ensures a review from one experiment/reviewer/assessment cannot be wrongly reused for another."""

    procedure: FinalAgentProcedureName
    writer: ResultsWriter
    _loaded: dict[tuple[str, str, str, str, str, str, str], dict] = field(default_factory=dict)
    _written: set[tuple[str, str, str, str, str, str, str]] = field(default_factory=set)

    def get(
        self,
        dataset: str,
        essay_id: str,
        criterion: str,
        source_model: str,
        source_experiment_id: str,
        reviewer_model: str,
        source_assessment_digest: str,
    ) -> dict | None:
        """Return the cached review for one assessment, or None on a cache miss.

        The cache key includes source_experiment_id, reviewer_model, and source_assessment_digest
        to ensure the review was created for exactly this variant, reviewer, and source assessment.
        """
        key = (
            dataset,
            essay_id,
            criterion,
            source_model,
            source_experiment_id,
            reviewer_model,
            source_assessment_digest,
        )
        return self._loaded.get(key)

    def store(
        self,
        dataset: str,
        essay_id: str,
        criterion: str,
        source_model: str,
        source_experiment_id: str,
        reviewer_model: str,
        source_assessment_digest: str,
        review: dict,
    ) -> None:
        """Persist a review turn's output and make it available to any later lookup in this run. Each key reaches the CSV once, so a same-run retry does not duplicate its row."""
        key = (
            dataset,
            essay_id,
            criterion,
            source_model,
            source_experiment_id,
            reviewer_model,
            source_assessment_digest,
        )
        self._loaded[key] = review
        if self.procedure == FinalAgentProcedureName.REVIEW or key in self._written:
            return
        self._written.add(key)
        self.writer.append_review(
            self.procedure, dataset, essay_id, criterion, source_model, review
        )


def build_review_cache(
    experiment: Experiment, reuse: ReuseConfig, writer: ResultsWriter
) -> ReviewCache | None:
    """Build the review cache for one grading run, or None for experiments that never perform a
    review turn at all. A Review run writes each review straight to its own reviews CSV rather
    than through this cache's `store` — but `--reuse-reviews-paths` still lets it look one up,
    so a restarted Review run can resume from a prior (partial) reviews CSV instead of the
    checkpoint DB."""
    procedure = experiment.final_agent_procedure.name
    if not experiment.experiment_type.reuses_results:
        return None
    paths = list(reuse.reviews_paths)
    loaded = load_reused_reviews(paths, procedure) if paths else {}
    written: set[tuple[str, str, str, str, str, str, str]] = set()
    if procedure != FinalAgentProcedureName.REVIEW and Path(writer.reviews_csv_path).exists():
        resumed = load_reused_reviews([writer.reviews_csv_path], procedure)
        loaded.update(resumed)
        written.update(resumed)
    return ReviewCache(procedure=procedure, writer=writer, _loaded=loaded, _written=written)
