"""Selects score-stratified few-shot reference examples for essay grading."""

import hashlib
import importlib
import math
import threading
from collections import defaultdict

import numpy as np

from abaes_eval_fwk.config.retrieval_config import ExamplesSelection, RetrievalConfig
from abaes_eval_fwk.core.models.assessment import HumanAssessment
from abaes_eval_fwk.core.models.experiment import CalibrationPool
from abaes_eval_fwk.core.models.rubric import Criterion, Rubric, Scale
from abaes_eval_fwk.core.retriever.retrieval_loader import (
    _bm25_scores,
    _get_model,
    _index_path_for_criterion,
    _load_and_embed,
    _pool_hash,
)


def _call_selector(
    selector: str,
    rubric: Rubric,
    calibration_pool: CalibrationPool,
    student_text: str,
    criterion: Criterion,
) -> list[HumanAssessment]:
    """Invoke a reader-supplied `module:function` with the built-in retriever's own arguments."""
    module_name, function_name = selector.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    return list(
        function(
            rubric=rubric,
            calibration_pool=calibration_pool,
            student_text=student_text,
            criterion=criterion,
        )
    )


def _hybrid_scores(
    dense: np.ndarray, bm25: np.ndarray, dense_weight: float, bm25_weight: float
) -> np.ndarray:
    """Combine normalized dense and BM25 scores with weights that sum to 1."""

    def _norm(v: np.ndarray) -> np.ndarray:
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi > lo else np.ones_like(v)

    return dense_weight * _norm(dense) + bm25_weight * _norm(bm25)


def _make_rng(rubric_id: str, student_text: str, seed: int = 42):
    """Create a deterministic random generator for an essay."""
    import random as _random

    seed = int(
        hashlib.sha256(f"{seed}|{rubric_id}|{student_text}".encode()).hexdigest()[:12],
        16,
    )
    return _random.Random(seed)  # nosec: deterministic seeding, not cryptographic


def _score_levels(score_points: list, score_min: float, score_max: float, k: int) -> list:
    """Group score points into at most `k` groups."""
    if not score_points or k <= 0:
        return []
    if len(score_points) <= k or score_max <= score_min:
        return [[s] for s in score_points]
    width = (score_max - score_min) / k
    levels: list = [[] for _ in range(k)]
    for s in score_points:
        idx = min(int((s - score_min) / width), k - 1)
        levels[idx].append(s)
    return [lvl for lvl in levels if lvl]


def _level_count(rubric: Rubric, max_examples: int, n_distinct: int, criterion_name: str) -> int:
    """Return the number of score-level groups to retrieve."""
    k = min(max_examples, n_distinct)
    if rubric.rubric_id == "set_7" and criterion_name == "holistic":
        k = math.ceil(k / 2)
    return max(k, 1)


def _resolve_pool_submissions(
    pool: list[HumanAssessment],
    criterion: Criterion,
) -> tuple[list[HumanAssessment], str]:
    """Return a criterion's pool entries, falling back to holistic entries."""
    matches = [s for s in pool if s.criterion.name == criterion.name]
    if matches or criterion.name == "holistic":
        return matches, criterion.name
    return [s for s in pool if s.criterion.name == "holistic"], "holistic"


def _scale_for(rubric: Rubric, criterion_name: str) -> Scale:
    return (
        rubric.total_score_range() if criterion_name == "holistic" else rubric.get_criterion_scale()
    )


# Caches the "holistic" query-side selection only — that's the one effective_criterion multiple
# distinct nominal criteria can share, so it's the only case two calls for the same essay would
# otherwise recompute an identical result (see select_reference_examples).
_selection_lock = threading.Lock()
_selection_cache: dict = {}


class Retriever:
    """Selects score-stratified few-shot reference examples for grading a submission."""

    @staticmethod
    def select_reference_examples(
        rubric: Rubric,
        calibration_pool: CalibrationPool | None,
        student_text: str,
        retrieval: RetrievalConfig,
        criterion: Criterion,
    ) -> list[HumanAssessment]:
        """Select score-stratified reference example essays for a student submission and criterion."""
        if retrieval.is_zero_shot:
            return []
        if not calibration_pool:
            raise ValueError("Few-shot retrieval requires a nonempty calibration pool")
        if retrieval.selector:
            return _call_selector(
                retrieval.selector, rubric, calibration_pool, student_text, criterion
            )

        human_assessments, effective_criterion = _resolve_pool_submissions(
            calibration_pool.human_assessments, criterion
        )
        if not human_assessments:
            raise ValueError(
                f"No calibration examples are available for criterion {criterion.name!r}"
            )
        human_assessments = sorted(human_assessments, key=lambda s: s.essay_id)
        pool_hash = _pool_hash(human_assessments)
        calibration_index_path = _index_path_for_criterion(
            calibration_pool.calibration_index_path, effective_criterion
        )

        # Only "holistic" can ever be requested by more than one distinct nominal criterion in the
        # same essay (see _resolve_pool_submissions), so it's the only case worth the cache/lock
        # overhead — a genuine per-criterion pool never collides with another call regardless.
        if effective_criterion != "holistic":
            return Retriever._compute_selection(
                rubric,
                human_assessments,
                effective_criterion,
                calibration_index_path,
                pool_hash,
                student_text,
                retrieval,
            )

        text_hash = hashlib.sha256(student_text.encode()).hexdigest()
        cache_key = (
            rubric.rubric_id,
            repr(rubric),
            effective_criterion,
            pool_hash,
            text_hash,
            retrieval,
        )
        if cache_key in _selection_cache:
            return _selection_cache[cache_key]
        with _selection_lock:
            if cache_key in _selection_cache:  # another thread computed it while we waited
                return _selection_cache[cache_key]
            picks = Retriever._compute_selection(
                rubric,
                human_assessments,
                effective_criterion,
                calibration_index_path,
                pool_hash,
                student_text,
                retrieval,
            )
            _selection_cache[cache_key] = picks
        return picks

    @staticmethod
    def _compute_selection(
        rubric: Rubric,
        human_assessments: list[HumanAssessment],
        effective_criterion: str,
        calibration_index_path: str,
        pool_hash: str,
        student_text: str,
        retrieval: RetrievalConfig,
    ) -> list[HumanAssessment]:
        """Embed/score the pool against one query and bucket-select from it — the expensive part
        `select_reference_examples` caches by (effective_criterion, pool_hash, student_text, ...).
        """
        rubric_id = rubric.rubric_id
        scale = _scale_for(rubric, effective_criterion)
        total_min, total_max = scale.min, scale.max

        selection = retrieval.examples_selection
        use_dense = selection in (ExamplesSelection.SEMANTIC, ExamplesSelection.HYBRID)

        if use_dense:
            pool, embeddings = _load_and_embed(
                calibration_index_path, pool_hash, human_assessments, retrieval
            )
        else:
            pool, embeddings = human_assessments, None

        by_score: dict = defaultdict(list)
        if use_dense:
            model = _get_model(retrieval)
            query_vec = model.encode(
                [student_text],
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=retrieval.embedding_batch_size,
                show_progress_bar=False,
            )[0]
            dense = (embeddings @ query_vec).astype(np.float32)
            if selection == ExamplesSelection.HYBRID:
                bm25 = _bm25_scores(calibration_index_path, pool_hash, pool, student_text)
                scores = _hybrid_scores(
                    dense, bm25, retrieval.hybrid_dense_weight, retrieval.hybrid_bm25_weight
                )
            else:
                scores = dense
            for i, s in enumerate(pool):
                by_score[s.human_score].append((float(scores[i]), s))
        elif selection == ExamplesSelection.LEXICAL:
            bm25 = _bm25_scores(calibration_index_path, pool_hash, pool, student_text)
            for i, s in enumerate(pool):
                by_score[s.human_score].append((float(bm25[i]), s))
        else:  # ExamplesSelection.RANDOM
            for s in pool:
                by_score[s.human_score].append((0.0, s))

        score_points = sorted(by_score.keys())
        k = _level_count(rubric, retrieval.max_examples, len(score_points), effective_criterion)
        levels = _score_levels(score_points, total_min, total_max, k)

        ranked_levels: list[list[HumanAssessment]] = []
        rng = _make_rng(rubric_id, student_text, retrieval.seed)
        for level in levels:
            candidates = [candidate for score in level for candidate in by_score[score]]
            if selection == ExamplesSelection.RANDOM:
                candidates = rng.sample(candidates, min(retrieval.n_ex_per_score, len(candidates)))
            else:
                candidates = sorted(
                    candidates, key=lambda candidate: (-candidate[0], candidate[1].essay_id)
                )[: retrieval.n_ex_per_score]
            ranked_levels.append([submission for _, submission in candidates])
        picks: list[HumanAssessment] = []
        for offset in range(retrieval.n_ex_per_score):
            for ranked in ranked_levels:
                if offset < len(ranked):
                    picks.append(ranked[offset])
                    if len(picks) == retrieval.max_examples:
                        return picks
        return picks
