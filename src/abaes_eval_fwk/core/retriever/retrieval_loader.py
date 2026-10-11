"""Lazy optional retrieval dependencies and safe content-addressed embedding caches."""

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

from abaes_eval_fwk.config.retrieval_config import ExamplesSelection, RetrievalConfig
from abaes_eval_fwk.core.models.assessment import HumanAssessment
from abaes_eval_fwk.core.models.experiment import ExperimentalSetup

_model_lock = threading.Lock()
_embed_lock = threading.Lock()
_bm25_lock = threading.Lock()
_embed_models: dict[tuple, Any] = {}  # settings tuple -> SentenceTransformer
_embed_cache: dict[tuple[str, ...], Any] = {}  # content hash -> np.ndarray
_bm25_cache: dict[tuple[str, ...], Any] = {}  # content hash -> BM25Okapi


def _default_retrieval_config() -> RetrievalConfig:
    """A RetrievalConfig for callers that never built one, built from the same YAML the
    full pipeline merges in — not a second, hardcoded copy of those values."""
    from abaes_eval_fwk.utils.resources import load_defaults

    defaults = load_defaults()["retrieval"]
    return RetrievalConfig(
        n_ex_per_score=defaults["n_ex_per_score"], max_examples=defaults["max_examples"]
    )


def _embedding_settings(retrieval: RetrievalConfig) -> tuple:
    return (
        retrieval.embedding_model,
        retrieval.embedding_device,
        retrieval.embedding_batch_size,
        retrieval.embedding_threads,
    )


def _get_model(retrieval: RetrievalConfig | None = None):
    """Load the selected embedding model only when dense retrieval is requested."""
    retrieval = retrieval or _default_retrieval_config()
    key = _embedding_settings(retrieval)
    with _model_lock:
        if key not in _embed_models:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Dense retrieval requires the optional dense dependencies; install 'abaes-eval-fwk[dense]'"
                ) from exc
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            torch.set_num_threads(retrieval.embedding_threads)
            _embed_models[key] = SentenceTransformer(
                retrieval.embedding_model, device=retrieval.embedding_device
            )
        return _embed_models[key]


def _pool_hash(pool: list[HumanAssessment]) -> str:
    """Hash ordered IDs, scores, criterion names, and essay text to protect cache alignment."""
    payload = [(s.essay_id, s.human_score, s.criterion.name, s.submission.text) for s in pool]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def _load_and_embed(
    calibration_index_path: str,
    pool_hash: str,
    pool: list[HumanAssessment],
    retrieval: RetrievalConfig | None = None,
):
    """Load or encode normalized vectors using an atomic NumPy cache keyed by model settings and contents."""
    retrieval = retrieval or _default_retrieval_config()
    signature = hashlib.sha256(
        json.dumps(["embedding-v1", pool_hash, _embedding_settings(retrieval)]).encode()
    ).hexdigest()
    key = (calibration_index_path, signature)
    if not pool:
        return pool, np.empty((0, 0), dtype=np.float32)
    with _embed_lock:
        if key in _embed_cache:
            return pool, _embed_cache[key]
        if not calibration_index_path:
            raise ValueError("Dense retrieval requires a calibration cache path")
        base = Path(calibration_index_path)
        destination = base.with_name(f"{base.stem}.{signature}.npy")
        if destination.is_file():
            embeddings = np.load(destination, allow_pickle=False)
            if (
                embeddings.ndim != 2
                or len(embeddings) != len(pool)
                or not np.isfinite(embeddings).all()
            ):
                raise ValueError(f"Invalid embedding cache: {destination}; remove it to rebuild")
        else:
            model = _get_model(retrieval)
            embeddings = np.asarray(
                model.encode(
                    [s.submission.text for s in pool],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=retrieval.embedding_batch_size,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
            if (
                embeddings.ndim != 2
                or len(embeddings) != len(pool)
                or not np.isfinite(embeddings).all()
            ):
                raise ValueError("Embedding model returned an invalid vector matrix")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
                np.save(handle, embeddings, allow_pickle=False)
            temporary.replace(destination)
        _embed_cache[key] = embeddings
        return pool, embeddings


def _load_bm25(calibration_index_path: str, pool_hash: str, pool: list[HumanAssessment]):
    """Build an in-memory BM25 index, raising a clear error when its optional dependency is absent."""
    if not pool:
        return None
    key = ("bm25-v1", calibration_index_path, pool_hash)
    with _bm25_lock:
        if key not in _bm25_cache:
            try:
                from rank_bm25 import BM25Okapi
            except ImportError as exc:
                raise ImportError(
                    "Lexical and hybrid retrieval require rank-bm25; install it with pip install rank-bm25"
                ) from exc
            _bm25_cache[key] = BM25Okapi([s.submission.text.lower().split() for s in pool])
        return _bm25_cache[key]


def _bm25_scores(
    calibration_index_path: str,
    pool_hash: str,
    pool: list[HumanAssessment],
    query: str,
) -> np.ndarray:
    """Return BM25 similarity normalized by the largest positive score."""
    if not pool:
        return np.empty(0, dtype=np.float32)
    bm25 = _load_bm25(calibration_index_path, pool_hash, pool)
    scores = np.asarray(bm25.get_scores(query.lower().split()), dtype=np.float32)
    maximum = scores.max()
    return scores / maximum if maximum > 0 else scores


def _index_path_for_criterion(calibration_index_path: str, criterion_name: str) -> str:
    """Use a safe criterion digest in retrieval cache filenames."""
    if not calibration_index_path:
        return ""
    base = Path(calibration_index_path)
    criterion_key = hashlib.sha256(criterion_name.encode()).hexdigest()[:16]
    return str(base.with_name(f"{base.stem}.{criterion_key}.faiss"))


class RetrievalLoader:
    """Warm optional dense and lexical indexes before concurrent grading."""

    @staticmethod
    def preload_indexes(setups: list[ExperimentalSetup], retrieval: RetrievalConfig) -> None:
        """Prepare the score pools required by the requested retrieval strategy."""
        if retrieval.is_zero_shot:
            return
        use_dense = retrieval.examples_selection in (
            ExamplesSelection.SEMANTIC,
            ExamplesSelection.HYBRID,
        )
        use_bm25 = retrieval.examples_selection in (
            ExamplesSelection.LEXICAL,
            ExamplesSelection.HYBRID,
        )
        for setup in setups:
            calibration_pool = setup.calibration_pool
            if not calibration_pool:
                raise ValueError(
                    "Few-shot retrieval requires a nonempty calibration pool; increase dataset size or select zero_shot"
                )
            if not (use_dense or use_bm25):
                continue
            for name in sorted({s.criterion.name for s in calibration_pool.human_assessments}):
                pool = sorted(
                    (s for s in calibration_pool.human_assessments if s.criterion.name == name),
                    key=lambda s: s.essay_id,
                )
                pool_hash = _pool_hash(pool)
                index_path = _index_path_for_criterion(
                    calibration_pool.calibration_index_path, name
                )
                if use_dense:
                    _load_and_embed(index_path, pool_hash, pool, retrieval)
                if use_bm25:
                    _load_bm25(index_path, pool_hash, pool)
