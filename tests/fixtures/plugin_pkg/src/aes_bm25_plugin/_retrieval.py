"""Few-shot example selection by Okapi BM25, computed offline and with no third-party dependency."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from math import log
from typing import Final

from abaes_eval_fwk.api import (
    Capability,
    ComponentCategory,
    ComponentConfig,
    ComponentContext,
    ComponentDescriptor,
    ComponentError,
    ErrorCode,
    ReferenceBatch,
    RetrievalRequest,
)
from pydantic import Field

__all__ = [
    "BM25_RETRIEVER_ID",
    "Bm25Retriever",
    "Bm25RetrieverConfig",
    "bm25_descriptor",
    "bm25_scores",
    "tokenize",
]

BM25_RETRIEVER_ID: Final[str] = "aes_bm25.retrieval.bm25"

# Word characters plus an intra-word apostrophe, so "don't" stays one term and punctuation does not.
_WORD: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize(text: str, char_ngram: int = 0) -> list[str]:
    """Lower-cased word terms, plus each word's character n-grams when `char_ngram` is 2 or more."""
    words = _WORD.findall(text.lower())
    if char_ngram < 2:
        return words
    # A '#' prefix keeps a character n-gram from ever colliding with a whole word as a term.
    grams = [
        f"#{word}:{word[offset : offset + char_ngram]}"
        for word in words
        for offset in range(len(word) - char_ngram + 1)
    ]
    return words + grams


def _idf(document_frequency: int, total: int) -> float:
    """The BM25 inverse document frequency, shifted so a term in every document still counts."""
    return log(1 + (total - document_frequency + 0.5) / (document_frequency + 0.5))


def bm25_scores(
    query: Sequence[str], documents: Sequence[Sequence[str]], k1: float, b: float
) -> list[float]:
    """Score each tokenized document against the tokenized query, in the order given."""
    if not documents:
        return []
    counts = [Counter(document) for document in documents]
    lengths = [len(document) for document in documents]
    average_length = sum(lengths) / len(lengths)
    frequency = {term: sum(1 for entry in counts if term in entry) for term in set(query)}
    scores: list[float] = []
    for entry, length in zip(counts, lengths, strict=True):
        # An all-empty pool has no average length, so the length normalisation falls back to k1.
        normalise = k1 * (1 - b + b * length / average_length) if average_length else k1
        scores.append(
            sum(
                _idf(frequency[term], len(counts))
                * entry[term]
                * (k1 + 1)
                / (entry[term] + normalise)
                for term in frequency
                if entry[term]
            )
        )
    return scores


class Bm25RetrieverConfig(ComponentConfig):
    """Term weighting, the query's character n-grams, and the cutoff and default depth."""

    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)
    char_ngram: int = Field(default=0, ge=0, le=4)
    min_score: float = Field(default=0.0, ge=0)
    default_examples: int = Field(default=3, ge=1)


class Bm25Retriever:
    """Selects the pool essays whose wording is closest to the submission, by BM25 rank."""

    def __init__(self, config: Bm25RetrieverConfig) -> None:
        self._config = config

    async def retrieve(
        self, request: RetrievalRequest, context: ComponentContext
    ) -> ReferenceBatch:
        """Return the best-scoring pool examples, never including the submission's own sample."""
        limit = request.max_examples or self._config.default_examples
        candidates = [example for example in request.pool if example.key != request.submission.key]
        scores = bm25_scores(
            tokenize(request.submission.text, self._config.char_ngram),
            [tokenize(example.text, self._config.char_ngram) for example in candidates],
            self._config.k1,
            self._config.b,
        )
        # The sample key breaks ties, so two runs over the same pool rank identically.
        ranked = sorted(
            zip(scores, candidates, strict=True),
            key=lambda pair: (-pair[0], pair[1].key.as_tuple()),
        )
        return ReferenceBatch(
            key=request.submission.key,
            strategy_id=BM25_RETRIEVER_ID,
            examples=[example for score, example in ranked if score > self._config.min_score][
                :limit
            ],
        )


def _build(
    config: ComponentConfig, bindings: Mapping[str, str], context: ComponentContext
) -> Bm25Retriever:
    """The `ComponentFactory` for this retriever, which declares no dependency bindings."""
    if not isinstance(config, Bm25RetrieverConfig):
        raise ComponentError(
            f"retriever {BM25_RETRIEVER_ID} requires a Bm25RetrieverConfig, not "
            f"{type(config).__name__}",
            code=ErrorCode.COMPONENT_INVALID,
            component_id=BM25_RETRIEVER_ID,
        )
    return Bm25Retriever(config)


def bm25_descriptor() -> ComponentDescriptor:
    """The retriever as data: its identity, contracts, config model, factory, and capabilities."""
    return ComponentDescriptor(
        component_id=BM25_RETRIEVER_ID,
        category=ComponentCategory.RETRIEVER,
        description="Okapi BM25 over word terms, with optional character n-grams",
        config_model=Bm25RetrieverConfig,
        factory=_build,
        input_contract="RetrievalRequest",
        output_contract="ReferenceBatch",
        # Ranking a fixed pool by term overlap is reproducible, and it neither models nor network.
        capabilities=frozenset({Capability.DETERMINISTIC}),
    )
