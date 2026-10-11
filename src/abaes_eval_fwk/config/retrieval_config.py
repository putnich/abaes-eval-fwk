"""Few-shot selection, embedding hardware, and deterministic retrieval configuration."""

import math
from dataclasses import dataclass
from enum import Enum

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ExamplesSelection(str, Enum):
    """Strategies for retrieving score-stratified calibration examples."""

    ZERO_SHOT = ""
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    LEXICAL = "lexical"
    RANDOM = "random"


@dataclass(frozen=True, kw_only=True)
class RetrievalConfig:
    """Few-shot settings with a total example cap and optional dense retrieval hardware.

    `n_ex_per_score` and `max_examples` have no default here: both are already
    defaulted in config/config_files/defaults/retrieval.yaml, which retrieval_config()
    in configuration.py always merges in before construction for the YAML-driven path.
    Standalone construction (retrieval_loader.py's bare fallback) supplies the same YAML
    values explicitly instead of duplicating them as a second Python-side default.

    `examples_selection` deliberately keeps its own default (`RANDOM`): it does not
    duplicate defaults/retrieval.yaml's `strategy: hybrid` — that YAML value is the
    project-wide recommended strategy, while `RANDOM` here is a distinct, simpler
    standalone default that several tests specifically exercise. Unifying the two would
    be a real behavior change, not a dead-code removal.
    """

    n_ex_per_score: int
    examples_selection: ExamplesSelection = ExamplesSelection.RANDOM
    max_examples: int
    hybrid_dense_weight: float = 0.5
    hybrid_bm25_weight: float = 0.5
    seed: int = 42
    embedding_model: str = EMBED_MODEL
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    embedding_threads: int = 1
    # Set when a run names a reader-supplied 'module:function' instead of a built-in policy. It
    # carries no behavior of its own; it identifies the selector in the run name and fingerprint.
    selector: str = ""

    def __post_init__(self) -> None:
        selection = "" if self.examples_selection == "zero_shot" else self.examples_selection
        object.__setattr__(self, "examples_selection", ExamplesSelection(selection))
        self.validate_retrieval_config()

    def validate_retrieval_config(self) -> None:
        if self.selector and (":" not in self.selector or self.selector.endswith(":")):
            raise ValueError("selector must be 'module:function' naming a selection callable")
        for name, minimum in (
            ("n_ex_per_score", 0),
            ("max_examples", 0),
            ("embedding_batch_size", 1),
            ("embedding_threads", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        for name in ("hybrid_dense_weight", "hybrid_bm25_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number between 0 and 1")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number between 0 and 1")
        if not math.isclose(self.hybrid_dense_weight + self.hybrid_bm25_weight, 1.0):
            raise ValueError("hybrid weights must sum to 1")
        if not isinstance(self.embedding_model, str) or not self.embedding_model.strip():
            raise ValueError("embedding_model must be a nonempty model identifier or path")
        if not isinstance(self.embedding_device, str) or not self.embedding_device.strip():
            raise ValueError(
                "embedding_device must be a nonempty device, such as cpu, cuda:0, or mps"
            )

    @property
    def is_zero_shot(self) -> bool:
        return (
            self.examples_selection == ExamplesSelection.ZERO_SHOT
            or self.n_ex_per_score == 0
            or self.max_examples == 0
        )
