"""Prior-results reuse configuration: which result and review CSVs a final agent works from."""

from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class ReuseConfig:
    """The prior runs a final agent works from, as given on the command line."""

    results_paths: tuple[str, ...] = ()  # --reuse-paths
    reviews_paths: tuple[str, ...] = ()  # --reuse-reviews-paths
    # --reuse-independent-path: the final agent's own prior run over the same essays, supplying the
    # independent assessment every adjudication branch that reassesses calls for.
    independent_path: str = ""

    def __post_init__(self) -> None:
        for name in ("results_paths", "reviews_paths"):
            object.__setattr__(self, name, tuple(getattr(self, name) or ()))
        object.__setattr__(self, "independent_path", self.independent_path or "")

    @cached_property
    def source_models(self) -> tuple[str, ...]:
        """The model behind each results path, read from the CSVs on first access. Imported locally so this config module doesn't depend on the experiment package."""
        from abaes_eval_fwk.experiment.setup.reuse_loader import source_model

        models = tuple(source_model(path) for path in self.results_paths)
        if len(set(models)) != len(models):
            raise ValueError("Candidate results must have distinct model identifiers")
        return models

    @cached_property
    def source_experiments(self) -> tuple[str, ...]:
        """The experiment_id behind each results path, naming the variant a review judged."""
        from abaes_eval_fwk.experiment.setup.reuse_loader import source_experiment

        return tuple(source_experiment(path) for path in self.results_paths)

    @cached_property
    def independent_experiment(self) -> str:
        """The experiment_id behind the independent-assessment path, or "" when none was given. Keyed into the checkpoint slug, since swapping it changes every reassessed score."""
        from abaes_eval_fwk.experiment.setup.reuse_loader import source_experiment

        return source_experiment(self.independent_path) if self.independent_path else ""

    def require(
        self, experiment_name: str, procedure: str, needs_independent: bool = False
    ) -> None:
        """Raise if the candidate pool — or, for an adjudicating procedure, the independent assessment — was not supplied. Only called where the procedure needs them."""
        if not self.results_paths:
            raise ValueError(
                f"Experiment '{experiment_name}': {procedure} requires --reuse path(s), got none"
            )
        if needs_independent and not self.independent_path:
            raise ValueError(
                f"Experiment '{experiment_name}': {procedure} reassesses from the final agent's "
                "own prior run — pass it with --reuse-independent-path"
            )
