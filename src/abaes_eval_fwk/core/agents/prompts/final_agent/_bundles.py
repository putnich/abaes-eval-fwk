"""Template bundle types that each procedure's prompts modules build one instance of."""

from typing import NamedTuple


class AdjudicationTemplates(NamedTuple):
    agreement_system: str
    agreement_user: str
    disagreement_system: str
    disagreement_user: str
    intro_system: str | None = None
    intro_user: str | None = None
    intro_fields: tuple[str, ...] = ()
    final_system: str | None = None
    final_user: str | None = None


class AdjudicationWithReviewTemplates(NamedTuple):
    intro_system: str
    intro_user: str
    intro_fields: tuple[str, ...]
    positive_system: str
    positive_user: str
    mixed_system: str
    mixed_user: str
    negative_system: str
    negative_user: str
    final_system: str | None = None
    final_user: str | None = None
