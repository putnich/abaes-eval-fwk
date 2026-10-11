"""The grading prompt templates a project sends, and the placeholder contract they must meet.

The templates are configuration rather than code: the prompt is the independent variable in most
grading comparisons, so a project states its own in `prompts.yaml` instead of forking the agents.
The built-in templates live in config_files/defaults/prompts.yaml and are the single source of
truth for the shipped wording; `prompts_config` in configuration.py merges a project's overrides
onto them, and the resolved text enters the run manifest and the resume fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from string import Formatter

# Every value `base_agent` supplies when it renders a grading prompt. A template naming anything
# else would fail at render time, mid-run, so it is rejected at load instead.
_COMMON_PLACEHOLDERS = frozenset(
    {
        "assignment_type",
        "assignment_prompt",
        "source_essay",
        "source_verification_step",
        "student_work",
        "score_min",
        "score_max",
        "examples",
        "examples_block",
        "examples_comparison_step",
        "brevity_instruction",
    }
)
# Without these a template would grade nothing (no submission, no scale, no rubric) or would
# silently discard the examples a few-shot retrieval strategy just selected.
_COMMON_REQUIRED = frozenset({"student_work", "score_min", "score_max", "examples_block"})
SLOTS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    # slot -> (allowed placeholders, required placeholders)
    "holistic": (
        _COMMON_PLACEHOLDERS | {"rubric_guidelines"},
        _COMMON_REQUIRED | {"rubric_guidelines"},
    ),
    "criterion_specialized": (
        _COMMON_PLACEHOLDERS | {"criterion_label", "criterion_guidelines"},
        _COMMON_REQUIRED | {"criterion_guidelines"},
    ),
}


def placeholders(template: str, slot: str) -> set[str]:
    """The placeholder names a template uses, rejecting unbalanced braces as the parse finds them."""
    try:
        return {name for _, name, _, _ in Formatter().parse(template) if name}
    except ValueError as exc:
        raise ValueError(f"prompts.{slot} is not a valid template: {exc}") from exc


@dataclass(frozen=True, kw_only=True)
class PromptsConfig:
    """One grading template per architecture, validated against what the agents supply.

    Neither field has a default here: both are already defaulted in
    config/config_files/defaults/prompts.yaml, which prompts_config() always merges in before
    construction, so a second Python-side copy of the shipped wording would only drift from it.
    """

    holistic: str
    criterion_specialized: str

    def __post_init__(self) -> None:
        for slot in SLOTS:
            self._validate(slot, getattr(self, slot))

    @staticmethod
    def _validate(slot: str, template: str) -> None:
        if not isinstance(template, str) or not template.strip():
            raise ValueError(f"prompts.{slot} must be a nonempty template")
        allowed, required = SLOTS[slot]
        used = placeholders(template, slot)
        if unknown := used - allowed:
            raise ValueError(
                f"prompts.{slot} uses placeholders the grader does not supply: "
                f"{sorted(unknown)}; available: {sorted(allowed)}"
            )
        if missing := required - used:
            raise ValueError(f"prompts.{slot} is missing required placeholders: {sorted(missing)}")

    def for_slot(self, slot: str) -> str:
        """The template for one architecture, by the name the configuration uses."""
        if slot not in SLOTS:
            raise KeyError(f"Unknown prompt slot {slot!r}; expected one of {sorted(SLOTS)}")
        return getattr(self, slot)


@cache
def default_prompts_config() -> PromptsConfig:
    """The shipped templates, read from the packaged defaults rather than duplicated in Python.

    Cached: the result is frozen and read on every agent construction that was given no project
    configuration of its own.
    """
    import yaml

    from abaes_eval_fwk.utils.resources import config_resource

    text = config_resource("defaults/prompts.yaml").read_text(encoding="utf-8")
    return PromptsConfig(**(yaml.safe_load(text) or {})["prompts"])


def read_template(value: object, slot: str, base_dir: str | Path | None) -> str:
    """Resolve one configured slot: an inline template, or `{file: path}` naming one.

    A `file:` path resolves against the project's benchmark.yaml, the same base every other
    configured path uses, so a prompt kept in its own file is found from wherever the run starts.
    """
    if isinstance(value, str):
        return value
    if not isinstance(value, dict) or set(value) != {"file"}:
        raise ValueError(
            f"prompts.{slot} must be a template string or a mapping with a single 'file' key"
        )
    reference = value["file"]
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"prompts.{slot}.file must be a path to a template file")
    base = Path(base_dir).expanduser().resolve() if base_dir else Path.cwd()
    path = Path(reference).expanduser()
    path = path if path.is_absolute() else base / path
    if not path.is_file():
        raise ValueError(f"prompts.{slot}.file not found: {path}")
    return path.read_text(encoding="utf-8")
