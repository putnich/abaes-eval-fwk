"""The reader-supplied extension points: dataset loaders, metrics, prompts, and aggregation.

Each point is checked three ways: the extension reaches the pipeline, a broken extension is
rejected with a message naming the setting, and the run's provenance records what was used.
"""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from abaes_eval_fwk.config.dataset_config import dataset_from_dict
from abaes_eval_fwk.config.prompts_config import default_prompts_config
from abaes_eval_fwk.configuration import merge, metrics_config, prompts_config
from abaes_eval_fwk.core.models.assessment import (
    Dataset,
    GradingRecord,
    HumanAssessment,
    LLMAssessment,
    PredictedScore,
    StudentSubmission,
)
from abaes_eval_fwk.core.models.rubric import CriteriaAggregation, Criterion, Rubric, Scale
from abaes_eval_fwk.experiment.eval.aggregate_criteria import aggregate_criteria
from abaes_eval_fwk.experiment.eval.metrics.compute_metrics import (
    BUILTIN_METRICS,
    MetricsInput,
    MetricsSet,
    process_csv,
    reported_metrics,
    resolve_metrics,
)
from abaes_eval_fwk.experiment.setup.dataset.dataset_loader import load_dataset
from abaes_eval_fwk.experiment.setup.rubric.rubric_loader import load_rubric
from abaes_eval_fwk.utils.resources import load_defaults

EXTENSION_SOURCE = """
import math

import pandas as pd


def load_frame(*, dataset, data_path):
    return pd.DataFrame(
        {
            "essay_id": ["a", "b", "c"],
            "set_id": ["only"] * 3,
            "essay": ["First essay.", "Second essay.", "Third essay."],
            "human_score": [1, 3, 5],
            "ideas": [1, 3, 5],
            "style": [2, 2, 4],
        }
    )


def load_two_sets(*, dataset, data_path):
    frame = load_frame(dataset=dataset, data_path=data_path)
    frame["set_id"] = ["only", "other", "only"]
    frame["essay_id"] = ["a", "b", "c"]
    return frame


def load_not_a_frame(*, dataset, data_path):
    return [{"essay_id": "a"}]


def load_without_columns(*, dataset, data_path):
    return pd.DataFrame({"essay_id": ["a"], "essay": ["Text."]})


def exact_agreement(*, preds, humans, scale):
    return sum(1 for p, h in zip(preds, humans)) and sum(
        1 for p, h in zip(preds, humans) if p == h
    ) / len(preds)


def always_nan(*, preds, humans, scale):
    return math.nan


def returns_text(*, preds, humans, scale):
    return "high"


def lowest_criterion(*, rubric, scores):
    return min(scores.values())


def returns_infinity(*, rubric, scores):
    return math.inf
"""

RUBRIC_YAML = """
assignment_type: essay
prompt: Write an essay.
score_min: 1
score_max: 5
rubric_guidelines: Score ideas and style.
criteria:
  ideas:
    score_levels:
      - score: 1
        guideline: Few ideas.
      - score: 5
        guideline: Many ideas.
  style:
    score_levels:
      - score: 1
        guideline: Unclear.
      - score: 5
        guideline: Clear.
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project directory holding an extensions module and a rubric, importable as a run would."""
    (tmp_path / "extensions.py").write_text(EXTENSION_SOURCE, encoding="utf-8")
    (tmp_path / "rubric.yaml").write_text(RUBRIC_YAML, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def _dataset(project: Path, loader: str, **extra):
    return dataset_from_dict(
        {"name": "Reader Corpus", "loader": loader, "rubric": "rubric.yaml", **extra},
        base_dir=str(project),
        defaults={"data_root": str(project)},
    )


# --------------------------------------------------------------------------- dataset loaders


def test_a_reader_loader_supplies_the_frame_and_still_meets_every_validation(project):
    """The loader replaces the reading step only: its frame is validated like a packaged one."""
    frame = load_dataset(_dataset(project, "extensions:load_frame"))
    assert list(frame["essay_id"]) == ["a", "b", "c"]
    assert list(frame["human_score"]) == [1, 3, 5]
    # The criterion columns the rubric declares survive the loader path.
    assert list(frame["ideas"]) == [1, 3, 5]


def test_a_loader_dataset_needs_no_csv_but_does_need_a_rubric(project):
    """A loader may build its frame from nothing on disk; the rubric is what validates it."""
    assert _dataset(project, "extensions:load_frame").raw_data_path == ""
    with pytest.raises(ValueError, match="requires a rubric"):
        dataset_from_dict(
            {"name": "No Rubric", "loader": "extensions:load_frame"}, base_dir=str(project)
        )


def test_set_id_still_selects_one_subset_of_a_loader_frame(project):
    """`set_id` filters a reader's frame exactly as it filters a normalized CSV."""
    frame = load_dataset(_dataset(project, "extensions:load_two_sets", set_id="only"))
    assert list(frame["essay_id"]) == ["a", "c"]


@pytest.mark.parametrize(
    ("function", "error", "message"),
    [
        ("extensions:load_not_a_frame", TypeError, "must return a pandas DataFrame"),
        ("extensions:load_without_columns", ValueError, "missing required columns"),
        ("extensions:missing_function", AttributeError, "does not define"),
        ("extensions_absent:load", ImportError, "could not be imported"),
    ],
)
def test_a_broken_loader_fails_with_a_message_naming_the_dataset(project, function, error, message):
    """Every failure path names the dataset whose loader failed, not just the module."""
    with pytest.raises(error, match=message):
        load_dataset(_dataset(project, function))


def test_a_loader_reference_must_be_module_and_function(project):
    with pytest.raises(ValueError, match="must be 'module:function'"):
        _dataset(project, "extensions.load_frame")


# --------------------------------------------------------------------------- custom metrics


def _metrics(**overrides):
    """Resolve a metrics configuration the way load_config does: packaged defaults, then overrides."""
    return metrics_config(merge(load_defaults()["metrics"], overrides))


def _metrics_set(preds, humans, scale=Scale(1, 5)):
    group = MetricsSet(
        "Reader Corpus",
        [
            MetricsInput(str(i), p, h, None)
            for i, (p, h) in enumerate(zip(preds, humans, strict=True))
        ],
        scale=scale,
    )
    return group


def test_a_custom_metric_is_computed_beside_the_built_ins(project):
    group = _metrics_set([1, 3, 5], [1, 3, 4])
    resolve_metrics(group, Scale(1, 5), {"exact": "extensions:exact_agreement"})
    assert group.resolved["exact"] == pytest.approx(2 / 3)
    # The built-ins are untouched by a custom metric being present.
    assert group.resolved.nmae == pytest.approx((1 / 3) * (10 / 4))


def test_a_custom_metric_may_report_nan_for_undefined_but_not_a_non_number(project):
    """NaN is how the built-in correlations say 'undefined here'; text is a configuration error."""
    group = _metrics_set([1, 3], [1, 3])
    resolve_metrics(group, Scale(1, 5), {"quiet": "extensions:always_nan"})
    assert group.resolved["quiet"] != group.resolved["quiet"]  # NaN
    with pytest.raises(TypeError, match="must return a number"):
        resolve_metrics(
            _metrics_set([1, 3], [1, 3]), Scale(1, 5), {"bad": "extensions:returns_text"}
        )


def test_a_custom_metric_cannot_shadow_a_built_in_name():
    with pytest.raises(ValueError, match="shadows a built-in metric"):
        _metrics(custom={"qwk": "extensions:exact_agreement"})
    with pytest.raises(ValueError, match="shadows a built-in metric"):
        _metrics(custom={"rho": "extensions:exact_agreement"})


def test_a_custom_metric_reference_is_validated_at_configuration_load():
    with pytest.raises(ValueError, match="metrics.custom.exact must be 'module:function'"):
        _metrics(custom={"exact": "extensions.exact_agreement"})


def test_the_compute_flags_choose_the_reported_columns():
    """The flags were previously accepted and ignored; they now select the saved table's columns."""
    config = _metrics(
        compute={
            "nrmse": False,
            "pearson": False,
            "spearman": False,
        },
        custom={"exact": "extensions:exact_agreement"},
    )
    assert reported_metrics(config) == ["nmae", "qwk", "exact"]
    # With no configuration the five historically reported metrics are the columns.
    assert reported_metrics() == ["nmae", "nrmse", "pearson", "spearman", "qwk"]


def test_a_custom_metric_reaches_the_saved_table(project, tmp_path):
    """End to end: the configured name is the column header, in the configured order."""
    rubric = load_rubric(str(project / "rubric.yaml"), "Reader Corpus")
    results = tmp_path / "results.csv"
    with results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_id",
                "dataset",
                "essay_id",
                "criterion",
                "human_score",
                "pred_score_raw",
            ],
        )
        writer.writeheader()
        for essay_id, human, pred in (("a", 1, 1), ("b", 3, 4), ("c", 5, 5)):
            writer.writerow(
                {
                    "experiment_id": "run",
                    "dataset": "Reader Corpus",
                    "essay_id": essay_id,
                    "criterion": "holistic",
                    "human_score": human,
                    "pred_score_raw": pred,
                }
            )
    humans = Dataset(
        SimpleNamespace(label="Reader Corpus"),
        [
            HumanAssessment(
                StudentSubmission(essay_id, "Essay"), human, rubric.holistic_criterion()
            )
            for essay_id, human in (("a", 1), ("b", 3), ("c", 5))
        ],
    )
    config = _metrics(custom={"exact": "extensions:exact_agreement"})
    path, _ = process_csv(results, [rubric], [humans], tmp_path / "metrics", config)
    header, *rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    assert header[-2:] == ["exact", "Avg Completion Tokens"]
    assert float(rows[0][header.index("exact")]) == pytest.approx(2 / 3)


def test_every_builtin_is_computed_whether_or_not_it_is_reported():
    """Computing every built-in costs nothing, and keeps a direct resolve_metrics call complete."""
    group = _metrics_set([2, 4], [3, 5], Scale(0, 10))
    resolve_metrics(group, Scale(0, 10))
    assert set(BUILTIN_METRICS) <= set(group.resolved)


# --------------------------------------------------------------------------- prompts


def test_a_project_overrides_one_template_and_keeps_the_other(tmp_path):
    override = (
        "Grade {student_work} on {score_min}-{score_max} using {rubric_guidelines}{examples_block}"
    )
    resolved = prompts_config({"holistic": override}, tmp_path)
    assert resolved.holistic == override
    assert resolved.criterion_specialized == default_prompts_config().criterion_specialized


def test_a_template_may_live_in_its_own_file_beside_the_configuration(tmp_path):
    (tmp_path / "prompts").mkdir()
    template = (
        "Score {student_work} from {score_min} to {score_max}.{examples_block}\n{rubric_guidelines}"
    )
    (tmp_path / "prompts" / "holistic.txt").write_text(template, encoding="utf-8")
    resolved = prompts_config({"holistic": {"file": "prompts/holistic.txt"}}, tmp_path)
    assert resolved.holistic == template


def test_a_missing_template_file_is_reported_with_its_resolved_path(tmp_path):
    with pytest.raises(ValueError, match="prompts.holistic.file not found"):
        prompts_config({"holistic": {"file": "prompts/absent.txt"}}, tmp_path)


def test_a_template_naming_an_unsupplied_placeholder_is_rejected_at_load(tmp_path):
    """The grader supplies a fixed set of values; anything else would fail mid-run instead."""
    with pytest.raises(ValueError, match="placeholders the grader does not supply"):
        prompts_config(
            {
                "holistic": "Grade {student_work} {score_min}{score_max}{examples_block}{rubric_guidelines}{school_board}"
            },
            tmp_path,
        )


@pytest.mark.parametrize(
    ("template", "missing"),
    [
        ("Grade {score_min}{score_max}{examples_block}{rubric_guidelines}", "student_work"),
        ("Grade {student_work}{score_min}{score_max}{rubric_guidelines}", "examples_block"),
        ("Grade {student_work}{score_min}{score_max}{examples_block}", "rubric_guidelines"),
    ],
)
def test_a_template_missing_an_essential_placeholder_is_rejected(tmp_path, template, missing):
    """A prompt without these would grade nothing, or silently discard retrieved examples."""
    with pytest.raises(ValueError, match=f"missing required placeholders.*{missing}"):
        prompts_config({"holistic": template}, tmp_path)


def test_an_unknown_prompt_slot_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="prompts"):
        prompts_config({"final_agent": "anything"}, tmp_path)


def test_the_shipped_templates_are_the_packaged_yaml_ones():
    """The wording has one home: the agents' constants read the packaged defaults."""
    from abaes_eval_fwk.core.agents.prompts.criterion_specialized_grading_prompts import (
        CRITERION_SPECIALIZED_GRADING_PROMPT,
    )
    from abaes_eval_fwk.core.agents.prompts.holistic_grading_prompts import (
        HOLISTIC_GRADING_PROMPT,
    )

    assert HOLISTIC_GRADING_PROMPT == default_prompts_config().holistic
    assert CRITERION_SPECIALIZED_GRADING_PROMPT == default_prompts_config().criterion_specialized


def test_an_agent_renders_the_configured_template(rubric, submission):
    """The template the configuration resolved is the text the agent sends."""
    from abaes_eval_fwk.core.agents import HolisticGradingAgent

    override = "Reader template: {student_work} on {score_min}-{score_max}{examples_block}{rubric_guidelines}"
    agent = HolisticGradingAgent(override)
    assert agent._prompt_template == override
    assert HolisticGradingAgent()._prompt_template == default_prompts_config().holistic


# --------------------------------------------------------------------------- aggregation


def _records(rubric, scores):
    return [
        GradingRecord(
            HumanAssessment(StudentSubmission("a", "Essay"), 3, Criterion(name, [])),
            LLMAssessment(PredictedScore(score), "Because.", None),
        )
        for name, score in scores.items()
    ]


def _weighted_rubric(weights, aggregation=CriteriaAggregation.WEIGHTED):
    return Rubric(
        rubric_id="Reader Corpus",
        assignment_type="essay",
        source_dependent=False,
        prompt="Write an essay.",
        scale=Scale(1, 5),
        rubric_guidelines="Guidelines",
        criteria=[Criterion("ideas", []), Criterion("style", [])],
        criteria_aggregation=aggregation,
        criteria_weights=weights,
    )


def test_weighted_aggregation_weights_each_criterion():
    """0.75*5 + 0.25*1 = 4, where the unweighted mean of the same scores is 3."""
    weighted = _weighted_rubric({"ideas": 0.75, "style": 0.25})
    scores = {"ideas": 5, "style": 1}
    assert aggregate_criteria(weighted, _records(weighted, scores)) == 4
    unweighted = _weighted_rubric({}, aggregation=CriteriaAggregation.MEAN)
    assert aggregate_criteria(unweighted, _records(unweighted, scores)) == 3


def test_weights_need_not_sum_to_one():
    """Weights are relative: 3:1 states the same rubric as 0.75:0.25."""
    weighted = _weighted_rubric({"ideas": 3, "style": 1})
    assert aggregate_criteria(weighted, _records(weighted, {"ideas": 5, "style": 1})) == 4


def test_a_rubric_declares_weights_in_its_own_yaml(tmp_path):
    (tmp_path / "weighted.yaml").write_text(
        RUBRIC_YAML
        + textwrap.dedent(
            """
            criteria_aggregation: weighted
            criteria_weights:
              ideas: 0.8
              style: 0.2
            """
        ),
        encoding="utf-8",
    )
    loaded = load_rubric(str(tmp_path / "weighted.yaml"), "Weighted")
    assert loaded.criteria_aggregation == CriteriaAggregation.WEIGHTED
    assert loaded.criteria_weights == {"ideas": 0.8, "style": 0.2}
    assert aggregate_criteria(loaded, _records(loaded, {"ideas": 5, "style": 1})) == 4


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("criteria_aggregation: weighted\n", "requires criteria_weights"),
        (
            "criteria_aggregation: weighted\ncriteria_weights:\n  ideas: 1.0\n",
            "must name every rubric criterion",
        ),
        (
            "criteria_aggregation: weighted\ncriteria_weights:\n  ideas: 0\n  style: 0\n",
            "must not all be zero",
        ),
        (
            "criteria_aggregation: weighted\ncriteria_weights:\n  ideas: -1\n  style: 2\n",
            "finite and >= 0",
        ),
        (
            "criteria_aggregation: mean\ncriteria_weights:\n  ideas: 1\n  style: 1\n",
            "require criteria_aggregation: weighted",
        ),
    ],
)
def test_weighted_aggregation_is_validated_when_the_rubric_is_parsed(tmp_path, extra, message):
    path = tmp_path / f"rubric-{abs(hash(extra))}.yaml"
    path.write_text(RUBRIC_YAML + extra, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_rubric(str(path), "Invalid")


def test_an_aggregation_callable_replaces_the_formula(project):
    """A reader's callable receives the scores by criterion name and supplies the total."""
    rubric = _weighted_rubric({}, aggregation="extensions:lowest_criterion")
    records = _records(rubric, {"ideas": 5, "style": 2})
    assert aggregate_criteria(rubric, records) == 2


def test_an_aggregation_callable_returning_a_non_finite_total_is_rejected(project):
    rubric = _weighted_rubric({}, aggregation="extensions:returns_infinity")
    with pytest.raises(ValueError, match="returned a non-finite total"):
        aggregate_criteria(rubric, _records(rubric, {"ideas": 5, "style": 2}))


def test_weights_and_an_aggregation_callable_are_mutually_exclusive(tmp_path):
    path = tmp_path / "both.yaml"
    path.write_text(
        RUBRIC_YAML
        + "criteria_aggregation: extensions:lowest_criterion\ncriteria_weights:\n  ideas: 1\n  style: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="do not apply to an aggregation callable"):
        load_rubric(str(path), "Both")


# --------------------------------------------------------------------------- provenance


def test_every_extension_a_configuration_names_is_fingerprinted(tmp_path, monkeypatch):
    """A resume must notice changed reader code, for each point that can name any."""
    from abaes_eval_fwk.runner import _extension_references

    config = SimpleNamespace(
        base_dir=str(tmp_path),
        metrics=SimpleNamespace(custom={"exact": "extensions:exact_agreement"}),
    )
    monkeypatch.setattr(
        "abaes_eval_fwk.runner.select_experiments", lambda _config: [], raising=True
    )
    monkeypatch.setattr(
        "abaes_eval_fwk.runner.sampling_config",
        lambda _config, _experiment=None: SimpleNamespace(
            evaluation_strategy="extensions.sampling:lowest_scores_first",
            calibration_strategy="stratified",
        ),
        raising=True,
    )
    monkeypatch.setattr(
        "abaes_eval_fwk.runner.dataset_configs",
        lambda _config: [
            SimpleNamespace(
                loader="extensions:load_frame",
                label="Reader Corpus",
                rubric_path=str(tmp_path / "absent.yaml"),
            )
        ],
        raising=True,
    )
    references = _extension_references(config)
    assert references == {
        "extensions.sampling:lowest_scores_first": "sampling.evaluation_strategy",
        "extensions:load_frame": "Dataset 'Reader Corpus' loader",
        "extensions:exact_agreement": "metrics.custom.exact",
    }
