"""DA/RA outcome counting and review cost, as the adjudication command reports them."""

import csv

import pandas as pd
import pytest

from abaes_eval_fwk.core.models.rubric import Criterion, Rubric, Scale
from abaes_eval_fwk.experiment.eval.metrics.procedure_outcomes import (
    disagreement_outcomes,
    read_reviews,
    review_outcomes,
    review_tokens,
    save_procedure_outcomes,
)

RUBRIC = Rubric(
    "Custom",
    "essay",
    False,
    "Write an essay",
    Scale(1, 6),
    "Guidelines",
    [Criterion("ideas", []), Criterion("style", [])],
)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def pool(tmp_path, scores_by_source):
    """One holistic results CSV per source; `scores_by_source[i][j]` scores essay j."""
    return [
        write_csv(
            tmp_path / f"source_{i}.csv",
            [
                dict(
                    dataset="Custom",
                    essay_id=str(j),
                    criterion="holistic",
                    human_score=3,
                    pred_score_raw=score,
                )
                for j, score in enumerate(scores)
            ],
        )  # fmt: skip
        for i, scores in enumerate(scores_by_source)
    ]


def review(essay, model, admissible, *, criterion="holistic", tokens="10", dataset="Custom"):
    return dict(
        dataset=dataset,
        essay_id=essay,
        criterion=criterion,
        source_model=model,
        is_admissible=admissible,
        completion_tokens=tokens,
    )


def test_da_disagrees_only_above_the_threshold(tmp_path):
    # Width 5: one-point splits give a normalized distance of 2/3 / 5 = 0.133, two points 0.267.
    sources = pool(tmp_path, [[3, 3, 2, 1], [3, 4, 4, 3], [3, 3, 2, 3]])
    outcomes = disagreement_outcomes(sources, {"Custom": RUBRIC}, threshold=0.2)
    assert [outcomes[("Custom", str(j), "holistic")].outcome for j in range(4)] == [
        "agreement",
        "agreement",
        "disagreement",
        "disagreement",
    ]
    assert outcomes[("Custom", "1", "holistic")].disagreement == pytest.approx(2 / 3 / 5)
    lenient = disagreement_outcomes(sources, {"Custom": RUBRIC}, threshold=0.3)
    assert {o.outcome for o in lenient.values()} == {"agreement"}


def test_da_uses_the_criterion_scale_for_criterion_items(tmp_path):
    rubric = Rubric(
        "Custom",
        "essay",
        False,
        "Prompt",
        Scale(0, 12),
        "Guidelines",
        [Criterion("ideas", [])],
        criteria_scale=Scale(0, 3),
    )
    sources = [
        write_csv(
            tmp_path / f"s{i}.csv",
            [
                dict(
                    dataset="Custom",
                    essay_id="0",
                    criterion="ideas",
                    human_score=6,
                    pred_score_raw=score,
                )
            ],
        )  # fmt: skip
        for i, score in enumerate([1, 2, 1])
    ]
    # A one-point split is small on the 0–12 total (0.056) but large on the 0–3 criterion
    # scale (0.222), and DA judges a criterion on its own scale.
    outcome = disagreement_outcomes(sources, {"Custom": rubric})[("Custom", "0", "ideas")]
    assert outcome.outcome == "disagreement"
    assert outcome.disagreement == pytest.approx(2 / 3 / 3)


def test_da_requires_matched_sources(tmp_path):
    first, second = pool(tmp_path, [[3, 3], [3]])
    with pytest.raises(ValueError, match="identical dataset, essay and criterion keys"):
        disagreement_outcomes([first, second], {"Custom": RUBRIC})
    with pytest.raises(ValueError, match="at least two"):
        disagreement_outcomes([first], {"Custom": RUBRIC})


def test_ra_outcome_follows_how_many_sources_are_admissible(tmp_path):
    path = write_csv(
        tmp_path / "reviews.csv",
        [
            review(essay, model, flag)
            for essay, flags in {"a": "111", "b": "101", "c": "000"}.items()
            for model, flag in zip(["m1", "m2", "m3"], flags)
        ],
    )
    outcomes = review_outcomes(read_reviews([path]))
    assert {key[1]: (o.outcome, o.admissible, o.reviewed) for key, o in outcomes.items()} == {
        "a": ("positive", 3, 3),
        "b": ("mixed", 2, 3),
        "c": ("negative", 0, 3),
    }


def test_ra_reads_one_file_per_source_like_one_combined_file(tmp_path):
    rows = [review(e, m, "1") for e in "ab" for m in ("m1", "m2")]
    combined = write_csv(tmp_path / "all.csv", rows)
    split = [
        write_csv(tmp_path / f"{m}.csv", [r for r in rows if r["source_model"] == m])
        for m in ("m1", "m2")
    ]
    assert review_outcomes(read_reviews([combined])) == review_outcomes(read_reviews(split))


@pytest.mark.parametrize("flag", ["True", "False", "true", "", "yes"])
def test_reviews_accept_only_one_and_zero(tmp_path, flag):
    path = write_csv(tmp_path / "reviews.csv", [review("a", "m1", flag)])
    with pytest.raises(ValueError, match="is_admissible must be 1 or 0"):
        read_reviews([path])


def test_ra_rejects_items_missing_a_source_review_and_duplicates(tmp_path):
    missing = write_csv(
        tmp_path / "missing.csv",
        [review("a", "m1", "1"), review("a", "m2", "1"), review("b", "m1", "1")],
    )
    with pytest.raises(ValueError, match="lacks reviews for \\['m2'\\]"):
        review_outcomes(read_reviews([missing]))
    duplicate = write_csv(tmp_path / "dup.csv", [review("a", "m1", "1")])
    with pytest.raises(ValueError, match="Duplicate review"):
        read_reviews([duplicate, duplicate])


def test_review_tokens_sum_criteria_per_essay_then_weight_datasets_equally(tmp_path):
    rows = [
        # Dataset A: one essay, two criteria → 30 tokens per essay.
        review("a1", "m1", "1", criterion="ideas", tokens="10", dataset="A"),
        review("a1", "m1", "1", criterion="style", tokens="20", dataset="A"),
        # Dataset B: two essays → (100 + 300) / 2 = 200 tokens per essay.
        review("b1", "m1", "1", criterion="ideas", tokens="40", dataset="B"),
        review("b1", "m1", "1", criterion="style", tokens="60", dataset="B"),
        review("b2", "m1", "1", criterion="ideas", tokens="100", dataset="B"),
        review("b2", "m1", "1", criterion="style", tokens="200", dataset="B"),
    ]
    tokens = review_tokens(read_reviews([write_csv(tmp_path / "r.csv", rows)]))
    by_dataset = dict(zip(tokens.Dataset, tokens["Avg Completion Tokens"], strict=True))
    assert by_dataset == {"A": 30, "B": 200, "Average": 115}
    assert dict(zip(tokens.Dataset, tokens.Essays, strict=True)) == {"A": 1, "B": 2, "Average": 3}


def test_reviews_without_recorded_tokens_still_give_outcomes(tmp_path):
    path = write_csv(tmp_path / "cached.csv", [review("a", "m1", "1", tokens="")])
    written = save_procedure_outcomes(tmp_path / "out", reviews=read_reviews([path]))
    assert [p.name for p in written] == ["outcomes.csv", "outcome_items.csv"]
    with pytest.raises(ValueError, match="lack completion_tokens"):
        review_tokens(read_reviews([path]))
    mixed = write_csv(
        tmp_path / "mixed.csv", [review("a", "m1", "1", tokens=""), review("b", "m1", "1")]
    )
    with pytest.raises(ValueError, match="some reviews but not others"):
        save_procedure_outcomes(tmp_path / "out", reviews=read_reviews([mixed]))


def test_summary_pools_items_across_datasets(tmp_path):
    rows = [review("a", "m1", "1", dataset="A")] + [
        review(e, "m1", "0", dataset="B") for e in ("b1", "b2", "b3")
    ]
    path = write_csv(tmp_path / "r.csv", rows)
    summary_path = save_procedure_outcomes(tmp_path / "out", reviews=read_reviews([path]))[0]
    summary = pd.read_csv(summary_path).set_index(["Dataset", "Outcome"])
    assert summary.loc[("All items", "positive"), "Count"] == 1
    assert summary.loc[("All items", "positive"), "Percent"] == 25
    assert summary.loc[("B", "negative"), "Percent"] == 100
