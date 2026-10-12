"""The replication package in results/study reproduces the paper's RQ4 procedure statistics.

Pins the DA/RA outcome shares quoted in the paper's RQ4 discussion and the "+Rev." column of its
completion-token table, and checks that the counted outcomes are the branches the DA and RA runs
actually took, by rebuilding every shipped final prediction from them.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from abaes_eval_fwk.configuration import dataset_configs, load_config
from abaes_eval_fwk.core.models.rubric import combine_criterion_scores
from abaes_eval_fwk.experiment.eval.baseline.build_pool_average_baseline import (
    matched_source_records,
)
from abaes_eval_fwk.experiment.eval.eval_adapter import procedure_outcomes
from abaes_eval_fwk.experiment.setup.rubric.rubric_loader import load_rubric
from abaes_eval_fwk.templates import initialize
from abaes_eval_fwk.utils.rounding import round_half_up

RESULTS = Path(__file__).resolve().parents[1] / "results" / "study"
pytestmark = pytest.mark.skipif(
    not RESULTS.is_dir(), reason="results/study ships with the repository, not the sdist"
)

# The initial assessment pools of the paper's RQ4 experiments, and the final agent's independent
# assessments (Gemma 4, reasoning enabled).
POOLS = {
    "holistic": [
        "RQ2/holistic/RQ2_holistic_llama.csv",
        "RQ3/holistic/RQ3_holistic_deepseek_r.csv",
        "RQ3/holistic/RQ3_holistic_qwen_r.csv",
    ],
    "criterion": [
        "RQ2/criterion/RQ2_criterion_llama.csv",
        "RQ2/criterion/RQ2_criterion_deepseek_nr.csv",
        "RQ3/criterion/RQ3_criterion_qwen_r.csv",
    ],
}
INDEPENDENT = {
    "holistic": "RQ3/holistic/RQ3_holistic_gemma_r.csv",
    "criterion": "RQ3/criterion/RQ3_criterion_gemma_r.csv",
}

# Paper, RQ4 discussion: shares of pools (holistic, criterion), in percent to one decimal.
PAPER_OUTCOME_PERCENT = {
    ("DA", "agreement"): (84.7, 86.0),
    ("RA", "positive"): (36.7, 35.6),
    ("RA", "mixed"): (56.0, 62.2),
    ("RA", "negative"): (7.3, 2.2),
}
# Paper, completion-token table, "+Rev." columns (holistic, criterion), rounded to a token.
PAPER_REVIEW_TOKENS = {
    "neuralmagic/DeepSeek-R1-Distill-Qwen-32B-FP8-dynamic": (1330, 5210),
    "neuralmagic-ent/Llama-3.3-70B-Instruct-FP8-dynamic": (1306, 5446),
    "Qwen/Qwen3.8-27B-FP8": (1060, 4147),
}
ARCHITECTURES = ("holistic", "criterion")


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    """The study template's dataset configurations; only their rubrics are read."""
    benchmark = initialize(tmp_path_factory.mktemp("replication") / "study", "study")
    return dataset_configs(load_config(benchmark, []))


@pytest.fixture(scope="module")
def outcomes(datasets, tmp_path_factory):
    """Run what `abaes-eval-fwk adjudication` runs for each architecture's pool and reviews."""
    output = tmp_path_factory.mktemp("adjudication")
    written = {}
    for architecture in ARCHITECTURES:
        paths = procedure_outcomes(
            [RESULTS / p for p in POOLS[architecture]],
            sorted((RESULTS / "RQ4" / "review" / architecture).glob("*.csv")),
            datasets,
            output / architecture,
        )
        written[architecture] = {
            path.stem: pd.read_csv(path, dtype={"Essay ID": str}) for path in paths
        }
    return written


@pytest.mark.parametrize(("procedure", "outcome"), list(PAPER_OUTCOME_PERCENT))
def test_outcome_shares_match_the_paper(outcomes, procedure, outcome):
    for architecture, expected in zip(
        ARCHITECTURES, PAPER_OUTCOME_PERCENT[(procedure, outcome)], strict=True
    ):
        summary = outcomes[architecture]["outcomes"]
        row = summary[
            (summary.Procedure == procedure)
            & (summary.Dataset == "All items")
            & (summary.Outcome == outcome)
        ].iloc[0]
        assert row.Items == (1200 if architecture == "holistic" else 5400)
        assert round(row.Percent, 1) == expected, architecture


def test_review_tokens_match_the_paper(outcomes):
    for index, architecture in enumerate(ARCHITECTURES):
        tokens = outcomes[architecture]["review_tokens"]
        average = tokens[tokens.Dataset == "Average"].set_index("Source Model")
        assert {
            model: round(value) for model, value in average["Avg Completion Tokens"].items()
        } == {model: expected[index] for model, expected in PAPER_REVIEW_TOKENS.items()}


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize(
    ("procedure", "results_name"),
    [("DA", "disagreement_based_adjudication"), ("RA", "review_based_adjudication")],
)
def test_outcomes_rebuild_every_shipped_final_prediction(
    outcomes, datasets, architecture, procedure, results_name
):
    """Resolve each item from its counted outcome as the procedure does, aggregate criteria with
    the rubric, and require the shipped final score for every essay."""
    sources = matched_source_records([RESULTS / p for p in POOLS[architecture]])
    independent = matched_source_records([RESULTS / INDEPENDENT[architecture]])[0]
    models = [pd.read_csv(RESULTS / p, nrows=1).model[0] for p in POOLS[architecture]]
    reviews = pd.concat(
        pd.read_csv(p, dtype=str, keep_default_na=False)
        for p in (RESULTS / "RQ4" / "review" / architecture).glob("*.csv")
    )
    admissible = {
        (r.dataset, r.essay_id, r.criterion, r.source_model): r.is_admissible == "1"
        for r in reviews.itertuples()
    }
    items = outcomes[architecture]["outcome_items"]
    items = items[items.Procedure == procedure]
    resolved: dict[tuple[str, str], dict[str, int]] = {}
    for item in items.itertuples():
        key = (item.Dataset, item._3, item.Criterion)
        pool = [source[key].llm_assessment.score.rounded_score for source in sources]
        own = independent[key].llm_assessment.score.rounded_score
        if procedure == "DA":
            score = own if item.Outcome == "disagreement" else round_half_up(sum(pool) / len(pool))
        else:
            kept = [s for s, m in zip(pool, models, strict=True) if admissible[(*key, m)]]
            score = {
                "positive": lambda: round_half_up(sum(kept) / len(kept)),
                "mixed": lambda: round_half_up((sum(kept) + own) / (len(kept) + 1)),
                "negative": lambda: own,
            }[item.Outcome]()
        resolved.setdefault(key[:2], {})[key[2]] = score
    rubrics = {d.label: load_rubric(d.rubric_path, d.label) for d in datasets}
    shipped = pd.read_csv(
        RESULTS / "RQ4" / architecture / f"RQ4_{architecture}_{results_name}_gemma_r.csv",
        dtype={"essay_id": str},
    )
    assert len(shipped) == len(resolved) == 1200
    for row in shipped.itertuples():
        scores = resolved[(row.dataset, row.essay_id)]
        final = (
            scores["holistic"]
            if architecture == "holistic"
            else combine_criterion_scores(rubrics[row.dataset], scores)
        )
        assert final == row.pred_score, (row.dataset, row.essay_id)


def adjudication_command(*args):
    environment = os.environ.copy()
    source_path = str(Path(__file__).parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (source_path, environment.get("PYTHONPATH", "")) if path
    )
    return subprocess.run(
        [sys.executable, "-m", "abaes_eval_fwk", "adjudication", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=300,
        env=environment,
    )


@pytest.mark.integration
def test_adjudication_command_writes_the_same_outcomes(outcomes, tmp_path):
    benchmark = initialize(tmp_path / "study", "study")
    result = adjudication_command(
        benchmark,
        *(RESULTS / p for p in POOLS["holistic"]),
        "--reviews",
        *sorted((RESULTS / "RQ4" / "review" / "holistic").glob("*.csv")),
        "--output",
        tmp_path / "out",
    )
    assert result.returncode == 0, result.stderr
    printed = [Path(line) for line in result.stdout.splitlines()]
    assert [p.name for p in printed] == ["outcomes.csv", "outcome_items.csv", "review_tokens.csv"]
    pd.testing.assert_frame_equal(
        pd.read_csv(printed[0]), outcomes["holistic"]["outcomes"], check_dtype=False
    )


@pytest.mark.integration
def test_adjudication_command_needs_sources_or_reviews(tmp_path):
    result = adjudication_command(initialize(tmp_path / "study", "study"))
    assert result.returncode == 1
    assert "needs pool source CSVs, --reviews, or both" in result.stderr


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_shipped_disagreement_pools_match_the_recomputed_outcomes(outcomes, architecture):
    """The DA pool files under RQ4/disagreement record the outcome and normalized disagreement
    of every pool; both must equal what the adjudication command recomputes from the sources."""
    shipped = pd.read_csv(
        RESULTS
        / "RQ4"
        / "disagreement"
        / architecture
        / f"RQ4_{architecture}_disagreement_pool_gemma_r.csv",
        dtype={"essay_id": str},
    )
    items = outcomes[architecture]["outcome_items"]
    items = items[items.Procedure == "DA"].rename(
        columns={"Dataset": "dataset", "Essay ID": "essay_id", "Criterion": "criterion"}
    )
    merged = shipped.merge(items, on=["dataset", "essay_id", "criterion"], validate="one_to_one")
    assert len(merged) == len(shipped) == len(items)
    assert (merged.outcome == merged.Outcome).all()
    assert merged.normalized_disagreement.to_numpy() == pytest.approx(
        merged.Disagreement.to_numpy(), abs=5e-5
    )
