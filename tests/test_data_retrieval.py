"""Regression checks for portable datasets, leakage-free sampling, and configurable retrieval."""

import builtins
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from abaes_eval_fwk.config.dataset_config import (
    DatasetConfig,
    DatasetFamily,
    DatasetNames,
    dataset_from_dict,
    get_dataset_config,
)
from abaes_eval_fwk.config.retrieval_config import ExamplesSelection, RetrievalConfig
from abaes_eval_fwk.core.models.assessment import HumanAssessment, StudentSubmission
from abaes_eval_fwk.core.models.experiment import CalibrationPool
from abaes_eval_fwk.core.models.rubric import Rubric, Scale
from abaes_eval_fwk.core.retriever import retrieval_loader, retriever
from abaes_eval_fwk.experiment.setup.calibration.calibration_loader import (
    load_calibration_pool,
)
from abaes_eval_fwk.experiment.setup.dataset.dataset_loader import load_dataset
from abaes_eval_fwk.experiment.setup.rubric.rubric_loader import load_rubric
from abaes_eval_fwk.experiment.setup.setup_builder import build_setup


@pytest.fixture
def custom(tmp_path):
    rubric_path = tmp_path / "rubric.yaml"
    rubric_path.write_text(
        yaml.safe_dump(
            {
                "assignment_type": "Essay",
                "prompt": "Explain your answer",
                "score_ranges": {"overall": [1, 5]},
                "criteria": {
                    "ideas": {
                        "score_levels": [
                            {"score": i, "guideline": f"Level {i}"} for i in range(1, 6)
                        ]
                    }
                },
            }
        )
    )
    rows = [
        {
            "essay_id": f"{score}-{i:03d}",
            "set_id": "01",
            "essay": f"Essay {score} {i}",
            "human_score": score,
            "ideas": score,
        }
        for score in range(1, 6)
        for i in range(60)
    ]
    csv_path = tmp_path / "essays.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return DatasetConfig(
        "Custom", data_root=tmp_path, normalized_csv=csv_path, rubric_file=rubric_path
    )


def make_pool(tmp_path, score_counts=(8, 8)):
    rubric = Rubric("Example", "Essay", False, "Prompt", Scale(1, len(score_counts)), "Guidelines")
    assessments = [
        HumanAssessment(
            StudentSubmission(f"{score}-{index}", f"Text {score} {index}"),
            score,
            rubric.holistic_criterion(),
        )
        for score, count in enumerate(score_counts, 1)
        for index in range(count)
    ]
    return rubric, CalibrationPool(assessments, index_path=str(tmp_path / "cache.faiss"))


def picks(rubric, pool, config):
    return retriever.Retriever.select_reference_examples(
        rubric, pool, "Query essay", config, rubric.holistic_criterion()
    )


def test_custom_paths_and_ids_do_not_depend_on_cwd(custom, tmp_path, monkeypatch):
    config = dataset_from_dict(
        {"name": "Custom", "csv": "essays.csv", "rubric": "rubric.yaml"},
        base_dir=tmp_path,
        defaults={"data_root": str(tmp_path), "seed": 19},
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    frame = load_dataset(config)
    assert config.normalized_csv == custom.normalized_csv
    assert config.seed == 19 and frame.set_id.unique().tolist() == ["01"]
    assert Path(config.calibration_pool_path).is_absolute()


@pytest.mark.parametrize("name", list(DatasetNames))
def test_all_builtin_loaders_keep_official_human_scores(name, tmp_path):
    config = get_dataset_config(name, data_root=tmp_path)
    rubric = load_rubric(config.rubric_path, config.label)
    assert Path(config.rubric_path).is_file()
    folder = Path(config.dataset_folder_path)
    folder.mkdir(parents=True)
    score = rubric.scale.max
    if config.family == DatasetFamily.ASAP_AES:
        row = {
            "essay_id": "001",
            "essay_set": config.set,
            "essay": "An essay",
            "rater1_domain1": score,
        }
        row.update({f"rater1_trait{i}": 3 for i in range(1, 5)})
        pd.DataFrame([row]).to_csv(config.raw_data_path, sep="\t", index=False)
    elif config.family == DatasetFamily.ASAP_2_0:
        pd.DataFrame(
            [
                {
                    "essay_id": "001",
                    "prompt_name": config.prompt_name,
                    "full_text": "An essay",
                    "score": score,
                }
            ]
        ).to_csv(config.raw_data_path, index=False)
    else:
        pd.DataFrame([{"text_id_kaggle": "001", "full_text": "An essay", "Overall": 1}]).to_csv(
            config.raw_data_path, index=False
        )
        row = {
            "text_id_kaggle": "001",
            **{column: 1 for column in config.raw_columns},
            "Overall_1": score,
        }
        pd.DataFrame([row]).to_csv(folder / config.raw_filenames[1], index=False)
    frame = load_dataset(config)
    assert frame.essay_id.tolist() == ["001"]
    assert frame.human_score.tolist() == [score]
    assert config.calibration_per_score == (
        8
        if name
        in {DatasetNames.ASAP_AES_SET_7, DatasetNames.ASAP_2_SET_1, DatasetNames.ASAP_2_SET_6}
        else 10
    )


@pytest.mark.parametrize(
    "column,value,match",
    [
        ("essay_id", None, "missing"),
        ("set_id", None, "missing"),
        ("essay", None, "text"),
        ("human_score", None, "finite"),
        ("human_score", float("inf"), "finite"),
        ("human_score", 6, "finite"),
        ("ideas", "nonsense", "finite"),
        ("ideas", 0, "finite"),
    ],
)
def test_invalid_rows_are_rejected_without_silent_drops(custom, column, value, match):
    frame = pd.read_csv(custom.normalized_csv, dtype={column: object})
    frame.loc[0, column] = value
    frame.to_csv(custom.normalized_csv, index=False)
    with pytest.raises(ValueError, match=match):
        load_dataset(custom)


def test_missing_columns_duplicates_and_empty_subsets_are_rejected(custom):
    frame = pd.read_csv(custom.normalized_csv)
    frame.loc[1, "essay_id"] = frame.loc[0, "essay_id"]
    frame.to_csv(custom.normalized_csv, index=False)
    with pytest.raises(ValueError, match="duplicate essay IDs"):
        load_dataset(custom)
    frame.drop(columns="human_score").to_csv(custom.normalized_csv, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(custom)
    frame.to_csv(custom.normalized_csv, index=False)
    with pytest.raises(ValueError, match="subset is empty"):
        load_dataset(replace(custom, subset_id="missing"))


def test_calibration_is_deterministic_disjoint_and_configurable(custom):
    first = build_setup(custom, False, sample_size=200)
    pool_ids = {s.essay_id for s in first.calibration_pool}
    eval_ids = {s.essay_id for s in first.sample.dataset.human_assessments}
    assert len(pool_ids) == 50 and len(eval_ids) == 200 and not pool_ids & eval_ids
    repeat = build_setup(custom, False, sample_size=200)
    assert [s.essay_id for s in first.calibration_pool] == [
        s.essay_id for s in repeat.calibration_pool
    ]
    frame = pd.read_csv(custom.normalized_csv, dtype={"set_id": "string"}).sample(
        frac=1, random_state=7
    )
    frame.to_csv(custom.normalized_csv, index=False)
    shuffled = build_setup(custom, False, sample_size=200)
    assert eval_ids == {s.essay_id for s in shuffled.sample.dataset.human_assessments}
    smaller = build_setup(replace(custom, calibration_per_score=2), False, sample_size=200)
    assert len({s.essay_id for s in smaller.calibration_pool}) == 10
    reseeded = build_setup(replace(custom, seed=99), False, sample_size=200)
    assert pool_ids != {s.essay_id for s in reseeded.calibration_pool}


def test_calibration_cache_tracks_text_changes(custom):
    rubric = load_rubric(custom.rubric_path, custom.label)
    frame = load_dataset(custom)
    first = load_calibration_pool(custom, frame, rubric)
    changed = frame.copy()
    changed["essay"] = "Changed text " + changed["essay"]
    second = load_calibration_pool(custom, changed, rubric)
    assert {s.essay_id for s in first} == {s.essay_id for s in second}
    assert all(s.submission.text.startswith("Changed text") for s in second)
    assert len(list(custom.calibration_root.glob("*.csv"))) == 2


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_nonpositive_or_noninteger_sample_counts_are_rejected(custom, count):
    with pytest.raises(ValueError, match="positive integer"):
        build_setup(custom, False, sample_size=count)


def test_requested_ids_must_exist_and_not_leak_calibration(custom):
    with pytest.raises(ValueError, match="missing from evaluation"):
        build_setup(custom, False, essay_ids=["missing"])
    with pytest.raises(ValueError, match="nonempty"):
        build_setup(custom, False, essay_ids=[])
    first = build_setup(custom, False, sample_size=2)
    with pytest.raises(ValueError, match="belong to calibration"):
        build_setup(custom, False, essay_ids=[first.calibration_pool.human_assessments[0].essay_id])


def test_retrieval_cap_seed_and_zero_counts_are_respected(tmp_path):
    rubric, pool = make_pool(tmp_path)
    config = RetrievalConfig(n_ex_per_score=4, max_examples=6)
    first = picks(rubric, pool, config)
    assert len(first) == 6 and {s.human_score for s in first} == {1, 2}
    assert first == picks(rubric, pool, config)
    assert first != picks(rubric, pool, replace(config, seed=43))
    assert len(picks(rubric, pool, replace(config, max_examples=2))) == 2
    assert picks(rubric, pool, replace(config, n_ex_per_score=0)) == []
    assert picks(rubric, pool, replace(config, max_examples=0)) == []


def test_random_does_not_import_optional_dense_or_lexical_packages(tmp_path, monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "sentence_transformers", "faiss", "rank_bm25"}:
            raise AssertionError(f"Unexpected optional import: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    rubric, pool = make_pool(tmp_path)
    config = RetrievalConfig(n_ex_per_score=1, max_examples=6)
    assert len(picks(rubric, replace(pool, index_path=""), config)) == 2


def test_missing_bm25_dependency_is_not_silently_replaced(tmp_path, monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "rank_bm25":
            raise ImportError("Not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    rubric, pool = make_pool(tmp_path)
    with pytest.raises(ImportError, match="rank-bm25"):
        picks(
            rubric,
            pool,
            RetrievalConfig(
                n_ex_per_score=1, max_examples=6, examples_selection=ExamplesSelection.LEXICAL
            ),
        )


def test_hybrid_weights_change_selection_and_are_part_of_cache_key(tmp_path, monkeypatch):
    rubric, pool = make_pool(tmp_path, (2,))
    monkeypatch.setattr(
        retriever,
        "_load_and_embed",
        lambda path, key, entries, config: (entries, np.array([[0.0, 1.0], [1.0, 0.0]])),
    )
    monkeypatch.setattr(
        retriever,
        "_get_model",
        lambda config: SimpleNamespace(encode=lambda *args, **kwargs: np.array([[1.0, 0.0]])),
    )
    monkeypatch.setattr(retriever, "_bm25_scores", lambda *args: np.array([1.0, 0.0]))
    config = RetrievalConfig(
        n_ex_per_score=1,
        examples_selection=ExamplesSelection.HYBRID,
        max_examples=1,
        hybrid_dense_weight=0,
        hybrid_bm25_weight=1,
    )
    assert picks(rubric, pool, config)[0].essay_id == "1-0"
    assert (
        picks(rubric, pool, replace(config, hybrid_dense_weight=1, hybrid_bm25_weight=0))[
            0
        ].essay_id
        == "1-1"
    )


def test_dense_cache_changes_with_text_and_embedding_model(tmp_path, monkeypatch):
    _, pool = make_pool(tmp_path, (2,))
    encoded = []

    def model(config):
        def encode(texts, **kwargs):
            encoded.append((config.embedding_model, texts, kwargs["batch_size"]))
            return np.ones((len(texts), 2), dtype=np.float32)

        return SimpleNamespace(encode=encode)

    monkeypatch.setattr(retrieval_loader, "_get_model", model)
    entries = pool.human_assessments
    config = RetrievalConfig(
        n_ex_per_score=1, max_examples=6, embedding_model="model-a", embedding_batch_size=7
    )
    loader = retrieval_loader._load_and_embed
    loader(pool.index_path, retrieval_loader._pool_hash(entries), entries, config)
    loader(pool.index_path, retrieval_loader._pool_hash(entries), entries, config)
    loader(
        pool.index_path,
        retrieval_loader._pool_hash(entries),
        entries,
        replace(config, embedding_model="model-b"),
    )
    entries = [
        replace(entry, submission=replace(entry.submission, text="New text")) for entry in entries
    ]
    loader(pool.index_path, retrieval_loader._pool_hash(entries), entries, config)
    assert len(encoded) == 3 and encoded[0][2] == 7
    assert len(list(tmp_path.glob("*.npy"))) == 3


@pytest.mark.parametrize(
    "changes",
    [
        {"max_examples": -1},
        {"n_ex_per_score": 1.2},
        {"hybrid_dense_weight": float("nan")},
        {"hybrid_dense_weight": 2},
        {"hybrid_dense_weight": 0.7, "hybrid_bm25_weight": 0.7},
        {"embedding_threads": 0},
        {"embedding_batch_size": 0},
    ],
)
def test_invalid_retrieval_settings_fail_early(changes):
    base = {"n_ex_per_score": 1, "max_examples": 6}
    with pytest.raises(ValueError):
        RetrievalConfig(**{**base, **changes})
