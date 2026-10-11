"""Exercise the public configuration and CLI, including complete offline workflows."""

import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pytest
import yaml

from abaes_eval_fwk.configuration import load_config, select_experiments
from abaes_eval_fwk.templates import initialize


def _as_mutable(config):
    """Convert a loaded `BenchmarkConfig` into a plain dict, stripped of loader-injected fields."""
    payload = asdict(config)
    payload.pop("config_path", None)
    payload.pop("base_dir", None)
    return payload


def command(*args, timeout=120):
    environment = os.environ.copy()
    source_path = str(Path(__file__).parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (source_path, environment.get("PYTHONPATH", "")) if path
    )
    return subprocess.run(
        [sys.executable, "-m", "abaes_eval_fwk", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
    )


def modify(path, change):
    try:
        config = load_config(path)
    except ValueError:
        config = yaml.safe_load(path.read_text())
    if is_dataclass(config):
        config = _as_mutable(config)
    else:
        config.pop("_config_path", None)
        config.pop("_base_dir", None)
    change(config)
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def test_demo_and_full_templates_validate_without_model_calls(tmp_path):
    demo = initialize(tmp_path / "playground", "playground")
    full = initialize(tmp_path / "study", "study")
    main = yaml.safe_load(full.read_text())
    assert set(main) == {"schema_version", "components"}
    assert set(main["components"]) == {
        "paths",
        "datasets",
        "sampling",
        "retrieval",
        "runtime",
        "models",
        "experiments",
        "prompts",
    }
    assert {f"rq{index}.yaml" for index in range(1, 5)} <= {
        path.name for path in (full.parent / "config").glob("rq*.yaml")
    }
    assert len(load_config(demo).experiments) == 31
    config = load_config(full)
    assert len(config.datasets) == 6
    ordered = select_experiments(config, ["rq4-criterion-review_based"])
    assert ordered[-1].name == "rq4-criterion-review_based"
    assert len(ordered) == 5
    result = command("run", full, "--experiment", "rq3-holistic-on", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert not (full.parent / "runs").exists()


@pytest.mark.parametrize(
    "assignment",
    [
        "sampling.sample_size=0",
        "runtime.max_workers=0",
        "retrieval.max_examples=-1",
        "sampling.typo=3",
        "models.a.streaming=123",
    ],
)
def test_invalid_overrides_fail(tmp_path, assignment):
    path = initialize(tmp_path / "playground", "playground")
    with pytest.raises((ValueError, TypeError)):
        load_config(path, [assignment])


def test_dependency_cycles_and_unknown_datasets_fail(tmp_path):
    path = initialize(tmp_path / "playground", "playground")
    # A self-reference is the smallest cycle; the last experiment's own name keeps this test
    # independent of which experiments the playground declares.
    modify(
        path,
        lambda cfg: cfg["experiments"][-1]["reuse"].update(
            {"independent": "@" + cfg["experiments"][-1]["name"]}
        ),
    )
    with pytest.raises(ValueError, match="Cyclic"):
        load_config(path)
    fresh = initialize(tmp_path / "unknown" / "playground", "playground")
    modify(fresh, lambda cfg: cfg["experiments"][0].update({"datasets": ["not-a-dataset"]}))
    with pytest.raises(ValueError, match="Unknown experiment datasets"):
        load_config(fresh)


def test_controlled_comparison_rejects_an_unintended_difference(tmp_path):
    path = initialize(tmp_path / "playground", "playground")
    config = load_config(path)
    criterion = next(item for item in config.experiments if item["name"] == "rq2-a-criterion")
    criterion["model"] = "b"
    path.write_text(yaml.safe_dump(_as_mutable(config), sort_keys=False))
    with pytest.raises(ValueError, match="effective differences=.*model"):
        load_config(path)


def test_reader_sampling_strategies_run_and_enter_split_configuration(tmp_path):
    path = initialize(tmp_path / "playground", "playground")
    module = path.parent / "study_sampling.py"
    module.write_text(
        """from pathlib import Path
from abaes_eval_fwk.experiment.setup.calibration.calibration_generator import generate_calibration_pool
from abaes_eval_fwk.experiment.setup.sample.data_sampler import stratified_quantile_rank_sample

def calibration(dataset, frame, rubric):
    Path(__file__).with_name("calibration.called").write_text("yes")
    return generate_calibration_pool(dataset, frame, rubric)

def evaluation(frame, n_per_set):
    Path(__file__).with_name("evaluation.called").write_text("yes")
    return stratified_quantile_rank_sample(frame, n_per_set)
"""
    )
    modify(
        path,
        lambda cfg: cfg["sampling"].update(
            {
                "evaluation_strategy": "study_sampling:evaluation",
                "calibration_strategy": "study_sampling:calibration",
            }
        ),
    )
    prepared = command("prepare", path)
    assert prepared.returncode == 0, prepared.stderr
    assert (path.parent / "calibration.called").read_text() == "yes"
    assert (path.parent / "evaluation.called").read_text() == "yes"
    split = json.loads(Path(prepared.stdout.strip()).read_text())
    assert split["sampling"]["evaluation_strategy"] == "study_sampling:evaluation"


def test_end_to_end_suite_metrics_review_reuse_and_resume(tmp_path):
    path = initialize(tmp_path / "playground", "playground")
    # The 31-experiment offline matrix completes in about three minutes on this machine, so the
    # timeout is set from that measured completion time, not to hide a hang.
    result = command("run", path, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    suite_path = next((path.parent / "runs" / "suites").glob("*.json"))
    suite = json.loads(suite_path.read_text())
    assert suite["status"] == "complete"
    assert len(suite["runs"]) == 31
    manifests = {
        run["experiment"]: json.loads(Path(run["manifest"]).read_text()) for run in suite["runs"]
    }
    # Review-only runs judge prior assessments and settle no score, so their results carry no score.
    review_only = {"rq4-holistic-review", "rq4-criterion-review"}
    criterion_rows = {
        "rq2-a-criterion",
        "rq2-b-criterion",
        "rq2-final-criterion",
        "rq3-criterion-reasoning",
    }
    subset_rows = {"rq2-a-holistic-one-dataset", "rq2-a-holistic-small-sample"}
    for name, manifest in manifests.items():
        assert manifest["status"] == "complete"
        with Path(manifest["results"]).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            rows = list(reader)
        if name in review_only:
            assert "pred_score_rounded" not in fields, name
            continue
        assert "pred_score_rounded" in fields, name
        expected = 24 if name in criterion_rows else 6 if name in subset_rows else 12
        assert len(rows) == expected, name
    final = manifests["rq4-criterion-review_based"]
    reviews_path = Path(final["reviews"])
    assert len(list(csv.DictReader(reviews_path.open()))) == 48
    run_dir = reviews_path.parent.parent
    resumed = command("run", path, "--resume", run_dir, timeout=900)
    assert resumed.returncode == 0, resumed.stderr
    assert len(list(csv.DictReader(reviews_path.open()))) == 48
    rejected = command("run", path, "--resume", run_dir, "--set", "retrieval.max_examples=1")
    assert rejected.returncode != 0
    assert "Resume rejected" in rejected.stderr
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "complete"
    comparison = tmp_path / "comparison.csv"
    compared = command(
        "compare",
        path,
        manifests["rq2-a-holistic"]["results"],
        manifests["rq2-a-criterion"]["results"],
        "--output",
        comparison,
    )
    assert compared.returncode == 0, compared.stderr
    assert next(csv.DictReader(comparison.open()))["p-value"] == "1.0"
    reusable = _as_mutable(load_config(path))
    spec = next(e for e in reusable["experiments"] if e["name"] == "rq4-criterion-review_based")
    spec["reuse"] = {
        "results": [
            manifests["rq2-a-criterion"]["results"],
            manifests["rq2-b-criterion"]["results"],
        ],
        "independent": manifests["rq2-final-criterion"]["results"],
        "reviews": [str(reviews_path)],
    }
    # The control baseline is not part of this single-experiment replay, so it is left out.
    spec.pop("control", None)
    reusable["experiments"] = [spec]
    reusable["output_dir"] = "replayed"
    replay_config = path.parent / "replay.yaml"
    replay_config.write_text(yaml.safe_dump(reusable))
    replayed = command("run", replay_config, timeout=900)
    assert replayed.returncode == 0, replayed.stderr


def test_pool_average_resume_checks_provenance_before_replacing_outputs(tmp_path):
    path = initialize(tmp_path / "playground", "playground")
    result = command("run", path, "--experiment", "rq4-holistic-pool_average")
    assert result.returncode == 0, result.stderr
    manifest_path = next(
        (path.parent / "runs" / "rq4-holistic-pool_average").glob("*/manifest.json")
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["fingerprint"]
    snapshots = json.loads((manifest_path.parent / "splits.json").read_text())
    results_path = Path(manifest["results"])
    rows = list(csv.DictReader(results_path.open()))
    assert sorted(i for snap in snapshots for i in snap["evaluation_ids"]) == sorted(
        row["essay_id"] for row in rows
    )
    assert all(snap["calibration_ids"] == [] for snap in snapshots)
    resumed = command("run", path, "--resume", manifest_path.parent)
    assert resumed.returncode == 0, resumed.stderr
    saved = {
        artifact: artifact.read_bytes()
        for artifact in (manifest_path, results_path, manifest_path.parent / "splits.json")
    }

    def reject(*args):
        rejected = command("run", path, "--resume", manifest_path.parent, *args)
        assert rejected.returncode != 0
        assert "Resume rejected" in rejected.stderr
        assert all(artifact.read_bytes() == contents for artifact, contents in saved.items())

    reject("--set", "runtime.max_workers=7")
    source = Path(manifest["resolved_reuse"]["results"][0])
    original = source.read_bytes()
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["pred_score_raw"] = rows[0]["pred_score_rounded"] = "4"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    reject()
    source.write_bytes(original)
    rubric_path = path.parent / "demo-rubric.yaml"
    rubric = yaml.safe_load(rubric_path.read_text())
    rubric["prompt"] = "Explain how students could improve their local park."
    rubric_path.write_text(yaml.safe_dump(rubric))
    reject()


def test_live_provider_protocol_with_mock_http():
    import httpx
    from langchain_openai import ChatOpenAI

    from abaes_eval_fwk.config.llm_config import LLMConfig
    from abaes_eval_fwk.core.agents.schemas.results import GradingResponse
    from abaes_eval_fwk.core.llm.llm_loader import structured_output

    requests = []

    def reply(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "future-model-id",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-test",
                                    "type": "function",
                                    "function": {
                                        "name": "GradingResponse",
                                        "arguments": json.dumps(
                                            {
                                                "score": 3,
                                                "confidence": 0.8,
                                                "justification": "Supported by the rubric.",
                                            }
                                        ),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
            },
        )

    client = ChatOpenAI(
        model="future-model-id",
        api_key="test-only",
        base_url="https://example.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(reply)),
        max_retries=0,
    )
    object.__setattr__(
        client,
        "_abaes_eval_fwk_config",
        LLMConfig(
            model_name="future-model-id",
            provider="openai",
            max_tokens=4096,
            timeout=120.0,
            streaming=False,
            tool_choice="required",
            structured_output_method="function_calling",
        ),
    )
    response = structured_output(client, GradingResponse).invoke(
        "Grade this installation-test essay."
    )
    assert response["parsed"].score == 3
    assert requests[0]["model"] == "future-model-id"
    assert requests[0]["tools"][0]["function"]["name"] == "GradingResponse"
    assert response["raw"].usage_metadata["total_tokens"] == 50


def test_partial_run_resumes_only_failed_submissions(tmp_path, monkeypatch):
    path = initialize(tmp_path / "playground", "playground")
    adapter = tmp_path / "fault_adapter.py"
    adapter.write_text(
        """import os
from pathlib import Path
from langchain_core.runnables import RunnableLambda
from abaes_eval_fwk.playground import DemoModel

class Model(DemoModel):
    def with_structured_output(self, schema, **kwargs):
        chain = super().with_structured_output(schema, **kwargs)
        def respond(messages):
            with Path(os.environ["TEST_CALLS"]).open("a") as handle:
                handle.write("call\\n")
            if os.environ.get("TEST_TRANSIENT_FAILURE") and "synthetic example 1 of set demo" in str(messages):
                raise RuntimeError("simulated transient failure")
            return chain.invoke(messages)
        return RunnableLambda(respond)

def create_model(config):
    return Model(config)
"""
    )
    modify(
        path,
        lambda cfg: cfg.update(
            {
                # A retrieval policy supplies examples, which the fault adapter keys on. Its control
                # baseline is left out, since the baseline is not part of this single-experiment run.
                "experiments": [
                    {
                        key: value
                        for key, value in next(
                            e for e in cfg["experiments"] if e["name"] == "rq1-random"
                        ).items()
                        if key != "control"
                    }
                ],
                "runtime": {"max_workers": 1, "max_attempts": 1},
                "sampling": {"sample_size": 2, "seed": 42},
                "calibration": {"max_per_score": 1},
            }
        ),
    )
    modify(
        path,
        lambda cfg: cfg["models"]["a"].update({"adapter_factory": "fault_adapter:create_model"}),
    )
    source_path = str(Path(__file__).parents[1] / "src")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + source_path)
    calls = tmp_path / "calls.txt"
    monkeypatch.setenv("TEST_CALLS", str(calls))
    monkeypatch.setenv("TEST_TRANSIENT_FAILURE", "1")
    failed = command("run", path)
    assert failed.returncode != 0
    manifest_path = next((path.parent / "runs" / "rq1-random").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "partial", failed.stderr
    # Four evaluation essays (sample_size per dataset, two datasets), one call each. The trigger
    # text also appears in the second dataset's essays, so two submissions fail and two complete.
    assert len(calls.read_text().splitlines()) == 4
    monkeypatch.delenv("TEST_TRANSIENT_FAILURE")
    resumed = command("run", path, "--resume", manifest_path.parent)
    assert resumed.returncode == 0, resumed.stderr
    # The resume re-grades only the two failed submissions: two new calls, none for the completed.
    assert len(calls.read_text().splitlines()) == 6
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    assert len(list(csv.DictReader(Path(manifest["results"]).open()))) == 4
