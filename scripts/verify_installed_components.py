"""Exercise the installed public component path without adding a checkout to the import path."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import yaml

from abaes_eval_fwk import api


class Events:
    def emit(self, event: api.ExecutionEvent) -> None:
        pass


class FirstCalibration:
    identifier = "smoke.split.first"
    config_model = api.ComponentConfig

    def apply(self, dataset, config):
        row = dataset.frame.iloc[0]
        return api.SplitAssignment(
            calibration=(
                api.SampleKey(
                    dataset_id=dataset.dataset_id,
                    set_id=str(row.set_id),
                    essay_id=str(row.essay_id),
                ),
            ),
            fingerprint="sha256:synthetic-first-row",
        )


async def verify(root: Path) -> dict:
    origins = {}
    for name in ("abaes_eval_fwk", "aes_example_extension"):
        origin = Path(importlib.import_module(name).__file__).resolve()
        assert origin.is_relative_to(Path(sys.prefix).resolve()), origin
        origins[name] = str(origin)
    registry = api.assemble()
    api.register_plugin(registry, "example_extension")
    rubric_document = {
        "assignment_type": "Essay",
        "prompt": "Explain",
        "score_ranges": {"overall": [0, 6]},
        "criteria": {"ideas": {"score_levels": []}, "style": {"score_levels": []}},
    }
    (root / "rubric.yaml").write_text(yaml.safe_dump(rubric_document), encoding="utf-8")
    (root / "essays.csv").write_text(
        "essay_id,set_id,essay,human_score\nshared,a,An essay on cats,1.5\n"
        "two,a,Another essay on cats,3.5\nshared,b,An essay on dogs,2.5\n"
        "two,b,Another essay on dogs,4.5\n",
        encoding="utf-8",
    )
    document = {
        "schema_version": 1,
        "id": "smoke",
        "loader": {
            "type": "aes.datasets.normalized_csv",
            "config": {
                "path": "essays.csv",
                "columns": {name: name for name in ("essay_id", "set_id", "essay")},
            },
        },
        "rubrics": {"a": "rubric.yaml", "b": "rubric.yaml"},
        "targets": {"overall": "human_score"},
    }
    manifest_path = root / "dataset.yaml"
    manifest_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    manifest = api.load_manifest(
        manifest_path, registered_loader_types=registry.component_ids("dataset_loader")
    )
    ctx = api.ComponentContext(run_id="installed-smoke", artifact_dir=root, emitter=Events())
    loader = registry.resolve(
        manifest.loader.type, api.NormalizedCsvLoaderConfig(manifest=manifest)
    ).build(ctx)
    bundle = await loader.load(api.DatasetLoadRequest(dataset_id=manifest.id), ctx)
    frame = pd.DataFrame(
        [
            {
                "set_id": submission.key.set_id,
                "essay_id": submission.key.essay_id,
                "human_score": bundle.reference_for(submission.key).overall,
            }
            for submission in bundle.submissions
        ]
    )
    policies = api.SplitPolicyRegistry()
    policies.register(FirstCalibration())
    assignment = api.generate_splits(
        api.SplitDataset.from_bundle(bundle, frame),
        evaluation=api.SplitConfig(type="aes.splits.quantile_rank", config={"sample_size": 1}),
        calibration=api.SplitConfig(type=FirstCalibration.identifier),
        registry=policies,
    )
    bundle = api.apply_split_assignment(bundle, assignment)
    rubrics = api.load_manifest_rubrics(manifest)
    predictions = []
    retriever = registry.resolve("example.retrieval.word_overlap").build(ctx)
    for key in api.evaluation_keys(bundle):
        submission = bundle.submissions[bundle.index_of(key)]
        rubric = rubrics[submission.rubric_id]
        batch = await retriever.retrieve(
            api.RetrievalRequest(
                submission=submission,
                rubric=rubric,
                max_examples=1,
                pool=api.permitted_reference_examples(bundle),
            ),
            ctx,
        )
        assert batch.examples and all(item.key in assignment.calibration for item in batch.examples)
        predictions.append(
            api.Prediction(
                key=key, raw_score=api.aggregate_criteria(rubric, [1.2, 1.8], api.NoOpPolicy())
            )
        )
    references = api.require_references(bundle, api.evaluation_keys(bundle))
    metric = (
        await registry.resolve("example.metric.mean_absolute_error")
        .build(ctx)
        .compute(
            api.MetricRequest(
                predictions=predictions,
                references=references,
                scale=api.ScaleIdentity(rubric_id=rubric.rubric_id, scale=rubric.scale),
                target=references[0].provenance,
            ),
            ctx,
        )
    )
    assert len(predictions) == 2 and metric.value == 2.5 and metric.n_samples == 2
    return {
        "python": sys.version.split()[0],
        "framework_version": version("abaes-eval-fwk"),
        "extension_version": version("aes-example-extension"),
        "origins": origins,
        "prediction_count": len(predictions),
        "metric": metric.model_dump(mode="json"),
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="mas-installed-components-") as directory:
        print(json.dumps(asyncio.run(verify(Path(directory))), indent=2))
