"""Portable commands for configuring, running, and comparing essay-scoring experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from abaes_eval_fwk import __version__
from abaes_eval_fwk.config.benchmark_config import BenchmarkConfig
from abaes_eval_fwk.config.experiment_spec import ExperimentSpec
from abaes_eval_fwk.configuration import (
    config_from_dict,
    dataset_configs,
    load_config,
    select_experiments,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="abaes-eval-fwk", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create a runnable playground or research study")
    init.add_argument("directory", type=Path)
    init.add_argument(
        "--template",
        choices=["playground", "study", "simulation", "demo", "full"],
        default="playground",
    )
    for name, help_text in [
        ("validate", "Validate configuration without contacting an LLM"),
        ("prepare", "Prepare calibration pools and evaluation splits"),
        ("run", "Run experiments and their declared dependencies"),
        ("metrics", "Recompute metrics from a results CSV"),
        ("compare", "Compare paired results with Wilcoxon tests"),
        ("baseline", "Average prior assessments using rubric aggregation"),
        ("adjudication", "Count DA/RA outcomes and review tokens from prior assessments"),
    ]:
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("config", type=Path)
        sub.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="PATH=VALUE",
            help="Override a dotted YAML setting",
        )
        if name in {"validate", "run"}:
            sub.add_argument(
                "--experiment",
                action="append",
                help="Select an experiment by name; repeat to select several",
            )
        if name == "validate":
            sub.add_argument(
                "--check-data", action="store_true", help="Also validate local datasets and rubrics"
            )
        if name == "run":
            sub.add_argument(
                "--dry-run",
                action="store_true",
                help="Print the ordered plan; perform no model calls",
            )
            sub.add_argument(
                "--resume",
                type=Path,
                help="Resume one run directory with identical configuration and inputs",
            )
        if name == "metrics":
            sub.add_argument("results", type=Path)
        if name == "compare":
            sub.add_argument("baseline", type=Path)
            sub.add_argument("variant", type=Path)
            sub.add_argument("--allow-partial", action="store_true")
            sub.add_argument("--alpha", type=float, default=0.05)
        if name == "baseline":
            sub.add_argument("sources", type=Path, nargs="+")
        if name == "adjudication":
            sub.add_argument(
                "sources",
                type=Path,
                nargs="*",
                help="The pool's source results CSVs, as for baseline; DA outcomes need them",
            )
            sub.add_argument(
                "--reviews",
                type=Path,
                nargs="+",
                default=[],
                help="Reviews CSVs of the pool's sources; RA outcomes and review tokens need them",
            )
            sub.add_argument(
                "--threshold",
                type=float,
                default=0.2,
                help="DA disagreement threshold, as the experiment's disagreement_threshold",
            )
        if name in {"metrics", "compare", "baseline", "adjudication"}:
            sub.add_argument("--output", type=Path)
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("request", type=Path)
    return parser


def _reuse_paths(config: BenchmarkConfig, spec: ExperimentSpec, completed: dict) -> dict:
    def resolve(reference: str, kind: str) -> str:
        if reference.startswith("@"):
            source = completed[reference[1:]]
            if source.get("status") != "complete" or not source.get(kind):
                raise ValueError(f"{reference} has no complete {kind} artifact")
            return source[kind]
        path = (Path(config.base_dir) / reference).resolve()
        if not path.is_file():
            raise ValueError(f"Reuse file not found: {path}")
        return str(path)

    reuse = spec.reuse
    return {
        "results": [resolve(path, "results") for path in reuse.results_paths],
        "reviews": [resolve(path, "reviews") for path in reuse.reviews_paths],
        "independent": resolve(reuse.independent_path, "results") if reuse.independent_path else "",
    }


def _run(config: BenchmarkConfig, args: argparse.Namespace) -> int:
    from abaes_eval_fwk.runner import write_json

    specs = select_experiments(config, args.experiment)
    raw_by_name = {item["name"]: item for item in config.experiments}
    if args.resume:
        previous = json.loads((args.resume / "manifest.json").read_text(encoding="utf-8"))
        specs = [spec for spec in specs if spec.name == previous["experiment"]]
        if len(specs) != 1:
            raise ValueError(
                "Resume requires the original experiment in the supplied configuration"
            )
    if args.dry_run:
        for spec in specs:
            print(
                f"{spec.name}: {spec.rq} / {spec.architecture} / {spec.procedure} / model={spec.model} / reasoning={spec.reasoning or 'default'}"
            )
        return 0
    completed: dict[str, dict] = {}
    suite_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    suite = {"suite_id": suite_id, "config": config, "status": "running", "runs": []}
    suite_path = Path(config.output_dir) / "suites" / f"{suite_id}.json"
    write_json(suite_path, suite)
    for spec in specs:
        run_dir = (
            args.resume.resolve() if args.resume else Path(config.output_dir) / spec.name / suite_id
        )
        try:
            reuse = (
                previous["resolved_reuse"] if args.resume else _reuse_paths(config, spec, completed)
            )
            request_path = run_dir / "request.json"
            write_json(
                request_path,
                {
                    "config": config,
                    "spec": raw_by_name[spec.name],
                    "run_dir": str(run_dir),
                    "reuse": reuse,
                    "resume": bool(args.resume),
                },
            )
            process = subprocess.run(
                [sys.executable, "-m", "abaes_eval_fwk", "_worker", str(request_path)],
                check=False,
            )
            manifest_path = run_dir / "manifest.json"
            manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists()
                else {"status": "failed"}
            )
            suite["runs"].append(
                {
                    "experiment": spec.name,
                    "manifest": str(manifest_path),
                    "status": manifest["status"],
                }
            )
            if process.returncode or manifest["status"] != "complete":
                raise RuntimeError(f"{spec.name} did not complete; see {manifest_path}")
            completed[spec.name] = manifest
            write_json(suite_path, suite)
        except Exception as exc:
            suite.update({"status": "failed", "error": str(exc)})
            write_json(suite_path, suite)
            raise
    suite["status"] = "complete"
    write_json(suite_path, suite)
    print(f"Suite: {suite_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            from abaes_eval_fwk.templates import initialize

            print(initialize(args.directory, args.template))
            return 0
        if args.command == "_worker":
            from abaes_eval_fwk.runner import execute

            request = json.loads(args.request.read_text(encoding="utf-8"))
            request["run_dir"] = Path(request["run_dir"])
            request["config"] = config_from_dict(request["config"])
            manifest = execute(**request)
            return 0 if manifest["status"] == "complete" else 1
        config = load_config(args.config, args.set)
        # Every command can reach a reader's `module:function` — a dataset loader during
        # `validate --check-data`, a custom metric during `metrics` — so the project's own
        # directory joins sys.path here rather than only inside a run.
        from abaes_eval_fwk.runner import _ensure_extensions_importable

        _ensure_extensions_importable(config)
        if args.command == "validate":
            specs = select_experiments(config, args.experiment)
            if args.check_data:
                from abaes_eval_fwk.experiment.setup.dataset.dataset_loader import load_dataset

                for dataset in dataset_configs(config):
                    print(f"{dataset.label}: {len(load_dataset(dataset))} validated submissions")
            print(f"Valid configuration: {len(specs)} experiments, {len(config.datasets)} datasets")
            return 0
        if args.command == "prepare":
            from abaes_eval_fwk.runner import prepare

            print(prepare(config))
            return 0
        if args.command == "run":
            return _run(config, args)
        from abaes_eval_fwk.experiment.eval.eval_adapter import (
            compare,
            evaluate,
            pool_average,
            procedure_outcomes,
        )

        datasets = dataset_configs(config)
        output = args.output or Path(config.output_dir) / "analysis"
        if args.command == "metrics":
            print(evaluate(args.results, datasets, output, config.metrics)[0])
        elif args.command == "compare":
            baseline_path, first = evaluate(
                args.baseline,
                datasets,
                output.parent / "comparison_metrics" / "baseline",
                config.metrics,
            )
            variant_path, second = evaluate(
                args.variant,
                datasets,
                output.parent / "comparison_metrics" / "variant",
                config.metrics,
            )
            destination = args.output or output / "comparison.csv"
            comparison_path = compare(
                first,
                second,
                destination,
                config=config.metrics,
                allow_partial=args.allow_partial,
                alpha=args.alpha,
            )
            print(baseline_path)
            print(variant_path)
            print(comparison_path)
        elif args.command == "baseline":
            destination = args.output or output / "pool_average_results.csv"
            print(pool_average(args.sources, datasets, destination))
        elif args.command == "adjudication":
            if not args.sources and not args.reviews:
                raise ValueError("adjudication needs pool source CSVs, --reviews, or both")
            destination = args.output or output / "adjudication"
            for path in procedure_outcomes(
                args.sources, args.reviews, datasets, destination, threshold=args.threshold
            ):
                print(path)
        return 0
    except (ValueError, TypeError, OSError, RuntimeError, KeyError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
