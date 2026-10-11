"""Execute one configured experiment with input fingerprints and inspectable run artifacts."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

from abaes_eval_fwk.config.benchmark_config import BenchmarkConfig
from abaes_eval_fwk.config.experiment_spec import PROCEDURES, ExperimentSpec, experiment_from_dict
from abaes_eval_fwk.configuration import (
    dataset_configs,
    model_config,
    retrieval_config,
    sampling_config,
    select_experiments,
)
from abaes_eval_fwk.utils.extensions import is_reference, module_of


def _rubric_payload(rubric: object) -> dict:
    """Serialize a rubric for a run snapshot, dropping the contract-only fields."""
    excluded = {"version", "score_policy", "aggregation"}
    if hasattr(rubric, "model_dump"):
        return rubric.model_dump(mode="json", exclude=excluded)
    if is_dataclass(rubric) and not isinstance(rubric, type):
        return {key: value for key, value in asdict(rubric).items() if key not in excluded}
    raise TypeError(f"Unsupported rubric type: {type(rubric).__name__}")


def _json_default(value: object) -> object:
    """Expand a dataclass (e.g. an embedded BenchmarkConfig) to a plain mapping; else stringify."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return str(value)


def write_json(path: Path, value: dict | list) -> None:
    """Atomically replace a UTF-8 JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hash() -> str:
    return _tree_hash(Path(__file__).parent)


def _tree_hash(root: Path) -> str:
    """Hash every Python source under a directory, so an imported helper counts as the component's."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _local_imports(path: Path, base: Path | None = None) -> set[str]:
    """The dotted module names a source file imports, including relative imports.

    Resolves relative imports (from .helper import x, from ..shared import y) relative to
    the source module's package. Absolute imports are returned as-is. Returns only module
    names that resolve within the base directory; external packages are excluded.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return set()

    # Compute the module package for this source file (for resolving relative imports)
    if base is None:
        module_package = None
    else:
        try:
            rel_path = path.resolve().relative_to(base.resolve())
            # For __init__.py files, the package is the directory itself
            # For other files, the package is the parent directory
            if path.name == "__init__.py":
                # __init__.py in foo/bar/__init__.py -> package is foo.bar
                pkg_path = rel_path.parent
            else:
                # selector.py in foo/selector.py -> package is foo
                pkg_path = rel_path.parent
            # Convert path to module notation: foo/bar -> foo.bar
            module_package = (
                str(pkg_path).replace("/", ".").replace("\\", ".") if pkg_path != Path(".") else ""
            )
        except ValueError:
            # path is not relative to base
            module_package = None

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # Absolute imports: import foo, import foo.bar
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                # Absolute relative imports: from foo import bar, from foo.bar import baz
                names.add(node.module)
                names.update(f"{node.module}.{alias.name}" for alias in node.names)
            elif node.level > 0 and module_package is not None:
                # Relative imports: from . import x, from ..y import z
                # Go up (level-1) levels from the current module's package
                pkg_parts = module_package.split(".") if module_package else []
                # level=1 means current package, level=2 means parent, etc.
                base_parts = pkg_parts[: max(0, len(pkg_parts) - (node.level - 1))]

                if node.module:
                    # from ..foo import bar -> resolve 'foo' relative to parent
                    resolved = (
                        ".".join(base_parts + node.module.split(".")) if base_parts else node.module
                    )
                    names.add(resolved)
                    names.update(f"{resolved}.{alias.name}" for alias in node.names)
                else:
                    # from . import foo, from .. import bar -> just the package names
                    resolved = ".".join(base_parts) if base_parts else ""
                    if resolved:
                        names.update(f"{resolved}.{alias.name}" for alias in node.names)

    return names


def _extension_sources(base: Path, module: str) -> list[Path] | None:
    """Every source file an extension is made of: its entry point plus what that code imports.

    Resolving the imports matters because an entry point is often a thin registration shim; the
    helper that actually implements retrieval is a sibling module, and a change there must move
    the fingerprint too. Only names resolving under the study directory are followed, so a
    third-party package never widens the bundle.
    """
    relative = module.replace(".", "/")
    entry = next(
        (c for c in (base / relative, base / f"{relative}.py") if c.is_dir() or c.is_file()),
        None,
    )
    if entry is None:
        return None
    sources: list[Path] = []
    pending = [entry] if entry.is_file() else sorted(entry.rglob("*.py"))
    seen: set[Path] = set()
    while pending:
        current = pending.pop()
        resolved = current.resolve()
        if resolved in seen or not current.is_file():
            continue
        seen.add(resolved)
        sources.append(current)
        # Include package __init__.py files in the fingerprint
        if current.name != "__init__.py":
            init_file = current.parent / "__init__.py"
            if init_file.is_file() and init_file.resolve() not in seen:
                pending.append(init_file)
        # Resolve imports with base directory context for relative imports
        for imported in _local_imports(current, base):
            candidate = base / imported.replace(".", "/")
            for target in (candidate, candidate.with_suffix(".py"), candidate / "__init__.py"):
                if target.is_file() and target.resolve() not in seen:
                    pending.append(target)
    return sorted(sources)


def _extension_hash(base: Path, module: str) -> str | None:
    """One extension's content digest, over its entry point and every local source it imports."""
    sources = _extension_sources(base, module)
    if sources is None:
        return None
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.resolve().relative_to(base.resolve())).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _extension_references(config: BenchmarkConfig) -> dict[str, str]:
    """Every reader-supplied `module:function` this configuration reaches, as reference -> the
    setting that named it.

    One registry, so a new extension point becomes part of a run's provenance by being declared
    here rather than by growing another bespoke hashing function. The label is only for error
    messages; the reference is what gets hashed, deduplicated across the experiments that share it.
    """
    references: dict[str, str] = {}
    experiments = select_experiments(config)
    for experiment in experiments:
        selector = retrieval_config(config, experiment).selector
        if is_reference(selector):
            references[selector] = f"Experiment {experiment.name!r} retrieval.strategy"
        adapter = model_config(config, experiment).adapter_factory
        if is_reference(adapter):
            references[adapter] = f"Model profile {experiment.model!r} adapter_factory"
    for sampling in [sampling_config(config), *(sampling_config(config, e) for e in experiments)]:
        for key in ("evaluation_strategy", "calibration_strategy"):
            strategy = getattr(sampling, key)
            if is_reference(strategy):
                references.setdefault(strategy, f"sampling.{key}")
    for dataset in dataset_configs(config):
        if is_reference(dataset.loader):
            references.setdefault(dataset.loader, f"Dataset {dataset.label!r} loader")
        # A rubric's aggregation callable is an input to every total score derived from criteria.
        # A missing rubric file is not silently tolerated anywhere else — dataset loading fails on
        # it moments later — so provenance does not need to raise for it a second time here.
        if Path(dataset.rubric_path).is_file():
            from abaes_eval_fwk.experiment.setup.rubric.rubric_loader import load_rubric

            aggregation = load_rubric(dataset.rubric_path, dataset.label).criteria_aggregation
            if is_reference(aggregation):
                references.setdefault(
                    str(aggregation), f"Rubric {dataset.label!r} criteria_aggregation"
                )
    for name, reference in config.metrics.custom.items():
        references.setdefault(reference, f"metrics.custom.{name}")
    return references


def _framework_root() -> Path:
    return Path(__file__).resolve().parent


def _installed_module_hash(module: str) -> str | None:
    """Hash an extension that is installed rather than kept beside the configuration.

    Returns the empty digest for a module inside this framework (the playground's own offline
    adapter is one): `source_sha256` already covers every framework source, so hashing it again
    would only make the fingerprint depend on the same bytes twice. A third-party install is
    hashed from its own files, without following its imports — bundling a dependency's entire
    tree is what `_extension_sources` deliberately avoids.
    """
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or spec.origin in {"built-in", "frozen"}:
        return None
    origin = Path(spec.origin).resolve()
    if origin.is_relative_to(_framework_root()):
        return ""
    roots = [Path(location) for location in (spec.submodule_search_locations or ())]
    sources = (
        sorted((path, root) for root in roots for path in root.rglob("*.py"))
        if roots
        else [(origin, origin.parent)]
    )
    digest = hashlib.sha256()
    for path, root in sources:
        # The path relative to the package root, so two same-named modules in different
        # subpackages cannot hash to the same bytes.
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _extension_hashes(config: BenchmarkConfig) -> dict[str, str]:
    """Hash the source behind every named extension, so a resume notices changed reader code."""
    base = config.base_dir
    if not base:
        return {}
    base_path = Path(base)
    hashes: dict[str, str] = {}
    for reference, label in sorted(_extension_references(config).items()):
        module = module_of(reference, label)
        digest = _extension_hash(base_path, module)
        if digest is None:
            digest = _installed_module_hash(module)
        if digest is None:
            raise ValueError(
                f"{label} names module {module!r}, which is neither importable nor present "
                f"under {base_path}; a resume could not detect a change to it"
            )
        if digest:
            hashes[reference] = digest
    return hashes


def environment(config: BenchmarkConfig | None = None) -> dict:
    """Record the client environment; remote server hardware is operator-supplied metadata."""
    packages = dict(
        sorted(
            (dist.metadata["Name"], dist.version)
            for dist in importlib.metadata.distributions()
            if dist.metadata.get("Name")
        )
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "source_sha256": _source_hash(),
        "extension_sha256": _extension_hashes(config) if config else {},
    }


def _snapshots(setups: list) -> list[dict]:
    snapshots = []
    for setup in setups:
        sample = list(setup.sample)
        pool = list(setup.calibration_pool)
        snapshots.append(
            {
                "dataset": setup.rubric.rubric_id,
                "rubric": _rubric_payload(setup.rubric),
                "evaluation_ids": sorted({item.essay_id for item in sample}),
                "calibration_ids": sorted({item.essay_id for item in pool}),
                "evaluation_sha256": hashlib.sha256(
                    json.dumps(
                        [
                            (
                                item.essay_id,
                                item.criterion.name,
                                item.human_score,
                                item.submission.text,
                            )
                            for item in sample
                        ],
                        ensure_ascii=False,
                    ).encode()
                ).hexdigest(),
                "calibration_sha256": hashlib.sha256(
                    json.dumps(
                        [
                            (
                                item.essay_id,
                                item.criterion.name,
                                item.human_score,
                                item.submission.text,
                            )
                            for item in pool
                        ],
                        ensure_ascii=False,
                    ).encode()
                ).hexdigest(),
            }
        )
    return snapshots


def _ensure_extensions_importable(config: BenchmarkConfig) -> None:
    """Put the config's own directory on sys.path, so a reader's `module:function` extension —
    which lives beside their config, not on the installed path — can be imported when it runs."""
    base = config.base_dir
    if not base:
        return
    base_path = str(Path(base).resolve())
    if base_path not in sys.path:
        sys.path.insert(0, base_path)


def build_experiment(config: BenchmarkConfig, spec: ExperimentSpec, reuse: dict):
    """Construct the original grading engine from independent experiment dimensions."""
    from abaes_eval_fwk.config.reuse_config import ReuseConfig
    from abaes_eval_fwk.core.models.experiment import (
        Experiment,
        ExperimentType,
        FinalAgentProcedure,
        ResearchQuestion,
    )
    from abaes_eval_fwk.experiment.setup.setup_builder import build_experiment as build_setups

    procedure = PROCEDURES[spec.procedure]
    experiment = Experiment(
        rq=ResearchQuestion(spec.rq),
        experiment_type=ExperimentType(spec.architecture),
        llm=model_config(config, spec),
        retrieval=retrieval_config(config, spec),
        prompts=config.prompts,
        final_agent_procedure=FinalAgentProcedure(name=procedure),
        description=spec.name,
    )
    sources = ReuseConfig(
        results_paths=tuple(reuse.get("results", ())),
        reviews_paths=tuple(reuse.get("reviews", ())),
        independent_path=reuse.get("independent", ""),
    )
    if sources.independent_path:
        from abaes_eval_fwk.experiment.setup.reuse_loader import source_model

        if source_model(sources.independent_path) != experiment.llm.model_name:
            raise ValueError("The independent assessment must come from the configured final model")
    datasets = dataset_configs(config, spec)
    sampling = sampling_config(config, spec)
    experiment = build_setups(
        experiment,
        datasets,
        sampling.sample_size,
        essay_ids=([str(i) for i in sampling.essay_ids] if sampling.essay_ids else None),
        reuse=sources,
        evaluation_strategy=sampling.evaluation_strategy,
        calibration_strategy=sampling.calibration_strategy,
    )
    if not all(len(list(setup.sample)) for setup in experiment.setups):
        raise ValueError("Every requested dataset must contain evaluation submissions")
    return experiment, sources, datasets


def prepare(config: BenchmarkConfig) -> Path:
    """Prepare reproducible calibration and evaluation splits without model requests or embeddings."""
    from abaes_eval_fwk.experiment.setup.setup_builder import build_setup

    _ensure_extensions_importable(config)
    sampling = sampling_config(config)
    setups = [
        build_setup(
            dataset,
            False,
            sampling.sample_size,
            essay_ids=[str(i) for i in sampling.essay_ids] if sampling.essay_ids else None,
            evaluation_strategy=sampling.evaluation_strategy,
            calibration_strategy=sampling.calibration_strategy,
        )
        for dataset in dataset_configs(config)
    ]
    snapshots = _snapshots(setups)
    identity = hashlib.sha256(
        json.dumps(snapshots, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    path = Path(config.output_dir) / "splits" / f"{identity}.json"
    write_json(path, {"sampling": asdict(sampling), "datasets": snapshots})
    return path


def _start_run(
    config: BenchmarkConfig,
    spec: ExperimentSpec,
    run_dir: Path,
    manifest: dict,
    previous: dict | None,
    snapshots: list[dict],
) -> None:
    """Verify resume provenance before replacing any saved run artifacts."""
    reuse = manifest["resolved_reuse"]
    inputs = {
        str(path): file_hash(path)
        for path in [
            *reuse.get("results", []),
            *reuse.get("reviews", []),
            *([reuse["independent"]] if reuse.get("independent") else []),
        ]
    }
    identity = {
        "spec": asdict(spec),
        "model": asdict(model_config(config, spec)),
        "retrieval": asdict(retrieval_config(config, spec)),
        "runtime": asdict(config.runtime),
        # The prompt text itself, not a flag naming it: an edited template asks the model for
        # something different, so a resume must reject it like any other changed input.
        "prompts": asdict(config.prompts),
        "sampling": asdict(sampling_config(config, spec)),
        "datasets": snapshots,
        "source_files": inputs,
        "source_sha256": manifest["environment"]["source_sha256"],
        # A reader's extension source is as much an input as the framework's own.
        "extension_sha256": manifest["environment"]["extension_sha256"],
        "packages": manifest["environment"]["packages"],
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()
    if previous and previous.get("fingerprint") != fingerprint:
        raise ValueError("Resume rejected: configuration, inputs, code, or dependencies changed")
    manifest.update({"fingerprint": fingerprint, "inputs": identity, "status": "running"})
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "splits.json", snapshots)


def execute(
    config: BenchmarkConfig, spec: dict, run_dir: Path, reuse: dict, resume: bool = False
) -> dict:
    """Run an experiment in its own process and retain failed checkpoints for explicit resume."""
    experiment_spec = experiment_from_dict(spec)
    subdirs = config.subdirs
    # Metrics and conversation logs default to run-specific directories to prevent independent runs
    # from overwriting each other. If subdirs are configured, they are trusted to be run-specific
    # or to include run-identifying information (e.g., suite ID).
    metrics_dir = subdirs.get("metrics", str(run_dir / "metrics"))
    conversations_dir = subdirs.get("conversations", str(run_dir / "conversations"))
    # Results, reviews, failures, checkpoints and logs stay inside this run's own directory;
    # two declared experiments can share a derived Experiment.name and would overwrite each other.
    results_dir = str(run_dir / "results")
    # Record the actual paths used in this run so the manifest identifies all artifacts
    artifact_paths = {
        "metrics_dir": metrics_dir,
        "conversations_dir": conversations_dir,
    }
    manifest_path = run_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if resume else None
    if previous is not None and previous.get("experiment") != experiment_spec.name:
        raise ValueError("Resume directory belongs to a different experiment")
    # Before environment(): provenance hashes the extensions this configuration names, and a
    # module kept beside the configuration is only importable once its directory is on sys.path.
    _ensure_extensions_importable(config)
    manifest = {
        "experiment": experiment_spec.name,
        "status": "preparing",
        "started_at": datetime.now(UTC).isoformat(),
        "config": config,
        "resolved_reuse": reuse,
        "environment": environment(config),
        "artifact_paths": artifact_paths,
    }
    if previous:
        manifest["previous_attempts"] = [
            *previous.get("previous_attempts", []),
            {key: previous.get(key) for key in ("status", "started_at", "finished_at", "error")},
        ]
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "run.log", encoding="utf-8"),
        ],
        force=True,
    )
    try:
        if experiment_spec.procedure == "pool_average":
            from abaes_eval_fwk.config.reuse_config import ReuseConfig
            from abaes_eval_fwk.experiment.eval.eval_adapter import evaluate, pool_average
            from abaes_eval_fwk.experiment.eval.metrics.compute_metrics import grading_records
            from abaes_eval_fwk.experiment.setup.setup_builder import build_setup

            datasets = dataset_configs(config, experiment_spec)
            needed = set(grading_records(Path(reuse["results"][0])))
            snapshots = _snapshots(
                [
                    build_setup(
                        dataset, False, reuse=ReuseConfig(results_paths=tuple(reuse["results"]))
                    )
                    for dataset in datasets
                    if dataset.label in needed
                ]
            )
            _start_run(config, experiment_spec, run_dir, manifest, previous, snapshots)
            csv_path = pool_average(
                [Path(p) for p in reuse["results"]],
                datasets,
                run_dir / "results" / f"{experiment_spec.name}_pool_average_results.csv",
                experiment_id=experiment_spec.name,
            )
            manifest.update(
                {
                    "results": str(csv_path),
                    "failures": None,
                    "reviews": None,
                    "source_files": {p: file_hash(p) for p in reuse["results"]},
                }
            )
            if config.runtime.compute_metrics:
                manifest["metrics"] = str(
                    evaluate(csv_path, datasets, Path(metrics_dir), config.metrics)[0]
                )
            manifest.update({"status": "complete", "finished_at": datetime.now(UTC).isoformat()})
            write_json(manifest_path, manifest)
            return manifest
        from abaes_eval_fwk.core.agents import base_agent
        from abaes_eval_fwk.core.agents.final_agent import AdjudicationProcedure
        from abaes_eval_fwk.core.grader.grader import Grader
        from abaes_eval_fwk.core.grader.grading_orchestrator import grade_dataframe
        from abaes_eval_fwk.experiment.eval.eval_adapter import evaluate
        from abaes_eval_fwk.utils.conversation_logging import configure_conversation_logging

        base_agent.MAX_ATTEMPTS = config.runtime.max_attempts
        base_agent.ALLOW_CONTEXT_RECOVERY = config.runtime.context_overflow == "truncate_examples"
        AdjudicationProcedure._DISAGREEMENT_THRESHOLD_POINTS = (
            experiment_spec.disagreement_threshold * 5
        )
        experiment, sources, datasets = build_experiment(config, experiment_spec, reuse)
        snapshots = _snapshots(experiment.setups)
        _start_run(config, experiment_spec, run_dir, manifest, previous, snapshots)
        # The declared name, not a constant: it is the token that keeps one experiment's
        # results, reviews, metrics and conversation logs from overwriting another's.
        run_id = experiment_spec.name
        if config.runtime.save_conversations:
            configure_conversation_logging(run_id, conversations_dir)
        with Grader(
            experiment,
            checkpoint_path=str(run_dir / "checkpoints" / "grader.sqlite"),
            reuse=sources,
            run_id=run_id,
            artifact_dir=run_dir,
        ) as grader:
            csv_path, failures = grade_dataframe(
                grader,
                max_workers=config.runtime.max_workers,
                run_id=run_id,
                results_dir=results_dir,
            )
        manifest.update({"results": csv_path, "failures": failures})
        review_files = sorted((run_dir / "results").glob("*_reviews.csv"))
        manifest["reviews"] = str(review_files[0]) if review_files else None
        if config.runtime.compute_metrics and not experiment.experiment_type.is_non_grading:
            metrics_path, _ = evaluate(Path(csv_path), datasets, Path(metrics_dir), config.metrics)
            manifest["metrics"] = str(metrics_path)
        manifest["status"] = "partial" if failures else "complete"
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        write_json(manifest_path, manifest)
        return manifest
    except Exception as exc:
        if previous and "fingerprint" not in manifest:
            raise
        manifest.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": datetime.now(UTC).isoformat(),
            }
        )
        write_json(manifest_path, manifest)
        raise
