"""Dataset configuration: identifies a dataset subset and derives its file paths."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from importlib.resources import files
from pathlib import Path
from typing import Any, Self

from ..utils.extensions import split_reference
from ..utils.paths import resolve_path


class DatasetFamily(str, Enum):
    """Dataset families — each may contain multiple prompt sets."""

    ASAP_AES = "ASAP AES"
    ASAP_2_0 = "ASAP 2.0"
    ELLIPSE = "ELLIPSE"
    CUSTOM = "custom"


class DatasetNames(StrEnum):
    """Every individually registered dataset subset, carrying its family and set number. Valued as its label, e.g."""

    _family: DatasetFamily
    _set: int | None

    ASAP_AES_SET_1 = (DatasetFamily.ASAP_AES, 1)
    ASAP_AES_SET_4 = (DatasetFamily.ASAP_AES, 4)
    ASAP_AES_SET_7 = (DatasetFamily.ASAP_AES, 7)
    ASAP_2_SET_1 = (DatasetFamily.ASAP_2_0, 1)
    ASAP_2_SET_6 = (DatasetFamily.ASAP_2_0, 6)
    ELLIPSE = (DatasetFamily.ELLIPSE, None)

    @property
    def family(self) -> DatasetFamily:
        return self._family

    @property
    def set(self) -> int | None:
        return self._set

    def __new__(cls, family: DatasetFamily, set: int | None) -> Self:
        label = family.value if set is None else f"{family.value} Set {set}"
        obj = str.__new__(cls, label)
        obj._value_ = label
        obj._family = family
        obj._set = set
        return obj

    @property
    def label(self) -> str:
        return self.value


_BASE_COLUMNS = ["essay_id", "set_id", "essay", "human_score"]


@dataclass(frozen=True)
class DatasetConfig:
    """Specification and file paths for an essay dataset subset."""

    dataset_name: DatasetNames | str
    data_root: Path = field(default_factory=lambda: Path("data"))
    calibration_root: Path | None = None
    cache_root: Path | None = None
    normalized_csv: Path | None = None
    rubric_file: Path | None = None
    raw_data_file: Path | None = None
    subset_id: str | None = None
    seed: int = 42
    calibration_per_score: int | None = None
    prompt_name: str | None = None
    column_map: dict[str, str] = field(default_factory=dict)
    base_columns: list[str] = field(default_factory=lambda: list(_BASE_COLUMNS))
    criteria_columns: list[str] = field(default_factory=list)
    raw_filenames: list[str] = field(default_factory=list)
    raw_columns: list[str] = field(default_factory=list)
    calibration_target_n: int | None = None
    # A reader's `module:function` returning this dataset's frame, for a corpus no built-in
    # family reads. It replaces the file-reading step only: the frame it returns goes through
    # exactly the same validation as a built-in family's, against this dataset's own rubric.
    loader: str = ""

    def __post_init__(self) -> None:
        for name in (
            "data_root",
            "calibration_root",
            "cache_root",
            "normalized_csv",
            "rubric_file",
            "raw_data_file",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser().resolve())
        if self.calibration_root is None:
            object.__setattr__(self, "calibration_root", self.data_root / "calibration")
        if self.cache_root is None:
            object.__setattr__(self, "cache_root", self.data_root / "cache")
        if self.loader:
            split_reference(self.loader, f"Dataset {self.label!r} loader")
            if self.rubric_file is None:
                raise ValueError(f"Dataset {self.label!r} has a loader and so requires a rubric")
        elif not isinstance(self.dataset_name, DatasetNames) and self.normalized_csv is None:
            raise ValueError("Custom datasets require normalized_csv and rubric_file")
        if self.normalized_csv is not None and self.rubric_file is None:
            raise ValueError("Normalized CSV datasets require a rubric YAML file")
        if not self.label.strip():
            raise ValueError("Dataset name cannot be empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        count = self.calibration_per_score
        if count is None:
            count = self.calibration_target_n if self.calibration_target_n is not None else 10
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("calibration_per_score must be a positive integer")
        object.__setattr__(self, "calibration_per_score", count)
        object.__setattr__(self, "calibration_target_n", count)

    @property
    def family(self) -> DatasetFamily:
        if self.loader or self.normalized_csv is not None:
            return DatasetFamily.CUSTOM
        # __post_init__ rejects a plain-string name unless it has a CSV or loader, so here it is a DatasetNames.
        assert isinstance(self.dataset_name, DatasetNames)
        return self.dataset_name.family

    @property
    def set(self) -> int | str | None:
        if self.family == DatasetFamily.CUSTOM:
            return self.subset_id
        assert isinstance(self.dataset_name, DatasetNames)
        return self.dataset_name.set

    @property
    def label(self) -> str:
        """Human-readable and rubric identifier, e.g. 'ASAP AES Set 1'."""
        return str(self.dataset_name)

    @property
    def slug(self) -> str:
        """Filename stem shared by this set's rubric, calibration pool, and index."""
        return re.sub(r"[^a-z0-9_.-]+", "_", self.label.lower()).strip("._") or "dataset"

    @property
    def dataset_folder_path(self) -> str:
        return str(self.data_root / "datasets" / self.family.value.replace(" ", "_"))

    @property
    def raw_data_path(self) -> str:
        explicit = self.normalized_csv or self.raw_data_file
        if explicit:
            return str(explicit)
        # A reader's loader need not read a file at all, so it is the one case with no path.
        if self.loader and not self.raw_filenames:
            return ""
        return str(Path(self.dataset_folder_path) / self.raw_filenames[0])

    @property
    def rubric_path(self) -> str:
        return str(
            self.rubric_file
            or files("abaes_eval_fwk").joinpath("resources", "rubrics", f"{self.slug}.yaml")
        )

    @property
    def calibration_pool_path(self) -> str:
        payload = [
            self.label,
            self.raw_data_path,
            self.rubric_path,
            self.seed,
            self.calibration_per_score,
            self.subset_id,
        ]
        key = hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]
        assert self.calibration_root is not None  # defaulted in __post_init__
        return str(self.calibration_root / f"{self.slug}.{key}.csv")

    @property
    def calibration_index_path(self) -> str:
        assert self.cache_root is not None  # defaulted in __post_init__
        return str(self.cache_root / "retrieval" / f"{self.slug}.faiss")


def _load_dataset_spec_from_config(dataset_name: str) -> dict[str, Any] | None:
    """Load dataset specification from datasets.yaml for a given dataset."""
    import yaml

    from abaes_eval_fwk.utils.resources import config_resource

    config = yaml.safe_load(config_resource("study/datasets.yaml").read_text(encoding="utf-8"))
    for dataset in config.get("datasets", []):
        if dataset.get("name") == dataset_name:
            return dataset
    return None


def get_dataset_config(
    name: DatasetNames | str,
    *,
    data_root: str | Path | None = None,
    calibration_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    seed: int = 42,
    calibration_per_score: int | None = None,
    rubric_path: str | Path | None = None,
    raw_data_path: str | Path | None = None,
    raw_filenames: list[str] | None = None,
) -> DatasetConfig:
    """Construct the DatasetConfig spec for one DatasetName directly."""
    from abaes_eval_fwk.utils.resources import rubric_resource

    name = DatasetNames(name)  # type: ignore[call-arg, arg-type]  # value lookup, not the (family, set) constructor
    spec = _load_dataset_spec_from_config(str(name))
    if not spec:
        raise ValueError(f"Unknown dataset in configuration: {name}")

    # Build raw_filenames from spec if not provided
    if raw_filenames is None:
        raw_filenames = []
        if "file" in spec:
            raw_filenames.append(spec["file"])
        if "per_rater_file" in spec:
            raw_filenames.append(spec["per_rater_file"])

    # Resolve rubric path from spec relative to package if not explicitly provided
    final_rubric_path = rubric_path
    if final_rubric_path is None and "rubric" in spec:
        rubric_spec_path = spec["rubric"]
        final_rubric_path = rubric_resource(rubric_spec_path)

    common = dict(
        calibration_root=calibration_root,
        cache_root=cache_root,
        seed=seed,
        calibration_per_score=calibration_per_score,
        rubric_file=final_rubric_path,
        raw_data_file=raw_data_path,
        column_map=spec.get("column_map", {}),
        criteria_columns=spec.get("criteria_columns", []),
        raw_columns=spec.get("raw_columns", []),
        raw_filenames=raw_filenames,
        calibration_target_n=spec.get("calibration_target_n"),
    )
    if data_root is not None:
        common["data_root"] = data_root

    # Add optional fields if present in spec
    if "prompt_name" in spec:
        common["prompt_name"] = spec["prompt_name"]

    return DatasetConfig(dataset_name=name, **common)


def dataset_from_dict(
    spec: dict[str, Any] | str,
    *,
    base_dir: str | Path | None = None,
    defaults: dict[str, Any] | None = None,
) -> DatasetConfig:
    """Resolve a built-in or normalized CSV dataset specification relative to its configuration file."""
    if not isinstance(spec, (str, dict)):
        raise ValueError("Each dataset specification must be a name or mapping")
    values = {**(defaults or {}), **({"name": spec} if isinstance(spec, str) else spec)}
    for primary, alias in (
        ("name", "dataset_name"),
        ("csv", "normalized_csv"),
        ("rubric", "rubric_path"),
    ):
        if primary in values and alias in values:
            raise ValueError(f"Configure only one of {primary} and {alias}")
    name = values.pop("name", values.pop("dataset_name", None))
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Each dataset requires a nonempty string name")
    base = Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd()

    loader = values.pop("loader", None)
    csv_path = resolve_path(values.pop("csv", values.pop("normalized_csv", None)), base)
    rubric_path = resolve_path(values.pop("rubric", values.pop("rubric_path", None)), base)
    file_path = values.pop("file", None)
    per_rater_file_path = values.pop("per_rater_file", None)
    # `data:` and `raw_data_path:` are the configured spellings; the field they set is
    # `raw_data_file`, which `DatasetConfig.raw_data_path` then derives the path from.
    for alias in ("data", "raw_data_path"):
        if alias in values:
            values["raw_data_file"] = values.pop(alias)
    for key in ("data_root", "calibration_root", "cache_root", "raw_data_file"):
        if key in values:
            values[key] = resolve_path(values[key], base)
    if csv_path is not None or loader is not None:
        criteria = values.pop("criteria_columns", [])
        if (
            not isinstance(criteria, list)
            or not all(isinstance(c, str) and c for c in criteria)
            or len(criteria) != len(set(criteria))
        ):
            raise ValueError("criteria_columns must be a list of unique nonempty criterion names")
        subset = values.pop("set_id", None)
        if csv_path is not None and "raw_data_file" in values:
            raise ValueError("csv and raw_data_path cannot both be configured")
        return DatasetConfig(
            dataset_name=str(name),
            normalized_csv=csv_path,
            rubric_file=rubric_path,
            criteria_columns=criteria,
            subset_id=None if subset is None else str(subset),
            **({"loader": loader} if loader is not None else {}),
            **values,
        )
    if "set_id" in values:
        raise ValueError("set_id is only supported for normalized CSV datasets")
    raw_filenames = []
    if file_path:
        raw_filenames.append(file_path)
    if per_rater_file_path:
        raw_filenames.append(per_rater_file_path)
    return DatasetConfig(
        dataset_name=DatasetNames(name),  # type: ignore[call-arg, arg-type]  # value lookup
        rubric_file=rubric_path,
        raw_filenames=raw_filenames,
        **values,
    )
