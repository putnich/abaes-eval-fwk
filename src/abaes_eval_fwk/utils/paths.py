"""Path resolution utilities for configuration files."""

from pathlib import Path


def resolve_path(value: str | None, base_dir: Path) -> Path | None:
    """Resolve a path relative to base_dir, expanding ~; returns None if value is None."""
    if value is None:
        return None
    path = Path(value).expanduser()
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()
