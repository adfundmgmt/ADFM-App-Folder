"""Retention for reproducible public-price basket snapshots."""

from pathlib import Path


def prune_basket_snapshots(directory: Path, max_snapshots: int = 16) -> None:
    """Retain the newest snapshot pairs; leave unrelated files untouched."""
    snapshots = sorted(
        directory.glob("basket_levels_*.pkl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for snapshot in snapshots[max(1, max_snapshots) :]:
        snapshot.unlink(missing_ok=True)
        snapshot.with_suffix(".json").unlink(missing_ok=True)
