"""Validate a Currency Tension Engine snapshot before it replaces production data."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REQUIRED_PARQUET_COLUMNS: dict[str, set[str]] = {
    "carry_grid_nominal.parquet": {"base"},
    "carry_grid_real.parquet": {"base"},
    "carry_history.parquet": {"date", "ccy", "kind", "real_2y", "nominal_2y"},
    "overlay_history.parquet": {"date", "ccy", "kind"},
    "overlays.parquet": {"ccy"},
    "pillar_history.parquet": {"date", "ccy", "pillar", "kind"},
    "pillar_scores.parquet": {"ccy"},
    "pos_history.parquet": {"date", "ccy"},
    "snapshot_history.parquet": {"date", "ccy", "kind"},
    "tension_map.parquet": {
        "ccy",
        "axis1_fundamental_struct",
        "axis2_stretch_struct",
        "axis1_fundamental_regime",
        "axis2_stretch_regime",
    },
    "tension_map_prev.parquet": {
        "ccy",
        "axis1_fundamental_struct",
        "axis2_stretch_struct",
    },
    "yields.parquet": {"date", "ccy", "tenor", "value", "source", "fetched_at"},
}
REQUIRED_TEXT_FILES = {"commentary.md", "commentary_meta.json", "warnings.json"}
EXPECTED_CURRENCIES = {"USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_snapshot(directory: Path) -> dict[str, object]:
    """Validate required files and schemas, returning a reproducibility manifest."""
    errors: list[str] = []
    file_details: dict[str, dict[str, object]] = {}

    for filename, required_columns in REQUIRED_PARQUET_COLUMNS.items():
        path = directory / filename
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"{filename}: missing or empty")
            continue

        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            errors.append(f"{filename}: unreadable parquet ({exc})")
            continue

        missing = sorted(required_columns.difference(frame.columns))
        if missing:
            errors.append(f"{filename}: missing columns {missing}")
        if frame.empty:
            errors.append(f"{filename}: contains no rows")
        if "ccy" in frame.columns:
            currencies = set(frame["ccy"].dropna().astype(str))
            unknown = sorted(currencies.difference(EXPECTED_CURRENCIES))
            if unknown:
                errors.append(f"{filename}: unexpected currencies {unknown}")

        file_details[filename] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "rows": len(frame),
            "columns": list(frame.columns),
        }

    for filename in sorted(REQUIRED_TEXT_FILES):
        path = directory / filename
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"{filename}: missing or empty")
            continue
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                errors.append(f"{filename}: invalid JSON ({exc})")
                continue
        file_details[filename] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    if errors:
        raise ValueError("Snapshot validation failed:\n- " + "\n- ".join(errors))

    return {
        "schema_version": 1,
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "files": file_details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    manifest = validate_snapshot(args.directory)
    output = args.manifest or args.directory / "snapshot_manifest.json"
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Validated {len(manifest['files'])} snapshot files; wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
