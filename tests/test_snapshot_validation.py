from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.validate_currency_snapshot import (
    REQUIRED_PARQUET_COLUMNS,
    REQUIRED_TEXT_FILES,
    validate_snapshot,
)


class CurrencySnapshotValidationTests(unittest.TestCase):
    def _write_snapshot(self, directory: Path) -> None:
        for filename, columns in REQUIRED_PARQUET_COLUMNS.items():
            values = {
                column: (
                    ["USD"]
                    if column == "ccy"
                    else ["2026-07-30"]
                    if column in {"date", "fetched_at"}
                    else ["test"]
                    if column in {"kind", "pillar", "source", "tenor", "base"}
                    else [1.0]
                )
                for column in columns
            }
            pd.DataFrame(values).to_parquet(directory / filename, index=False)
        for filename in REQUIRED_TEXT_FILES:
            content = "{}" if filename.endswith(".json") else "Snapshot commentary"
            (directory / filename).write_text(content, encoding="utf-8")

    def test_valid_snapshot_produces_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_snapshot(directory)

            manifest = validate_snapshot(directory)

            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(
                set(manifest["files"]),
                set(REQUIRED_PARQUET_COLUMNS) | REQUIRED_TEXT_FILES,
            )

    def test_missing_required_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_snapshot(directory)
            pd.DataFrame({"ccy": ["USD"]}).to_parquet(
                directory / "tension_map.parquet", index=False
            )

            with self.assertRaisesRegex(ValueError, "missing columns"):
                validate_snapshot(directory)

    def test_invalid_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_snapshot(directory)
            (directory / "warnings.json").write_text("{", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                validate_snapshot(directory)
