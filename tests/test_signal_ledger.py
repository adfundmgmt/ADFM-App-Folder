from __future__ import annotations

import multiprocessing
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from adfm_core.signal_ledger import (
    latest_score_changes,
    load_signal_history,
    record_signal_snapshot,
)


def snapshot(data_through: str, score: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Data Through": data_through,
                "Signal": "Credit sponsorship",
                "Key": "credit",
                "Group": "Credit",
                "Composite": score,
                "Impulse": score / 2.0,
                "Confidence": 1.0,
            }
        ]
    )


def _write_capture(args):
    path, number = args
    row = snapshot("2026-07-29", float(number))
    row["Key"] = f"signal-{number}"
    record_signal_snapshot(row, path)


class SignalLedgerTests(unittest.TestCase):
    def test_ledger_preserves_same_date_versions_and_compares_prior_date(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.parquet"
            record_signal_snapshot(
                snapshot("2026-07-29", 0.10),
                path,
                captured_at=datetime(2026, 7, 29, 22, tzinfo=timezone.utc),
            )
            record_signal_snapshot(
                snapshot("2026-07-30", 0.25),
                path,
                captured_at=datetime(2026, 7, 30, 22, tzinfo=timezone.utc),
            )
            record_signal_snapshot(
                snapshot("2026-07-30", 0.30),
                path,
                captured_at=datetime(2026, 7, 30, 23, tzinfo=timezone.utc),
            )

            history = load_signal_history(path)
            changes = latest_score_changes(history)
            self.assertEqual(len(history), 3)
            self.assertAlmostEqual(changes.loc[0, "Previous Composite"], 0.10)
            self.assertAlmostEqual(changes.loc[0, "Change Since Prior"], 0.20)

    def test_identical_rerun_is_idempotent_but_revision_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.parquet"
            record_signal_snapshot(snapshot("2026-07-29", 0.10), path)
            original = load_signal_history(path)
            record_signal_snapshot(snapshot("2026-07-29", 0.10), path)
            pd.testing.assert_frame_equal(original, load_signal_history(path))
            record_signal_snapshot(snapshot("2026-07-29", 0.20), path)
            self.assertEqual(load_signal_history(path)["Composite"].tolist(), [0.10, 0.20])

    def test_concurrent_processes_preserve_every_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.parquet"
            with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
                list(pool.map(_write_capture, [(path, n) for n in range(12)]))
            self.assertEqual(len(load_signal_history(path)), 12)

    def test_corrupt_ledger_does_not_get_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.parquet"
            path.write_bytes(b"original-corrupt-content")
            with self.assertRaises(ValueError):
                record_signal_snapshot(snapshot("2026-07-29", 0.10), path)
            self.assertEqual(path.read_bytes(), b"original-corrupt-content")

    def test_configurable_default_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "persistent.parquet"
            with patch.dict("os.environ", {"ADFM_SIGNAL_LEDGER_PATH": str(path)}):
                record_signal_snapshot(snapshot("2026-07-29", 0.10))
            self.assertTrue(path.exists())

    def test_missing_ledger_has_stable_empty_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            history = load_signal_history(Path(directory) / "missing.parquet")
            self.assertTrue(history.empty)
            self.assertIn("Composite", history.columns)


if __name__ == "__main__":
    unittest.main()
