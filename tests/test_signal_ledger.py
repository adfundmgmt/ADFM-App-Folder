from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

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


class SignalLedgerTests(unittest.TestCase):
    def test_ledger_upserts_same_date_and_compares_prior_date(self):
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
            self.assertEqual(len(history), 2)
            self.assertAlmostEqual(changes.loc[0, "Previous Composite"], 0.10)
            self.assertAlmostEqual(changes.loc[0, "Change Since Prior"], 0.20)

    def test_missing_ledger_has_stable_empty_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            history = load_signal_history(Path(directory) / "missing.parquet")
            self.assertTrue(history.empty)
            self.assertIn("Composite", history.columns)


if __name__ == "__main__":
    unittest.main()
