import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from adfm_core import public_signal_capture as capture
from adfm_core.signal_ledger import load_signal_history


class PublicSignalCaptureTests(unittest.TestCase):
    def test_capture_records_public_computation_and_skips_stale_observation(self):
        dates = pd.bdate_range("2026-01-01", "2026-09-28")
        prices = pd.DataFrame({"Close": 100.0, "Volume": 1000.0}, index=dates)
        diagnostics = pd.DataFrame({"RSI": 75.0}, index=dates)
        signal = pd.Series(False, index=dates)
        signal.iloc[-1] = True
        def compute(data, symbol, profile, settings, window):
            return diagnostics, signal, "Fixture signal", "CFTC unavailable"
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "signals.parquet"
            now = datetime(2026, 9, 29, 23, tzinfo=timezone.utc)
            actual, failures = capture.capture_public_signals(["CL=F"], path=ledger, captured_at=now, history_loader=lambda _: prices, compute=compute)
            self.assertFalse(failures)
            self.assertEqual(actual.iloc[0]["Composite"], 1.0)
            self.assertEqual(load_signal_history(ledger).iloc[0]["Data Through"], "2026-09-28")
            old, failures = capture.capture_public_signals(["CL=F"], path=ledger, captured_at=datetime(2026, 10, 20, tzinfo=timezone.utc), history_loader=lambda _: prices, compute=compute)
            self.assertTrue(old.empty)
            self.assertIn("stale", failures["CL=F"])
            self.assertEqual(len(load_signal_history(ledger)), 1)

    def test_unknown_symbol_failure_leaves_valid_capture_and_audit_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            snapshot = pd.DataFrame([{"Key": "public", "Composite": 1.0}])
            capture.write_capture_audit(snapshot, path)
            original = path.read_bytes()
            capture.write_capture_audit(snapshot, path)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(ValueError):
                capture.write_capture_audit(pd.DataFrame([{"Key": "different"}]), path)
            self.assertEqual(path.read_bytes(), original)

    def test_default_capture_uses_completed_futures_history(self):
        with patch("adfm_core.commodity_top_exhaustion_page.load_contract_history", return_value=pd.DataFrame()) as loader:
            _, failures = capture.capture_public_signals(["CL=F"])
        self.assertIn("CL=F", failures)
        loader.assert_called_once_with("CL=F")


if __name__ == "__main__":
    unittest.main()
