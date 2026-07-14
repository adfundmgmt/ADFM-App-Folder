"""Tests for typed catalog and session-level provider observability."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from adfm_core.observability import current_data_health, record_data_load


class ObservabilityTests(unittest.TestCase):
    @patch("adfm_core.observability.st.session_state", new_callable=dict)
    def test_data_load_status_reports_provider_counts_and_last_date(self, state: dict[str, object]) -> None:
        index = pd.DatetimeIndex(["2026-07-10", "2026-07-13"])
        frames = {"SPY": pd.DataFrame({"Close": [100.0, 101.0]}, index=index)}

        event = record_data_load("Yahoo Finance", frames, ("SPY", "QQQ"))

        self.assertEqual(event.returned_symbols, 1)
        self.assertEqual(event.failed_symbols, 1)
        self.assertEqual(event.data_through, "2026-07-13")
        self.assertEqual(current_data_health(), event)
        self.assertIn("adfm_data_health", state)


if __name__ == "__main__":
    unittest.main()
