"""Availability and historical-vintage behavior, independent of providers."""
import unittest
from unittest.mock import Mock

import pandas as pd

from adfm_core import point_in_time as pit


class PointInTimeTests(unittest.TestCase):
    def test_later_release_and_revision_never_enter_earlier_decision(self):
        records = pd.DataFrame([
            {"observation_date": "2020-01-01", "available_at": "2020-02-03", "vintage_date": "2020-02-03", "value": 1.0},
            {"observation_date": "2020-01-01", "available_at": "2020-03-03", "vintage_date": "2020-03-03", "value": 9.0},
            {"observation_date": "2020-02-01", "available_at": None, "vintage_date": "2020-02-25", "value": 99.0},
        ])
        self.assertTrue(pit.known_observations(records, "2020-02-01").empty)
        early = pit.known_observations(records, "2020-02-28")
        self.assertEqual(early.tolist(), [1.0])
        self.assertEqual(pit.known_observations(records, "2020-03-04").tolist(), [9.0])

    def test_month_start_regime_uses_one_verified_vintage_and_unavailable_is_unknown(self):
        def loader(symbol, start, end, *, vintage):
            self.assertEqual(vintage, "2020-05-31")
            self.assertEqual(end, vintage)
            values = pd.Series([2.0, 1.9, 1.8, 1.5], index=pd.date_range("2020-01-01", periods=4, freq="MS"))
            return values, {"vintage": vintage, "source": "FRED API / ALFRED"}
        result = pit.fed_regimes_at_month_start(pd.period_range("2020-06", periods=1, freq="M"), loader=loader)
        self.assertEqual(result.iloc[0]["fed_regime"], "Cutting")
        self.assertTrue(pd.isna(result.iloc[0]["is_recession"]))
        wrong = pit.fed_regimes_at_month_start(result.index, loader=lambda *a, **kw: (pd.Series([1.0]), {"source": "latest revised"}))
        self.assertEqual(wrong.iloc[0]["fed_regime"], "Unknown")

    def test_fred_release_records_keep_official_realtime_start_as_availability(self):
        response = Mock()
        response.json.return_value = {"count": 2, "observations": [
            {"date": "2020-01-01", "realtime_start": "2020-02-03", "realtime_end": "2020-03-02", "value": "1.0"},
            {"date": "2020-01-01", "realtime_start": "2020-03-03", "realtime_end": "9999-12-31", "value": "9.0"},
        ]}
        transport = Mock(return_value=response)
        records = pit.fetch_fred_availability_records("FEDFUNDS", "2020-01-01", "2020-06-01", key="fixture-key", transport=transport)
        self.assertEqual(pit.known_observations(records, "2020-02-15").tolist(), [1.0])
        params = transport.call_args.kwargs["params"]
        self.assertEqual(params["output_type"], 1)
        self.assertEqual(params["realtime_start"], "1776-07-04")
        self.assertEqual(records["available_at"].iloc[0], "2020-02-03")

    def test_bulk_records_produce_same_asof_regime_and_future_revision_is_excluded(self):
        records = pd.DataFrame([
            {"observation_date": f"2020-{month:02d}-01", "available_at": f"2020-{month+1:02d}-03", "vintage_date": f"2020-{month+1:02d}-03", "value": value}
            for month, value in [(1, 2), (2, 1.9), (3, 1.8), (4, 1.5)]
        ] + [{"observation_date": "2020-04-01", "available_at": "2020-06-03", "vintage_date": "2020-06-03", "value": 99}])
        result = pit.fed_regimes_from_availability(pd.period_range("2020-06", periods=1, freq="M"), records)
        self.assertEqual(result.iloc[0]["fed_regime"], "Cutting")
        self.assertEqual(result.iloc[0]["fedfunds"], 1.5)

    def test_cftc_actual_record_and_strict_unknown_and_holiday_schedule(self):
        self.assertEqual(pit.cftc_publication("2023-02-14", strict=True)[0], pd.Timestamp("2023-03-08"))
        self.assertTrue(pd.isna(pit.cftc_publication("2026-09-08", strict=True)[0]))
        date, basis = pit.cftc_publication("2026-06-16", strict=False)
        self.assertEqual(date, pd.Timestamp("2026-06-22"))
        self.assertIn("schedule", basis)
        self.assertTrue(pd.isna(pit.cftc_publication("2019-01-01", strict=False)[0]))
        # A planned backlog date is not evidence of an actual release.
        self.assertTrue(pd.isna(pit.cftc_publication("2025-11-25", strict=True)[0]))
        self.assertEqual(pit.cftc_publication("2025-11-04", strict=False)[0], pd.Timestamp("2025-12-09"))


if __name__ == "__main__":
    unittest.main()
