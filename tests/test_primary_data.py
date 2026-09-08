from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

from adfm_core.data_registry import SeriesDefinition
from adfm_core.fred_store import (
    FredError,
    FredStore,
    download,
    normalize,
    path_for,
    request,
    write_record,
)
from adfm_core.macro_history import align_available_observations, percentile_context
from adfm_core.primary_data import fetch_fred_series


class FredStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.store = FredStore(self.root / "runtime", self.root / "shipped")
        self.clock = patch("adfm_core.fred_store.utcnow", return_value=pd.Timestamp("2026-09-08T18:00:00Z"))
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.series = pd.Series([4., np.nan, 4.2], index=pd.to_datetime(["2026-09-02", "2026-09-03", "2026-09-04"]), name="DGS10")
        self.metadata = {"fetched_at": "2026-09-08T10:00:00Z", "requested_start": "1900-01-01", "requested_end": "2026-09-08", "source": "FRED CSV", "units": "Percent", "frequency": "Daily", "vintage": None}

    def save(self):
        path = path_for(self.store.snapshot_dir, "DGS10")
        write_record(path, self.series, self.metadata)
        return path

    @patch("adfm_core.fred_store.download")
    def test_shipped_history_serves_without_network_and_keeps_missing(self, network):
        self.save()
        result = self.store.get("DGS10", "2026-01-01", "2026-09-08")
        network.assert_not_called()
        self.assertEqual(result.metadata["status"], "CACHED")
        self.assertEqual(result.metadata["data_through"], "2026-09-04")
        self.assertTrue(pd.isna(result.series.iloc[1]))

    @patch("adfm_core.fred_store.download", side_effect=requests.Timeout("url?api_key=secret"))
    def test_failure_preserves_last_good_and_redacts_errors(self, network):
        path = self.save()
        original = path.read_bytes()
        result = self.store.get("DGS10", "2026-01-01", "2026-09-08", refresh=True)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(result.metadata["delivery"], "last-good fallback")
        self.assertNotIn("secret", result.metadata["error"])
        self.store.get("DGS10", "2026-01-01", "2026-09-08")
        self.assertEqual(network.call_count, 1)

    def test_stale_offline_data_is_not_presented_as_current(self):
        self.series.index -= pd.Timedelta(days=60)
        self.save()
        result = self.store.get("DGS10", "2026-01-01", "2026-09-08", offline=True)
        self.assertEqual(result.metadata["status"], "STALE")
        self.assertEqual(result.metadata["fetched_at"], self.metadata["fetched_at"])

    @patch("adfm_core.fred_store.download")
    def test_refresh_replaces_revisions_without_filling_missing(self, network):
        self.save()
        revised = self.series.copy()
        revised.iloc[0] = 3.9
        network.return_value = revised, {"source": "FRED API", "units": "Percent", "frequency": "Daily"}
        result = self.store.get("DGS10", "2026-01-01", "2026-09-08", refresh=True)
        self.assertEqual(result.series.iloc[0], 3.9)
        self.assertTrue(pd.isna(result.series.iloc[1]))
        reloaded = self.store.get("DGS10", "2026-01-01", "2026-09-08", offline=True)
        pd.testing.assert_series_equal(result.series, reloaded.series, check_freq=False)

    @patch("adfm_core.fred_store.download")
    def test_shortened_government_history_is_rejected(self, network):
        self.series = pd.Series(4., index=pd.bdate_range("2025-01-01", periods=430), name="DGS10")
        self.save()
        network.return_value = self.series.iloc[-50:], {"source": "FRED API"}
        result = self.store.get("DGS10", "2025-01-01", "2026-09-08", refresh=True)
        self.assertIn("truncated", result.metadata["error"])
        self.assertGreater(len(result.series), 400)

    @patch("adfm_core.fred_store.download")
    def test_partial_provider_failure_does_not_erase_other_series(self, network):
        self.save()
        network.side_effect = FredError("Unavailable")
        definitions = tuple(SeriesDefinition(s.lower(), s, s, "FRED", "Rates", "Yield") for s in ("DGS10", "DGS2"))
        panel, status = fetch_fred_series(definitions, start="2026-01-01", end="2026-09-08", store=self.store)
        self.assertIn("dgs10", panel)
        self.assertNotIn("dgs2", panel)
        self.assertEqual(status.set_index("symbol").loc["DGS2", "status"], "FAILED")

    def test_corrupt_runtime_does_not_hide_valid_shipped_snapshot(self):
        self.save()
        self.store.cache_dir.mkdir()
        path_for(self.store.cache_dir, "DGS10").write_bytes(b"not gzip")
        self.assertFalse(self.store.get("DGS10", "2026-01-01", "2026-09-08", offline=True).series.empty)

    def test_vintage_never_reuses_current_snapshot(self):
        self.save()
        result = self.store.get("DGS10", "2026-01-01", "2026-09-08", offline=True, vintage="2026-09-01")
        self.assertTrue(result.series.empty)
        with self.assertRaises(FredError):
            download("DGS10", "2026-01-01", "2026-09-08", vintage="2026-09-01")


class FredTransportTests(unittest.TestCase):
    def test_invalid_units_values_and_dates_are_rejected(self):
        for dates, values in [(["bad"], [4.]), (["2026-01-01"] * 2, [4., 5.]), (["2026-01-01"], [float("inf")]), (["2026-01-01"], [450.]), (["2026-01-01"], ["broken"])]:
            with self.subTest(values=values), self.assertRaises(FredError):
                normalize("DGS10", pd.DataFrame({"DGS10": values}, index=dates))

    @patch("adfm_core.fred_store.time.sleep")
    def test_transient_retry_honors_bounded_retry_after(self, sleep):
        session = Mock()
        session.get.side_effect = [Mock(status_code=429, headers={"Retry-After": "999"}), Mock(status_code=200)]
        request(session, "https://example.test", {})
        sleep.assert_called_once_with(5.)
        self.assertEqual(session.get.call_count, 2)

    @patch("adfm_core.fred_store.time.sleep")
    def test_invalid_request_is_not_retried(self, sleep):
        session = Mock()
        session.get.return_value = Mock(status_code=400)
        with self.assertRaises(FredError):
            request(session, "https://example.test", {"api_key": "secret"})
        self.assertEqual(session.get.call_count, 1)
        sleep.assert_not_called()

    @patch("adfm_core.fred_store.requests.Session")
    def test_api_vintage_and_natural_units_are_explicit(self, factory):
        session = factory.return_value.__enter__.return_value
        session.get.side_effect = [Mock(status_code=200, json=Mock(return_value={"seriess": [{"units": "Percent", "frequency": "Daily"}]})),
                                   Mock(status_code=200, json=Mock(return_value={"count": 2, "observations": [{"date": "2026-01-02", "value": "4.2"}, {"date": "2026-01-05", "value": "."}]}))]
        values, metadata = download("DGS10", "2026-01-01", "2026-02-01", key="secret", vintage="2026-02-01")
        params = session.get.call_args.kwargs["params"]
        self.assertEqual(params["realtime_start"], "2026-02-01")
        self.assertEqual(params["realtime_end"], "2026-02-01")
        self.assertEqual(params["units"], "lin")
        self.assertTrue(pd.isna(values.iloc[1]))
        self.assertEqual(metadata["source"], "FRED API / ALFRED")

    @patch("adfm_core.fred_store.requests.Session")
    def test_api_metadata_unit_mismatch_is_rejected(self, factory):
        session = factory.return_value.__enter__.return_value
        session.get.return_value = Mock(status_code=200, json=Mock(return_value={"seriess": [{"units": "Basis Points", "frequency": "Daily"}]}))
        with self.assertRaisesRegex(FredError, "units"):
            download("DGS10", "2026-01-01", "2026-02-01", key="secret")

    def test_cache_identifier_cannot_escape_its_directory(self):
        with self.assertRaises(FredError):
            path_for(Path("cache"), "../../secrets")

    @patch("adfm_core.fred_store.PAGE_SIZE", 2)
    @patch("adfm_core.fred_store.requests.Session")
    def test_api_reads_every_observation_page(self, factory):
        session = factory.return_value.__enter__.return_value
        payloads = [{"seriess": [{"units": "Percent", "frequency": "Daily"}]},
                    {"count": 3, "observations": [{"date": "2026-01-02", "value": "4"}, {"date": "2026-01-05", "value": "4.1"}]},
                    {"count": 3, "observations": [{"date": "2026-01-06", "value": "4.2"}]}]
        session.get.side_effect = [Mock(status_code=200, json=Mock(return_value=p)) for p in payloads]
        values, _ = download("DGS10", "2026-01-01", "2026-02-01", key="secret")
        self.assertEqual(len(values), 3)
        self.assertEqual([c.kwargs["params"]["offset"] for c in session.get.call_args_list[1:]], [0, 2])


class MacroHistoryTests(unittest.TestCase):
    def test_three_year_credit_history_is_not_labeled_five_years(self):
        series = pd.Series(np.arange(780), index=pd.bdate_range("2023-09-01", periods=780))
        value, label = percentile_context(series, 5)
        self.assertEqual(value, 1.)
        self.assertIn("available", label)
        self.assertNotEqual(label, "5Y")

    def test_full_history_keeps_requested_window(self):
        series = pd.Series(np.arange(1700), index=pd.bdate_range("2020-01-01", periods=1700))
        self.assertEqual(percentile_context(series, 5)[1], "5Y")

    def test_alignment_cannot_backfill_or_extend_stale_rates(self):
        dates = pd.date_range("2026-01-01", periods=20)
        series = pd.Series([4.], index=[dates[2]])
        result = align_available_observations(series, dates, max_age_days=7)
        self.assertTrue(result.iloc[:2].isna().all())
        self.assertEqual(result.iloc[9], 4.)
        self.assertTrue(result.iloc[10:].isna().all())
