"""First native basket slice: original arithmetic and missing-data boundary."""
import numpy as np
import pandas as pd
from datetime import datetime
import pytest

from adfm_api.main import BasketParameters, create_app
from adfm_engine import baskets_service
from adfm_engine.services import DataUnavailable


def test_baskets_preserve_original_table_calculations(monkeypatch):
    b = baskets_service.b
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 26, 12, 0, tzinfo=tz)
    monkeypatch.setattr(baskets_service, "datetime", FixedDateTime)
    monkeypatch.setattr(b, "CATEGORIES", {"Sample": {"Pair": ["AAA", "BBB"]}})
    index = pd.bdate_range("2023-01-02", "2026-09-25")
    count = len(index)
    prices = pd.DataFrame({
        "AAA": 100 * np.cumprod(np.full(count, 1.001)),
        "BBB": 90 * np.cumprod(np.full(count, 1.0005)),
        "SPY": 100 * np.cumprod(np.full(count, 1.0003)),
    }, index=index)
    monkeypatch.setattr(b, "fetch_daily_levels", lambda *_args, **_kwargs: (prices, {"source": "yahoo", "returned_tickers": 3}))
    result = baskets_service.load_baskets(categories=["Sample"])
    assert result["as_of"] == "2026-09-25"
    assert result["rows"][0]["Members"] == "2/2"
    assert result["rows"][0]["%5D"] > 0
    assert result["rows"][0]["vs SPY YTD"] > 0


def test_baskets_reject_missing_benchmark(monkeypatch):
    b = baskets_service.b
    monkeypatch.setattr(b, "CATEGORIES", {"Sample": {"Solo": ["AAA"]}})
    monkeypatch.setattr(b, "fetch_daily_levels", lambda *_args, **_kwargs: (pd.DataFrame({"AAA": [100.]}, index=[pd.Timestamp("2026-09-25")]), {"source": "yahoo"}))
    with pytest.raises(DataUnavailable, match="SPY"):
        baskets_service.load_baskets(categories=["Sample"])


def test_rejects_unknown_categories():
    with pytest.raises(ValueError):
        BasketParameters(categories=["Invented"])


def test_raw_definitions_are_available_without_price_downloads(monkeypatch):
    from fastapi.testclient import TestClient
    from adfm_engine.baskets_legacy_math import CATEGORIES
    monkeypatch.setenv("ADFM_ENV", "development")
    monkeypatch.delenv("ADFM_GATEWAY_TOKEN", raising=False)
    with TestClient(create_app()) as client:
        response = client.get("/v1/basket-definitions")
    assert response.status_code == 200
    assert response.json()["categories"] == CATEGORIES


def test_recent_price_snapshot_skips_repeat_download(monkeypatch, tmp_path):
    from adfm_engine import baskets_legacy_math as b
    monkeypatch.setattr(b, "CACHE_DIR", tmp_path)
    dates = pd.bdate_range("2026-09-21", "2026-09-25")
    levels = pd.DataFrame({"AAA": [1, 2, 3, 4, 5], "SPY": [10, 11, 12, 13, 14]}, index=dates)
    key = b._cache_key(["AAA", "SPY"], pd.Timestamp("2026-09-20"))
    assert b.save_last_good_levels(levels, {}, key) is None
    monkeypatch.setattr(b, "_download_close", lambda *_a, **_kw: pytest.fail("unnecessary price download"))
    returned, meta = b.fetch_daily_levels(["AAA", "SPY"], pd.Timestamp("2026-09-20"), pd.Timestamp("2026-09-27"))
    pd.testing.assert_frame_equal(returned, levels)
    assert meta["source"] == "recent_cache"
