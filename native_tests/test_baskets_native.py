"""First native basket slice: original arithmetic and missing-data boundary."""
import numpy as np
import pandas as pd
from datetime import datetime
import pytest

from adfm_api.main import BasketParameters
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
