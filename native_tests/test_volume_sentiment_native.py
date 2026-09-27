"""The native page keeps the causal volume signal and descriptive outcomes."""
import numpy as np
import pandas as pd

from adfm_engine import volume_sentiment_service as service


def test_volume_signal_uses_prior_session_baseline_and_returns_events(monkeypatch):
    dates = pd.bdate_range("2025-01-02", periods=400)
    close = 100 * np.cumprod(1 + np.sin(np.arange(400) / 11) * .003 + .0003)
    volume = np.full(400, 1_000_000.0)
    volume[::17] = 4_000_000
    frame = pd.DataFrame({
        "Open": close * .997, "High": close * 1.01,
        "Low": close * .99, "Close": close,
        "Adj Close": close, "Volume": volume,
    }, index=dates)
    monkeypatch.setattr(service, "fetch_daily_ohlcv", lambda *args, **kwargs: ({"QQQ": frame}, pd.DataFrame()))
    result = service.load_volume_sentiment(symbol="QQQ", lookback_months=48)
    assert result["as_of"] == str(dates[-1].date())
    assert result["events"]
    assert result["outcomes"]
    assert result["figure"]["data"]
    assert result["latest"]["Volume_Ratio"] is not None


def test_missing_volume_reports_unavailable(monkeypatch):
    dates = pd.bdate_range("2025-01-02", periods=150)
    frame = pd.DataFrame({"Open": 100., "High": 101., "Low": 99.,
                          "Close": 100., "Adj Close": 100., "Volume": 0.}, index=dates)
    monkeypatch.setattr(service, "fetch_daily_ohlcv", lambda *args, **kwargs: ({"QQQ": frame}, pd.DataFrame()))
    import pytest
    with pytest.raises(service.DataUnavailable, match="exchange volume"):
        service.load_volume_sentiment()
