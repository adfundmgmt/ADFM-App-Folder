"""Check native transport against the independent historical sizing contract."""
import importlib.util
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adfm_engine import position_sizing_service as service
from adfm_engine import position_sizing_math as native


@pytest.fixture
def frames():
    dates = pd.bdate_range("2018-01-02", periods=1540)
    rng = np.random.default_rng(23)
    common = rng.normal(.0003, .012, len(dates))
    result = {}
    for i, symbol in enumerate(("AAPL", *service.BENCHMARKS)):
        returns = common * (.85 + i * .09) + rng.normal(0, .003, len(dates))
        prices = 100 * np.cumprod(1 + returns)
        result[symbol] = pd.DataFrame({"Open": prices * 1.002, "High": prices * 1.018,
                                       "Low": prices * .982, "Close": prices,
                                       "Adj Close": prices, "Volume": np.full(len(dates), 2_000_000)}, index=dates)
    return result


def test_native_math_matches_original_without_streamlit_import():
    path = Path(__file__).parents[1] / "adfm_core" / "position_sizing.py"
    spec = importlib.util.spec_from_file_location("original_position_math", path)
    import sys
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    args = dict(conviction=4, max_nav_loss=.0125, stop_distance=.08,
                current_volatility=.33, historical_median_volatility=.26,
                event_move=.13, tail_move=.12, portfolio_nav=5_000_000,
                median_dollar_volume=2_000_000, hold_through_earnings=True)
    assert asdict(native.calculate_sizing(**args)) == asdict(original.calculate_sizing(**args))
    assert native.HORIZON_TRADING_DAYS == original.HORIZON_TRADING_DAYS


def test_position_analysis_contains_all_original_sections(monkeypatch, frames):
    monkeypatch.setattr(service, "fetch_daily_ohlcv", lambda symbols, period: (frames, pd.DataFrame(columns=["Ticker", "Reason"])))
    monkeypatch.setattr(service, "_earnings_dates", lambda symbol: ())
    result = service.load_position_sizing()
    assert result["data_through"] == frames["AAPL"].index[-1].date().isoformat()
    assert len(result["caps"]) == 6
    assert len(result["paths"]) > 100
    assert sum(result["touch"][key] for key in ("target_first", "stop_first", "same_day", "neither")) == result["touch"]["sample_count"]
    assert len(result["sensitivities"]) == 6
    assert len(result["simulation"]["sessions"]) == 63
    first = result["simulation"]["sessions"][0]
    assert first["balance"] == pytest.approx(5_000_000 * (1 + first["nav_return"]))
    assert result["event_basis"].startswith("90th-percentile overnight gap")


def test_direction_and_horizon_alter_outcomes(monkeypatch, frames):
    monkeypatch.setattr(service, "fetch_daily_ohlcv", lambda symbols, period: (frames, pd.DataFrame(columns=["Ticker", "Reason"])))
    monkeypatch.setattr(service, "_earnings_dates", lambda symbol: ())
    long = service.load_position_sizing(direction="Long", horizon_label="1 month", sampling_mode="Chronological regime replay", seed=42)
    short = service.load_position_sizing(direction="Short", horizon_label="1 month", sampling_mode="Chronological regime replay", seed=42)
    assert long["trade"]["stop_distance"] == pytest.approx(.08)
    assert short["trade"]["stop_distance"] == pytest.approx(.08)
    assert len(short["simulation"]["sessions"]) == 21
    assert short["simulation"]["sessions"][0]["ticker_return"] == pytest.approx(-long["simulation"]["sessions"][0]["ticker_return"])
    assert short["simulation"]["sessions"][0]["date"] == long["simulation"]["sessions"][0]["date"]
