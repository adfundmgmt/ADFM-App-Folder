import numpy as np
import pandas as pd
import pytest
from adfm_engine import seasonality_math as m
from adfm_engine import seasonality_service as service
from adfm_api.main import SeasonalityParameters


def test_filtered_matrix_retains_realized_calendar_values(monkeypatch):
    dates = pd.bdate_range("2015-12-01", "2026-08-31")
    prices = pd.Series(100 * np.cumprod(np.full(len(dates), 1.0003)), index=dates)
    table = m.build_filter_table(prices, pd.DataFrame(), pd.DataFrame())
    monkeypatch.setattr(service, "_history", lambda *args: (prices, "^SPX", table))
    result = service.load_monthly_seasonality(lookback="5Y", cycle="Election years", month=1, year=2024)
    assert result["summary"]["sample_months"] == 1
    assert result["sample_years"] >= 1
    assert result["matrix"][0]["Jan"] == pytest.approx(result["profile"][0]["mean_total"])
    assert result["matrix"][1]["Jan"] is not None
    assert result["audit"] and all(row["pres_cycle_bucket"] == "Election years" for row in result["audit"])
    assert result["path_figure"]["data"][2]["y"][0] == 0


def test_rejects_inverted_custom_sample():
    try:
        SeasonalityParameters(lookback="Custom", start_year=2025, end_year=2020)
    except ValueError:
        pass
    else:
        raise AssertionError("Inverted sample accepted")
