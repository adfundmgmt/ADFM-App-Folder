import importlib.util
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from adfm_api.main import create_app
from adfm_engine.analytics.rate_of_change import add_trading_session_axis, compute_features
from adfm_engine.charts.rate_of_change import ROC_PERIODS, TIMEFRAME_MAP
from adfm_engine.data.market import drop_unfinished_daily_session
from adfm_engine.serialization import figure_json
from adfm_engine.services import rate_of_change
from reference_roc import original


def history(count=540, gap=False):
    rng = np.random.default_rng(1729)
    close = 100 * np.exp(np.cumsum(rng.normal(.0003, .015, count)))
    frame = pd.DataFrame({"Open": close * .998, "High": close * 1.015, "Low": close * .985, "Close": close, "Adj Close": close * .97, "Volume": rng.integers(1000, 100000, count)}, index=pd.bdate_range("2022-01-03", periods=count))
    if gap:
        frame.loc[frame.index[100], "Close"] = np.nan
        frame.loc[frame.index[130], "High"] = np.nan
        frame = frame.drop(frame.index[200:204])
    return frame


@pytest.mark.parametrize("roc", ROC_PERIODS)
@pytest.mark.parametrize("view", ["Candlestick", "Line"])
@pytest.mark.parametrize("inflections", [False, True])
@pytest.mark.parametrize("gap", [False, True])
def test_whole_original_chart_and_calculations_match(roc, view, inflections, gap):
    frame = history(gap=gap)
    expected, capture = original(frame, roc=roc, view=view, inflections=inflections)
    actual = rate_of_change(frame, roc=roc, view=view, inflections=inflections)
    calculated = add_trading_session_axis(compute_features(frame, ROC_PERIODS[roc]).dropna(subset=["ROC", "Second_Derivative"]))
    pd.testing.assert_frame_equal(calculated, expected["feat"], check_exact=True)
    assert actual["figure"] == figure_json(capture.figure)
    assert actual["warnings"] == capture.warnings
    json.dumps(actual, allow_nan=False)


@pytest.mark.parametrize("count,roc", [(35, "10D"), (60, "252D"), (270, "252D"), (6600, "63D"), (13200, "252D")])
def test_short_and_decades_long_history(count, roc):
    frame = history(count=count)
    _, capture = original(frame, roc=roc)
    actual = rate_of_change(frame, roc=roc)
    assert actual["warnings"] == capture.warnings
    assert actual["figure"] == (figure_json(capture.figure) if capture.figure is not None else None)


def test_completed_daily_session_policy_is_retained():
    frame = history(10)
    today = frame.index[-1]
    before = datetime(today.year, today.month, today.day, 15, 30, tzinfo=ZoneInfo("America/New_York"))
    after = before.replace(hour=16, minute=15)
    assert len(drop_unfinished_daily_session(frame, before)) == 9
    assert len(drop_unfinished_daily_session(frame, after)) == 10


def test_api_validation_authentication_and_json_contract(monkeypatch):
    monkeypatch.setenv("ADFM_ENV", "production")
    monkeypatch.setenv("ADFM_GATEWAY_TOKEN", "test-only-" * 8)
    import adfm_api.main as api
    monkeypatch.setattr(api, "load_rate_of_change", lambda **kw: rate_of_change(history(), **kw))
    with TestClient(create_app()) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/v1/rate-of-change").status_code == 401
        headers = {"Authorization": "Bearer " + "test-only-" * 8}
        for window in TIMEFRAME_MAP:
            response = client.get("/v1/rate-of-change", params={"window": window}, headers=headers)
            assert response.status_code == 200
            assert len(response.json()["figure"]["data"]) == 9
        for query in ({"roc": "5D"}, {"window": "2Y"}, {"symbol": "https://bad.test"}, {"unexpected": "1"}):
            assert client.get("/v1/rate-of-change", params=query, headers=headers).status_code == 422
        assert client.get("/v1/rate-of-change", params={"symbol": " tlt "}, headers=headers).json()["parameters"]["symbol"] == "TLT"


def test_runtime_has_no_streamlit():
    assert importlib.util.find_spec("streamlit") is None
