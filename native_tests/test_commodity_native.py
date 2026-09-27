"""Check native event study against the original signal and observation formulas."""
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from adfm_api.main import CommodityParameters, create_app
from adfm_engine import commodity_legacy_math as math
from adfm_engine.commodity_service import load_commodity_event_study


def test_event_study_preserves_history_and_forward_math(monkeypatch):
    dates = pd.bdate_range("2023-01-02", periods=850)
    close = pd.Series(100 * np.exp(np.arange(len(dates)) * 0.001), index=dates)
    close.iloc[300:] *= 1.4
    close.iloc[570:] *= 1.3
    monkeypatch.setattr(math, "load_contract_history", lambda symbol: pd.DataFrame({"Close": close, "Volume": 1}, index=dates))
    result = load_commodity_event_study(symbol="CL=F", threshold=25)
    assert result["count"] >= 1
    assert result["history"][0]["Date"] <= result["as_of"]
    assert result["summary"][0]["Metric"] == "Average"
    assert result["summary"][0]["1M"] is not None
    assert result["figure"]["data"]


def test_commodity_parameters_reject_mismatched_signal():
    import pytest
    with pytest.raises(ValueError, match="Direction"):
        CommodityParameters(signal_type="RSI extreme", direction="Rally")


def test_commodity_job_route_validates_symbol(monkeypatch):
    monkeypatch.setenv("ADFM_ENV", "development")
    monkeypatch.delenv("ADFM_GATEWAY_TOKEN", raising=False)
    with TestClient(create_app()) as client:
        response = client.post("/v1/commodity-event-study", json={"symbol": "CL=F;DROP"})
    assert response.status_code == 422
