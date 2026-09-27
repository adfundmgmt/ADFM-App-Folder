"""Preserve rotation rankings, data coverage and both charts at the native boundary."""
import numpy as np
import pandas as pd
from adfm_engine import sector_rotation_legacy_math as rotation
from adfm_engine.sector_rotation_service import load_sector_rotation


def test_sector_rotation_keeps_original_ranking_and_coverage(monkeypatch):
    dates = pd.bdate_range("2023-01-02", periods=780)
    rng = np.random.default_rng(17)
    universe = rotation.build_universe("Major sectors only")
    prices = pd.DataFrame({
        ticker: 100 * np.exp(np.cumsum(rng.normal(.0003, .009, len(dates))))
        for ticker in [*universe["Ticker"], "SPY"]
    }, index=dates)
    monkeypatch.setattr(rotation, "fetch_prices", lambda *_args, **_kwargs: prices)
    result = load_sector_rotation(universe="Major sectors only")
    assert result["coverage"] == 11
    assert result["requested"] == 11
    assert result["rows"][0]["Rank"] == 1
    assert result["rotation_chart"]["data"]
    assert result["rs_chart"]["data"]
