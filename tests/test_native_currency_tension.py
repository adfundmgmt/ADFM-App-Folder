import pandas as pd
import pytest

from adfm_engine import currency_tension_service as cte
from adfm_api.main import CurrencyTensionParameters


def test_month_selector_includes_completed_daily_month_end_only():
    dates = [pd.Timestamp("2026-07-31"), pd.Timestamp("2026-08-31"), pd.Timestamp("2026-09-25")]
    rows = [{"date": day, "kind": "month_end" if day.month == 7 else "daily", "ccy": ccy,
             "axis1_fundamental_struct": .3, "axis2_stretch_struct": -.2}
            for day in dates for ccy in cte.CURRENCIES]
    assert cte._months(pd.DataFrame(rows), "struct") == ["2026-07-31", "2026-08-31"]


def test_custom_weights_recompose_original_two_axes():
    pillars = pd.DataFrame([{"ccy": "USD", "pillar": k, "struct": v}
                            for k, v in zip(cte.PILLAR_AXIS, (1, 3, -1, 1, 1, 2))])
    result = cte._weighted_axes(pillars, "struct", cte.DEFAULT_WEIGHTS).iloc[0]
    assert result.axis1_fundamental_struct == pytest.approx(1)
    assert result.axis2_stretch_struct == pytest.approx(5 / 3)
    weights = {**cte.DEFAULT_WEIGHTS, "G_valuation": 0}
    assert cte._weighted_axes(pillars, "struct", weights).iloc[0].axis2_stretch_struct == 1


def test_carry_is_antisymmetric_and_currency_weight_validation():
    values = pd.DataFrame({"ccy": ["USD", "EUR", "JPY"], "real_2y": [2.5, 1.25, -1.5]})
    grid = cte._carry_grid(values, "real_2y")
    assert grid[0]["EUR"] == 1.25
    assert grid[1]["USD"] == -1.25
    assert grid[0]["USD"] == 0
    with pytest.raises(ValueError):
        CurrencyTensionParameters(weights={**cte.DEFAULT_WEIGHTS, "A_growth": 3.5})
    with pytest.raises(ValueError):
        CurrencyTensionParameters(weights={**cte.DEFAULT_WEIGHTS, "A_growth": 0, "B_inflation": 0, "C_external": 0, "D_fiscal": 0})
