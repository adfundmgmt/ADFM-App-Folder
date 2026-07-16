import numpy as np
import pandas as pd

from adfm_core.monthly_returns_matrix import _compound_pct, _display_years, _prepare_month_table


def test_compound_pct_uses_geometric_linking():
    assert _compound_pct([10.0, -5.0]) == 4.5
    assert np.isnan(_compound_pct([np.nan, None]))


def test_prepare_month_table_recovers_period_fields():
    raw = pd.DataFrame(
        {"total_ret": [1.5, -2.0]},
        index=pd.PeriodIndex(["2025-12", "2026-01"], freq="M"),
    )
    prepared = _prepare_month_table(raw)
    assert prepared["year"].tolist() == [2025, 2026]
    assert prepared["month"].tolist() == [12, 1]


def test_display_years_keeps_current_year_first():
    raw = pd.DataFrame(
        {
            "year": [2022, 2023, 2024, 2025],
            "month": [12, 12, 12, 12],
            "total_ret": [1.0, 2.0, 3.0, 4.0],
        },
        index=pd.period_range("2022-12", periods=4, freq="12M"),
    )
    years = _display_years(raw, years_to_show=5, as_of=pd.Timestamp("2026-07-16"))
    assert years == [2026, 2025, 2024, 2023, 2022]
