import numpy as np
import pandas as pd

from adfm_core.regime_math import (
    grouped_weighted_composite,
    rolling_percentile_previous,
)


def test_rolling_percentile_uses_prior_observations_only():
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = rolling_percentile_previous(values, window=3, min_periods=3, scale=100.0)

    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == 100.0


def test_grouped_composite_prevents_proxy_count_from_overweighting_group():
    scores = pd.DataFrame(
        {
            "credit_a": [3.0],
            "credit_b": [3.0],
            "credit_c": [3.0],
            "volatility": [-3.0],
        }
    )
    specs = [
        {"name": "credit_a", "category": "Credit", "weight": 1.0},
        {"name": "credit_b", "category": "Credit", "weight": 1.0},
        {"name": "credit_c", "category": "Credit", "weight": 1.0},
        {"name": "volatility", "category": "Volatility", "weight": 1.0},
    ]

    sleeves, composite, breadth, coverage = grouped_weighted_composite(
        scores, specs, min_groups=2
    )

    assert sleeves.loc[0, "Credit"] == 3.0
    assert sleeves.loc[0, "Volatility"] == -3.0
    assert np.isclose(composite.iloc[0], 0.0)
    assert breadth.iloc[0] == 50.0
    assert coverage.iloc[0] == 100.0
