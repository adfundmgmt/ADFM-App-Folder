from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cte.transform.pairwise import grid_from_values
from cte.transform.zscore import _roll_z, dual_horizon_z


class CurrencyTensionMathTests(unittest.TestCase):
    def test_pairwise_grid_is_antisymmetric_with_zero_diagonal(self) -> None:
        values = pd.Series({"USD": 4.0, "EUR": 2.5, "JPY": 1.0})

        grid = grid_from_values(values)

        np.testing.assert_allclose(grid.to_numpy(), -grid.to_numpy().T)
        np.testing.assert_allclose(np.diag(grid), 0.0)
        self.assertEqual(grid.loc["USD", "JPY"], 3.0)

    def test_rolling_zscore_is_sorted_and_finite_after_warmup(self) -> None:
        index = pd.date_range("2018-01-01", periods=84, freq="MS")
        values = pd.Series(np.linspace(-2.0, 3.0, len(index)), index=index)

        result = _roll_z(values.sample(frac=1.0, random_state=7), years=2)

        self.assertTrue(result.index.is_monotonic_increasing)
        self.assertTrue(np.isfinite(result.dropna()).all())
        self.assertGreater(result.iloc[-1], 0)

    def test_multi_horizon_scores_are_isolated_by_currency_and_metric(self) -> None:
        dates = pd.date_range("2008-01-01", periods=216, freq="MS")
        frame = pd.concat(
            [
                pd.DataFrame(
                    {
                        "date": dates,
                        "ccy": currency,
                        "metric": metric,
                        "value": np.linspace(offset, offset + 10.0, len(dates)),
                    }
                )
                for currency, metric, offset in (
                    ("USD", "growth", 0.0),
                    ("EUR", "growth", 100.0),
                    ("USD", "inflation", -50.0),
                )
            ],
            ignore_index=True,
        )

        result = dual_horizon_z(frame)

        self.assertEqual(len(result), len(frame))
        self.assertEqual(
            set(result.columns),
            {"date", "ccy", "metric", "value", "struct_z", "regime_z", "secular_z"},
        )
        latest = result.groupby(["ccy", "metric"]).tail(1)
        self.assertTrue(latest["regime_z"].notna().all())
