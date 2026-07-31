from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from adfm_core.data_registry import market_symbols


def market_fixture() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    index = pd.bdate_range("2023-01-02", periods=700)
    time = np.arange(len(index), dtype=float)
    frames = {}
    for position, symbol in enumerate(market_symbols()):
        values = 100.0 * np.exp((0.0002 + position * 0.00001) * time)
        if symbol == "^VIX":
            values = 30.0 * np.exp(-0.0002 * time)
        frames[symbol] = pd.DataFrame(
            {
                "Open": values,
                "High": values * 1.01,
                "Low": values * 0.99,
                "Close": values,
                "Adj Close": values,
                "Volume": 1_000_000,
            },
            index=index,
        )
    return frames, pd.DataFrame(columns=["Ticker", "Reason"])


def macro_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2023-01-02", periods=700)
    time = np.arange(len(index), dtype=float)
    panel = pd.DataFrame(
        {
            "dgs2": np.linspace(1.0, 4.0, len(index)),
            "dgs10": np.linspace(2.0, 5.0, len(index)),
            "t10yie": 2.0 + np.sin(time / 40) * 0.2,
            "walcl": np.linspace(5_000, 8_000, len(index)),
            "tga": np.linspace(300, 600, len(index)),
            "rrp": np.linspace(100, 1_000, len(index)),
        },
        index=index,
    )
    status = pd.DataFrame(
        [
            {
                "key": "test",
                "symbol": "TEST",
                "provider": "FRED",
                "data_through": index[-1].date().isoformat(),
                "observations": len(index),
                "status": "OK",
                "error": None,
            }
        ]
    )
    return panel, status


class ResearchPageSmokeTests(unittest.TestCase):
    def test_all_new_pages_render_without_exceptions(self) -> None:
        frames, missing = market_fixture()
        macro, source_status = macro_fixture()
        pages = (
            "21_Daily_PM_Brief.py",
            "22_Signal_Attribution_Diagnostics.py",
            "23_Historical_Regime_Analogs.py",
            "24_Primary_Source_Monitor.py",
            "25_Decision_Journal.py",
            "26_Reliability_Alerts.py",
        )
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "adfm_core.market_data.fetch_daily_ohlcv",
                    return_value=(frames, missing),
                )
            )
            stack.enter_context(
                patch(
                    "adfm_core.primary_data.fetch_fred_series",
                    return_value=(macro, source_status),
                )
            )
            stack.enter_context(
                patch(
                    "adfm_core.operations.new_alert_transitions",
                    return_value=[],
                )
            )
            for page in pages:
                app = AppTest.from_file(
                    f"pages/{page}", default_timeout=60
                ).run()
                self.assertEqual(
                    list(app.exception),
                    [],
                    msg=f"{page} raised a runtime exception",
                )


if __name__ == "__main__":
    unittest.main()
