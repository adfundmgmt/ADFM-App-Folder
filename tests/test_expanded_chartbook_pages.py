"""Render tests for the always-expanded leadership and ratio chartbooks."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
LEADERSHIP_PAGE = ROOT / "pages" / "8_Equity_Leadership_and_Rotation.py"
RATIO_PAGE = ROOT / "pages" / "11_Cross_Asset_Ratio_Chartbook.py"


def fake_yfinance_download(tickers, **_kwargs) -> pd.DataFrame:
    if isinstance(tickers, str):
        requested = [ticker for ticker in tickers.replace(",", " ").split() if ticker]
    else:
        requested = list(tickers)

    index = pd.bdate_range("2021-01-04", periods=1_500)
    step = np.arange(len(index), dtype=float)
    data = {}
    for ticker in requested:
        seed = sum(ord(character) for character in ticker)
        drift = 0.00008 + (seed % 17) * 0.000015
        cycle = 0.025 * np.sin(step / (18.0 + seed % 23) + seed % 11)
        data[(ticker, "Close")] = (60.0 + seed % 80) * np.exp(drift * step + cycle)

    return pd.DataFrame(data, index=index)


class ExpandedChartbookPageTests(unittest.TestCase):
    @patch("yfinance.download", side_effect=fake_yfinance_download)
    def test_leadership_page_renders_rotation_map_and_all_25_charts(self, _download):
        app = AppTest.from_file(str(LEADERSHIP_PAGE)).run(timeout=60)

        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.get("plotly_chart")), 26)
        self.assertEqual(len(app.dataframe), 0)
        self.assertFalse(any(item.label == "Relationship" for item in app.selectbox))
        rotation_spec = json.loads(app.get("plotly_chart")[0].proto.spec)
        for trace in rotation_spec["data"]:
            self.assertNotIn("customdata[2]:", trace["hovertemplate"])
            for row in trace["customdata"]:
                for value in row[2:6]:
                    self.assertIsNotNone(re.fullmatch(r"[+-]\d+\.\d{2}%", value))
        body = "\n".join(block.value for block in app.markdown)
        self.assertNotIn("Current read", body)
        self.assertNotIn("Leadership by Family", body)
        self.assertNotIn("Multi-Horizon Leadership Matrix", body)
        self.assertNotIn("Leadership Ranking", body)
        self.assertNotIn("Selected Detail", body)

    @patch("yfinance.download", side_effect=fake_yfinance_download)
    def test_ratio_page_renders_all_50_charts_without_view_gate(self, _download):
        app = AppTest.from_file(str(RATIO_PAGE)).run(timeout=60)

        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.get("plotly_chart")), 50)
        self.assertEqual(len(app.radio), 0)
        self.assertFalse(any(item.label == "Relationship" for item in app.selectbox))


if __name__ == "__main__":
    unittest.main()
