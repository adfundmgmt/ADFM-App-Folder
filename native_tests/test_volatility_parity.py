import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adfm_engine.data.market import adjusted_ohlcv, unique_tickers
from adfm_engine.palette import PASTEL
from adfm_engine.serialization import figure_json, records
from adfm_engine.volatility_service import volatility
from reference_roc import Capture, StopPage

SOURCES = json.loads((Path(__file__).parent / "fixtures/volatility_baseline.json").read_text())["sources"]


class VolCapture(Capture):
    def __init__(self, values):
        super().__init__(values)
        self.figures = []
    def header(self, *a, **kw): pass
    def caption(self, *a, **kw): pass
    def spinner(self, *a, **kw): return self
    def form(self, *a, **kw): return self
    def expander(self, *a, **kw): return self
    def form_submit_button(self, *a, **kw): return True
    def tabs(self, labels): return [self] * len(labels)
    def selectbox(self, label, *a, **kw): return self.controls[label]
    def text_input(self, label, *a, **kw): return self.controls[label]
    def dataframe(self, *a, **kw): pass
    def plotly_chart(self, fig, **kw): self.figures.append(fig)


def fixture(optional):
    rng = np.random.default_rng(13)
    index = pd.bdate_range("2020-01-01", periods=850)
    frames = {}
    symbols = ["^NDX", "^GSPC", "^VXN", "^VIX", "SOXX", "QEW", "QQQ"] if optional else ["^NDX", "^GSPC"]
    for symbol in symbols:
        close = 100 * np.exp(np.cumsum(rng.normal(.0003, .013, len(index))))
        frames[symbol] = pd.DataFrame(dict(Open=close, High=close * 1.01, Low=close * .99, Close=close, **{"Adj Close":close*.98}, Volume=np.full(len(index),1000)), index=index)
    frames["^GSPC"] = frames["^GSPC"].drop(index[200:207])
    return frames


def original(frames, rvol, norm):
    capture = VolCapture({"Primary ticker": "^NDX", "Comparison ticker": "^GSPC", "Primary implied-vol ticker": "^VXN", "Comparison implied-vol ticker": "^VIX", "Price history": "5y", "Synthetic VIX window": rvol, "Z-score lookback": norm})
    namespace = dict(__name__=__name__, st=capture, PASTEL=PASTEL, adjusted_ohlcv=adjusted_ohlcv, unique_tickers=unique_tickers, fetch_daily_ohlcv=lambda *a, **kw: (frames,pd.DataFrame()))
    for name in ("PageHeader", "dataframe_download", "inject_explorer_style", "render_footer", "render_page_header", "render_section_header", "render_sidebar_about", "render_status_line", "configure_yfinance_cache"):
        namespace[name] = lambda *a, **kw: None
    exec(compile(SOURCES["adfm_core/relative_volatility.py"], "original-vol-math", "exec"), namespace)
    tree = ast.parse(SOURCES["pages/13_Relative_Volatility_Lab.py"])
    tree.body = [n for n in tree.body if not (isinstance(n,ast.Import) and any(a.name == "streamlit" for a in n.names)) and not (isinstance(n,ast.ImportFrom) and (n.module or "").startswith("adfm_core"))]
    exec(compile(tree, "original-volatility", "exec"), namespace)
    return namespace,capture


@pytest.mark.parametrize("rvol,norm", [(5,21),(10,63),(21,126),(42,252),(63,504),(126,1260),(252,252),(21,252)])
@pytest.mark.parametrize("optional", [True,False])
def test_all_figures_tables_download_and_optional_failure(rvol,norm,optional):
    frames = fixture(optional)
    expected,capture = original(frames,rvol,norm)
    actual = volatility(frames,rvol_window=rvol,normalization_window=norm)
    assert [actual["overview"],actual["normalized"]] == [figure_json(f) for f in capture.figures]
    assert actual["z_table"] == records(expected["z_table"])
    assert actual["data"] == records(expected["display"])
    assert actual["csv"] == expected["export"].reset_index().to_csv(index=False)
    assert actual["warnings"] == capture.warnings
    assert actual["data_through"] == expected["as_of"].date().isoformat()


def test_blank_optional_symbols_are_supported():
    result = volatility(fixture(False),primary_implied="",comparison_implied="")
    assert len(result["z_table"]) == 2
    assert result["overview"] is not None
