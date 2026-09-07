"""Compare the entire original page's math and figures on identical inputs."""
import ast
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adfm_engine.analytics.leadership_universe import ALL_SPECS, LEADERSHIP_FAMILIES
from adfm_engine.leadership_service import DETAIL_SPANS, STATES, leadership
from adfm_engine.serialization import figure_json, records
from reference_roc import Capture, StopPage

SOURCES = json.loads((Path(__file__).parent / "fixtures/leadership_baseline.json").read_text())["sources"]


class LeadershipCapture(Capture):
    def __init__(self, controls):
        super().__init__(controls)
        self.figures = []
    def header(self, *a, **kw): pass
    def subheader(self, *a, **kw): pass
    def caption(self, *a, **kw): pass
    def spinner(self, *a, **kw): return self
    def multiselect(self, label, **kw): return self.controls[label]
    def selectbox(self, label, **kw): return self.controls[label]
    def columns(self, count, **kw): return [self] * count
    def plotly_chart(self, fig, **kw): self.figures.append(fig)


def fixture(missing=False):
    symbols = sorted({t for spec in ALL_SPECS for t in (spec.ticker_1, spec.ticker_2)})
    rng = np.random.default_rng(8)
    values = 100 * np.exp(np.cumsum(rng.normal(.0003, .012, (1400, len(symbols))), axis=0))
    result = pd.DataFrame(values, columns=symbols, index=pd.bdate_range("2020-01-01", periods=1400))
    if missing:
        result = result.drop(columns=["CQQQ", "KRE"])
        result.loc[result.index[:-40], "RWJ"] = np.nan
    return result


def baseline(closes, families, states, history):
    from adfm_engine.palette import PASTEL
    capture = LeadershipCapture({"Leadership families": families, "Rotation states": states, "Detail-chart history": history})
    namespace = {"__name__": __name__, "st": capture, "PASTEL": PASTEL}
    for name in ("PageHeader", "render_footer", "render_page_header", "render_sidebar_about"):
        namespace[name] = lambda *a, **kw: None
    exec(compile(SOURCES["adfm_core/leadership.py"], "original-leadership-math", "exec"), namespace)
    page = ast.parse(SOURCES["pages/8_Equity_Leadership_&_Rotation.py"])
    page.body = [n for n in page.body if not (isinstance(n, ast.Import) and any(a.name == "streamlit" for a in n.names)) and not (isinstance(n, ast.ImportFrom) and (n.module or "").startswith("adfm_core"))]
    # Swap ONLY the provider boundary, leaving page orchestration and figures intact.
    for i, node in enumerate(page.body):
        if isinstance(node, ast.FunctionDef) and node.name == "fetch_closes":
            page.body[i] = ast.parse("fetch_closes = supplied_closes").body[0]
    namespace["supplied_closes"] = lambda *a, **kw: closes.copy()
    try: exec(compile(ast.fix_missing_locations(page), "original-leadership-page", "exec"), namespace)
    except StopPage: pass
    return namespace, capture


@pytest.mark.parametrize("history", DETAIL_SPANS)
@pytest.mark.parametrize("families,states", [(list(LEADERSHIP_FAMILIES), STATES), (["S&P 500 Sector Leadership"], ["Leading", "Weakening"]), (["China / U.S. Leadership", "Inter-Sector Leadership"], STATES)])
@pytest.mark.parametrize("missing", [False, True])
def test_every_score_chart_hover_and_filter(history, families, states, missing):
    closes = fixture(missing)
    namespace, capture = baseline(closes, families, states, history)
    actual = leadership(closes, families=families, states=states, history=history)
    assert actual["rows"] == records(namespace["visible"])
    figures = ([actual["rotation"]] if actual["rotation"] else []) + [chart["figure"] for chart in actual["charts"]]
    assert figures == [figure_json(fig) for fig in capture.figures]
    assert actual["warnings"] == capture.warnings
    assert actual["unavailable"] == namespace["unavailable"]


def test_filtering_does_not_change_rank_universe():
    full = leadership(fixture())
    filtered = leadership(fixture(), families=["S&P 500 Sector Leadership"])
    scores = {r["Pair"]: r["Leadership Score"] for r in full["rows"]}
    assert all(row["Leadership Score"] == scores[row["Pair"]] for row in filtered["rows"])
    assert full["charts"][0]["key"] == "XLK/SPY"
    assert len(full["charts"]) == 25


def test_no_selected_states_returns_empty_results():
    actual = leadership(fixture(), states=[])
    assert not actual["charts"] and not actual["rows"] and actual["rotation"] is None
