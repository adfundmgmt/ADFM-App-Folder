from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from html import escape as html_escape
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from adfm_core.leadership import build_leadership_frame
from adfm_core.palette import PASTEL
from adfm_core.ui import PageHeader, render_footer, render_page_header, render_sidebar_about

warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


# ============================== Page config ==============================
st.set_page_config(layout="wide", page_title="Equity Leadership & Rotation")
render_page_header(
    PageHeader(
        title="Equity Leadership & Rotation",
        description=(
            "A focused scanner for all eleven S&P 500 sectors versus SPY, China versus U.S. equities, "
            "market breadth, and economically paired inter-sector leadership."
        ),
        eyebrow="ADFM Equity Leadership",
    )
)

st.markdown(
    """
    <style>
    .leadership-chart-heading {
        margin: 1rem 0 .25rem;
        color: #000000;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.02rem;
        font-weight: 700;
        line-height: 1.3;
    }
    .relationship-note {
        margin: 0 0 .55rem;
        color: #4a4a4a;
        font-family: Arial, Helvetica, sans-serif;
        font-size: .74rem;
        line-height: 1.45;
    }
    .method-note {
        border-top: 1px solid #bdbdbd;
        margin: 1rem 0 0;
        padding-top: .65rem;
        color: #555555;
        font-family: Arial, Helvetica, sans-serif;
        font-size: .72rem;
        line-height: 1.5;
    }
    @media (max-width: 760px) {
        .leadership-chart-heading { font-size: .96rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================== Universe =================================
@dataclass(frozen=True)
class RatioSpec:
    ticker_1: str
    ticker_2: str
    label: str
    note: str

    @property
    def key(self) -> str:
        return f"{self.ticker_1}/{self.ticker_2}"


LEADERSHIP_FAMILIES: Dict[str, List[RatioSpec]] = {
    "S&P 500 Sector Leadership": [
        RatioSpec("XLK", "SPY", "Technology / S&P 500", "Technology sector leadership versus the broad market."),
        RatioSpec("XLC", "SPY", "Communication Services / S&P 500", "Communication-services leadership versus the broad market."),
        RatioSpec("XLY", "SPY", "Consumer Discretionary / S&P 500", "Consumer-cycle leadership versus the broad market."),
        RatioSpec("XLF", "SPY", "Financials / S&P 500", "Financial-sector leadership and balance-sheet sensitivity versus the broad market."),
        RatioSpec("XLI", "SPY", "Industrials / S&P 500", "Industrial, capex, and cyclical-growth leadership versus the broad market."),
        RatioSpec("XLV", "SPY", "Health Care / S&P 500", "Health-care leadership versus the broad market."),
        RatioSpec("XLP", "SPY", "Consumer Staples / S&P 500", "Defensive-consumption leadership versus the broad market."),
        RatioSpec("XLE", "SPY", "Energy / S&P 500", "Energy and inflation-beta leadership versus the broad market."),
        RatioSpec("XLB", "SPY", "Materials / S&P 500", "Materials and upstream cyclicality versus the broad market."),
        RatioSpec("XLU", "SPY", "Utilities / S&P 500", "Defensive-duration and power-sector leadership versus the broad market."),
        RatioSpec("XLRE", "SPY", "Real Estate / S&P 500", "Rate-sensitive real-estate leadership versus the broad market."),
    ],
    "China / U.S. Leadership": [
        RatioSpec("FXI", "SPY", "China Large Caps / S&P 500", "China policy and growth beta versus U.S. equity leadership."),
        RatioSpec("MCHI", "SPY", "Broad China / S&P 500", "Broad investable Chinese equities versus U.S. equity leadership."),
        RatioSpec("KWEB", "QQQ", "China Internet / Nasdaq 100", "Chinese internet and platform companies versus U.S. large-cap technology."),
        RatioSpec("CQQQ", "QQQ", "China Technology / Nasdaq 100", "Chinese technology leadership versus the U.S. technology benchmark."),
        RatioSpec("ASHR", "SPY", "China A-Shares / S&P 500", "Mainland-listed China exposure and domestic policy transmission versus U.S. equities."),
    ],
    "Breadth / Alternative Weighting": [
        RatioSpec("QQQE", "QQQ", "Equal Weight Nasdaq 100 / Nasdaq 100", "Nasdaq breadth versus mega-cap concentration."),
        RatioSpec("RSP", "SPY", "Equal Weight S&P 500 / S&P 500", "Median-stock participation versus capitalization-weighted leadership."),
        RatioSpec("RWJ", "IJR", "Revenue Weight Small Caps / Cap Weight Small Caps", "A live small-cap alternative-weighting proxy using the same S&P 600 universe."),
    ],
    "Inter-Sector Leadership": [
        RatioSpec("SMH", "IGV", "Semiconductors / Software", "AI hardware and compute leadership versus software and the application layer."),
        RatioSpec("XLF", "XLK", "Financials / Technology", "Nominal-growth and curve-sensitive leadership versus long-duration technology."),
        RatioSpec("XLI", "XLU", "Industrials / Utilities", "Cyclical growth and capex leadership versus defensive duration."),
        RatioSpec("XLY", "XLP", "Consumer Discretionary / Staples", "Consumer-cycle leadership versus defensive consumption."),
        RatioSpec("XLE", "XLK", "Energy / Technology", "Hard-asset and inflation beta versus long-duration growth."),
        RatioSpec("KRE", "XLF", "Regional Banks / Financials", "Regional-bank funding and credit sensitivity versus diversified financials."),
    ],
}

ALL_SPECS = [spec for specs in LEADERSHIP_FAMILIES.values() for spec in specs]
SPEC_BY_KEY = {spec.key: spec for spec in ALL_SPECS}
FAMILY_BY_KEY = {spec.key: family for family, specs in LEADERSHIP_FAMILIES.items() for spec in specs}


# ============================== Sidebar ==================================
with st.sidebar:
    render_sidebar_about("8_Equity_Leadership_&_Rotation.py")
    st.header("Scanner")
    selected_families = st.multiselect(
        "Leadership families",
        options=list(LEADERSHIP_FAMILIES.keys()),
        default=list(LEADERSHIP_FAMILIES.keys()),
    )
    detail_spans = {"6 Months": 180, "1 Year": 365, "3 Years": 365 * 3, "5 Years": 365 * 5}
    detail_span_key = st.selectbox("Detail-chart history", options=list(detail_spans.keys()), index=2)

if not selected_families:
    st.warning("Select at least one leadership family in the sidebar.")
    st.stop()


# ============================== Data =====================================
def clean_ticker(ticker: str) -> str:
    return str(ticker).strip().upper()


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        normalized = clean_ticker(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output


def chunked(items: Sequence[str], size: int) -> Iterable[List[str]]:
    for index in range(0, len(items), size):
        yield list(items[index : index + size])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_closes(tickers: Tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    ticker_list = unique_keep_order(tickers)
    if not ticker_list:
        return pd.DataFrame()

    def extract_close(block: pd.DataFrame) -> Optional[pd.Series]:
        if block is None or block.empty:
            return None
        for field in ("Close", "Adj Close"):
            if field in block.columns:
                return pd.to_numeric(block[field], errors="coerce")
        numeric = block.select_dtypes(include=[np.number])
        return None if numeric.empty else pd.to_numeric(numeric.iloc[:, 0], errors="coerce")

    def normalize(raw: pd.DataFrame, requested: List[str]) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()
        output = pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            level_0 = set(raw.columns.get_level_values(0).astype(str)).intersection(requested)
            level_1 = set(raw.columns.get_level_values(1).astype(str)).intersection(requested)
            if level_0:
                for ticker in requested:
                    if ticker not in level_0:
                        continue
                    try:
                        close = extract_close(raw[ticker])
                        if close is not None:
                            output[ticker] = close
                    except Exception:
                        continue
            elif level_1:
                for ticker in requested:
                    for field in ("Close", "Adj Close"):
                        try:
                            if (field, ticker) in raw.columns:
                                output[ticker] = pd.to_numeric(raw[(field, ticker)], errors="coerce")
                                break
                        except Exception:
                            continue
        elif len(requested) == 1:
            close = extract_close(raw)
            if close is not None:
                output[requested[0]] = close

        if output.empty:
            return output
        output.index = pd.to_datetime(output.index).tz_localize(None)
        output = output.sort_index()
        output = output.loc[~output.index.duplicated(keep="last")]
        return output.ffill().dropna(how="all")

    frames = []
    for batch in chunked(ticker_list, 30):
        try:
            raw = yf.download(
                tickers=batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            normalized = normalize(raw, batch)
            if not normalized.empty:
                frames.append(normalized)
                continue
        except Exception:
            pass
        try:
            raw = yf.download(
                tickers=batch,
                period="max",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            normalized = normalize(raw, batch)
            if not normalized.empty:
                frames.append(normalized)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()
    output = pd.concat(frames, axis=1)
    return output.loc[:, ~output.columns.duplicated()].sort_index().ffill().dropna(how="all")


def raw_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    first, second = numerator.align(denominator, join="inner")
    return (first / second).replace([np.inf, -np.inf], np.nan).dropna()


market_date = date.today()
history_start = market_date - timedelta(days=365 * 5 + 120)
tickers = unique_keep_order(ticker for spec in ALL_SPECS for ticker in (spec.ticker_1, spec.ticker_2))

with st.spinner("Building the leadership map..."):
    closes = fetch_closes(tuple(tickers), history_start, market_date + timedelta(days=1))

if closes.empty:
    st.error("Failed to download leadership data.")
    st.stop()

ratios: Dict[str, pd.Series] = {}
unavailable: List[str] = []
for spec in ALL_SPECS:
    if spec.ticker_1 not in closes.columns or spec.ticker_2 not in closes.columns:
        unavailable.append(spec.key)
        continue
    series = raw_ratio(closes[spec.ticker_1], closes[spec.ticker_2])
    if len(series) < 22:
        unavailable.append(spec.key)
        continue
    ratios[spec.key] = series

metadata = pd.DataFrame(
    {
        "Family": [FAMILY_BY_KEY[spec.key] for spec in ALL_SPECS],
        "Relationship": [spec.label for spec in ALL_SPECS],
        "Pair": [spec.key for spec in ALL_SPECS],
        "Note": [spec.note for spec in ALL_SPECS],
    },
    index=[spec.key for spec in ALL_SPECS],
)
leadership = build_leadership_frame(ratios, metadata)
if leadership.empty:
    st.error("The available histories were insufficient to calculate leadership scores.")
    st.stop()

visible = leadership.loc[leadership["Family"].isin(selected_families)].copy()
with st.sidebar:
    state_order = ["Leading", "Improving", "Weakening", "Lagging"]
    selected_states = st.multiselect("Rotation states", options=state_order, default=state_order)

visible = visible.loc[visible["State"].isin(selected_states)].copy() if selected_states else visible.iloc[0:0]
if visible.empty:
    st.warning("No relationships match the selected family and state filters.")
    st.stop()


# ============================== Presentation =============================
STATE_COLORS = {
    "Leading": PASTEL["sage"],
    "Improving": PASTEL["blue"],
    "Weakening": PASTEL["amber"],
    "Lagging": PASTEL["rose"],
}


def fmt_signed_percent(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:+.2%}"


def make_rotation_map(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for state in ("Leading", "Improving", "Weakening", "Lagging"):
        subset = frame.loc[frame["State"] == state]
        if subset.empty:
            continue
        custom = [
            [
                row["Relationship"],
                row["Family"],
                fmt_signed_percent(row["1W"]),
                fmt_signed_percent(row["1M"]),
                fmt_signed_percent(row["3M"]),
                fmt_signed_percent(row["6M"]),
            ]
            for _, row in subset.iterrows()
        ]
        figure.add_trace(
            go.Scatter(
                x=subset["Leadership Score"],
                y=subset["Acceleration"],
                mode="markers+text",
                name=state,
                text=subset["Pair"],
                textposition="top center",
                textfont={"size": 9, "color": "#202020"},
                marker={"size": 10, "color": STATE_COLORS[state], "line": {"color": "#ffffff", "width": 1}},
                customdata=custom,
                hovertemplate=(
                    "<b>%{text}</b><br>%{customdata[0]}<br>%{customdata[1]}"
                    "<br>Score %{x:.1f}<br>Acceleration %{y:.1f}"
                    "<br>1W %{customdata[2]} · 1M %{customdata[3]}"
                    "<br>3M %{customdata[4]} · 6M %{customdata[5]}<extra></extra>"
                ),
            )
        )
    figure.add_hline(y=0, line_color="#777777", line_width=1)
    figure.add_vline(x=0, line_color="#777777", line_width=1)
    figure.update_layout(
        height=540,
        margin={"l": 45, "r": 25, "t": 20, "b": 55},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Arial, Helvetica, sans-serif", "color": "#202020", "size": 11},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0},
        hoverlabel={"bgcolor": "white", "font": {"color": "black"}},
    )
    figure.update_xaxes(title="Leadership score", range=[-112, 112], showgrid=True, gridcolor="#e8e8e8", zeroline=False)
    y_limit = max(55.0, float(np.nanmax(np.abs(frame["Acceleration"]))) * 1.22)
    figure.update_yaxes(title="Momentum acceleration", range=[-y_limit, y_limit], showgrid=True, gridcolor="#e8e8e8", zeroline=False)
    return figure


def make_detail_figure(series: pd.Series, start: pd.Timestamp) -> go.Figure:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    display = clean.loc[start:].copy()
    if display.empty:
        display = clean.copy()
    base = float(display.iloc[0])
    rebased = clean / base * 100.0
    view = rebased.loc[display.index.min() :]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=view.index, y=view, mode="lines", name="Ratio", line={"color": "#000000", "width": 2.2}, hovertemplate="%{y:.2f}<extra>Ratio</extra>"))
    for window, color in ((21, PASTEL["lavender"]), (50, PASTEL["blue"]), (200, PASTEL["rose"])):
        moving_average = rebased.rolling(window, min_periods=max(10, window // 2)).mean().loc[view.index]
        figure.add_trace(go.Scatter(x=moving_average.index, y=moving_average, mode="lines", name=f"{window}D", line={"color": color, "width": 1.25}, hovertemplate=f"%{{y:.2f}}<extra>{window}D</extra>"))
    figure.add_trace(go.Scatter(x=[view.index[-1]], y=[view.iloc[-1]], mode="markers", marker={"color": "#000000", "size": 7}, showlegend=False, hovertemplate="%{y:.2f}<extra>Last</extra>"))
    figure.update_layout(
        height=390,
        margin={"l": 50, "r": 25, "t": 20, "b": 45},
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        font={"family": "Arial, Helvetica, sans-serif", "color": "#202020", "size": 11},
        legend={"orientation": "h", "y": 1.02, "x": 0},
    )
    figure.update_xaxes(showgrid=True, gridcolor="#e8e8e8", zeroline=False)
    figure.update_yaxes(title="Rebased ratio", showgrid=True, gridcolor="#e8e8e8", zeroline=False)
    return figure


st.subheader("Rotation Map")
st.caption("Right of zero denotes stronger leadership; above zero denotes improving short-term momentum.")
st.plotly_chart(make_rotation_map(visible), width="stretch", config={"displayModeBar": False, "responsive": True}, key="leadership-rotation-map")

detail_start = pd.Timestamp(market_date - timedelta(days=detail_spans[detail_span_key]))
st.subheader("Leadership Charts")
st.caption("All selected relationships are displayed below. Charts are grouped by leadership family and remain fully expanded.")

for family in selected_families:
    family_keys = [
        spec.key
        for spec in LEADERSHIP_FAMILIES[family]
        if spec.key in visible.index
    ]
    if not family_keys:
        continue

    st.markdown(f"### {family}")
    for row_start in range(0, len(family_keys), 2):
        columns = st.columns(2, gap="large")
        for column_index, detail_key in enumerate(family_keys[row_start : row_start + 2]):
            detail_row = visible.loc[detail_key]
            detail_spec = SPEC_BY_KEY[detail_key]
            with columns[column_index]:
                st.markdown(
                    f"<div class='leadership-chart-heading'>{html_escape(detail_spec.label)} ({html_escape(detail_key)})</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='relationship-note'>{html_escape(detail_spec.note)}</div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    make_detail_figure(ratios[detail_key], detail_start),
                    width="stretch",
                    config={"displayModeBar": False, "responsive": True},
                    key=f"leadership-detail-{detail_key}",
                )
                st.caption(
                    f"State {detail_row['State']} | Score {detail_row['Leadership Score']:+.1f} | "
                    f"Acceleration {detail_row['Acceleration']:+.1f} | 1W {fmt_signed_percent(detail_row['1W'])} | "
                    f"1M {fmt_signed_percent(detail_row['1M'])} | 3M {fmt_signed_percent(detail_row['3M'])} | "
                    f"6M {fmt_signed_percent(detail_row['6M'])} | As of {detail_row['As Of']}"
                )
                st.markdown("---")

if unavailable:
    st.caption("Unavailable this session: " + ", ".join(sorted(set(unavailable))))

st.markdown(
    "<div class='method-note'>Leadership Score is a weighted cross-sectional rank of 1W, 1M, 3M, and 6M relative returns using 20%, 35%, 30%, and 15% weights. "
    "Acceleration is the average 1W/1M rank minus the average 3M/6M rank. Scores are comparative within the fixed 25-relationship universe.</div>",
    unsafe_allow_html=True,
)

render_footer()
