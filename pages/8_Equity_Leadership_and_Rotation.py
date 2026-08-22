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

from adfm_core.leadership import build_leadership_frame, summarize_families
from adfm_core.palette import PASTEL
from adfm_core.ui import PageHeader, render_footer, render_page_header

warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


# ============================== Page config ==============================
st.set_page_config(layout="wide", page_title="Equity Leadership & Rotation")
render_page_header(
    PageHeader(
        title="Equity Leadership & Rotation",
        description=(
            "A systematic scanner for the equity relationships that are leading, improving, "
            "weakening, or lagging across short- and medium-term horizons."
        ),
        eyebrow="ADFM Equity Leadership",
    )
)

st.markdown(
    """
    <style>
    .leadership-read {
        border-top: 1px solid #000000;
        border-bottom: 1px solid #000000;
        margin: .15rem 0 1.15rem;
        padding: .8rem 0 .85rem;
        color: #171717;
        font-family: Arial, Helvetica, sans-serif;
        font-size: .88rem;
        line-height: 1.55;
    }
    .leadership-read-label {
        margin-bottom: .25rem;
        color: #000000;
        font-size: .66rem;
        font-weight: 800;
        letter-spacing: .13em;
        text-transform: uppercase;
    }
    .relationship-note {
        max-width: 980px;
        margin: -.2rem 0 .7rem;
        color: #4a4a4a;
        font-family: Arial, Helvetica, sans-serif;
        font-size: .8rem;
        line-height: 1.5;
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
        .leadership-read { font-size: .84rem; }
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
    "AI / Technology Leadership": [
        RatioSpec("SMH", "IGV", "Semiconductors / Software", "AI hardware and compute leadership versus software and application-layer leadership."),
        RatioSpec("SMH", "QQQ", "Semiconductors / Nasdaq 100", "Tests whether chips are leading technology rather than merely participating."),
        RatioSpec("IGV", "QQQ", "Software / Nasdaq 100", "Tracks whether long-duration software is regaining leadership within technology."),
        RatioSpec("XLK", "SPY", "Technology / S&P 500", "High-level technology leadership and broadening away from technology."),
        RatioSpec("XLF", "XLK", "Financials / Technology", "Nominal-growth and curve-sensitive breadth versus duration-heavy growth."),
    ],
    "Breadth / Style Leadership": [
        RatioSpec("IWM", "SPY", "Small Caps / Large Caps", "Domestic, economically sensitive small-cap participation versus large-cap leadership."),
        RatioSpec("RSP", "SPY", "Equal Weight / Cap Weight S&P 500", "Median-stock participation versus mega-cap concentration."),
        RatioSpec("IJH", "SPY", "Mid Caps / Large Caps", "Broadening with less unprofitable-company contamination than small caps."),
        RatioSpec("IWD", "IWF", "Value / Growth", "Reflation and nominal-growth leadership versus duration-sensitive growth."),
        RatioSpec("MTUM", "QUAL", "Momentum / Quality", "Winner-chasing versus earnings and balance-sheet durability; reversals flag factor deleveraging."),
    ],
    "Cyclicals / Defensives": [
        RatioSpec("XLY", "XLP", "Consumer Discretionary / Staples", "Consumer-cycle and equity risk appetite versus defensive consumption."),
        RatioSpec("XLI", "XLU", "Industrials / Utilities", "Cyclical growth and capex expectations versus defensive duration."),
        RatioSpec("XLF", "XLU", "Financials / Utilities", "Economic activity and curve health versus defensive demand."),
        RatioSpec("XLE", "XLK", "Energy / Technology", "Hard-asset and inflation beta versus long-duration growth."),
        RatioSpec("KRE", "XLF", "Regional Banks / Large Financials", "Domestic credit and funding conditions versus diversified money-center balance sheets."),
    ],
    "Domestic Cyclical Internals": [
        RatioSpec("ITB", "XLU", "Homebuilders / Utilities", "Housing and rate-sensitive cyclicality versus defensive duration."),
        RatioSpec("XHB", "SPY", "Homebuilders / S&P 500", "Housing-cycle leadership versus the broad equity market."),
        RatioSpec("XRT", "SPY", "Retail / S&P 500", "Consumer breadth and lower-income demand sensitivity versus the broad market."),
        RatioSpec("XME", "SPY", "Metals & Mining / S&P 500", "Cyclical materials and industrial-inflation beta versus the broad market."),
        RatioSpec("URA", "XLU", "Uranium / Utilities", "Nuclear-fuel and power-scarcity exposure versus regulated utilities."),
    ],
    "Global Equity Leadership": [
        RatioSpec("EEM", "SPY", "Emerging Markets / United States", "Global-liquidity participation versus U.S. exceptionalism and dollar tightness."),
        RatioSpec("EFA", "SPY", "Developed Ex-US / United States", "Developed international leadership versus U.S. equities."),
        RatioSpec("EWJ", "SPY", "Japan / United States", "Japanese equity leadership versus U.S. equities."),
        RatioSpec("DXJ", "EWJ", "Hedged Japan / Unhedged Japan", "Japanese equity leadership after separating yen translation effects."),
        RatioSpec("FXI", "SPY", "China / United States", "China beta and policy impulse versus U.S. equity leadership."),
    ],
}

ALL_SPECS = [spec for specs in LEADERSHIP_FAMILIES.values() for spec in specs]
SPEC_BY_KEY = {spec.key: spec for spec in ALL_SPECS}
FAMILY_BY_KEY = {spec.key: family for family, specs in LEADERSHIP_FAMILIES.items() for spec in specs}


# ============================== Sidebar ==================================
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Rank equity leadership and identify rotation before it is obvious in the broad indexes.

        **How to read it**
        - A positive score means the numerator ranks in the stronger half of the 25-ratio universe.
        - Acceleration compares 1W/1M ranks with 3M/6M ranks.
        - Leading and lagging confirm established trends; improving and weakening identify transitions.
        - Use Cross-Asset Ratio Chartbook for full historical inspection and custom pairs.

        **Data source:** Yahoo Finance adjusted daily close history.
        """
    )
    st.markdown("---")
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
STATE_BACKGROUNDS = {
    "Leading": "background-color: #E2F0D9; color: #000000",
    "Improving": "background-color: #D9EAF7; color: #000000",
    "Weakening": "background-color: #FFF2CC; color: #000000",
    "Lagging": "background-color: #FCE4D6; color: #000000",
}


def current_read(frame: pd.DataFrame) -> str:
    family_frame = summarize_families(frame)
    strongest = family_frame.iloc[0]
    weakest = family_frame.iloc[-1]
    breadth = float((frame["Leadership Score"] >= 0).mean())
    parts = [
        f"{strongest['Family']} carries the strongest aggregate leadership score ({strongest['Leadership Score']:+.1f}), "
        f"while {weakest['Family']} is weakest ({weakest['Leadership Score']:+.1f}).",
        f"{breadth:.0%} of the displayed relationships rank in the stronger half of the universe.",
    ]
    improving = frame.loc[frame["State"] == "Improving"].sort_values("Acceleration", ascending=False)
    weakening = frame.loc[frame["State"] == "Weakening"].sort_values("Acceleration")
    if not improving.empty:
        row = improving.iloc[0]
        parts.append(f"The clearest positive inflection is {row['Pair']} ({row['Relationship']}), with acceleration of {row['Acceleration']:+.1f}.")
    if not weakening.empty:
        row = weakening.iloc[0]
        parts.append(f"The clearest loss of momentum is {row['Pair']} ({row['Relationship']}), with acceleration of {row['Acceleration']:+.1f}.")
    return " ".join(parts)


def state_cell(value: object) -> str:
    return STATE_BACKGROUNDS.get(str(value), "")


def score_cell(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number >= 25:
        return "background-color: #E2F0D9; color: #000000"
    if number <= -25:
        return "background-color: #FCE4D6; color: #000000"
    return "background-color: #F2F2F2; color: #000000"


def make_rotation_map(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for state in ("Leading", "Improving", "Weakening", "Lagging"):
        subset = frame.loc[frame["State"] == state]
        if subset.empty:
            continue
        custom = np.column_stack(
            [subset["Relationship"], subset["Family"], subset["1W"] * 100, subset["1M"] * 100, subset["3M"] * 100, subset["6M"] * 100]
        )
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
                    "<br>1W %{customdata[2]:+.1f}% · 1M %{customdata[3]:+.1f}%"
                    "<br>3M %{customdata[4]:+.1f}% · 6M %{customdata[5]:+.1f}%<extra></extra>"
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


def make_momentum_heatmap(frame: pd.DataFrame) -> go.Figure:
    columns = ["1W", "1M", "3M", "6M"]
    values = frame[columns].to_numpy(dtype=float) * 100.0
    finite = np.abs(values[np.isfinite(values)])
    limit = max(2.0, float(np.nanpercentile(finite, 92))) if finite.size else 2.0
    text = np.empty(values.shape, dtype=object)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            text[row_index, column_index] = "n/a" if not np.isfinite(value) else f"{value:+.1f}%"
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=columns,
            y=frame["Pair"],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale=[[0.0, PASTEL["rose"]], [0.5, "#FBFBF8"], [1.0, PASTEL["sage"]]],
            zmin=-limit,
            zmax=limit,
            zmid=0,
            xgap=1,
            ygap=1,
            colorbar={"title": "Relative<br>return", "ticksuffix": "%", "thickness": 12},
            customdata=np.column_stack([frame["Relationship"], frame["State"]]),
            hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>%{x}: %{z:+.2f}%<br>%{customdata[1]}<extra></extra>",
        )
    )
    figure.update_layout(
        height=max(500, 26 * len(frame) + 125),
        margin={"l": 85, "r": 35, "t": 15, "b": 40},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Arial, Helvetica, sans-serif", "color": "#202020", "size": 11},
    )
    figure.update_yaxes(autorange="reversed", tickfont={"size": 10})
    figure.update_xaxes(side="top", tickfont={"size": 11})
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
        height=500,
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


st.markdown(
    "<div class='leadership-read'><div class='leadership-read-label'>Current read</div>"
    f"{html_escape(current_read(visible))}</div>",
    unsafe_allow_html=True,
)

family_summary = summarize_families(visible)
st.subheader("Leadership by Family")
family_display = family_summary[["Rank", "Family", "Leadership Score", "Acceleration", "Leading Share", "1M", "3M"]].copy()
family_styler = (
    family_display.style.format({"Leadership Score": "{:+.1f}", "Acceleration": "{:+.1f}", "Leading Share": "{:.0%}", "1M": "{:+.1%}", "3M": "{:+.1%}"})
    .map(score_cell, subset=["Leadership Score", "Acceleration"])
    .set_properties(**{"text-align": "right"}, subset=["Rank", "Leadership Score", "Acceleration", "Leading Share", "1M", "3M"])
)
st.dataframe(family_styler, width="stretch", hide_index=True, height=212)

st.subheader("Rotation Map")
st.caption("Right of zero denotes stronger leadership; above zero denotes improving short-term momentum.")
st.plotly_chart(make_rotation_map(visible), width="stretch", config={"displayModeBar": False, "responsive": True}, key="leadership-rotation-map")

st.subheader("Multi-Horizon Leadership Matrix")
st.caption("Relationships are ordered by composite leadership score. Green denotes numerator outperformance.")
st.plotly_chart(make_momentum_heatmap(visible), width="stretch", config={"displayModeBar": False, "responsive": True}, key="leadership-momentum-heatmap")

st.subheader("Leadership Ranking")
ranking_columns = ["Rank", "Family", "Relationship", "Pair", "State", "Leadership Score", "Acceleration", "1W", "1M", "3M", "6M", "Trend"]
ranking_display = visible[ranking_columns].copy()
ranking_display["Rank"] = np.arange(1, len(ranking_display) + 1)
ranking_styler = (
    ranking_display.style.format({"Leadership Score": "{:+.1f}", "Acceleration": "{:+.1f}", "1W": "{:+.1%}", "1M": "{:+.1%}", "3M": "{:+.1%}", "6M": "{:+.1%}"})
    .map(state_cell, subset=["State"])
    .map(score_cell, subset=["Leadership Score", "Acceleration"])
)
st.dataframe(ranking_styler, width="stretch", hide_index=True, height=560)

with st.sidebar:
    st.markdown("---")
    st.header("Detail")
    detail_key = st.selectbox("Relationship", options=visible["Pair"].tolist(), format_func=lambda key: f"{key} — {SPEC_BY_KEY[key].label}")

detail_row = visible.loc[detail_key]
detail_spec = SPEC_BY_KEY[detail_key]
detail_start = pd.Timestamp(market_date - timedelta(days=detail_spans[detail_span_key]))
st.subheader(f"Selected Detail · {detail_key}")
st.markdown(
    f"<div class='relationship-note'><strong>{html_escape(detail_spec.label)}</strong> · {html_escape(detail_spec.note)}</div>",
    unsafe_allow_html=True,
)
st.plotly_chart(make_detail_figure(ratios[detail_key], detail_start), width="stretch", config={"displayModeBar": False, "responsive": True}, key=f"leadership-detail-{detail_key}")
st.caption(
    f"State {detail_row['State']} | Score {detail_row['Leadership Score']:+.1f} | Acceleration {detail_row['Acceleration']:+.1f} | "
    f"1W {detail_row['1W']:+.1%} | 1M {detail_row['1M']:+.1%} | 3M {detail_row['3M']:+.1%} | 6M {detail_row['6M']:+.1%} | As of {detail_row['As Of']}"
)

if unavailable:
    st.caption("Unavailable this session: " + ", ".join(sorted(set(unavailable))))

st.markdown(
    "<div class='method-note'>Leadership Score is a weighted cross-sectional rank of 1W, 1M, 3M, and 6M relative returns using 20%, 35%, 30%, and 15% weights. "
    "Acceleration is the average 1W/1M rank minus the average 3M/6M rank. Scores are comparative within the fixed 25-relationship universe.</div>",
    unsafe_allow_html=True,
)

render_footer()
