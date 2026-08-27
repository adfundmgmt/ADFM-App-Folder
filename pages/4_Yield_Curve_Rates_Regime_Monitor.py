from __future__ import annotations

from datetime import date, timedelta
from html import escape
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from adfm_core.palette import PASTEL, PASTEL_RATES_SCALE
from adfm_core.ui import (
    PageHeader,
    inject_institutional_tool_finish,
    render_footer,
    render_page_header,
)

TITLE = "Yield Curve Rates Regime Monitor"

st.set_page_config(
    page_title=TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

YAHOO_YIELD_TICKERS: Dict[str, Dict[str, object]] = {
    "^IRX": {"label": "3M", "field": "Y3M", "years": 0.25},
    "^FVX": {"label": "5Y", "field": "Y5", "years": 5.0},
    "^TNX": {"label": "10Y", "field": "Y10", "years": 10.0},
    "^TYX": {"label": "30Y", "field": "Y30", "years": 30.0},
}

YIELD_LABELS = {
    str(v["field"]): str(v["label"]) for v in YAHOO_YIELD_TICKERS.values()
}
YIELD_TICKER_TO_FIELD = {
    ticker: str(meta["field"]) for ticker, meta in YAHOO_YIELD_TICKERS.items()
}
FIELD_TO_TICKER = {
    field: ticker for ticker, field in YIELD_TICKER_TO_FIELD.items()
}

PERIODS: Dict[str, Dict[str, object]] = {
    "Today": {"kind": "row", "rows": 1, "threshold": 4},
    "1W": {"kind": "calendar", "days": 7, "threshold": 8},
    "1M": {"kind": "calendar", "months": 1, "threshold": 15},
    "3M": {"kind": "calendar", "months": 3, "threshold": 30},
    "YTD": {"kind": "ytd", "threshold": 35},
}

CURVE_OPTIONS = ["3m10y", "5s10s", "10s30s", "5s30s"]

COLORS = {
    "ink": "#111111",
    "muted": "#666666",
    "border": "#d0d0d0",
    "soft": "#f5f5f3",
    "blue": PASTEL["blue"],
    "purple": PASTEL["lavender"],
    "green": PASTEL["sage"],
    "red": PASTEL["rose"],
    "amber": PASTEL["amber"],
    "slate": PASTEL["slate_blue"],
    "grey": "#A8ADB5",
}

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 3.75rem;
            padding-bottom: 2.00rem;
            max-width: 1560px;
        }

        .metric-card {
            background: #ffffff;
            border-top: 1px solid #bdbdbd;
            border-bottom: 1px solid #bdbdbd;
            border-left: 0;
            border-right: 0;
            border-radius: 0;
            padding: 10px 8px 9px;
            min-height: 88px;
            box-shadow: none;
        }

        .metric-label {
            font-size: 0.67rem;
            color: #555555;
            text-transform: uppercase;
            letter-spacing: 0.075em;
            margin-bottom: 0.34rem;
            font-weight: 800;
        }

        .metric-value {
            font-size: 1.10rem;
            font-weight: 780;
            color: #111111;
            line-height: 1.15;
            letter-spacing: -0.018em;
        }

        .metric-footnote {
            font-size: 0.72rem;
            color: #666666;
            margin-top: 0.34rem;
            line-height: 1.34;
        }

        .section-title {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.15rem;
            font-weight: 700;
            color: #111111;
            margin-top: 1.15rem;
            margin-bottom: 0.22rem;
            letter-spacing: -0.01em;
            border-bottom: 1px solid #111111;
            padding-bottom: 0.34rem;
        }

        .small-note {
            font-size: 0.78rem;
            color: #666666;
            margin-top: 0.28rem;
            margin-bottom: 0.55rem;
            line-height: 1.42;
        }

        .note-box {
            background: #ffffff;
            border-top: 1px solid #c9c9c9;
            border-bottom: 1px solid #c9c9c9;
            border-left: 0;
            border-right: 0;
            border-radius: 0;
            padding: 10px 0;
            color: #303030;
            font-size: 0.84rem;
            line-height: 1.48;
        }

        .data-note {
            color: #666666;
            font-size: 0.75rem;
            line-height: 1.42;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #bdbdbd;
            border-radius: 0;
            overflow: hidden;
        }

        @media (max-width: 760px) {
            .metric-card {
                min-height: 78px;
                padding: 8px 5px;
            }
            .metric-value {
                font-size: 1rem;
            }
            .section-title {
                font-size: 1.05rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

inject_institutional_tool_finish()


def safe_float(x: object) -> float:
    try:
        value = float(x)
        return value if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def latest(series: pd.Series) -> float:
    clean = series.dropna()
    return safe_float(clean.iloc[-1]) if not clean.empty else np.nan


def latest_date(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    clean = df.dropna(how="all")
    if clean.empty:
        return None
    return pd.Timestamp(clean.index[-1])


def value_on_or_before(series: pd.Series, target: pd.Timestamp) -> float:
    clean = series.dropna().sort_index()
    if clean.empty:
        return np.nan
    subset = clean.loc[clean.index <= target]
    if subset.empty:
        return np.nan
    return safe_float(subset.iloc[-1])


def first_value_on_or_after(series: pd.Series, target: pd.Timestamp) -> float:
    clean = series.dropna().sort_index()
    if clean.empty:
        return np.nan
    subset = clean.loc[clean.index >= target]
    if subset.empty:
        return np.nan
    return safe_float(subset.iloc[0])


def anchor_value(series: pd.Series, period: str) -> float:
    clean = series.dropna().sort_index()
    if len(clean) < 2:
        return np.nan

    last_idx = pd.Timestamp(clean.index[-1])
    spec = PERIODS[period]

    if spec["kind"] == "row":
        rows = int(spec.get("rows", 1))
        if len(clean) <= rows:
            return np.nan
        return safe_float(clean.iloc[-rows - 1])

    if spec["kind"] == "calendar":
        if "months" in spec:
            target = last_idx - pd.DateOffset(months=int(spec["months"]))
        else:
            target = last_idx - pd.DateOffset(days=int(spec.get("days", 0)))
        return value_on_or_before(clean, target)

    if spec["kind"] == "ytd":
        jan_first = pd.Timestamp(date(last_idx.year, 1, 1))
        return first_value_on_or_after(clean, jan_first)

    return np.nan


def change_bp(series: pd.Series, period: str) -> float:
    clean = series.dropna().sort_index()
    if len(clean) < 2:
        return np.nan
    last_value = safe_float(clean.iloc[-1])
    anchor = anchor_value(clean, period)
    if not np.isfinite(last_value) or not np.isfinite(anchor):
        return np.nan
    return float((last_value - anchor) * 100.0)


def fmt_pct(x: float) -> str:
    return "N/A" if not np.isfinite(x) else f"{x:.2f}%"


def fmt_bp(x: float) -> str:
    return "N/A" if not np.isfinite(x) else f"{x:+.0f} bp"


def normalize_yahoo_yield_series(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").astype(float)
    median = out.dropna().tail(260).median()
    if np.isfinite(median) and median > 20:
        out = out / 10.0
    return out


def extract_close_frame(
    raw: pd.DataFrame, tickers: Tuple[str, ...]
) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
    diagnostics: List[str] = []

    if raw is None or raw.empty:
        return pd.DataFrame(), ("Yahoo download returned an empty frame.",)

    data = pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = list(raw.columns.get_level_values(0))
        level1 = list(raw.columns.get_level_values(1))

        if "Close" in level0:
            data = raw["Close"].copy()
        elif "Adj Close" in level0:
            data = raw["Adj Close"].copy()
        elif "Close" in level1:
            data = raw.xs("Close", axis=1, level=1).copy()
        elif "Adj Close" in level1:
            data = raw.xs("Adj Close", axis=1, level=1).copy()
        else:
            return pd.DataFrame(), (
                "Yahoo frame did not include Close or Adj Close columns.",
            )
    else:
        close_col = (
            "Close"
            if "Close" in raw.columns
            else "Adj Close"
            if "Adj Close" in raw.columns
            else None
        )
        if close_col is None:
            return pd.DataFrame(), ("Yahoo frame did not include a Close column.",)
        data = raw[[close_col]].copy()
        if len(tickers) == 1:
            data.columns = [tickers[0]]

    data.columns = [str(c) for c in data.columns]
    data.index = pd.to_datetime(data.index, errors="coerce")
    data = data.loc[data.index.notna()]
    data = data.sort_index()
    data = data[~data.index.duplicated(keep="last")]
    data = data.apply(pd.to_numeric, errors="coerce")

    for ticker in tickers:
        if ticker not in data.columns:
            diagnostics.append(f"{ticker}: missing from Yahoo close frame")

    present_cols = [ticker for ticker in tickers if ticker in data.columns]
    if not present_cols:
        return pd.DataFrame(), tuple(
            diagnostics + ["No requested tickers were present in the Yahoo close frame."]
        )

    return data[present_cols].dropna(how="all"), tuple(diagnostics)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_yahoo_close(
    tickers: Tuple[str, ...], start_date: date, end_date: date
) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
    diagnostics: List[str] = []

    try:
        import yfinance as yf
    except Exception as exc:
        return pd.DataFrame(), (
            f"yfinance import failed: {type(exc).__name__}: {exc}",
        )

    try:
        raw = yf.download(
            list(tickers),
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            group_by="column",
            threads=True,
            timeout=12,
        )
    except Exception as exc:
        return pd.DataFrame(), (
            f"Yahoo download failed: {type(exc).__name__}: {exc}",
        )

    data, extract_diag = extract_close_frame(raw, tickers)
    diagnostics.extend(extract_diag)
    return data, tuple(diagnostics)


def split_yahoo_yields(close: pd.DataFrame) -> pd.DataFrame:
    yield_cols = [
        ticker for ticker in YAHOO_YIELD_TICKERS if ticker in close.columns
    ]
    yields = (
        close[yield_cols].copy()
        if yield_cols
        else pd.DataFrame(index=close.index)
    )
    for ticker in yield_cols:
        yields[ticker] = normalize_yahoo_yield_series(yields[ticker])
    yields = yields.rename(columns=YIELD_TICKER_TO_FIELD)
    return yields.ffill().dropna(how="all")


def add_derived_yahoo_rates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"Y10", "Y3M"}.issubset(out.columns):
        out["3m10y"] = out["Y10"] - out["Y3M"]
    if {"Y10", "Y5"}.issubset(out.columns):
        out["5s10s"] = out["Y10"] - out["Y5"]
    if {"Y30", "Y10"}.issubset(out.columns):
        out["10s30s"] = out["Y30"] - out["Y10"]
    if {"Y30", "Y5"}.issubset(out.columns):
        out["5s30s"] = out["Y30"] - out["Y5"]

    return out


def available_curve_columns(df: pd.DataFrame) -> List[str]:
    return [
        c
        for c in CURVE_OPTIONS
        if c in df.columns and df[c].dropna().any()
    ]


def label_for_series(col: str) -> str:
    labels = {
        "Y3M": "3M",
        "Y5": "5Y",
        "Y10": "10Y",
        "Y30": "30Y",
        "3m10y": "3M/10Y",
        "5s10s": "5s10s",
        "10s30s": "10s30s",
        "5s30s": "5s30s",
    }
    return labels.get(col, col)


def classify_regime(
    df: pd.DataFrame, period: str, curve_col: str
) -> Tuple[str, str, str]:
    ten = change_bp(df["Y10"], period) if "Y10" in df else np.nan
    curve = change_bp(df[curve_col], period) if curve_col in df else np.nan
    threshold = float(PERIODS[period]["threshold"])

    if not np.isfinite(ten):
        return (
            "Insufficient Data",
            "Need a valid Yahoo 10Y series.",
            COLORS["amber"],
        )

    if not np.isfinite(curve):
        if ten > threshold:
            return (
                "Rates Rising",
                f"10Y yield up {fmt_bp(ten)} over {period}; selected curve unavailable.",
                COLORS["red"],
            )
        if ten < -threshold:
            return (
                "Rates Falling",
                f"10Y yield down {fmt_bp(ten)} over {period}; selected curve unavailable.",
                COLORS["green"],
            )
        return (
            "Range / Mixed",
            f"10Y move is inside the {threshold:.0f} bp signal band.",
            COLORS["amber"],
        )

    if abs(ten) < threshold and abs(curve) < threshold:
        return (
            "Range / Mixed",
            f"10Y and {label_for_series(curve_col)} are inside the {threshold:.0f} bp signal band.",
            COLORS["amber"],
        )
    if ten > threshold and curve > threshold:
        return (
            "Bear Steepener",
            f"10Y up {fmt_bp(ten)}; {label_for_series(curve_col)} steepened {fmt_bp(curve)} over {period}.",
            COLORS["red"],
        )
    if ten > threshold and curve < -threshold:
        return (
            "Bear Flattener",
            f"10Y up {fmt_bp(ten)}; {label_for_series(curve_col)} flattened {fmt_bp(curve)} over {period}.",
            COLORS["red"],
        )
    if ten < -threshold and curve > threshold:
        return (
            "Bull Steepener",
            f"10Y down {fmt_bp(ten)}; {label_for_series(curve_col)} steepened {fmt_bp(curve)} over {period}.",
            COLORS["green"],
        )
    if ten < -threshold and curve < -threshold:
        return (
            "Bull Flattener",
            f"10Y down {fmt_bp(ten)}; {label_for_series(curve_col)} flattened {fmt_bp(curve)} over {period}.",
            COLORS["green"],
        )
    if ten > threshold:
        return (
            "Bearish Rates Impulse",
            f"10Y up {fmt_bp(ten)}; curve signal is mixed.",
            COLORS["red"],
        )
    if ten < -threshold:
        return (
            "Bullish Rates Impulse",
            f"10Y down {fmt_bp(ten)}; curve signal is mixed.",
            COLORS["green"],
        )
    return (
        "Curve Signal",
        f"10Y quiet, but {label_for_series(curve_col)} moved {fmt_bp(curve)} over {period}.",
        COLORS["amber"],
    )


def regime_read(regime: str) -> str:
    reads = {
        "Bear Steepener": (
            "Long-end yields are rising and the curve is steepening. The market is "
            "adding duration pressure while raising the probability that nominal growth, "
            "term premium, fiscal supply, or some combination of the three is dominating."
        ),
        "Bear Flattener": (
            "Yields are rising while the curve compresses. That is a tighter-policy or "
            "front-end pressure regime rather than a clean reflation signal."
        ),
        "Bull Steepener": (
            "Yields are falling while the curve steepens. The market is moving toward "
            "easier policy, weaker growth, or a stronger duration bid."
        ),
        "Bull Flattener": (
            "Yields are falling while the curve flattens. The long end is outperforming "
            "the front of the available curve, consistent with a stronger duration bid."
        ),
        "Bearish Rates Impulse": (
            "The outright level move matters more than curve shape. Duration pressure is "
            "rising, but the selected curve is not confirming a clean steepening or flattening regime."
        ),
        "Bullish Rates Impulse": (
            "The outright level move is supportive for duration, while the selected curve "
            "has not moved enough to define a clean steepener or flattener."
        ),
        "Curve Signal": (
            "Curve shape is moving more than the outright 10Y level. The information is in "
            "policy-path and term-premium repricing rather than a broad duration shock."
        ),
        "Range / Mixed": (
            "Neither the 10Y level nor the selected curve has cleared the regime threshold. "
            "Treat the rates tape as range-bound until one side breaks."
        ),
    }
    return reads.get(
        regime,
        "Signal quality is low. Check data freshness and missing Yahoo yield symbols.",
    )


def period_matrix(df: pd.DataFrame, rows: List[str]) -> pd.DataFrame:
    out: List[Dict[str, object]] = []

    for col in rows:
        if col not in df.columns or df[col].dropna().empty:
            continue

        row = {
            "Series": label_for_series(col),
            "Latest": latest(df[col]),
        }
        for period in PERIODS:
            row[period] = change_bp(df[col], period)
        out.append(row)

    return pd.DataFrame(out)


def metric_card(
    label: str, value: str, footnote: str, color: str
) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value" style="color:{color};">{escape(value)}</div>
            <div class="metric-footnote">{escape(footnote)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clean_plot_layout(
    fig: go.Figure, height: int, y_title: Optional[str] = None
) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=18, t=24, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        font=dict(color="#334155", size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#d9dee7",
        tickfont=dict(color="#64748b"),
    )
    fig.update_yaxes(
        gridcolor="#edf0f4",
        zeroline=False,
        title_text=y_title,
        tickfont=dict(color="#64748b"),
    )
    return fig


def chart_display_mode() -> Dict[str, object]:
    return {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "responsive": True,
    }


def curve_comparison_values(
    curve_data: pd.DataFrame, compare_window: str
) -> Tuple[pd.Series, List[float]]:
    latest_curve = curve_data.iloc[-1]
    last_idx = pd.Timestamp(curve_data.index[-1])

    if compare_window == "1W":
        compare_target = last_idx - pd.DateOffset(days=7)
    elif compare_window == "1M":
        compare_target = last_idx - pd.DateOffset(months=1)
    elif compare_window == "3M":
        compare_target = last_idx - pd.DateOffset(months=3)
    else:
        compare_target = pd.Timestamp(date(last_idx.year, 1, 1))

    comparison_curve: List[float] = []
    for tenor in curve_data.columns:
        if compare_window == "YTD":
            comparison_curve.append(
                first_value_on_or_after(curve_data[tenor], compare_target)
            )
        else:
            comparison_curve.append(
                value_on_or_before(curve_data[tenor], compare_target)
            )

    return latest_curve, comparison_curve


def history_chart(
    rates: pd.DataFrame,
    selected_curve: str,
    available_yields: List[str],
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
    )

    yield_style = {
        "Y3M": (COLORS["slate"], 1.4),
        "Y5": (COLORS["grey"], 1.5),
        "Y10": (COLORS["blue"], 2.4),
        "Y30": (COLORS["purple"], 2.0),
    }

    for col in available_yields:
        color, width = yield_style.get(col, (COLORS["grey"], 1.5))
        current = latest(rates[col])
        fig.add_trace(
            go.Scatter(
                x=rates.index,
                y=rates[col],
                mode="lines",
                name=f"{label_for_series(col)}  {fmt_pct(current)}",
                line=dict(color=color, width=width),
                hovertemplate=(
                    f"<b>{label_for_series(col)}</b><br>"
                    "%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    curve_bp = rates[selected_curve] * 100.0
    current_curve = latest(curve_bp)
    fig.add_trace(
        go.Scatter(
            x=rates.index,
            y=curve_bp,
            mode="lines",
            name=f"{label_for_series(selected_curve)}  {fmt_bp(current_curve)}",
            line=dict(color=COLORS["amber"], width=2.1),
            hovertemplate=(
                f"<b>{label_for_series(selected_curve)}</b><br>"
                "%{x|%b %d, %Y}<br>%{y:.0f} bp<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    fig.add_hline(
        y=0,
        line_width=1,
        line_color="#94a3b8",
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text="Yield (%)",
        row=1,
        col=1,
        gridcolor="#edf0f4",
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="Curve (bp)",
        row=2,
        col=1,
        gridcolor="#edf0f4",
        zeroline=False,
    )
    fig.update_xaxes(showgrid=False, row=1, col=1)
    fig.update_xaxes(showgrid=False, row=2, col=1)

    fig.update_layout(
        height=570,
        margin=dict(l=12, r=18, t=30, b=24),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        font=dict(color="#334155", size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
    )
    return fig


with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        A focused U.S. rates monitor. It separates the outright Treasury yield
        level from curve shape and shows how both are changing across multiple
        horizons.

        Cross-asset confirmation and generic shock-day rankings are intentionally
        excluded. Those belong in other ADFM tools.
        """
    )

    st.divider()
    st.header("Controls")

    lookback_label = st.selectbox(
        "History",
        ["6M", "1Y", "2Y", "3Y", "5Y", "10Y"],
        index=4,
    )
    lookback_days_map = {
        "6M": 190,
        "1Y": 380,
        "2Y": 760,
        "3Y": 1140,
        "5Y": 1900,
        "10Y": 3800,
    }
    lookback_days = lookback_days_map[lookback_label]

    regime_period = st.radio(
        "Regime window",
        list(PERIODS.keys()),
        index=2,
        horizontal=True,
    )
    selected_curve = st.selectbox(
        "Curve gauge",
        CURVE_OPTIONS,
        index=0,
    )
    curve_compare = st.radio(
        "Curve comparison",
        ["1W", "1M", "3M", "YTD"],
        index=1,
        horizontal=True,
    )

    show_table = st.checkbox("Show raw Yahoo table", value=False)
    show_status = st.checkbox("Show Yahoo download status", value=False)


render_page_header(
    PageHeader(
        title=TITLE,
        description=(
            "Treasury yield levels, curve shape, and rates-regime direction "
            "across the available U.S. curve points."
        ),
        eyebrow="ADFM Macro + Rates",
    )
)

start = date.today() - timedelta(days=lookback_days + 10)
end = date.today()
all_tickers = tuple(YAHOO_YIELD_TICKERS.keys())

with st.spinner("Loading Yahoo Finance yield data..."):
    yahoo_close, diagnostics = fetch_yahoo_close(
        all_tickers,
        start,
        end,
    )

if yahoo_close.empty:
    st.error("No usable Yahoo Finance yield data loaded.")
    if diagnostics:
        with st.expander("Yahoo download status", expanded=True):
            st.code("\n".join(diagnostics[-80:]))
    st.stop()

rates_raw = split_yahoo_yields(yahoo_close)
rates = add_derived_yahoo_rates(rates_raw)

if rates.empty or "Y10" not in rates.columns or rates["Y10"].dropna().empty:
    st.error(
        "No usable 10Y Treasury yield loaded from Yahoo Finance. "
        "The page needs ^TNX to classify the rates regime."
    )
    if diagnostics:
        with st.expander("Yahoo download status", expanded=True):
            st.code("\n".join(diagnostics[-80:]))
    st.stop()

curve_cols = available_curve_columns(rates)
if not curve_cols:
    st.error(
        "Yahoo loaded the 10Y yield, but not enough curve points "
        "to calculate a curve spread."
    )
    st.stop()

if selected_curve not in curve_cols:
    selected_curve = curve_cols[0]

last_obs = latest_date(rates)
if last_obs is not None:
    age_days = (
        pd.Timestamp(date.today()) - last_obs.normalize()
    ).days
    st.caption(
        f"Source: Yahoo Finance via yfinance. Last yield observation: "
        f"{last_obs.date()}. Yield set: ^IRX 3M, ^FVX 5Y, ^TNX 10Y, ^TYX 30Y."
    )
    if age_days > 4:
        st.warning(
            f"Last Yahoo yield observation is {last_obs.date()}. "
            "The rates tape may be stale."
        )

missing_yield_tickers = [
    ticker
    for ticker in YAHOO_YIELD_TICKERS
    if ticker not in yahoo_close.columns
]
if missing_yield_tickers:
    st.warning(
        "Missing Yahoo yield symbols: "
        + ", ".join(missing_yield_tickers)
        + ". Outputs are recalculated from available data only."
    )

if show_status:
    with st.expander("Yahoo download status", expanded=False):
        available = [
            ticker
            for ticker in all_tickers
            if ticker in yahoo_close.columns
            and yahoo_close[ticker].dropna().any()
        ]
        st.markdown(
            "<div class='data-note'>Available tickers: "
            + escape(", ".join(available))
            + "</div>",
            unsafe_allow_html=True,
        )
        if diagnostics:
            st.code("\n".join(diagnostics[-80:]))

regime, regime_note, regime_color = classify_regime(
    rates,
    regime_period,
    selected_curve,
)

cards = [
    ("Regime", regime, regime_note, regime_color),
    (
        "3M Treasury",
        fmt_pct(latest(rates["Y3M"])) if "Y3M" in rates else "N/A",
        f"{regime_period} {fmt_bp(change_bp(rates['Y3M'], regime_period))}"
        if "Y3M" in rates
        else "Unavailable",
        COLORS["slate"],
    ),
    (
        "5Y Treasury",
        fmt_pct(latest(rates["Y5"])) if "Y5" in rates else "N/A",
        f"{regime_period} {fmt_bp(change_bp(rates['Y5'], regime_period))}"
        if "Y5" in rates
        else "Unavailable",
        COLORS["grey"],
    ),
    (
        "10Y Treasury",
        fmt_pct(latest(rates["Y10"])),
        f"{regime_period} {fmt_bp(change_bp(rates['Y10'], regime_period))}",
        COLORS["blue"],
    ),
    (
        "30Y Treasury",
        fmt_pct(latest(rates["Y30"])) if "Y30" in rates else "N/A",
        f"{regime_period} {fmt_bp(change_bp(rates['Y30'], regime_period))}"
        if "Y30" in rates
        else "Unavailable",
        COLORS["purple"],
    ),
    (
        label_for_series(selected_curve),
        fmt_bp(latest(rates[selected_curve]) * 100.0),
        f"{regime_period} {fmt_bp(change_bp(rates[selected_curve], regime_period))}",
        COLORS["amber"],
    ),
]

for col, card in zip(st.columns(6), cards):
    with col:
        metric_card(*card)

st.markdown(
    "<div class='section-title'>Read-through</div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div class='note-box'>{escape(regime_read(regime))}</div>",
    unsafe_allow_html=True,
)

available_yields = [
    c
    for c in ["Y3M", "Y5", "Y10", "Y30"]
    if c in rates.columns and rates[c].dropna().any()
]
curve_data = rates[available_yields].dropna(how="all")

left, right = st.columns([1.03, 0.97])

with left:
    st.markdown(
        "<div class='section-title'>Yield Curve Snapshot</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='small-note'>Current curve versus the selected lookback. "
        "This shows where the repricing is concentrated by tenor.</div>",
        unsafe_allow_html=True,
    )

    if len(available_yields) < 2 or curve_data.empty:
        st.info(
            "At least two Yahoo yield tenors are needed for the curve snapshot."
        )
    else:
        latest_curve, comparison_curve = curve_comparison_values(
            curve_data,
            curve_compare,
        )

        x_vals = [
            float(
                YAHOO_YIELD_TICKERS[
                    FIELD_TO_TICKER[c]
                ]["years"]
            )
            for c in available_yields
        ]
        x_labels = [YIELD_LABELS[c] for c in available_yields]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=latest_curve.values,
                mode="lines+markers",
                name="Latest",
                line=dict(color=COLORS["blue"], width=3),
                marker=dict(size=7),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=comparison_curve,
                mode="lines+markers",
                name=f"{curve_compare} ago",
                line=dict(
                    color=COLORS["grey"],
                    width=1.7,
                    dash="dash",
                ),
                marker=dict(size=6),
            )
        )
        fig.update_layout(
            xaxis=dict(
                title="Tenor",
                tickvals=x_vals,
                ticktext=x_labels,
            ),
            yaxis=dict(title="Yield (%)"),
        )
        st.plotly_chart(
            clean_plot_layout(fig, 360),
            use_container_width=True,
            config=chart_display_mode(),
        )

with right:
    st.markdown(
        "<div class='section-title'>Rates Pressure Matrix</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='small-note'>Basis-point changes by horizon. "
        "This is the fastest way to see whether pressure is front-end, long-end, "
        "or curve-driven.</div>",
        unsafe_allow_html=True,
    )

    matrix_rows = ["Y3M", "Y5", "Y10", "Y30", selected_curve]
    matrix = period_matrix(rates, matrix_rows)

    if matrix.empty:
        st.info("No rates pressure matrix available.")
    else:
        heat_cols = list(PERIODS.keys())
        z = matrix[heat_cols].to_numpy(dtype=float)
        text = np.full(z.shape, "", dtype=object)
        finite_mask = np.isfinite(z)
        text[finite_mask] = np.vectorize(
            lambda v: f"{v:+.0f}"
        )(z[finite_mask])

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=heat_cols,
                y=matrix["Series"],
                colorscale=PASTEL_RATES_SCALE,
                zmid=0,
                text=text,
                texttemplate="%{text}",
                colorbar=dict(title="bp", thickness=10),
                hovertemplate=(
                    "<b>%{y}</b><br>%{x}: %{z:+.0f} bp<extra></extra>"
                ),
            )
        )
        fig.update_layout(xaxis=dict(side="top"))
        st.plotly_chart(
            clean_plot_layout(fig, 360),
            use_container_width=True,
            config=chart_display_mode(),
        )

st.markdown(
    "<div class='section-title'>Rates History</div>",
    unsafe_allow_html=True,
)
selected_curve_level = latest(rates[selected_curve]) * 100.0
selected_curve_move = change_bp(
    rates[selected_curve],
    regime_period,
)
st.markdown(
    (
        "<div class='small-note'>Outright yields and curve shape are separated "
        "into two aligned panels so the level move does not get distorted by a "
        "second axis. "
        f"{escape(label_for_series(selected_curve))} is currently "
        f"{escape(fmt_bp(selected_curve_level))} and has moved "
        f"{escape(fmt_bp(selected_curve_move))} over {escape(regime_period)}."
        "</div>"
    ),
    unsafe_allow_html=True,
)

st.plotly_chart(
    history_chart(
        rates,
        selected_curve,
        available_yields,
    ),
    use_container_width=True,
    config=chart_display_mode(),
)

if show_table:
    st.markdown(
        "<div class='section-title'>Raw Yahoo Data</div>",
        unsafe_allow_html=True,
    )

    table = pd.DataFrame(index=yahoo_close.index)
    for ticker, meta in YAHOO_YIELD_TICKERS.items():
        field = str(meta["field"])
        label = str(meta["label"])
        if field in rates.columns:
            table[f"{label} Yield"] = rates[field]

    for curve in CURVE_OPTIONS:
        if curve in rates.columns:
            table[f"{label_for_series(curve)} bp"] = (
                rates[curve] * 100.0
            )

    st.dataframe(
        table.tail(260),
        use_container_width=True,
    )

st.markdown(
    """
    <div class='data-note'>
        Method note: this page uses Yahoo Finance Treasury-yield symbols only.
        It does not estimate missing 2Y yields, real yields, breakevens, or
        cross-asset confirmation. Missing observations remain unavailable.
    </div>
    """,
    unsafe_allow_html=True,
)

render_footer()
