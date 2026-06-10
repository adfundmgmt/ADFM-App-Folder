from __future__ import annotations

from datetime import date, timedelta
from html import escape
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


TITLE = "Rates Regime Monitor"

YAHOO_YIELD_TICKERS: Dict[str, Dict[str, object]] = {
    "^IRX": {"label": "3M", "field": "Y3M", "years": 0.25},
    "^FVX": {"label": "5Y", "field": "Y5", "years": 5.0},
    "^TNX": {"label": "10Y", "field": "Y10", "years": 10.0},
    "^TYX": {"label": "30Y", "field": "Y30", "years": 30.0},
}

YAHOO_MARKET_TICKERS: Dict[str, str] = {
    "SHY": "SHY 1-3Y Treasury",
    "IEF": "IEF 7-10Y Treasury",
    "TLT": "TLT 20Y+ Treasury",
    "SPY": "SPY S&P 500",
    "QQQ": "QQQ Nasdaq 100",
    "IWM": "IWM Russell 2000",
    "HYG": "HYG High Yield",
    "LQD": "LQD IG Credit",
    "GLD": "GLD Gold",
    "UUP": "UUP Dollar",
}

YIELD_LABELS = {str(v["field"]): str(v["label"]) for v in YAHOO_YIELD_TICKERS.values()}
YIELD_TICKER_TO_FIELD = {ticker: str(meta["field"]) for ticker, meta in YAHOO_YIELD_TICKERS.items()}
FIELD_TO_TICKER = {field: ticker for ticker, field in YIELD_TICKER_TO_FIELD.items()}

PERIODS: Dict[str, Dict[str, object]] = {
    "Today": {"kind": "row", "rows": 1, "threshold": 4},
    "1W": {"kind": "calendar", "days": 7, "threshold": 8},
    "1M": {"kind": "calendar", "months": 1, "threshold": 15},
    "3M": {"kind": "calendar", "months": 3, "threshold": 30},
    "YTD": {"kind": "ytd", "threshold": 35},
}

CURVE_OPTIONS = ["3m10y", "5s10s", "10s30s", "5s30s"]

COLORS = {
    "ink": "#0f172a",
    "muted": "#64748b",
    "border": "#e2e8f0",
    "panel": "#ffffff",
    "soft": "#f8fafc",
    "blue": "#2563eb",
    "purple": "#7c3aed",
    "green": "#059669",
    "red": "#dc2626",
    "amber": "#d97706",
    "slate": "#475569",
    "grey": "#94a3b8",
}

st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.20rem;
            padding-bottom: 2.00rem;
            max-width: 1560px;
        }

        .adfm-title {
            font-size: 1.72rem;
            line-height: 1.12;
            font-weight: 760;
            margin-bottom: 0.18rem;
            color: #0f172a;
            letter-spacing: -0.025em;
        }

        .adfm-subtitle {
            font-size: 0.92rem;
            color: #64748b;
            margin-bottom: 1.05rem;
        }

        .metric-card {
            background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 13px 15px 12px 15px;
            min-height: 102px;
            box-shadow: 0 1px 5px rgba(15, 23, 42, 0.045);
        }

        .metric-label {
            font-size: 0.70rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.060em;
            margin-bottom: 0.42rem;
        }

        .metric-value {
            font-size: 1.23rem;
            font-weight: 760;
            color: #0f172a;
            line-height: 1.18;
            letter-spacing: -0.02em;
        }

        .metric-footnote {
            font-size: 0.75rem;
            color: #64748b;
            margin-top: 0.43rem;
            line-height: 1.34;
        }

        .section-title {
            font-size: 1.01rem;
            font-weight: 760;
            color: #0f172a;
            margin-top: 1.00rem;
            margin-bottom: 0.45rem;
            letter-spacing: -0.01em;
        }

        .small-note {
            font-size: 0.80rem;
            color: #64748b;
            margin-top: -0.24rem;
            margin-bottom: 0.50rem;
            line-height: 1.38;
        }

        .note-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 13px;
            padding: 11px 13px;
            color: #475569;
            font-size: 0.84rem;
            line-height: 1.45;
        }

        .data-note {
            color: #64748b;
            font-size: 0.78rem;
            line-height: 1.40;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.65rem 0.75rem;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            overflow: hidden;
        }

        [data-testid="stSidebar"] .stRadio > label,
        [data-testid="stSidebar"] .stSelectbox > label,
        [data-testid="stSidebar"] .stCheckbox > label {
            color: #0f172a;
            font-weight: 620;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def return_pct(series: pd.Series, period: str) -> float:
    clean = series.dropna().sort_index()
    if len(clean) < 2:
        return np.nan

    last_value = safe_float(clean.iloc[-1])
    anchor = anchor_value(clean, period)

    if not np.isfinite(last_value) or not np.isfinite(anchor) or anchor == 0:
        return np.nan
    return float((last_value / anchor - 1.0) * 100.0)


def fmt_pct(x: float) -> str:
    return "N/A" if not np.isfinite(x) else f"{x:.2f}%"


def fmt_bp(x: float) -> str:
    return "N/A" if not np.isfinite(x) else f"{x:+.0f} bps"


def fmt_num(x: float, decimals: int = 1) -> str:
    return "N/A" if not np.isfinite(x) else f"{x:,.{decimals}f}"


def fmt_ret(x: float) -> str:
    return "N/A" if not np.isfinite(x) else f"{x:+.2f}%"


def normalize_yahoo_yield_series(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").astype(float)
    median = out.dropna().tail(260).median()

    # Yahoo/Cboe yield indices can occasionally arrive in 10x notation depending on endpoint behavior.
    # The display target here is yield in percent, e.g. 4.55 for a 4.55% 10Y Treasury yield.
    if np.isfinite(median) and median > 20:
        out = out / 10.0

    return out


def extract_close_frame(raw: pd.DataFrame, tickers: Tuple[str, ...]) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
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
            return pd.DataFrame(), ("Yahoo frame did not include Close or Adj Close columns.",)
    else:
        close_col = "Close" if "Close" in raw.columns else "Adj Close" if "Adj Close" in raw.columns else None
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
        return pd.DataFrame(), tuple(diagnostics + ["No requested tickers were present in the Yahoo close frame."])

    return data[present_cols].dropna(how="all"), tuple(diagnostics)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_yahoo_close(tickers: Tuple[str, ...], start_date: date, end_date: date) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
    diagnostics: List[str] = []

    try:
        import yfinance as yf
    except Exception as exc:
        return pd.DataFrame(), (f"yfinance import failed: {type(exc).__name__}: {exc}",)

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
        return pd.DataFrame(), (f"Yahoo download failed: {type(exc).__name__}: {exc}",)

    data, extract_diag = extract_close_frame(raw, tickers)
    diagnostics.extend(extract_diag)

    return data, tuple(diagnostics)


def split_yahoo_data(close: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    yield_cols = [ticker for ticker in YAHOO_YIELD_TICKERS if ticker in close.columns]
    market_cols = [ticker for ticker in YAHOO_MARKET_TICKERS if ticker in close.columns]

    yields = close[yield_cols].copy() if yield_cols else pd.DataFrame(index=close.index)
    for ticker in yield_cols:
        yields[ticker] = normalize_yahoo_yield_series(yields[ticker])
    yields = yields.rename(columns=YIELD_TICKER_TO_FIELD)

    prices = close[market_cols].copy() if market_cols else pd.DataFrame(index=close.index)
    return yields.ffill().dropna(how="all"), prices.ffill().dropna(how="all")


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
    return [c for c in CURVE_OPTIONS if c in df.columns and df[c].dropna().any()]


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


def classify_regime(df: pd.DataFrame, period: str, curve_col: str) -> Tuple[str, str, str]:
    ten = change_bp(df["Y10"], period) if "Y10" in df else np.nan
    curve = change_bp(df[curve_col], period) if curve_col in df else np.nan
    threshold = float(PERIODS[period]["threshold"])

    if not np.isfinite(ten):
        return "Insufficient Data", "Need a valid Yahoo 10Y series.", COLORS["amber"]

    if not np.isfinite(curve):
        if ten > threshold:
            return "Rates Rising", f"10Y yield up {fmt_bp(ten)} over {period}; selected curve unavailable.", COLORS["red"]
        if ten < -threshold:
            return "Rates Falling", f"10Y yield down {fmt_bp(ten)} over {period}; selected curve unavailable.", COLORS["green"]
        return "Range / Mixed", f"10Y move is inside the {threshold:.0f} bp signal band.", COLORS["amber"]

    if abs(ten) < threshold and abs(curve) < threshold:
        return "Range / Mixed", f"10Y and {label_for_series(curve_col)} are inside the {threshold:.0f} bp signal band.", COLORS["amber"]
    if ten > threshold and curve > threshold:
        return "Bear Steepener", f"10Y up {fmt_bp(ten)}; {label_for_series(curve_col)} steepened {fmt_bp(curve)} over {period}.", COLORS["red"]
    if ten > threshold and curve < -threshold:
        return "Bear Flattener", f"10Y up {fmt_bp(ten)}; {label_for_series(curve_col)} flattened {fmt_bp(curve)} over {period}.", COLORS["red"]
    if ten < -threshold and curve > threshold:
        return "Bull Steepener", f"10Y down {fmt_bp(ten)}; {label_for_series(curve_col)} steepened {fmt_bp(curve)} over {period}.", COLORS["green"]
    if ten < -threshold and curve < -threshold:
        return "Bull Flattener", f"10Y down {fmt_bp(ten)}; {label_for_series(curve_col)} flattened {fmt_bp(curve)} over {period}.", COLORS["green"]
    if ten > threshold:
        return "Bearish Rates Impulse", f"10Y up {fmt_bp(ten)}; curve signal is mixed.", COLORS["red"]
    if ten < -threshold:
        return "Bullish Rates Impulse", f"10Y down {fmt_bp(ten)}; curve signal is mixed.", COLORS["green"]
    return "Curve Signal", f"10Y quiet, but {label_for_series(curve_col)} moved {fmt_bp(curve)} over {period}.", COLORS["amber"]


def regime_read(regime: str) -> str:
    reads = {
        "Bear Steepener": "Long-end yields are rising and the curve is steepening. That is the most hostile mix for duration, unprofitable growth, bond proxies, and levered balance sheets unless equity markets are underwriting a genuine nominal-growth acceleration.",
        "Bear Flattener": "Yields are rising while the curve compresses. The tape is pricing tighter financial conditions, policy pressure, or term-premium stress without a clean reflation impulse. Small caps, credit beta, and long-duration equities usually deserve the first risk check.",
        "Bull Steepener": "Yields are falling while the curve steepens. The market is usually moving toward Fed-cut pricing, weaker growth, or a flight-to-duration bid. Duration can work, but the equity read depends on whether liquidity is improving or earnings risk is rising.",
        "Bull Flattener": "Yields are falling while the curve flattens. The long-end bid is stronger than the reflation impulse. Quality duration can screen better than cyclicals, but the regime is not automatically bullish for equities.",
        "Bearish Rates Impulse": "The level move matters more than curve shape. Duration is the pressure point. Watch TLT, QQQ/SPY, HYG/IEF, and small-cap relative performance for confirmation.",
        "Bullish Rates Impulse": "The level move is supportive for duration. Confirmation should come from TLT stabilization, softer credit stress, and better performance from equity-duration assets.",
        "Curve Signal": "The curve is moving more than the outright 10Y level. The trade is about policy path, term premium, growth expectations, and front-end anchoring rather than duration alone.",
        "Range / Mixed": "No clean rates impulse. The better read is cross-asset confirmation: whether duration, credit, small caps, and equity-duration ratios are confirming or rejecting the yield move.",
    }
    return reads.get(regime, "Signal quality is low. Check data freshness and missing Yahoo tickers before using the classification.")


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


def build_market_signal_frame(prices: pd.DataFrame) -> pd.DataFrame:
    signals = pd.DataFrame(index=prices.index)

    for ticker, label in YAHOO_MARKET_TICKERS.items():
        if ticker in prices.columns and prices[ticker].dropna().any():
            signals[label] = prices[ticker]

    ratio_defs = {
        "QQQ/SPY Equity Duration": ("QQQ", "SPY"),
        "IWM/SPY Small-Cap Beta": ("IWM", "SPY"),
        "HYG/IEF Credit vs Duration": ("HYG", "IEF"),
        "LQD/IEF IG Credit vs Rates": ("LQD", "IEF"),
        "GLD/SPY Hard Asset Rel.": ("GLD", "SPY"),
        "UUP/SPY Dollar Pressure": ("UUP", "SPY"),
    }

    for label, (num, den) in ratio_defs.items():
        if num in prices.columns and den in prices.columns:
            ratio = prices[num] / prices[den]
            if ratio.dropna().any():
                signals[label] = ratio

    return signals.ffill().dropna(how="all")


def market_return_matrix(signals: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for col in signals.columns:
        if signals[col].dropna().empty:
            continue

        row = {"Series": col}
        for period in PERIODS:
            row[period] = return_pct(signals[col], period)
        rows.append(row)

    return pd.DataFrame(rows)


def build_shock_table(rates: pd.DataFrame, prices: pd.DataFrame, curve_col: str, lookback_days: int, top_n: int) -> pd.DataFrame:
    if "Y10" not in rates.columns or len(rates) < 5:
        return pd.DataFrame()

    pieces = pd.DataFrame(index=rates.index)
    for col in ["Y10", "Y30", curve_col]:
        if col in rates.columns:
            pieces[col] = rates[col].diff() * 100.0

    if "TLT" in prices.columns:
        pieces["TLT"] = prices["TLT"].pct_change() * 100.0

    if pieces.dropna(how="all").empty:
        return pd.DataFrame()

    recent = pieces.tail(lookback_days).copy()
    weights = {"Y10": 1.00, "Y30": 0.75, curve_col: 0.70, "TLT": 0.55}

    score = pd.Series(0.0, index=recent.index)
    for col in recent.columns:
        score = score.add(recent[col].abs().fillna(0.0) * weights.get(col, 0.50), fill_value=0.0)

    shock = recent.copy()
    shock["Shock Score"] = score
    shock = shock.dropna(how="all")
    if shock.empty:
        return pd.DataFrame()

    component_cols = [c for c in ["Y10", "Y30", curve_col, "TLT"] if c in shock.columns]
    shock["Driver"] = shock[component_cols].abs().idxmax(axis=1).replace(
        {
            "Y10": "10Y",
            "Y30": "30Y",
            curve_col: label_for_series(curve_col),
            "TLT": "TLT",
        }
    )

    rename = {
        "Y10": "10Y bp",
        "Y30": "30Y bp",
        curve_col: f"{label_for_series(curve_col)} bp",
        "TLT": "TLT %",
    }

    shock = shock.sort_values("Shock Score", ascending=False).head(top_n).sort_index()
    shock = shock.rename(columns=rename)
    shock.insert(0, "Date", shock.index.date)

    return shock.reset_index(drop=True)


def metric_card(label: str, value: str, footnote: str, color: str) -> None:
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


def clean_plot_layout(fig: go.Figure, height: int, y_title: Optional[str] = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=20, b=18),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#eef2f7", zeroline=False, title_text=y_title)
    return fig


def chart_display_mode() -> Dict[str, object]:
    return {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "responsive": True,
    }


with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Yahoo Finance-only rates monitor. It uses liquid Yahoo yield symbols for the available U.S. curve points and confirms the signal through ETF and relative-performance proxies.

        FRED, Treasury CSVs, real yields, and breakevens are intentionally removed.
        """
    )

    st.divider()
    st.header("Controls")

    lookback_label = st.selectbox("History", ["6M", "1Y", "2Y", "3Y", "5Y", "10Y"], index=4)
    lookback_days_map = {"6M": 190, "1Y": 380, "2Y": 760, "3Y": 1140, "5Y": 1900, "10Y": 3800}
    lookback_days = lookback_days_map[lookback_label]

    regime_period = st.radio("Regime window", list(PERIODS.keys()), index=2, horizontal=True)
    selected_curve = st.selectbox("Curve gauge", CURVE_OPTIONS, index=0)
    curve_compare = st.radio("Curve comparison", ["1W", "1M", "3M", "YTD"], index=1, horizontal=True)

    show_full_history = st.checkbox("Show full history chart", value=True)
    show_market_confirmation = st.checkbox("Show market confirmation", value=True)
    show_table = st.checkbox("Show raw Yahoo table", value=False)
    show_status = st.checkbox("Show Yahoo download status", value=False)


st.markdown(f"<div class='adfm-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='adfm-subtitle'>Yahoo Finance-only curve pressure, duration stress, relative-performance confirmation, and outlier rate days.</div>",
    unsafe_allow_html=True,
)

start = date.today() - timedelta(days=lookback_days + 10)
end = date.today()
all_tickers = tuple(YAHOO_YIELD_TICKERS.keys()) + tuple(YAHOO_MARKET_TICKERS.keys())

with st.spinner("Loading Yahoo Finance data..."):
    yahoo_close, diagnostics = fetch_yahoo_close(all_tickers, start, end)

if yahoo_close.empty:
    st.error("No usable Yahoo Finance data loaded.")
    if diagnostics:
        with st.expander("Yahoo download status", expanded=True):
            st.code("\n".join(diagnostics[-80:]))
    st.stop()

rates_raw, prices = split_yahoo_data(yahoo_close)
rates = add_derived_yahoo_rates(rates_raw)

if rates.empty or "Y10" not in rates.columns or rates["Y10"].dropna().empty:
    st.error("No usable 10Y Treasury yield loaded from Yahoo Finance. The page needs ^TNX to classify the rates regime.")
    if diagnostics:
        with st.expander("Yahoo download status", expanded=True):
            st.code("\n".join(diagnostics[-80:]))
    st.stop()

curve_cols = available_curve_columns(rates)
if not curve_cols:
    st.error("Yahoo loaded the 10Y yield, but not enough curve points to calculate a curve spread.")
    if diagnostics:
        with st.expander("Yahoo download status", expanded=True):
            st.code("\n".join(diagnostics[-80:]))
    st.stop()

if selected_curve not in curve_cols:
    selected_curve = curve_cols[0]

last_obs = latest_date(rates)
if last_obs is not None:
    age_days = (pd.Timestamp(date.today()) - last_obs.normalize()).days
    st.caption(
        f"Source: Yahoo Finance via yfinance. Last yield observation: {last_obs.date()}. "
        "Yield set: ^IRX 3M, ^FVX 5Y, ^TNX 10Y, ^TYX 30Y."
    )
    if age_days > 4:
        st.warning(f"Last Yahoo yield observation is {last_obs.date()}. The rates tape may be stale.")

missing_yield_tickers = [ticker for ticker in YAHOO_YIELD_TICKERS if ticker not in yahoo_close.columns]
if missing_yield_tickers:
    st.warning("Missing Yahoo yield symbols: " + ", ".join(missing_yield_tickers) + ". Outputs are recalculated from the available Yahoo data only.")

if show_status:
    with st.expander("Yahoo download status", expanded=False):
        available = [ticker for ticker in all_tickers if ticker in yahoo_close.columns and yahoo_close[ticker].dropna().any()]
        st.markdown(
            "<div class='data-note'>Available tickers: "
            + escape(", ".join(available))
            + "</div>",
            unsafe_allow_html=True,
        )
        if diagnostics:
            st.code("\n".join(diagnostics[-80:]))

regime, regime_note, regime_color = classify_regime(rates, regime_period, selected_curve)
market_signals = build_market_signal_frame(prices)

tlt_return = return_pct(prices["TLT"], regime_period) if "TLT" in prices.columns else np.nan
qqq_spy = market_signals["QQQ/SPY Equity Duration"] if "QQQ/SPY Equity Duration" in market_signals.columns else pd.Series(dtype=float)
qqq_spy_return = return_pct(qqq_spy, regime_period) if not qqq_spy.empty else np.nan

cards = [
    ("Regime", regime, regime_note, regime_color),
    ("10Y Treasury", fmt_pct(latest(rates["Y10"])), f"{regime_period} {fmt_bp(change_bp(rates['Y10'], regime_period))}", COLORS["blue"]),
    ("30Y Treasury", fmt_pct(latest(rates["Y30"])) if "Y30" in rates else "N/A", f"{regime_period} {fmt_bp(change_bp(rates['Y30'], regime_period))}" if "Y30" in rates else "Unavailable", COLORS["purple"]),
    (label_for_series(selected_curve), fmt_bp(latest(rates[selected_curve]) * 100.0), f"{regime_period} {fmt_bp(change_bp(rates[selected_curve], regime_period))}", COLORS["amber"]),
    ("TLT Check", fmt_ret(tlt_return), f"QQQ/SPY {fmt_ret(qqq_spy_return)} over {regime_period}", COLORS["green"] if np.isfinite(tlt_return) and tlt_return > 0 else COLORS["red"] if np.isfinite(tlt_return) and tlt_return < 0 else COLORS["slate"]),
]

for col, card in zip(st.columns(5), cards):
    with col:
        metric_card(*card)

st.markdown("<div class='section-title'>Read-through</div>", unsafe_allow_html=True)
st.markdown(f"<div class='note-box'>{escape(regime_read(regime))}</div>", unsafe_allow_html=True)

available_yields = [c for c in ["Y3M", "Y5", "Y10", "Y30"] if c in rates.columns and rates[c].dropna().any()]
curve_data = rates[available_yields].dropna(how="all")

left, right = st.columns([1.03, 0.97])

with left:
    st.markdown("<div class='section-title'>Yahoo Yield Curve Snapshot</div>", unsafe_allow_html=True)

    if len(available_yields) < 2 or curve_data.empty:
        st.info("At least two Yahoo yield tenors are needed for the curve snapshot.")
    else:
        latest_curve = curve_data.iloc[-1]
        last_idx = pd.Timestamp(curve_data.index[-1])

        if curve_compare == "1W":
            compare_target = last_idx - pd.DateOffset(days=7)
        elif curve_compare == "1M":
            compare_target = last_idx - pd.DateOffset(months=1)
        elif curve_compare == "3M":
            compare_target = last_idx - pd.DateOffset(months=3)
        else:
            compare_target = pd.Timestamp(date(last_idx.year, 1, 1))

        comparison_curve = []
        for tenor in available_yields:
            if curve_compare == "YTD":
                comparison_curve.append(first_value_on_or_after(curve_data[tenor], compare_target))
            else:
                comparison_curve.append(value_on_or_before(curve_data[tenor], compare_target))

        x_vals = [float(YAHOO_YIELD_TICKERS[FIELD_TO_TICKER[c]]["years"]) for c in available_yields]
        x_labels = [YIELD_LABELS[c] for c in available_yields]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=latest_curve.values,
                mode="lines+markers",
                name="Latest",
                line=dict(color=COLORS["blue"], width=3),
                marker=dict(size=8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=comparison_curve,
                mode="lines+markers",
                name=f"{curve_compare} ago",
                line=dict(color=COLORS["grey"], width=2, dash="dash"),
                marker=dict(size=7),
            )
        )
        fig.update_layout(
            xaxis=dict(title="Tenor", tickvals=x_vals, ticktext=x_labels),
            yaxis=dict(title="Yield (%)"),
        )
        st.plotly_chart(clean_plot_layout(fig, 390), use_container_width=True, config=chart_display_mode())

with right:
    st.markdown("<div class='section-title'>Rates Pressure Matrix</div>", unsafe_allow_html=True)

    matrix_rows = ["Y3M", "Y5", "Y10", "Y30", selected_curve]
    matrix = period_matrix(rates, matrix_rows)

    if matrix.empty:
        st.info("No rates pressure matrix available.")
    else:
        heat_cols = list(PERIODS.keys())
        z = matrix[heat_cols].to_numpy(dtype=float)
        text = np.full(z.shape, "", dtype=object)
        finite_mask = np.isfinite(z)
        text[finite_mask] = np.round(z[finite_mask], 0).astype(int).astype(str)

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=heat_cols,
                y=matrix["Series"],
                colorscale="RdYlGn_r",
                zmid=0,
                text=text,
                texttemplate="%{text}",
                colorbar=dict(title="bps"),
            )
        )
        fig.update_layout(xaxis=dict(side="top"))
        st.plotly_chart(clean_plot_layout(fig, 390), use_container_width=True, config=chart_display_mode())

if show_market_confirmation:
    st.markdown("<div class='section-title'>Yahoo Market Confirmation Tape</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='small-note'>Returns and relative-performance ratios from Yahoo ETFs. This replaces unavailable FRED real-yield and breakeven panels with liquid market proxies.</div>",
        unsafe_allow_html=True,
    )

    market_matrix = market_return_matrix(market_signals)

    if market_matrix.empty:
        st.info("No market confirmation data available from Yahoo tickers.")
    else:
        preferred_order = [
            "TLT 20Y+ Treasury",
            "IEF 7-10Y Treasury",
            "SHY 1-3Y Treasury",
            "QQQ/SPY Equity Duration",
            "IWM/SPY Small-Cap Beta",
            "HYG/IEF Credit vs Duration",
            "LQD/IEF IG Credit vs Rates",
            "GLD/SPY Hard Asset Rel.",
            "UUP/SPY Dollar Pressure",
        ]
        market_matrix["_order"] = market_matrix["Series"].map({name: i for i, name in enumerate(preferred_order)}).fillna(99)
        market_matrix = market_matrix.sort_values(["_order", "Series"]).drop(columns="_order")

        heat_cols = list(PERIODS.keys())
        z = market_matrix[heat_cols].to_numpy(dtype=float)
        text = np.full(z.shape, "", dtype=object)
        finite_mask = np.isfinite(z)
        text[finite_mask] = np.vectorize(lambda v: f"{v:+.1f}%")(z[finite_mask])

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=heat_cols,
                y=market_matrix["Series"],
                colorscale="RdYlGn",
                zmid=0,
                text=text,
                texttemplate="%{text}",
                colorbar=dict(title="%"),
            )
        )
        fig.update_layout(xaxis=dict(side="top"))
        st.plotly_chart(clean_plot_layout(fig, 430), use_container_width=True, config=chart_display_mode())

st.markdown("<div class='section-title'>Outlier Rate Days</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='small-note'>Weighted daily shock score from 10Y, 30Y, selected curve, and TLT. The goal is to flag sessions worth reviewing, not to forecast the next move.</div>",
    unsafe_allow_html=True,
)

shock = build_shock_table(rates, prices, selected_curve, lookback_days=min(len(rates), 300), top_n=10)

if shock.empty:
    st.info("Not enough Yahoo history to calculate outlier days.")
else:
    chart = shock.copy()
    chart["Date Label"] = chart["Date"].astype(str)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart["Date Label"],
            y=chart["Shock Score"],
            text=chart["Driver"],
            textposition="outside",
            name="Shock score",
            marker=dict(color=COLORS["blue"]),
        )
    )
    fig.update_layout(
        xaxis=dict(title="Date"),
        yaxis=dict(title="Weighted shock"),
        showlegend=False,
    )
    st.plotly_chart(clean_plot_layout(fig, 330), use_container_width=True, config=chart_display_mode())

    display_shock = shock.copy()
    for col in display_shock.columns:
        if col.endswith("bp") or col == "Shock Score":
            display_shock[col] = display_shock[col].map(lambda x: fmt_num(safe_float(x), 0))
        if col.endswith("%"):
            display_shock[col] = display_shock[col].map(lambda x: fmt_ret(safe_float(x)))
    st.dataframe(display_shock, use_container_width=True, hide_index=True)

if show_full_history:
    st.markdown("<div class='section-title'>Level + Curve History</div>", unsafe_allow_html=True)

    hist = make_subplots(specs=[[{"secondary_y": True}]])

    hist.add_trace(
        go.Scatter(
            x=rates.index,
            y=rates["Y10"],
            name="10Y",
            line=dict(color=COLORS["blue"], width=2.1),
        ),
        secondary_y=False,
    )

    if "Y30" in rates.columns:
        hist.add_trace(
            go.Scatter(
                x=rates.index,
                y=rates["Y30"],
                name="30Y",
                line=dict(color=COLORS["purple"], width=1.7),
            ),
            secondary_y=False,
        )

    if "Y5" in rates.columns:
        hist.add_trace(
            go.Scatter(
                x=rates.index,
                y=rates["Y5"],
                name="5Y",
                line=dict(color=COLORS["grey"], width=1.3),
            ),
            secondary_y=False,
        )

    hist.add_trace(
        go.Scatter(
            x=rates.index,
            y=rates[selected_curve] * 100.0,
            name=f"{label_for_series(selected_curve)} bp",
            line=dict(color=COLORS["amber"], width=2.0),
        ),
        secondary_y=True,
    )

    hist.update_yaxes(title_text="Yield (%)", secondary_y=False)
    hist.update_yaxes(title_text="Curve bps", secondary_y=True)
    st.plotly_chart(clean_plot_layout(hist, 430), use_container_width=True, config=chart_display_mode())

if show_table:
    st.markdown("<div class='section-title'>Raw Yahoo Data</div>", unsafe_allow_html=True)

    table = pd.DataFrame(index=yahoo_close.index)
    for ticker, meta in YAHOO_YIELD_TICKERS.items():
        field = str(meta["field"])
        label = str(meta["label"])
        if field in rates.columns:
            table[f"{label} Yield"] = rates[field]

    for curve in CURVE_OPTIONS:
        if curve in rates.columns:
            table[f"{label_for_series(curve)} bp"] = rates[curve] * 100.0

    for ticker, label in YAHOO_MARKET_TICKERS.items():
        if ticker in prices.columns:
            table[label] = prices[ticker]

    st.dataframe(table.tail(260), use_container_width=True)

st.markdown(
    """
    <div class='data-note'>
        Method note: Yahoo Finance does not provide the same full macro set as FRED inside this page. The monitor therefore uses only Yahoo-native yield symbols and ETF/ratio confirmation. It does not estimate 2Y yields, real yields, or breakevens.
    </div>
    """,
    unsafe_allow_html=True,
)
