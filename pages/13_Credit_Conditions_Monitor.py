
from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


# ============================== Page config ==============================

TITLE = "Credit Conditions Monitor"
SUBTITLE = (
    "Market-implied credit stress, bank pressure, loan appetite, EM credit, "
    "and rates / volatility backdrop."
)

st.set_page_config(layout="wide", page_title=TITLE, initial_sidebar_state="expanded")


# ============================== Style ====================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.75rem;
            padding-bottom: 2.25rem;
            max-width: 1580px;
        }

        section[data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #e5e7eb;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 0.95rem 1.0rem;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stMetricLabel"] {
            color: #64748b;
            font-weight: 700;
            letter-spacing: 0.035em;
            text-transform: uppercase;
            font-size: 0.72rem;
        }

        div[data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 760;
            font-size: 1.30rem;
        }

        .section-title {
            font-size: 1.03rem;
            font-weight: 760;
            color: #0f172a;
            margin-top: 1.00rem;
            margin-bottom: 0.20rem;
            letter-spacing: -0.01em;
        }

        .section-subtitle {
            font-size: 0.82rem;
            color: #64748b;
            margin-top: 0;
            margin-bottom: 0.55rem;
            line-height: 1.42;
        }

        .note-box {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 13px 15px;
            color: #475569;
            font-size: 0.86rem;
            line-height: 1.48;
            margin-top: 0.95rem;
        }

        .warn-box {
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 14px;
            padding: 11px 13px;
            color: #92400e;
            font-size: 0.84rem;
            line-height: 1.45;
            margin-top: 0.55rem;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================== Defaults =================================

MARKET_TICKERS: Tuple[str, ...] = (
    "HYG",
    "JNK",
    "LQD",
    "BKLN",
    "SRLN",
    "EMB",
    "KRE",
    "XLF",
    "SPY",
    "QQQ",
    "IWM",
    "TLT",
    "IEF",
    "SHY",
    "UUP",
    "GLD",
    "^VIX",
    "^TNX",
    "^IRX",
)

DISPLAY_NAMES: Dict[str, str] = {
    "HYG": "HY Credit",
    "JNK": "HY Credit 2",
    "LQD": "IG Credit",
    "BKLN": "Leveraged Loans",
    "SRLN": "Senior Loans",
    "EMB": "EM USD Debt",
    "KRE": "Regional Banks",
    "XLF": "Financials",
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Small Caps",
    "TLT": "Long Duration",
    "IEF": "Intermediate Duration",
    "SHY": "Bills / Front End",
    "UUP": "Dollar Proxy",
    "GLD": "Gold",
    "^VIX": "VIX",
    "^TNX": "10Y Yield",
    "^IRX": "3M Bill Yield",
}

LOOKBACK_DAYS: Dict[str, Optional[int]] = {
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "2 Years": 365 * 2,
    "3 Years": 365 * 3,
    "5 Years": 365 * 5,
    "10 Years": 365 * 10,
}

FOCUS_WINDOWS: Dict[str, object] = {
    "1D": 1,
    "1W": 7,
    "1M": 30,
    "3M": 91,
    "6M": 182,
    "YTD": "YTD",
}

TRADING_WINDOWS: Dict[str, int] = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
    "3Y": 756,
    "5Y": 1260,
}

COLORS = {
    "green": "#15803d",
    "soft_green": "#dcfce7",
    "red": "#b91c1c",
    "soft_red": "#fee2e2",
    "amber": "#b45309",
    "soft_amber": "#fef3c7",
    "blue": "#1d4ed8",
    "purple": "#6d28d9",
    "teal": "#0f766e",
    "slate": "#334155",
    "muted": "#64748b",
    "border": "#e5e7eb",
    "grid": "#e2e8f0",
    "dark": "#0f172a",
}

LINE_COLORS = [
    "#1d4ed8",
    "#b91c1c",
    "#15803d",
    "#6d28d9",
    "#b45309",
    "#0f766e",
    "#4338ca",
    "#be123c",
    "#475569",
]


# ============================== Specs ====================================

@dataclass(frozen=True)
class ProxySpec:
    name: str
    numerator: str
    denominator: str
    read_through: str


@dataclass(frozen=True)
class RateGridSpec:
    name: str
    source: str
    source_type: str  # rates or market
    mode: str         # diff or pct
    risk_sign: int    # +1 means higher z is tighter; -1 means lower z is tighter


PROXY_SPECS: Tuple[ProxySpec, ...] = (
    ProxySpec("HYG/LQD", "HYG", "LQD", "HY beta versus IG credit. Higher means credit sponsorship is improving."),
    ProxySpec("JNK/LQD", "JNK", "LQD", "Second HY confirmation versus IG credit."),
    ProxySpec("BKLN/LQD", "BKLN", "LQD", "Leveraged loan sponsorship versus IG credit."),
    ProxySpec("SRLN/LQD", "SRLN", "LQD", "Senior loan confirmation versus IG credit."),
    ProxySpec("EMB/LQD", "EMB", "LQD", "EM USD credit versus domestic IG credit."),
    ProxySpec("KRE/SPY", "KRE", "SPY", "Regional banks versus the market. Lower exposes credit transmission risk."),
    ProxySpec("XLF/SPY", "XLF", "SPY", "Financials versus the market. Lower means credit-sensitive equities are losing sponsorship."),
    ProxySpec("IWM/SPY", "IWM", "SPY", "Small caps versus large caps. Lower can flag tighter domestic liquidity."),
    ProxySpec("HYG/SHY", "HYG", "SHY", "HY credit versus front-end Treasury collateral."),
    ProxySpec("LQD/TLT", "LQD", "TLT", "IG credit versus pure duration."),
)

RATE_GRID_SPECS: Tuple[RateGridSpec, ...] = (
    RateGridSpec("10Y Yield", "10Y Yield", "rates", "diff", +1),
    RateGridSpec("3M Bill Yield", "3M Bill Yield", "rates", "diff", +1),
    RateGridSpec("10Y-3M Slope", "10Y-3M Slope", "rates", "diff", +1),
    RateGridSpec("VIX", "^VIX", "market", "diff", +1),
    RateGridSpec("TLT", "TLT", "market", "pct", -1),
    RateGridSpec("IEF", "IEF", "market", "pct", -1),
    RateGridSpec("SHY", "SHY", "market", "pct", -1),
)


# ============================== Sidebar ==================================

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Credit conditions dashboard for regime framing, sponsorship confirmation, and tape discipline.

        **How to read it**
        - Rising HY/IG, banks/SPY, loans/IG, and EM/IG usually signal improving credit sponsorship.
        - Rising VIX, deeper HY drawdowns, and rate volatility usually signal tighter risk appetite.
        - The rates volatility index is built from trailing move z-scores across the curve and duration proxies.
        """
    )

    st.markdown("---")
    st.header("Controls")

    lookback_key = st.selectbox(
        "Chart lookback",
        list(LOOKBACK_DAYS.keys()),
        index=list(LOOKBACK_DAYS.keys()).index("3 Years"),
    )

    focus_window = st.selectbox(
        "Focus window",
        ["1D", "1W", "1M", "3M", "6M", "YTD"],
        index=2,
    )

    stress_window = st.selectbox(
        "Stress percentile window",
        [63, 126, 252, 504],
        index=2,
        format_func=lambda x: f"{x} trading days",
    )

    smoothing_days = st.selectbox(
        "Composite smoothing",
        [1, 3, 5, 10],
        index=1,
        format_func=lambda x: "None" if x == 1 else f"{x} days",
    )

    show_raw = st.checkbox("Show raw data", value=False)


# ============================== Dates ====================================

now = datetime.today()
yf_end = now + timedelta(days=1)
hist_start = now - timedelta(days=365 * 25)

lookback_days = LOOKBACK_DAYS[lookback_key]
display_start = pd.Timestamp(now - timedelta(days=lookback_days)) if lookback_days else pd.Timestamp(datetime(now.year, 1, 1))


# ============================== Helpers ==================================

def clean_ticker(ticker: str) -> str:
    return str(ticker).strip().upper()


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []

    for item in items:
        item = clean_ticker(item)
        if item and item not in seen:
            seen.add(item)
            out.append(item)

    return out


def chunked(items: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_closes(tickers: Tuple[str, ...], start: datetime, end: datetime) -> pd.DataFrame:
    ticker_list = unique_keep_order(tickers)

    if not ticker_list:
        return pd.DataFrame()

    def _extract_close(block: pd.DataFrame) -> Optional[pd.Series]:
        if block is None or block.empty:
            return None

        for field in ("Close", "Adj Close"):
            if field in block.columns:
                return pd.to_numeric(block[field], errors="coerce")

        numeric = block.select_dtypes(include=[np.number])
        if numeric.empty:
            return None

        return pd.to_numeric(numeric.iloc[:, 0], errors="coerce")

    def _normalize(raw: pd.DataFrame, requested: List[str]) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()

        out = pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            level0 = raw.columns.get_level_values(0).astype(str)
            level1 = raw.columns.get_level_values(1).astype(str)

            tickers_in_level0 = set(level0).intersection(requested)
            tickers_in_level1 = set(level1).intersection(requested)

            if tickers_in_level0:
                for ticker in requested:
                    if ticker not in tickers_in_level0:
                        continue

                    try:
                        close = _extract_close(raw[ticker])
                        if close is not None:
                            out[ticker] = close
                    except Exception:
                        continue

            elif tickers_in_level1:
                for ticker in requested:
                    if ticker not in tickers_in_level1:
                        continue

                    for field in ("Close", "Adj Close"):
                        try:
                            if (field, ticker) in raw.columns:
                                out[ticker] = pd.to_numeric(raw[(field, ticker)], errors="coerce")
                                break
                        except Exception:
                            continue

        else:
            if len(requested) == 1:
                close = _extract_close(raw)
                if close is not None:
                    out[requested[0]] = close

        if out.empty:
            return pd.DataFrame()

        out.index = pd.to_datetime(out.index).tz_localize(None)
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]
        out = out.ffill()
        out = out.dropna(how="all")

        return out

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
            normalized = _normalize(raw, batch)

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
            normalized = _normalize(raw, batch)

            if not normalized.empty:
                frames.append(normalized)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out = out.sort_index().ffill().dropna(how="all")

    return out


def latest(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def latest_date(df: pd.DataFrame) -> str:
    if df.empty:
        return "N/A"

    clean = df.dropna(how="all")
    if clean.empty:
        return "N/A"

    return pd.to_datetime(clean.index[-1]).strftime("%Y-%m-%d")


def fmt_num(x: float, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"

    return f"{x:.{digits}f}"


def fmt_price(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"

    if abs(x) >= 100:
        return f"{x:,.1f}"

    return f"{x:,.2f}"


def fmt_pct(x: float, digits: int = 1, signed: bool = True) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"

    sign = "+" if signed and x > 0 else ""
    return f"{sign}{x:.{digits}f}%"


def fmt_level(x: float, suffix: str = "") -> str:
    if x is None or not np.isfinite(x):
        return "n/a"

    return f"{x:.2f}{suffix}"


def fmt_move(raw_value: float, mode: str) -> str:
    if raw_value is None or not np.isfinite(raw_value):
        return "n/a"

    if mode == "pct":
        return f"{raw_value:+.1f}%"

    return f"{raw_value:+.2f}"


def first_valid_on_or_after(series: pd.Series, ts: pd.Timestamp) -> Tuple[pd.Timestamp, float]:
    s = series.dropna()

    if s.empty:
        return pd.NaT, np.nan

    sub = s.loc[ts:]

    if not sub.empty:
        return sub.index[0], float(sub.iloc[0])

    return s.index[-1], float(s.iloc[-1])


def last_valid_on_or_before(series: pd.Series, ts: pd.Timestamp) -> Tuple[pd.Timestamp, float]:
    s = series.dropna()

    if s.empty:
        return pd.NaT, np.nan

    sub = s.loc[:ts]

    if not sub.empty:
        return sub.index[-1], float(sub.iloc[-1])

    return s.index[0], float(s.iloc[0])


def asof_value(series: pd.Series, target: pd.Timestamp) -> float:
    clean = series.dropna().sort_index()

    if clean.empty:
        return np.nan

    eligible = clean.loc[clean.index <= target]

    if eligible.empty:
        return np.nan

    return float(eligible.iloc[-1])


def lookback_target(label: str) -> pd.Timestamp:
    today = pd.Timestamp(datetime.today().date())

    if label == "YTD":
        return pd.Timestamp(datetime(today.year, 1, 1))

    days = FOCUS_WINDOWS.get(label, 30)

    if isinstance(days, int):
        return today - pd.Timedelta(days=days)

    return today - pd.Timedelta(days=30)


def pct_change_since(series: pd.Series, target: pd.Timestamp) -> float:
    clean = series.dropna().sort_index()

    if clean.empty:
        return np.nan

    now_val = latest(clean)
    then_val = asof_value(clean, target)

    if not np.isfinite(now_val) or not np.isfinite(then_val) or then_val == 0:
        return np.nan

    return float(now_val / then_val - 1.0)


def absolute_change_since(series: pd.Series, target: pd.Timestamp) -> float:
    clean = series.dropna().sort_index()

    if clean.empty:
        return np.nan

    now_val = latest(clean)
    then_val = asof_value(clean, target)

    if not np.isfinite(now_val) or not np.isfinite(then_val):
        return np.nan

    return float(now_val - then_val)


def ytd_change(series: pd.Series) -> float:
    s = series.dropna()

    if s.empty:
        return np.nan

    latest_idx = s.index[-1]
    year_start = pd.Timestamp(datetime(latest_idx.year, 1, 1))
    _, base_val = first_valid_on_or_after(s, year_start)
    latest_val = float(s.iloc[-1])

    if not np.isfinite(base_val) or base_val == 0:
        return np.nan

    return latest_val / base_val - 1.0


def period_change(series: pd.Series, periods: int) -> float:
    s = series.dropna()

    if len(s) <= periods:
        return np.nan

    prev = s.iloc[-periods - 1]
    latest_val = s.iloc[-1]

    if not np.isfinite(prev) or prev == 0:
        return np.nan

    return latest_val / prev - 1.0


def drawdown_pct(series: pd.Series, window: int = 126) -> float:
    clean = series.dropna().tail(window)

    if clean.empty:
        return np.nan

    peak = clean.cummax().iloc[-1]

    if not np.isfinite(peak) or peak == 0:
        return np.nan

    return float((clean.iloc[-1] / peak - 1.0) * 100.0)


def rolling_drawdown(series: pd.Series, window: int = 126) -> pd.Series:
    clean = series.astype(float).copy()
    peak = clean.rolling(window=window, min_periods=30).max()

    return clean.divide(peak).subtract(1.0)


def zscore(series: pd.Series, window: int = 252) -> float:
    clean = series.dropna().tail(window)

    if len(clean) < 30:
        return np.nan

    std = clean.std()

    if not np.isfinite(std) or std == 0:
        return np.nan

    return float((clean.iloc[-1] - clean.mean()) / std)


def rolling_percentile_last(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    clean = series.astype(float).copy()

    def pct_rank(values: np.ndarray) -> float:
        s = pd.Series(values).dropna()

        if len(s) < min_periods:
            return np.nan

        return float(s.rank(pct=True).iloc[-1])

    return clean.rolling(window=window, min_periods=min_periods).apply(pct_rank, raw=True)


def rebase(series_or_df, base_date: pd.Timestamp, base: float = 100.0):
    if isinstance(series_or_df, pd.Series):
        s = series_or_df.replace([np.inf, -np.inf], np.nan).dropna()

        if s.empty:
            return pd.Series(dtype=float)

        _, base_val = last_valid_on_or_before(s, base_date)

        if not np.isfinite(base_val) or base_val == 0:
            return pd.Series(dtype=float)

        return s.divide(base_val).multiply(base)

    df = series_or_df.replace([np.inf, -np.inf], np.nan).dropna(how="all").copy()

    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)

    for col in df.columns:
        out[col] = rebase(df[col], base_date, base)

    return out.dropna(how="all")


def stress_color(score: float) -> str:
    if not np.isfinite(score):
        return COLORS["muted"]
    if score >= 0.35:
        return COLORS["red"]
    if score <= -0.35:
        return COLORS["green"]
    return COLORS["amber"]


def classify_credit(score: float) -> str:
    if score >= 0.60:
        return "Stress / De-risk"
    if score >= 0.25:
        return "Tightening"
    if score > -0.20:
        return "Neutral / Watch"
    if score > -0.60:
        return "Constructive"
    return "Easy / Risk-On"


def classify_component(score: float) -> str:
    if not np.isfinite(score):
        return "n/a"
    if score >= 0.60:
        return "Stress"
    if score >= 0.25:
        return "Tightening"
    if score <= -0.60:
        return "Easy"
    if score <= -0.25:
        return "Constructive"
    return "Neutral"


def chart_layout(height: int = 390, showlegend: bool = True) -> Dict[str, object]:
    return dict(
        height=height,
        margin=dict(l=12, r=12, t=12, b=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=12, color=COLORS["slate"]),
        hovermode="x unified",
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
    )


def apply_axis_style(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=COLORS["border"],
        tickfont=dict(color=COLORS["muted"]),
    )
    fig.update_yaxes(
        gridcolor=COLORS["grid"],
        zeroline=False,
        linecolor=COLORS["border"],
        tickfont=dict(color=COLORS["muted"]),
    )

    return fig


def rolling_change_zscore(series: pd.Series, periods: int, mode: str = "diff") -> Tuple[float, float]:
    clean = series.dropna().astype(float)

    if len(clean) <= periods + 30:
        return np.nan, np.nan

    if mode == "pct":
        changes = clean.pct_change(periods) * 100.0
    else:
        changes = clean.diff(periods)

    changes = changes.dropna()

    if len(changes) < 30:
        return np.nan, np.nan

    current = float(changes.iloc[-1])
    std = float(changes.std())

    if not np.isfinite(std) or std == 0:
        return current, np.nan

    z_val = float((current - float(changes.mean())) / std)

    return current, z_val


def rolling_abs_z_index(series_map: Dict[str, Tuple[pd.Series, str]], windows: Dict[str, int]) -> pd.Series:
    component_frames = []

    for _, (series, mode) in series_map.items():
        clean = series.dropna().astype(float)

        if clean.empty:
            continue

        z_cols = []

        for _, periods in windows.items():
            if len(clean) <= periods + 60:
                continue

            if mode == "pct":
                changes = clean.pct_change(periods) * 100.0
            else:
                changes = clean.diff(periods)

            rolling_mean = changes.rolling(504, min_periods=126).mean()
            rolling_std = changes.rolling(504, min_periods=126).std()
            z = (changes - rolling_mean).divide(rolling_std.replace(0, np.nan))
            z_cols.append(z.abs())

        if z_cols:
            component_frames.append(pd.concat(z_cols, axis=1).mean(axis=1))

    if not component_frames:
        return pd.Series(dtype=float)

    index = pd.concat(component_frames, axis=1).mean(axis=1)
    return index.replace([np.inf, -np.inf], np.nan).dropna()


# ============================== Header ===================================

st.title(TITLE)
st.caption(SUBTITLE)


# ============================== Load data ================================

with st.spinner("Downloading market history..."):
    market = fetch_closes(tuple(MARKET_TICKERS), hist_start, yf_end)

if market.empty:
    st.error("Failed to download market data.")
    st.stop()

loaded_tickers = set(market.columns)
missing_tickers = [t for t in MARKET_TICKERS if t not in loaded_tickers]

if missing_tickers:
    st.markdown(
        f"""
        <div class="warn-box">
            <b>Partial load:</b> {', '.join(missing_tickers)} did not load. The dashboard is using available tickers and skipping unavailable signals automatically.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================== Derived series ============================

proxy = pd.DataFrame(index=market.index)
proxy_reads: Dict[str, str] = {}

for spec in PROXY_SPECS:
    if spec.numerator in market.columns and spec.denominator in market.columns:
        denom = market[spec.denominator].replace(0, np.nan)
        proxy[spec.name] = market[spec.numerator].divide(denom)
        proxy_reads[spec.name] = spec.read_through

proxy = proxy.replace([np.inf, -np.inf], np.nan).dropna(how="all").ffill()

rates = pd.DataFrame(index=market.index)

if "^TNX" in market.columns:
    rates["10Y Yield"] = market["^TNX"] / 10.0

if "^IRX" in market.columns:
    rates["3M Bill Yield"] = market["^IRX"] / 10.0

if {"^TNX", "^IRX"}.issubset(market.columns):
    rates["10Y-3M Slope"] = (market["^TNX"] - market["^IRX"]) / 10.0

rates = rates.replace([np.inf, -np.inf], np.nan).dropna(how="all").ffill()


# ============================== Credit stress =============================

stress_components = pd.DataFrame(index=market.index)

for name in [
    "HYG/LQD",
    "JNK/LQD",
    "BKLN/LQD",
    "SRLN/LQD",
    "EMB/LQD",
    "KRE/SPY",
    "XLF/SPY",
    "IWM/SPY",
    "HYG/SHY",
    "LQD/TLT",
]:
    if name in proxy.columns:
        pct = rolling_percentile_last(proxy[name], window=stress_window)
        stress_components[name] = -(pct * 2.0 - 1.0)

if "HYG" in market.columns:
    stress_components["HYG Drawdown"] = np.clip(-rolling_drawdown(market["HYG"], 126) * 8.0, -1.0, 1.0)

if "JNK" in market.columns:
    stress_components["JNK Drawdown"] = np.clip(-rolling_drawdown(market["JNK"], 126) * 8.0, -1.0, 1.0)

if "^VIX" in market.columns:
    vix_pct = rolling_percentile_last(market["^VIX"], window=stress_window)
    stress_components["VIX"] = vix_pct * 2.0 - 1.0

stress_components = stress_components.dropna(how="all").ffill()

if stress_components.empty:
    credit_score_ts = pd.Series(dtype=float)
    credit_score = 0.0
    current_components = pd.Series(dtype=float)
else:
    credit_score_ts = stress_components.mean(axis=1).dropna()

    if smoothing_days > 1:
        credit_score_ts = credit_score_ts.rolling(smoothing_days, min_periods=1).mean()

    credit_score = latest(credit_score_ts) if not credit_score_ts.empty else 0.0
    current_components = stress_components.dropna(how="all").iloc[-1].dropna().sort_values(ascending=False)

credit_regime = classify_credit(credit_score)

if len(current_components) > 0:
    tightening_count = int((current_components >= 0.25).sum())
    easing_count = int((current_components <= -0.25).sum())
    stress_breadth = tightening_count / len(current_components)
else:
    tightening_count = 0
    easing_count = 0
    stress_breadth = np.nan


# ============================== Rates volatility index ====================

def get_grid_series(spec: RateGridSpec) -> pd.Series:
    if spec.source_type == "rates":
        return rates[spec.source] if spec.source in rates.columns else pd.Series(dtype=float)

    return market[spec.source] if spec.source in market.columns else pd.Series(dtype=float)


rate_vol_series_map: Dict[str, Tuple[pd.Series, str]] = {}

for spec in RATE_GRID_SPECS:
    s = get_grid_series(spec)
    if not s.empty:
        rate_vol_series_map[spec.name] = (s, spec.mode)

rates_vol_index = rolling_abs_z_index(rate_vol_series_map, TRADING_WINDOWS)
rates_vol_score = latest(rates_vol_index) if not rates_vol_index.empty else np.nan

rate_grid_z = pd.DataFrame(index=[s.name for s in RATE_GRID_SPECS], columns=list(TRADING_WINDOWS.keys()), dtype=float)
rate_grid_text = pd.DataFrame(index=[s.name for s in RATE_GRID_SPECS], columns=list(TRADING_WINDOWS.keys()), dtype=object)
rate_grid_hover = pd.DataFrame(index=[s.name for s in RATE_GRID_SPECS], columns=list(TRADING_WINDOWS.keys()), dtype=object)

for spec in RATE_GRID_SPECS:
    s = get_grid_series(spec)

    for label, periods in TRADING_WINDOWS.items():
        raw_move, z_val = rolling_change_zscore(s, periods=periods, mode=spec.mode)
        signed_risk_z = z_val * spec.risk_sign if np.isfinite(z_val) else np.nan

        rate_grid_z.loc[spec.name, label] = signed_risk_z
        rate_grid_text.loc[spec.name, label] = "" if not np.isfinite(signed_risk_z) else f"{signed_risk_z:.1f}"
        rate_grid_hover.loc[spec.name, label] = (
            f"Move: {fmt_move(raw_move, spec.mode)}<br>"
            f"Risk z-score: {fmt_num(signed_risk_z, 2)}"
        )


# ============================== Metric cards ==============================

target = lookback_target(focus_window)

hyg_lqd_move = pct_change_since(proxy["HYG/LQD"], target) * 100 if "HYG/LQD" in proxy.columns else np.nan
kre_spy_move = pct_change_since(proxy["KRE/SPY"], target) * 100 if "KRE/SPY" in proxy.columns else np.nan
bkl_lqd_move = pct_change_since(proxy["BKLN/LQD"], target) * 100 if "BKLN/LQD" in proxy.columns else np.nan
vix_level = latest(market["^VIX"]) if "^VIX" in market.columns else np.nan
vix_z = zscore(market["^VIX"], 252) if "^VIX" in market.columns else np.nan

cols = st.columns(6)

cols[0].metric(
    "Credit Regime",
    credit_regime,
    f"Composite {credit_score:+.2f}",
)

cols[1].metric(
    "Stress Breadth",
    fmt_pct(stress_breadth * 100, 0, signed=False),
    f"{tightening_count} tightening / {easing_count} easing",
)

cols[2].metric(
    "HY vs IG",
    fmt_pct(hyg_lqd_move, 1, signed=True),
    f"HYG/LQD {focus_window}",
)

cols[3].metric(
    "Bank Beta",
    fmt_pct(kre_spy_move, 1, signed=True),
    f"KRE/SPY {focus_window}",
)

cols[4].metric(
    "Loan Appetite",
    fmt_pct(bkl_lqd_move, 1, signed=True),
    f"BKLN/LQD {focus_window}",
)

cols[5].metric(
    "Rates Vol Index",
    fmt_num(rates_vol_score, 2),
    f"VIX {fmt_price(vix_level)} | z {fmt_num(vix_z, 2)}",
)


# ============================== Top charts ================================

top_left, top_right = st.columns([1.15, 0.85])

with top_left:
    st.markdown('<div class="section-title">Credit Stress Pulse</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Composite of credit, bank, loan, EM, drawdown, and volatility signals. Higher means tighter credit conditions.</div>',
        unsafe_allow_html=True,
    )

    if credit_score_ts.empty:
        st.info("Composite stress could not be calculated with the available data.")
    else:
        view = credit_score_ts.loc[display_start:]
        if view.empty:
            view = credit_score_ts

        fig = go.Figure()
        fig.add_hrect(y0=0.60, y1=1.05, fillcolor=COLORS["soft_red"], opacity=0.45, line_width=0)
        fig.add_hrect(y0=0.25, y1=0.60, fillcolor=COLORS["soft_amber"], opacity=0.45, line_width=0)
        fig.add_hrect(y0=-1.05, y1=-0.25, fillcolor=COLORS["soft_green"], opacity=0.35, line_width=0)

        fig.add_trace(
            go.Scatter(
                x=view.index,
                y=view.values,
                mode="lines",
                name="Composite Stress",
                line=dict(color=stress_color(credit_score), width=2.8),
            )
        )

        fig.add_hline(y=0, line_width=1, line_color=COLORS["grid"])
        fig.add_hline(y=0.60, line_width=1, line_dash="dot", line_color=COLORS["red"])
        fig.add_hline(y=0.25, line_width=1, line_dash="dot", line_color=COLORS["amber"])
        fig.add_hline(y=-0.25, line_width=1, line_dash="dot", line_color=COLORS["green"])

        fig.update_layout(**chart_layout(height=395, showlegend=False))
        fig.update_yaxes(title_text="Stress score", range=[-1.05, 1.05])
        apply_axis_style(fig)
        st.plotly_chart(fig, use_container_width=True)

with top_right:
    st.markdown('<div class="section-title">Pressure Map</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Current component scores. Positive bars are tightening. Negative bars are easing.</div>',
        unsafe_allow_html=True,
    )

    if current_components.empty:
        st.info("Pressure map unavailable.")
    else:
        bar_df = current_components.sort_values(ascending=True).reset_index()
        bar_df.columns = ["Signal", "Stress"]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=bar_df["Stress"],
                y=bar_df["Signal"],
                orientation="h",
                marker=dict(
                    color=[stress_color(v) for v in bar_df["Stress"]],
                    line=dict(color="rgba(15, 23, 42, 0.10)", width=1),
                ),
                hovertemplate="%{y}: %{x:.2f}<extra></extra>",
                name="Stress",
            )
        )

        fig.add_vline(x=0, line_width=1, line_color=COLORS["grid"])
        fig.add_vline(x=0.25, line_width=1, line_dash="dot", line_color=COLORS["amber"])
        fig.add_vline(x=0.60, line_width=1, line_dash="dot", line_color=COLORS["red"])
        fig.add_vline(x=-0.25, line_width=1, line_dash="dot", line_color=COLORS["green"])

        fig.update_layout(**chart_layout(height=395, showlegend=False))
        fig.update_xaxes(title_text="Stress score", range=[-1.05, 1.05])
        apply_axis_style(fig)
        st.plotly_chart(fig, use_container_width=True)


# ============================== Rates module ==============================

rate_left, rate_right = st.columns([1.0, 1.0])

with rate_left:
    st.markdown('<div class="section-title">Rates Volatility Index</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Average absolute z-score of trailing 1M, 3M, 6M, 1Y, 3Y, and 5Y moves across rates, curve, duration, and VIX proxies.</div>',
        unsafe_allow_html=True,
    )

    if rates_vol_index.empty:
        st.info("Rates volatility index unavailable.")
    else:
        rv_view = rates_vol_index.loc[display_start:]
        if rv_view.empty:
            rv_view = rates_vol_index

        fig = go.Figure()
        fig.add_hrect(y0=1.50, y1=max(3.5, float(rv_view.max()) * 1.05), fillcolor=COLORS["soft_red"], opacity=0.35, line_width=0)
        fig.add_hrect(y0=1.00, y1=1.50, fillcolor=COLORS["soft_amber"], opacity=0.35, line_width=0)

        fig.add_trace(
            go.Scatter(
                x=rv_view.index,
                y=rv_view.values,
                mode="lines",
                name="Rates Vol Index",
                line=dict(color=COLORS["purple"], width=2.6),
                hovertemplate="%{y:.2f}<extra>Rates Vol Index</extra>",
            )
        )

        fig.add_hline(y=1.0, line_width=1, line_dash="dot", line_color=COLORS["amber"])
        fig.add_hline(y=1.5, line_width=1, line_dash="dot", line_color=COLORS["red"])
        fig.add_hline(y=0.0, line_width=1, line_color=COLORS["grid"])

        fig.update_layout(**chart_layout(height=400, showlegend=False))
        fig.update_yaxes(title_text="Average |z-score|")
        apply_axis_style(fig)
        st.plotly_chart(fig, use_container_width=True)

with rate_right:
    st.markdown('<div class="section-title">Rates / Volatility Regime Grid</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Risk-adjusted trailing move z-scores. Positive cells mean the move is tightening versus its own history. Negative cells mean easing.</div>',
        unsafe_allow_html=True,
    )

    if rate_grid_z.dropna(how="all").empty:
        st.info("Rates / volatility grid unavailable.")
    else:
        fig = go.Figure(
            data=go.Heatmap(
                z=rate_grid_z.values.astype(float),
                x=list(rate_grid_z.columns),
                y=list(rate_grid_z.index),
                zmin=-2.5,
                zmax=2.5,
                colorscale=[
                    [0.00, "#15803d"],
                    [0.20, "#86efac"],
                    [0.50, "#f8fafc"],
                    [0.80, "#fca5a5"],
                    [1.00, "#b91c1c"],
                ],
                text=rate_grid_text.values,
                texttemplate="%{text}",
                textfont={"size": 12},
                customdata=rate_grid_hover.values,
                colorbar=dict(title="risk z", thickness=12, len=0.82),
                hovertemplate="%{y}<br>%{x}<br>%{customdata}<extra></extra>",
            )
        )

        fig.update_layout(**chart_layout(height=400, showlegend=False))
        fig.update_xaxes(side="top", tickfont=dict(color=COLORS["slate"]))
        fig.update_yaxes(tickfont=dict(color=COLORS["slate"]))
        st.plotly_chart(fig, use_container_width=True)


# ============================== Confirmation charts =======================

conf_left, conf_right = st.columns([1.0, 1.0])

with conf_left:
    st.markdown('<div class="section-title">Credit Risk Appetite</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Relative proxy basket rebased to 100. Rising lines generally indicate improving credit sponsorship.</div>',
        unsafe_allow_html=True,
    )

    proxy_cols = [c for c in ["HYG/LQD", "JNK/LQD", "BKLN/LQD", "EMB/LQD", "KRE/SPY", "XLF/SPY"] if c in proxy.columns]

    if not proxy_cols:
        st.info("Relative credit proxy data unavailable.")
    else:
        rebased = rebase(proxy[proxy_cols], display_start, 100.0)
        rebased = rebased.loc[display_start:]

        fig = go.Figure()

        for i, col_name in enumerate(rebased.columns):
            fig.add_trace(
                go.Scatter(
                    x=rebased.index,
                    y=rebased[col_name],
                    mode="lines",
                    name=col_name,
                    line=dict(color=LINE_COLORS[i % len(LINE_COLORS)], width=2),
                )
            )

        fig.add_hline(y=100, line_width=1, line_color=COLORS["grid"])
        fig.update_layout(**chart_layout(height=385, showlegend=True))
        fig.update_yaxes(title_text="Rebased to 100")
        apply_axis_style(fig)
        st.plotly_chart(fig, use_container_width=True)

with conf_right:
    st.markdown('<div class="section-title">Macro Confirmation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Cross-asset confirmation around credit: duration, dollar, gold, equities, and small caps.</div>',
        unsafe_allow_html=True,
    )

    macro_cols = [c for c in ["SPY", "QQQ", "IWM", "TLT", "IEF", "UUP", "GLD"] if c in market.columns]

    if not macro_cols:
        st.info("Macro confirmation data unavailable.")
    else:
        macro = market[macro_cols].rename(columns=DISPLAY_NAMES)
        rebased = rebase(macro, display_start, 100.0)
        rebased = rebased.loc[display_start:]

        fig = go.Figure()

        for i, col_name in enumerate(rebased.columns):
            fig.add_trace(
                go.Scatter(
                    x=rebased.index,
                    y=rebased[col_name],
                    mode="lines",
                    name=col_name,
                    line=dict(color=LINE_COLORS[i % len(LINE_COLORS)], width=2),
                )
            )

        fig.add_hline(y=100, line_width=1, line_color=COLORS["grid"])
        fig.update_layout(**chart_layout(height=385, showlegend=True))
        fig.update_yaxes(title_text="Rebased to 100")
        apply_axis_style(fig)
        st.plotly_chart(fig, use_container_width=True)


# ============================== Credit tape ===============================

st.markdown('<div class="section-title">Credit Tape</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Decision table across ratio proxies, ETFs, rates, and volatility. Ratio and ETF moves are percentage changes. Rates and VIX are absolute moves.</div>',
    unsafe_allow_html=True,
)

tape_rows: List[Dict[str, object]] = []


def add_ratio_tape(name: str, series: pd.Series, read: str) -> None:
    current_stress = latest(stress_components[name]) if name in stress_components.columns else np.nan

    tape_rows.append(
        {
            "Signal": name,
            "Latest": fmt_num(latest(series), 3),
            "1W": fmt_pct(pct_change_since(series, lookback_target("1W")) * 100),
            "1M": fmt_pct(pct_change_since(series, lookback_target("1M")) * 100),
            "3M": fmt_pct(pct_change_since(series, lookback_target("3M")) * 100),
            "6M": fmt_pct(pct_change_since(series, lookback_target("6M")) * 100),
            "YTD": fmt_pct(pct_change_since(series, lookback_target("YTD")) * 100),
            "Z": fmt_num(zscore(series, 252), 2),
            "Stress": fmt_num(current_stress, 2),
            "Regime": classify_component(current_stress),
            "Read-through": read,
        }
    )


def add_price_tape(ticker: str, series: pd.Series, read: str) -> None:
    drawdown_name = f"{ticker} Drawdown"
    current_stress = latest(stress_components[drawdown_name]) if drawdown_name in stress_components.columns else np.nan

    tape_rows.append(
        {
            "Signal": DISPLAY_NAMES.get(ticker, ticker),
            "Latest": fmt_price(latest(series)),
            "1W": fmt_pct(pct_change_since(series, lookback_target("1W")) * 100),
            "1M": fmt_pct(pct_change_since(series, lookback_target("1M")) * 100),
            "3M": fmt_pct(pct_change_since(series, lookback_target("3M")) * 100),
            "6M": fmt_pct(pct_change_since(series, lookback_target("6M")) * 100),
            "YTD": fmt_pct(pct_change_since(series, lookback_target("YTD")) * 100),
            "Z": fmt_num(zscore(series, 252), 2),
            "Stress": fmt_num(current_stress, 2),
            "Regime": classify_component(current_stress) if np.isfinite(current_stress) else "Price",
            "Read-through": read,
        }
    )


def add_absolute_tape(name: str, series: pd.Series, latest_formatter, read: str) -> None:
    tape_rows.append(
        {
            "Signal": name,
            "Latest": latest_formatter(latest(series)),
            "1W": fmt_num(absolute_change_since(series, lookback_target("1W")), 2),
            "1M": fmt_num(absolute_change_since(series, lookback_target("1M")), 2),
            "3M": fmt_num(absolute_change_since(series, lookback_target("3M")), 2),
            "6M": fmt_num(absolute_change_since(series, lookback_target("6M")), 2),
            "YTD": fmt_num(absolute_change_since(series, lookback_target("YTD")), 2),
            "Z": fmt_num(zscore(series, 252), 2),
            "Stress": fmt_num(latest(stress_components[name]) if name in stress_components.columns else np.nan, 2),
            "Regime": classify_component(latest(stress_components[name])) if name in stress_components.columns else "Macro",
            "Read-through": read,
        }
    )


for name in [
    "HYG/LQD",
    "JNK/LQD",
    "BKLN/LQD",
    "SRLN/LQD",
    "EMB/LQD",
    "KRE/SPY",
    "XLF/SPY",
    "IWM/SPY",
    "HYG/SHY",
    "LQD/TLT",
]:
    if name in proxy.columns:
        add_ratio_tape(name, proxy[name], proxy_reads.get(name, "Relative credit proxy."))

for ticker, read in [
    ("HYG", "HY ETF price. Persistent drawdown is a market-implied spread warning."),
    ("JNK", "Second high-yield ETF confirmation."),
    ("LQD", "IG credit price. Useful for separating credit and duration stress imperfectly."),
    ("BKLN", "Loan ETF price. Tracks floating-rate credit appetite."),
    ("EMB", "EM hard-currency debt price. Flags global credit pressure."),
    ("KRE", "Regional bank equity proxy. Often leads tightening in credit transmission."),
    ("SPY", "Equity beta confirmation."),
]:
    if ticker in market.columns:
        add_price_tape(ticker, market[ticker], read)

if "^VIX" in market.columns:
    add_absolute_tape("VIX", market["^VIX"], fmt_price, "Equity volatility. Higher means tighter risk appetite.")

if "10Y Yield" in rates.columns:
    add_absolute_tape("10Y Yield", rates["10Y Yield"], lambda x: f"{x:.2f}%" if np.isfinite(x) else "n/a", "Rates pressure. Move shown in percentage points.")

if "3M Bill Yield" in rates.columns:
    add_absolute_tape("3M Bill Yield", rates["3M Bill Yield"], lambda x: f"{x:.2f}%" if np.isfinite(x) else "n/a", "Front-end policy rate pressure. Move shown in percentage points.")

if "10Y-3M Slope" in rates.columns:
    add_absolute_tape("10Y-3M Slope", rates["10Y-3M Slope"], lambda x: f"{x:.2f}%" if np.isfinite(x) else "n/a", "Curve slope. Move shown in percentage points.")

if tape_rows:
    tape = pd.DataFrame(tape_rows)
    st.dataframe(tape, use_container_width=True, hide_index=True, height=480)
else:
    st.info("No tape signals loaded.")


# ============================== Interpretation ============================

if credit_score >= 0.60:
    readthrough = (
        "Credit proxies are in stress mode. When HY underperforms IG, KRE loses to SPY, drawdowns deepen, "
        "and VIX rises together, the tape is saying de-risking pressure has moved into credit transmission."
    )
elif credit_score >= 0.25:
    readthrough = (
        "Credit is tightening. Equity rallies need confirmation from HYG/LQD and KRE/SPY here. "
        "If those ratios fail while VIX stays bid or HY drawdowns deepen, the rally is losing credit sponsorship."
    )
elif credit_score <= -0.25:
    readthrough = (
        "Credit is supportive. HY, loans, banks, or volatility proxies are not confirming broad funding stress. "
        "That keeps the burden of proof on equity bears unless the bank and HY ratios roll over."
    )
else:
    readthrough = (
        "Credit is neutral. The composite is not strong enough by itself. "
        "The next useful break should come from HYG/LQD, KRE/SPY, BKLN/LQD, and VIX moving in the same direction."
    )

st.markdown(
    f"""
    <div class="note-box">
        <b>Read-through:</b> {readthrough}<br>
        <b>Framework:</b> This is a market-implied monitor. Outputs are proxy signals from liquid instruments rather than cash OAS series.<br>
        <b>Latest close:</b> {latest_date(market)}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================== Raw data =================================

if show_raw:
    with st.expander("Raw Close Data", expanded=False):
        renamed_market = market.rename(columns=DISPLAY_NAMES)
        st.dataframe(renamed_market.tail(500), use_container_width=True)

    with st.expander("Derived Proxy Data", expanded=False):
        if proxy.empty:
            st.info("No proxy data calculated.")
        else:
            st.dataframe(proxy.tail(500), use_container_width=True)

    with st.expander("Stress Components", expanded=False):
        if stress_components.empty:
            st.info("No stress components calculated.")
        else:
            st.dataframe(stress_components.tail(500), use_container_width=True)

    with st.expander("Rates / Volatility Grid Data", expanded=False):
        st.dataframe(rate_grid_z, use_container_width=True)

    with st.expander("Rates Volatility Index", expanded=False):
        if rates_vol_index.empty:
            st.info("No rates volatility index calculated.")
        else:
            st.dataframe(rates_vol_index.rename("Rates Vol Index").to_frame().tail(500), use_container_width=True)

st.caption("© 2026 AD Fund Management LP")
