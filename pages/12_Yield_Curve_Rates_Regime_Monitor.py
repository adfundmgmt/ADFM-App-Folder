from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from html import escape
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


TITLE = "Yield Curve Rates Regime"

PERIODS: Dict[str, Dict[str, object]] = {
    "Today": {"kind": "row", "rows": 1, "threshold": 4},
    "1W": {"kind": "calendar", "days": 7, "threshold": 8},
    "1M": {"kind": "calendar", "months": 1, "threshold": 15},
    "3M": {"kind": "calendar", "months": 3, "threshold": 30},
    "YTD": {"kind": "ytd", "threshold": 35},
}

MATURITY_ORDER = ["2Y", "5Y", "10Y", "30Y"]
MATURITY_YEARS = {"2Y": 2.0, "5Y": 5.0, "10Y": 10.0, "30Y": 30.0}

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


@dataclass(frozen=True)
class YieldInstrument:
    country: str
    region: str
    bucket: str
    maturity: str
    tickers: Tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.country}|{self.maturity}"


GLOBAL_YIELD_CANDIDATES: Tuple[YieldInstrument, ...] = (
    YieldInstrument("United States", "Americas", "DM", "2Y", ("US2YT=X", "US2YT=RR")),
    YieldInstrument("United States", "Americas", "DM", "5Y", ("^FVX", "US5YT=X", "US5YT=RR")),
    YieldInstrument("United States", "Americas", "DM", "10Y", ("^TNX", "US10YT=X", "US10YT=RR")),
    YieldInstrument("United States", "Americas", "DM", "30Y", ("^TYX", "US30YT=X", "US30YT=RR")),

    YieldInstrument("Germany", "Europe", "DM", "2Y", ("DE2YT=RR",)),
    YieldInstrument("Germany", "Europe", "DM", "5Y", ("DE5YT=RR",)),
    YieldInstrument("Germany", "Europe", "DM", "10Y", ("DE10YT=RR",)),
    YieldInstrument("Germany", "Europe", "DM", "30Y", ("DE30YT=RR",)),

    YieldInstrument("United Kingdom", "Europe", "DM", "2Y", ("GB2YT=RR",)),
    YieldInstrument("United Kingdom", "Europe", "DM", "5Y", ("GB5YT=RR",)),
    YieldInstrument("United Kingdom", "Europe", "DM", "10Y", ("GB10YT=RR",)),
    YieldInstrument("United Kingdom", "Europe", "DM", "30Y", ("GB30YT=RR",)),

    YieldInstrument("Japan", "Asia", "DM", "2Y", ("JP2YT=RR", "JP2YT=XX")),
    YieldInstrument("Japan", "Asia", "DM", "5Y", ("JP5YT=RR", "JP5YT=XX")),
    YieldInstrument("Japan", "Asia", "DM", "10Y", ("JP10YT=RR", "JP10YT=XX")),
    YieldInstrument("Japan", "Asia", "DM", "30Y", ("JP30YT=RR", "JP30YT=XX")),

    YieldInstrument("France", "Europe", "DM", "2Y", ("FR2YT=RR",)),
    YieldInstrument("France", "Europe", "DM", "5Y", ("FR5YT=RR",)),
    YieldInstrument("France", "Europe", "DM", "10Y", ("FR10YT=RR",)),
    YieldInstrument("France", "Europe", "DM", "30Y", ("FR30YT=RR",)),

    YieldInstrument("Italy", "Europe", "DM", "2Y", ("IT2YT=RR",)),
    YieldInstrument("Italy", "Europe", "DM", "5Y", ("IT5YT=RR",)),
    YieldInstrument("Italy", "Europe", "DM", "10Y", ("IT10YT=RR",)),
    YieldInstrument("Italy", "Europe", "DM", "30Y", ("IT30YT=RR",)),

    YieldInstrument("Spain", "Europe", "DM", "2Y", ("ES2YT=RR",)),
    YieldInstrument("Spain", "Europe", "DM", "5Y", ("ES5YT=RR",)),
    YieldInstrument("Spain", "Europe", "DM", "10Y", ("ES10YT=RR",)),
    YieldInstrument("Spain", "Europe", "DM", "30Y", ("ES30YT=RR",)),

    YieldInstrument("Netherlands", "Europe", "DM", "2Y", ("NL2YT=RR",)),
    YieldInstrument("Netherlands", "Europe", "DM", "5Y", ("NL5YT=RR",)),
    YieldInstrument("Netherlands", "Europe", "DM", "10Y", ("NL10YT=RR",)),
    YieldInstrument("Netherlands", "Europe", "DM", "30Y", ("NL30YT=RR",)),

    YieldInstrument("Switzerland", "Europe", "DM", "2Y", ("CH2YT=RR",)),
    YieldInstrument("Switzerland", "Europe", "DM", "5Y", ("CH5YT=RR",)),
    YieldInstrument("Switzerland", "Europe", "DM", "10Y", ("CH10YT=RR",)),
    YieldInstrument("Switzerland", "Europe", "DM", "30Y", ("CH30YT=RR",)),

    YieldInstrument("Canada", "Americas", "DM", "2Y", ("CA2YT=RR",)),
    YieldInstrument("Canada", "Americas", "DM", "5Y", ("CA5YT=RR",)),
    YieldInstrument("Canada", "Americas", "DM", "10Y", ("CA10YT=RR",)),
    YieldInstrument("Canada", "Americas", "DM", "30Y", ("CA30YT=RR",)),

    YieldInstrument("Australia", "Asia", "DM", "2Y", ("AU2YT=RR",)),
    YieldInstrument("Australia", "Asia", "DM", "5Y", ("AU5YT=RR",)),
    YieldInstrument("Australia", "Asia", "DM", "10Y", ("AU10YT=RR",)),
    YieldInstrument("Australia", "Asia", "DM", "30Y", ("AU30YT=RR",)),

    YieldInstrument("China", "Asia", "EM", "2Y", ("CN2YT=RR",)),
    YieldInstrument("China", "Asia", "EM", "5Y", ("CN5YT=RR",)),
    YieldInstrument("China", "Asia", "EM", "10Y", ("CN10YT=RR",)),
    YieldInstrument("China", "Asia", "EM", "30Y", ("CN30YT=RR",)),

    YieldInstrument("South Korea", "Asia", "DM", "2Y", ("KR2YT=RR",)),
    YieldInstrument("South Korea", "Asia", "DM", "5Y", ("KR5YT=RR",)),
    YieldInstrument("South Korea", "Asia", "DM", "10Y", ("KR10YT=RR",)),
    YieldInstrument("South Korea", "Asia", "DM", "30Y", ("KR30YT=RR",)),

    YieldInstrument("India", "Asia", "EM", "2Y", ("IN2YT=RR",)),
    YieldInstrument("India", "Asia", "EM", "5Y", ("IN5YT=RR",)),
    YieldInstrument("India", "Asia", "EM", "10Y", ("IN10YT=RR",)),
    YieldInstrument("India", "Asia", "EM", "30Y", ("IN30YT=RR",)),

    YieldInstrument("Brazil", "Americas", "EM", "2Y", ("BR2YT=RR",)),
    YieldInstrument("Brazil", "Americas", "EM", "5Y", ("BR5YT=RR",)),
    YieldInstrument("Brazil", "Americas", "EM", "10Y", ("BR10YT=RR",)),
    YieldInstrument("Brazil", "Americas", "EM", "30Y", ("BR30YT=RR",)),

    YieldInstrument("Mexico", "Americas", "EM", "2Y", ("MX2YT=RR",)),
    YieldInstrument("Mexico", "Americas", "EM", "5Y", ("MX5YT=RR",)),
    YieldInstrument("Mexico", "Americas", "EM", "10Y", ("MX10YT=RR",)),
    YieldInstrument("Mexico", "Americas", "EM", "30Y", ("MX30YT=RR",)),

    YieldInstrument("South Africa", "EMEA", "EM", "2Y", ("ZA2YT=RR",)),
    YieldInstrument("South Africa", "EMEA", "EM", "5Y", ("ZA5YT=RR",)),
    YieldInstrument("South Africa", "EMEA", "EM", "10Y", ("ZA10YT=RR",)),
    YieldInstrument("South Africa", "EMEA", "EM", "30Y", ("ZA30YT=RR",)),

    YieldInstrument("Greece", "Europe", "DM", "2Y", ("GR2YT=RR",)),
    YieldInstrument("Greece", "Europe", "DM", "5Y", ("GR5YT=RR",)),
    YieldInstrument("Greece", "Europe", "DM", "10Y", ("GR10YT=RR",)),
    YieldInstrument("Greece", "Europe", "DM", "30Y", ("GR30YT=RR",)),

    YieldInstrument("Portugal", "Europe", "DM", "2Y", ("PT2YT=RR",)),
    YieldInstrument("Portugal", "Europe", "DM", "5Y", ("PT5YT=RR",)),
    YieldInstrument("Portugal", "Europe", "DM", "10Y", ("PT10YT=RR",)),
    YieldInstrument("Portugal", "Europe", "DM", "30Y", ("PT30YT=RR",)),
)

MARKET_TICKERS: Dict[str, str] = {
    "TLT": "TLT Long U.S. Duration",
    "IEF": "IEF Intermediate U.S. Duration",
    "SHY": "SHY Front-End U.S.",
    "BWX": "BWX Global ex-U.S. Sov.",
    "IGOV": "IGOV International Sov.",
    "EMB": "EMB EM Sovereign Credit",
    "HYG": "HYG U.S. High Yield",
    "LQD": "LQD U.S. IG Credit",
    "SPY": "SPY S&P 500",
    "QQQ": "QQQ Nasdaq 100",
    "IWM": "IWM Russell 2000",
    "UUP": "UUP Dollar",
    "GLD": "GLD Gold",
}

st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.10rem;
            padding-bottom: 2.00rem;
            max-width: 1560px;
        }

        .adfm-title {
            font-size: 1.74rem;
            line-height: 1.10;
            font-weight: 770;
            margin-bottom: 0.18rem;
            color: #0f172a;
            letter-spacing: -0.030em;
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
            font-size: 1.22rem;
            font-weight: 770;
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
            font-weight: 770;
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

        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            overflow: hidden;
        }

        [data-testid="stSidebar"] .stRadio > label,
        [data-testid="stSidebar"] .stSelectbox > label,
        [data-testid="stSidebar"] .stCheckbox > label,
        [data-testid="stSidebar"] .stNumberInput > label {
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


def unique_preserve_order(values: Iterable[str]) -> Tuple[str, ...]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)


def normalize_yield_series(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").astype(float)
    median = out.dropna().tail(260).median()

    # Cboe Treasury yield indices may arrive as 45.4 for a 4.54% yield through some endpoints.
    if np.isfinite(median) and median > 20:
        out = out / 10.0

    # Guardrail for decimal-form rates, while leaving low-yield countries such as Switzerland untouched.
    if np.isfinite(median) and 0 < median < 0.25:
        out = out * 100.0

    return out.replace([np.inf, -np.inf], np.nan)


def extract_close_frame(raw: pd.DataFrame, tickers: Tuple[str, ...]) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
    diagnostics: List[str] = []

    if raw is None or raw.empty:
        return pd.DataFrame(), ("Market-data download returned an empty frame.",)

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
            return pd.DataFrame(), ("Market-data frame did not include Close or Adj Close columns.",)
    else:
        close_col = "Close" if "Close" in raw.columns else "Adj Close" if "Adj Close" in raw.columns else None
        if close_col is None:
            return pd.DataFrame(), ("Market-data frame did not include a Close column.",)
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
            diagnostics.append(f"{ticker}: missing from close frame")

    present_cols = [ticker for ticker in tickers if ticker in data.columns]
    if not present_cols:
        return pd.DataFrame(), tuple(diagnostics + ["No requested tickers were present in the close frame."])

    return data[present_cols].dropna(how="all"), tuple(diagnostics)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_close_prices(tickers: Tuple[str, ...], start_date: date, end_date: date) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
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
        return pd.DataFrame(), (f"Market-data download failed: {type(exc).__name__}: {exc}",)

    data, extract_diag = extract_close_frame(raw, tickers)
    diagnostics.extend(extract_diag)
    return data, tuple(diagnostics)


def choose_yield_series(
    close: pd.DataFrame,
    instruments: Tuple[YieldInstrument, ...],
    min_obs: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    series_map: Dict[str, pd.Series] = {}
    meta_rows: List[Dict[str, object]] = []

    for instrument in instruments:
        selected_ticker: Optional[str] = None
        selected_series = pd.Series(dtype=float)

        for ticker in instrument.tickers:
            if ticker not in close.columns:
                continue
            candidate = normalize_yield_series(close[ticker]).dropna()
            if len(candidate) >= min_obs:
                selected_ticker = ticker
                selected_series = normalize_yield_series(close[ticker])
                break

        if selected_ticker is None:
            continue

        series_map[instrument.key] = selected_series
        meta_rows.append(
            {
                "Country": instrument.country,
                "Region": instrument.region,
                "Bucket": instrument.bucket,
                "Maturity": instrument.maturity,
                "Ticker": selected_ticker,
                "Key": instrument.key,
            }
        )

    if not series_map:
        return pd.DataFrame(), pd.DataFrame()

    yields = pd.DataFrame(series_map).sort_index()
    yields = yields[~yields.index.duplicated(keep="last")]
    yields = yields.ffill().dropna(how="all")
    meta = pd.DataFrame(meta_rows)
    return yields, meta


def split_market_prices(close: pd.DataFrame) -> pd.DataFrame:
    cols = [ticker for ticker in MARKET_TICKERS if ticker in close.columns and close[ticker].dropna().any()]
    if not cols:
        return pd.DataFrame(index=close.index)
    return close[cols].apply(pd.to_numeric, errors="coerce").ffill().dropna(how="all")


def available_countries(meta: pd.DataFrame, maturity: Optional[str] = None) -> List[str]:
    if meta.empty:
        return []
    frame = meta.copy()
    if maturity:
        frame = frame[frame["Maturity"] == maturity]
    return sorted(frame["Country"].dropna().unique().tolist())


def key_for(country: str, maturity: str) -> str:
    return f"{country}|{maturity}"


def country_meta(meta: pd.DataFrame, country: str) -> Dict[str, str]:
    row = meta[meta["Country"] == country].head(1)
    if row.empty:
        return {"Region": "", "Bucket": ""}
    return {"Region": str(row.iloc[0]["Region"]), "Bucket": str(row.iloc[0]["Bucket"])}


def filter_meta(meta: pd.DataFrame, region_filter: str, bucket_filter: str, maturity: str) -> pd.DataFrame:
    frame = meta[meta["Maturity"] == maturity].copy()
    if region_filter != "All":
        frame = frame[frame["Region"] == region_filter]
    if bucket_filter != "All":
        frame = frame[frame["Bucket"] == bucket_filter]
    return frame


def build_country_snapshot(yields: pd.DataFrame, meta: pd.DataFrame, maturity: str, period: str, region_filter: str, bucket_filter: str) -> pd.DataFrame:
    frame = filter_meta(meta, region_filter, bucket_filter, maturity)
    rows: List[Dict[str, object]] = []

    for _, row in frame.iterrows():
        key = str(row["Key"])
        if key not in yields.columns:
            continue
        series = yields[key].dropna()
        if series.empty:
            continue
        rows.append(
            {
                "Country": str(row["Country"]),
                "Region": str(row["Region"]),
                "Bucket": str(row["Bucket"]),
                "Ticker": str(row["Ticker"]),
                "Latest": latest(series),
                "Today": change_bp(series, "Today"),
                "1W": change_bp(series, "1W"),
                "1M": change_bp(series, "1M"),
                "3M": change_bp(series, "3M"),
                "YTD": change_bp(series, "YTD"),
                "Selected Move": change_bp(series, period),
                "Last Date": pd.Timestamp(series.index[-1]).date(),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Latest", ascending=False).reset_index(drop=True)


def classify_global_regime(snapshot: pd.DataFrame, period: str) -> Tuple[str, str, str]:
    if snapshot.empty or "Selected Move" not in snapshot.columns:
        return "Insufficient Data", "No valid sovereign yield series loaded for the selected universe.", COLORS["amber"]

    threshold = float(PERIODS[period]["threshold"])
    moves = pd.to_numeric(snapshot["Selected Move"], errors="coerce").dropna()
    if moves.empty:
        return "Insufficient Data", "Loaded yield levels, but not enough history for the selected window.", COLORS["amber"]

    up = int((moves > threshold).sum())
    down = int((moves < -threshold).sum())
    quiet = int(len(moves) - up - down)
    up_share = up / len(moves)
    down_share = down / len(moves)
    median_move = float(moves.median())

    if up_share >= 0.60:
        return "Global Bearish Rates Impulse", f"{up}/{len(moves)} markets are up more than {threshold:.0f} bps; median move {fmt_bp(median_move)} over {period}.", COLORS["red"]
    if down_share >= 0.60:
        return "Global Bullish Rates Impulse", f"{down}/{len(moves)} markets are down more than {threshold:.0f} bps; median move {fmt_bp(median_move)} over {period}.", COLORS["green"]
    if up > down and median_move > 0:
        return "Rates Pressure Building", f"{up} markets are above the signal band, {down} are below it, and {quiet} are range-bound.", COLORS["red"]
    if down > up and median_move < 0:
        return "Duration Bid Broadening", f"{down} markets are below the signal band, {up} are above it, and {quiet} are range-bound.", COLORS["green"]
    return "Mixed Global Rates Tape", f"No clean breadth signal. Median move {fmt_bp(median_move)} over {period}; dispersion matters more than direction.", COLORS["amber"]


def regime_read(regime: str) -> str:
    reads = {
        "Global Bearish Rates Impulse": "The pressure is broad, not local. When a majority of sovereign curves are selling off together, the signal usually migrates from duration into equity duration, credit beta, fiscal risk, FX pressure, and crowded growth multiples.",
        "Global Bullish Rates Impulse": "The duration bid is broad. The better question is whether yields are falling because liquidity is easing or because growth is deteriorating. Equity confirmation should come through credit, small caps, and QQQ/SPY rather than the rates move alone.",
        "Rates Pressure Building": "The move is not yet universal, but the skew is higher. Watch whether the high-yield countries are leading for fiscal reasons or whether core DM duration is dragging the world with it.",
        "Duration Bid Broadening": "The skew is lower. That can help duration and long-multiple equities, but it is less constructive if credit, small caps, and cyclicals fail to confirm.",
        "Mixed Global Rates Tape": "The global signal is fragmented. Treat country-level dispersion as the trade: fiscal stress, policy divergence, currency pressure, and local inflation surprises matter more than a single global duration call.",
    }
    return reads.get(regime, "Signal quality is low. Check the loaded universe and the available market-data tickers before using the classification.")


def build_pressure_matrix(yields: pd.DataFrame, meta: pd.DataFrame, maturity: str, period_rows: List[str], region_filter: str, bucket_filter: str, top_n: int) -> pd.DataFrame:
    snapshot = build_country_snapshot(yields, meta, maturity, "1M", region_filter, bucket_filter)
    if snapshot.empty:
        return pd.DataFrame()

    snapshot["abs_move"] = snapshot["1M"].abs()
    selected = snapshot.sort_values(["Latest", "abs_move"], ascending=[False, False]).head(top_n)["Country"].tolist()
    rows: List[Dict[str, object]] = []

    for country in selected:
        key = key_for(country, maturity)
        if key not in yields.columns:
            continue
        row = {"Country": country}
        for p in period_rows:
            row[p] = change_bp(yields[key], p)
        rows.append(row)

    return pd.DataFrame(rows)


def build_market_signal_frame(prices: pd.DataFrame) -> pd.DataFrame:
    signals = pd.DataFrame(index=prices.index)

    for ticker, label in MARKET_TICKERS.items():
        if ticker in prices.columns and prices[ticker].dropna().any():
            signals[label] = prices[ticker]

    ratio_defs = {
        "QQQ/SPY Equity Duration": ("QQQ", "SPY"),
        "IWM/SPY Small-Cap Beta": ("IWM", "SPY"),
        "HYG/IEF Credit vs Duration": ("HYG", "IEF"),
        "LQD/IEF IG Credit vs Rates": ("LQD", "IEF"),
        "BWX/TLT Global ex-U.S. vs U.S.": ("BWX", "TLT"),
        "EMB/IGOV EM vs DM Sov.": ("EMB", "IGOV"),
        "UUP/SPY Dollar Pressure": ("UUP", "SPY"),
        "GLD/SPY Hard Asset Rel.": ("GLD", "SPY"),
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


def build_outlier_days(yields: pd.DataFrame, meta: pd.DataFrame, maturity: str, prices: pd.DataFrame, lookback_days: int, top_n: int) -> pd.DataFrame:
    keys = meta[meta["Maturity"] == maturity]["Key"].tolist()
    keys = [key for key in keys if key in yields.columns]
    if not keys:
        return pd.DataFrame()

    daily_bp = yields[keys].diff() * 100.0
    recent = daily_bp.tail(lookback_days).copy()
    if recent.dropna(how="all").empty:
        return pd.DataFrame()

    abs_moves = recent.abs()
    shock_score = abs_moves.mean(axis=1, skipna=True)

    if "TLT" in prices.columns:
        tlt_shock = prices["TLT"].pct_change().abs() * 12.0
        shock_score = shock_score.add(tlt_shock.reindex(shock_score.index), fill_value=0.0)

    driver_key = abs_moves.idxmax(axis=1)
    key_to_country = dict(zip(meta["Key"], meta["Country"]))

    shock = pd.DataFrame(
        {
            "Date": recent.index.date,
            "Shock Score": shock_score,
            "Driver": driver_key.map(key_to_country),
            "Avg Move bp": recent.mean(axis=1, skipna=True),
            "Max Abs bp": abs_moves.max(axis=1, skipna=True),
        },
        index=recent.index,
    ).dropna(subset=["Shock Score"])

    if shock.empty:
        return pd.DataFrame()

    shock = shock.sort_values("Shock Score", ascending=False).head(top_n).sort_index()
    return shock.reset_index(drop=True)


def curve_for_country(yields: pd.DataFrame, country: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for maturity in MATURITY_ORDER:
        key = key_for(country, maturity)
        if key in yields.columns and yields[key].dropna().any():
            rows.append({"Maturity": maturity, "Years": MATURITY_YEARS[maturity], "Yield": latest(yields[key])})
    return pd.DataFrame(rows)


def curve_comparison_for_country(yields: pd.DataFrame, country: str, compare_period: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for maturity in MATURITY_ORDER:
        key = key_for(country, maturity)
        if key not in yields.columns or yields[key].dropna().empty:
            continue
        series = yields[key]
        rows.append(
            {
                "Maturity": maturity,
                "Years": MATURITY_YEARS[maturity],
                "Latest": latest(series),
                "Prior": anchor_value(series, compare_period),
            }
        )
    return pd.DataFrame(rows)


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


def chart_config() -> Dict[str, object]:
    return {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "responsive": True,
    }


with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Global sovereign-rate monitor for benchmark yields, cross-market breadth, curve stress, and duration confirmation. The page validates available symbols at runtime and drops markets that do not load.
        """
    )

    st.divider()
    st.header("Controls")

    lookback_label = st.selectbox("History", ["6M", "1Y", "2Y", "3Y", "5Y", "10Y"], index=4)
    lookback_days_map = {"6M": 190, "1Y": 380, "2Y": 760, "3Y": 1140, "5Y": 1900, "10Y": 3800}
    lookback_days = lookback_days_map[lookback_label]

    maturity = st.radio("Benchmark maturity", ["10Y", "30Y", "5Y", "2Y"], index=0, horizontal=True)
    regime_period = st.radio("Signal window", list(PERIODS.keys()), index=2, horizontal=True)
    region_filter = st.selectbox("Region", ["All", "Americas", "Europe", "Asia", "EMEA"], index=0)
    bucket_filter = st.selectbox("Market bucket", ["All", "DM", "EM"], index=0)
    compare_period = st.radio("Curve comparison", ["1W", "1M", "3M", "YTD"], index=1, horizontal=True)

    max_ranked = st.slider("Markets shown", 8, 24, 16, 1)
    min_obs = st.slider("Minimum observations", 5, 60, 20, 5)

    show_market_confirmation = st.checkbox("Show market confirmation", value=True)
    show_curve = st.checkbox("Show selected-country curve", value=True)
    show_history = st.checkbox("Show history chart", value=True)
    show_table = st.checkbox("Show loaded data table", value=False)
    show_status = st.checkbox("Show data status", value=False)


st.markdown(f"<div class='adfm-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='adfm-subtitle'>Sovereign yield pressure, cross-market breadth, curve stress, duration confirmation, and outlier rate days.</div>",
    unsafe_allow_html=True,
)

start = date.today() - timedelta(days=lookback_days + 10)
end = date.today()
yield_tickers = unique_preserve_order(ticker for item in GLOBAL_YIELD_CANDIDATES for ticker in item.tickers)
market_tickers = tuple(MARKET_TICKERS.keys())
all_tickers = unique_preserve_order((*yield_tickers, *market_tickers))

with st.spinner("Loading rates and market data..."):
    close, diagnostics = fetch_close_prices(all_tickers, start, end)

if close.empty:
    st.error("No usable market data loaded.")
    if diagnostics:
        with st.expander("Data status", expanded=True):
            st.code("\n".join(diagnostics[-100:]))
    st.stop()

yields, loaded_meta = choose_yield_series(close, GLOBAL_YIELD_CANDIDATES, min_obs=min_obs)
prices = split_market_prices(close)

if yields.empty or loaded_meta.empty:
    st.error("No usable sovereign yield series loaded from the configured universe.")
    if diagnostics:
        with st.expander("Data status", expanded=True):
            st.code("\n".join(diagnostics[-100:]))
    st.stop()

available_for_maturity = loaded_meta[loaded_meta["Maturity"] == maturity]
if available_for_maturity.empty:
    fallback_maturity = loaded_meta["Maturity"].value_counts().index[0]
    st.warning(f"No valid {maturity} yield series loaded. Falling back to {fallback_maturity}.")
    maturity = str(fallback_maturity)

snapshot = build_country_snapshot(yields, loaded_meta, maturity, regime_period, region_filter, bucket_filter)
if snapshot.empty:
    st.error("No valid markets matched the selected filters.")
    st.stop()

last_obs = latest_date(yields)
if last_obs is not None:
    age_days = (pd.Timestamp(date.today()) - last_obs.normalize()).days
    if age_days > 5:
        st.warning(f"Latest loaded rates observation is {last_obs.date()}. Some markets may be stale.")

if show_status:
    with st.expander("Data status", expanded=False):
        loaded_yields = loaded_meta[["Country", "Maturity", "Ticker", "Region", "Bucket"]].sort_values(["Country", "Maturity"])
        st.markdown(
            f"<div class='data-note'>Loaded yield series: {len(loaded_yields)}. Loaded market proxies: {len(prices.columns)}. Last rates observation: {last_obs.date() if last_obs is not None else 'N/A'}.</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(loaded_yields, use_container_width=True, hide_index=True)
        if diagnostics:
            st.code("\n".join(diagnostics[-100:]))

regime, regime_note, regime_color = classify_global_regime(snapshot, regime_period)
highest = snapshot.sort_values("Latest", ascending=False).iloc[0]
biggest_up = snapshot.sort_values("Selected Move", ascending=False).iloc[0]
biggest_down = snapshot.sort_values("Selected Move", ascending=True).iloc[0]

us_key = key_for("United States", maturity)
us_yield = latest(yields[us_key]) if us_key in yields.columns else np.nan
us_move = change_bp(yields[us_key], regime_period) if us_key in yields.columns else np.nan

tlt_return = return_pct(prices["TLT"], regime_period) if "TLT" in prices.columns else np.nan
bwx_return = return_pct(prices["BWX"], regime_period) if "BWX" in prices.columns else np.nan

cards = [
    ("Global Regime", regime, regime_note, regime_color),
    (f"Highest {maturity}", f"{highest['Country']} {fmt_pct(safe_float(highest['Latest']))}", f"{regime_period} {fmt_bp(safe_float(highest['Selected Move']))}", COLORS["purple"]),
    ("Biggest Yield Rise", f"{biggest_up['Country']} {fmt_bp(safe_float(biggest_up['Selected Move']))}", f"Latest {fmt_pct(safe_float(biggest_up['Latest']))}", COLORS["red"]),
    ("Biggest Yield Fall", f"{biggest_down['Country']} {fmt_bp(safe_float(biggest_down['Selected Move']))}", f"Latest {fmt_pct(safe_float(biggest_down['Latest']))}", COLORS["green"]),
    (f"U.S. {maturity}", fmt_pct(us_yield), f"{regime_period} {fmt_bp(us_move)} | TLT {fmt_ret(tlt_return)}", COLORS["blue"]),
]

for col, card in zip(st.columns(5), cards):
    with col:
        metric_card(*card)

st.markdown("<div class='section-title'>Read-through</div>", unsafe_allow_html=True)
st.markdown(f"<div class='note-box'>{escape(regime_read(regime))}</div>", unsafe_allow_html=True)

left, right = st.columns([1.02, 0.98])

with left:
    st.markdown(f"<div class='section-title'>Global {maturity} Yield Ranking</div>", unsafe_allow_html=True)
    rank = snapshot.sort_values("Latest", ascending=False).head(max_ranked).copy()
    rank["Label"] = rank["Country"] + "  " + rank["Latest"].map(lambda x: fmt_pct(safe_float(x)))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=rank["Latest"],
            y=rank["Country"],
            orientation="h",
            text=rank["Latest"].map(lambda x: fmt_pct(safe_float(x))),
            textposition="outside",
            marker=dict(color=COLORS["blue"]),
            hovertemplate="%{y}<br>Yield: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis=dict(title="Yield (%)"), showlegend=False)
    st.plotly_chart(clean_plot_layout(fig, 430), use_container_width=True, config=chart_config())

with right:
    st.markdown("<div class='section-title'>Global Yield Pressure Matrix</div>", unsafe_allow_html=True)
    matrix = snapshot.copy().sort_values("Latest", ascending=False).head(max_ranked)
    heat_cols = list(PERIODS.keys())
    z = matrix[heat_cols].to_numpy(dtype=float)
    text = np.full(z.shape, "", dtype=object)
    finite_mask = np.isfinite(z)
    text[finite_mask] = np.round(z[finite_mask], 0).astype(int).astype(str)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=heat_cols,
            y=matrix["Country"],
            colorscale="RdYlGn_r",
            zmid=0,
            text=text,
            texttemplate="%{text}",
            colorbar=dict(title="bps"),
        )
    )
    fig.update_layout(xaxis=dict(side="top"), yaxis=dict(autorange="reversed"))
    st.plotly_chart(clean_plot_layout(fig, 430), use_container_width=True, config=chart_config())

st.markdown("<div class='section-title'>Yield Level vs Move</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='small-note'>Latest {maturity} yield versus {regime_period} basis-point move. Upper-right markets combine high nominal yield with fresh pressure.</div>",
    unsafe_allow_html=True,
)

scatter = snapshot.copy().head(max_ranked)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=scatter["Latest"],
        y=scatter["Selected Move"],
        mode="markers+text",
        text=scatter["Country"],
        textposition="top center",
        marker=dict(
            size=11,
            color=scatter["Selected Move"],
            colorscale="RdYlGn_r",
            cmid=0,
            showscale=True,
            colorbar=dict(title="bps"),
            line=dict(width=0.6, color="#ffffff"),
        ),
        hovertemplate="%{text}<br>Yield: %{x:.2f}%<br>Move: %{y:+.0f} bps<extra></extra>",
    )
)
fig.add_hline(y=0, line_width=1, line_dash="dot", line_color=COLORS["grey"])
fig.update_layout(xaxis=dict(title=f"Latest {maturity} Yield (%)"), yaxis=dict(title=f"{regime_period} Move (bps)"), showlegend=False)
st.plotly_chart(clean_plot_layout(fig, 410), use_container_width=True, config=chart_config())

if show_curve:
    country_options = available_countries(loaded_meta)
    default_country = "United States" if "United States" in country_options else country_options[0]
    selected_country = st.selectbox("Country curve", country_options, index=country_options.index(default_country))

    curve_comp = curve_comparison_for_country(yields, selected_country, compare_period)
    st.markdown(f"<div class='section-title'>{escape(selected_country)} Curve Snapshot</div>", unsafe_allow_html=True)

    if curve_comp.empty or len(curve_comp) < 2:
        st.info("At least two maturities are needed for the selected-country curve snapshot.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=curve_comp["Years"],
                y=curve_comp["Latest"],
                mode="lines+markers",
                name="Latest",
                line=dict(color=COLORS["blue"], width=3),
                marker=dict(size=8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=curve_comp["Years"],
                y=curve_comp["Prior"],
                mode="lines+markers",
                name=f"{compare_period} ago",
                line=dict(color=COLORS["grey"], width=2, dash="dash"),
                marker=dict(size=7),
            )
        )
        fig.update_layout(
            xaxis=dict(title="Tenor", tickvals=curve_comp["Years"], ticktext=curve_comp["Maturity"]),
            yaxis=dict(title="Yield (%)"),
        )
        st.plotly_chart(clean_plot_layout(fig, 390), use_container_width=True, config=chart_config())

if show_market_confirmation:
    st.markdown("<div class='section-title'>Market Confirmation Tape</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='small-note'>Duration, credit, FX, equity-duration, and global sovereign ETF proxies. This panel checks whether asset prices confirm or reject the rates signal.</div>",
        unsafe_allow_html=True,
    )

    market_signals = build_market_signal_frame(prices)
    market_matrix = market_return_matrix(market_signals)

    if market_matrix.empty:
        st.info("No market confirmation proxies loaded.")
    else:
        preferred_order = [
            "TLT Long U.S. Duration",
            "IEF Intermediate U.S. Duration",
            "BWX Global ex-U.S. Sov.",
            "IGOV International Sov.",
            "EMB EM Sovereign Credit",
            "HYG U.S. High Yield",
            "LQD U.S. IG Credit",
            "QQQ/SPY Equity Duration",
            "IWM/SPY Small-Cap Beta",
            "HYG/IEF Credit vs Duration",
            "LQD/IEF IG Credit vs Rates",
            "BWX/TLT Global ex-U.S. vs U.S.",
            "EMB/IGOV EM vs DM Sov.",
            "UUP/SPY Dollar Pressure",
            "GLD/SPY Hard Asset Rel.",
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
        st.plotly_chart(clean_plot_layout(fig, 440), use_container_width=True, config=chart_config())

st.markdown("<div class='section-title'>Outlier Global Rate Days</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='small-note'>Average absolute daily move across loaded {maturity} markets, with the largest country driver identified. Designed to surface sessions worth reviewing.</div>",
    unsafe_allow_html=True,
)

shock = build_outlier_days(yields, loaded_meta, maturity, prices, lookback_days=min(len(yields), 300), top_n=10)
if shock.empty:
    st.info("Not enough loaded history to calculate outlier days.")
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
    fig.update_layout(xaxis=dict(title="Date"), yaxis=dict(title="Weighted shock"), showlegend=False)
    st.plotly_chart(clean_plot_layout(fig, 330), use_container_width=True, config=chart_config())

    display_shock = shock.copy()
    for col in ["Shock Score", "Avg Move bp", "Max Abs bp"]:
        if col in display_shock.columns:
            display_shock[col] = display_shock[col].map(lambda x: fmt_num(safe_float(x), 0))
    st.dataframe(display_shock, use_container_width=True, hide_index=True)

if show_history:
    st.markdown("<div class='section-title'>Benchmark History</div>", unsafe_allow_html=True)
    history_countries = snapshot.sort_values("Latest", ascending=False).head(min(8, max_ranked))["Country"].tolist()

    fig = go.Figure()
    for country in history_countries:
        key = key_for(country, maturity)
        if key not in yields.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=yields.index,
                y=yields[key],
                name=country,
                mode="lines",
                line=dict(width=1.7),
            )
        )
    fig.update_layout(yaxis=dict(title="Yield (%)"))
    st.plotly_chart(clean_plot_layout(fig, 430), use_container_width=True, config=chart_config())

if show_table:
    st.markdown("<div class='section-title'>Loaded Sovereign Yields</div>", unsafe_allow_html=True)
    table = snapshot.copy()
    display_cols = ["Country", "Region", "Bucket", "Ticker", "Latest", "Today", "1W", "1M", "3M", "YTD", "Last Date"]
    for col in ["Latest"]:
        table[col] = table[col].map(lambda x: fmt_pct(safe_float(x)))
    for col in ["Today", "1W", "1M", "3M", "YTD"]:
        table[col] = table[col].map(lambda x: fmt_bp(safe_float(x)))
    st.dataframe(table[display_cols], use_container_width=True, hide_index=True)
