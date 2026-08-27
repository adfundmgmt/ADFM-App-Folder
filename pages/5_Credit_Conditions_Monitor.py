from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from pandas_datareader import data as web

from adfm_core.data_registry import PRIMARY_MACRO_SERIES, SeriesDefinition
from adfm_core.market_data import configure_yfinance_cache
from adfm_core.palette import PASTEL, PASTEL_20
from adfm_core.primary_data import fetch_fred_series
from adfm_core.ui import (
    PageHeader,
    inject_explorer_style,
    render_footer,
    render_kpi_cards,
    render_page_header,
    render_selection_note,
)

configure_yfinance_cache()
warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None

TITLE = "Credit Conditions Monitor"
SUBTITLE = (
    "Separates credit-spread stress from outright funding-cost pressure, then "
    "checks banks, loans, volatility, and global sovereign yields for confirmation."
)

st.set_page_config(layout="wide", page_title=TITLE, initial_sidebar_state="expanded")
inject_explorer_style()

st.markdown(
    """
    <style>
        .section-title {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.13rem;
            font-weight: 700;
            color: #111111;
            margin-top: 1.15rem;
            margin-bottom: .22rem;
            letter-spacing: -.012em;
            border-bottom: 1px solid #111111;
            padding-bottom: .34rem;
        }
        .section-subtitle {
            max-width: 1180px;
            font-size: .79rem;
            color: #5b6470;
            line-height: 1.46;
            margin: .30rem 0 .62rem;
        }
        .source-line {
            color: #666666;
            font-size: .72rem;
            line-height: 1.45;
            margin: .18rem 0 .62rem;
        }
        .quality-line {
            border-top: 1px solid #d8d8d8;
            border-bottom: 1px solid #d8d8d8;
            padding: .52rem 0;
            margin: .42rem 0 .72rem;
            color: #3f3f3f;
            font-size: .76rem;
            line-height: 1.5;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 0 !important;
        }
        @media (max-width: 760px) {
            .section-title { font-size: 1.04rem; }
            .section-subtitle { font-size: .76rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

COLORS = {
    "green": PASTEL["sage"],
    "red": PASTEL["rose"],
    "orange": PASTEL["coral"],
    "amber": PASTEL["amber"],
    "blue": PASTEL["blue"],
    "purple": PASTEL["lavender"],
    "teal": PASTEL["teal"],
    "grey": "#A8ADB5",
    "slate": "#334155",
    "muted": "#64748b",
    "grid": "#e5e7eb",
    "dark": "#111111",
}
LINE_COLORS = list(PASTEL_20)

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
    "IWM",
    "TLT",
    "IEF",
    "UUP",
    "^VIX",
)

DISPLAY_NAMES: Dict[str, str] = {
    "HYG": "High Yield",
    "JNK": "High Yield 2",
    "LQD": "Investment Grade",
    "BKLN": "Leveraged Loans",
    "SRLN": "Senior Loans",
    "EMB": "EM USD Debt",
    "KRE": "Regional Banks",
    "XLF": "Financials",
    "SPY": "S&P 500",
    "IWM": "Russell 2000",
    "TLT": "20Y+ Treasuries",
    "IEF": "7-10Y Treasuries",
    "UUP": "U.S. Dollar",
    "^VIX": "VIX",
}

FOCUS_WINDOWS = ["5D", "1M", "3M", "YTD", "1Y"]
GLOBAL_WINDOWS = ["5D", "1M", "YTD", "1Y", "3Y", "5Y"]

PRIMARY_BY_KEY = {definition.key: definition for definition in PRIMARY_MACRO_SERIES}
CREDIT_FRED_DEFINITIONS: Tuple[SeriesDefinition, ...] = (
    PRIMARY_BY_KEY["hy_oas"],
    PRIMARY_BY_KEY["dgs10"],
    PRIMARY_BY_KEY["dgs30"],
    SeriesDefinition(
        "ig_oas",
        "US Corporate OAS",
        "BAMLC0A0CM",
        "Federal Reserve FRED",
        "Credit",
        "ICE BofA US Corporate option-adjusted spread.",
        5,
    ),
    SeriesDefinition(
        "bbb_oas",
        "US BBB OAS",
        "BAMLC0A4CBBB",
        "Federal Reserve FRED",
        "Credit",
        "ICE BofA BBB US Corporate option-adjusted spread.",
        5,
    ),
)

SOVEREIGN_UNIVERSE: Tuple[dict, ...] = (
    {"country": "United States", "label": "U.S.", "group": "Developed", "stooq": "10YUSY.B", "fred": "IRLTLT01USM156N"},
    {"country": "Japan", "label": "Japan", "group": "Developed", "stooq": "10YJPY.B", "fred": "IRLTLT01JPM156N"},
    {"country": "Australia", "label": "Australia", "group": "Developed", "stooq": "10YAUY.B", "fred": "IRLTLT01AUM156N"},
    {"country": "Canada", "label": "Canada", "group": "Developed", "stooq": "10YCAY.B", "fred": "IRLTLT01CAM156N"},
    {"country": "Germany", "label": "Germany", "group": "Developed", "stooq": "10YDEY.B", "fred": "IRLTLT01DEM156N"},
    {"country": "France", "label": "France", "group": "Developed", "stooq": "10YFRY.B", "fred": "IRLTLT01FRM156N"},
    {"country": "Italy", "label": "Italy", "group": "Developed", "stooq": "10YITY.B", "fred": "IRLTLT01ITM156N"},
    {"country": "Spain", "label": "Spain", "group": "Developed", "stooq": "10YESY.B", "fred": "IRLTLT01ESM156N"},
    {"country": "United Kingdom", "label": "U.K.", "group": "Developed", "stooq": "10YGBY.B", "fred": "IRLTLT01GBM156N"},
    {"country": "Switzerland", "label": "Switzerland", "group": "Developed", "stooq": "10YCHY.B", "fred": "IRLTLT01CHM156N"},
    {"country": "Netherlands", "label": "Netherlands", "group": "Developed", "stooq": "10YNLY.B", "fred": "IRLTLT01NLM156N"},
    {"country": "Belgium", "label": "Belgium", "group": "Developed", "stooq": "10YBEY.B", "fred": "IRLTLT01BEM156N"},
    {"country": "Portugal", "label": "Portugal", "group": "Developed", "stooq": "10YPTY.B", "fred": "IRLTLT01PTM156N"},
    {"country": "Sweden", "label": "Sweden", "group": "Developed", "stooq": "10YSEY.B", "fred": "IRLTLT01SEM156N"},
    {"country": "Norway", "label": "Norway", "group": "Developed", "stooq": "10YNOY.B", "fred": "IRLTLT01NOM156N"},
    {"country": "New Zealand", "label": "New Zealand", "group": "Developed", "stooq": "10YNZY.B", "fred": "IRLTLT01NZM156N"},
    {"country": "Turkey", "label": "Turkey", "group": "Emerging", "stooq": "10YTRY.B", "fred": "IRLTLT01TRM156N"},
    {"country": "South Korea", "label": "South Korea", "group": "Emerging", "stooq": "10YKRY.B", "fred": "IRLTLT01KRM156N"},
    {"country": "Poland", "label": "Poland", "group": "Emerging", "stooq": "10YPLY.B", "fred": "IRLTLT01PLM156N"},
    {"country": "Czechia", "label": "Czechia", "group": "Emerging", "stooq": "10YCZY.B", "fred": "IRLTLT01CZM156N"},
    {"country": "Hungary", "label": "Hungary", "group": "Emerging", "stooq": "10YHUY.B", "fred": "IRLTLT01HUM156N"},
    {"country": "Brazil", "label": "Brazil", "group": "Emerging", "stooq": "10YBRY.B", "fred": "IRLTLT01BRM156N"},
    {"country": "Mexico", "label": "Mexico", "group": "Emerging", "stooq": "10YMXY.B", "fred": "IRLTLT01MXM156N"},
    {"country": "India", "label": "India", "group": "Emerging", "stooq": "10YINY.B", "fred": "IRLTLT01INM156N"},
    {"country": "China", "label": "China", "group": "Emerging", "stooq": "10YCNY.B", "fred": "IRLTLT01CNM156N"},
    {"country": "Indonesia", "label": "Indonesia", "group": "Emerging", "stooq": "10YIDY.B", "fred": "IRLTLT01IDM156N"},
    {"country": "Malaysia", "label": "Malaysia", "group": "Emerging", "stooq": "10YMYY.B", "fred": "IRLTLT01MYM156N"},
    {"country": "South Africa", "label": "South Africa", "group": "Emerging", "stooq": "10YZAY.B", "fred": "IRLTLT01ZAM156N"},
    {"country": "Colombia", "label": "Colombia", "group": "Emerging", "stooq": "10YCOY.B", "fred": "IRLTLT01COM156N"},
    {"country": "Chile", "label": "Chile", "group": "Emerging", "stooq": "10YCLY.B", "fred": "IRLTLT01CLM156N"},
)

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** separate two different credit problems that are often conflated.

        - **Spread stress** asks whether markets are charging more default/liquidity premium.
        - **Funding-cost pressure** asks whether the risk-free base rate itself is expensive.
        - **Market confirmation** checks HY, loans, banks, EM debt, and volatility.
        - **Global 10Y moves** shows where sovereign borrowing costs are actually repricing.
        """
    )
    st.markdown("---")
    st.header("Controls")
    focus_window = st.selectbox("Credit move window", FOCUS_WINDOWS, index=1)
    global_window = st.selectbox("Global 10Y move window", GLOBAL_WINDOWS, index=3)
    history_label = st.selectbox(
        "Chart history", ["1 Year", "3 Years", "5 Years", "10 Years"], index=1
    )
    show_raw = st.checkbox("Show data audit", value=False)

HISTORY_DAYS = {
    "1 Year": 365,
    "3 Years": 365 * 3,
    "5 Years": 365 * 5,
    "10 Years": 365 * 10,
}


def clean_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    out = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].sort_index()
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    return out


def latest(series: pd.Series | None) -> float:
    s = clean_series(series)
    return float(s.iloc[-1]) if not s.empty else np.nan


def latest_timestamp(series: pd.Series | None) -> Optional[pd.Timestamp]:
    s = clean_series(series)
    return pd.Timestamp(s.index[-1]) if not s.empty else None


def asof_value(series: pd.Series | None, target: pd.Timestamp) -> float:
    s = clean_series(series)
    if s.empty:
        return np.nan
    eligible = s.loc[s.index <= pd.Timestamp(target)]
    return float(eligible.iloc[-1]) if not eligible.empty else np.nan


def first_on_or_after(series: pd.Series | None, target: pd.Timestamp) -> float:
    s = clean_series(series)
    if s.empty:
        return np.nan
    eligible = s.loc[s.index >= pd.Timestamp(target)]
    return float(eligible.iloc[0]) if not eligible.empty else np.nan


def focus_target(label: str, asof: pd.Timestamp) -> pd.Timestamp:
    if label == "5D":
        return asof - pd.Timedelta(days=8)
    if label == "1M":
        return asof - pd.DateOffset(months=1)
    if label == "3M":
        return asof - pd.DateOffset(months=3)
    if label == "YTD":
        return pd.Timestamp(asof.year, 1, 1)
    if label == "1Y":
        return asof - pd.DateOffset(years=1)
    if label == "3Y":
        return asof - pd.DateOffset(years=3)
    if label == "5Y":
        return asof - pd.DateOffset(years=5)
    return asof - pd.DateOffset(months=1)


def pct_move(series: pd.Series | None, label: str) -> float:
    s = clean_series(series)
    if len(s) < 2:
        return np.nan
    now = float(s.iloc[-1])
    if label == "5D" and len(s) >= 6:
        base = float(s.iloc[-6])
    elif label == "YTD":
        base = first_on_or_after(s, pd.Timestamp(s.index[-1].year, 1, 1))
    else:
        base = asof_value(s, focus_target(label, pd.Timestamp(s.index[-1])))
    if not np.isfinite(base) or base == 0:
        return np.nan
    return (now / base - 1.0) * 100.0


def absolute_move(series: pd.Series | None, label: str, scale: float = 1.0) -> float:
    s = clean_series(series)
    if len(s) < 2:
        return np.nan
    now = float(s.iloc[-1])
    if label == "5D" and len(s) >= 6:
        base = float(s.iloc[-6])
    elif label == "YTD":
        base = first_on_or_after(s, pd.Timestamp(s.index[-1].year, 1, 1))
    else:
        base = asof_value(s, focus_target(label, pd.Timestamp(s.index[-1])))
    if not np.isfinite(base):
        return np.nan
    return (now - base) * scale


def trailing_percentile(series: pd.Series | None, years: int) -> float:
    s = clean_series(series)
    if s.empty:
        return np.nan
    cutoff = pd.Timestamp(s.index[-1]) - pd.DateOffset(years=years)
    window = s.loc[s.index >= cutoff]
    if len(window) < 30:
        return np.nan
    return float((window <= window.iloc[-1]).mean())


def fmt_pct(value: float, digits: int = 2, signed: bool = True) -> str:
    if not np.isfinite(value):
        return "N/A"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value:.{digits}f}%"


def fmt_bp(value: float, digits: int = 0) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:+.{digits}f} bp"


def fmt_yield(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.2f}%"


def fmt_percentile(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value * 100:.0f}th pct"


def chart_layout(height: int = 390, showlegend: bool = True) -> dict:
    return dict(
        height=height,
        margin=dict(l=12, r=16, t=26, b=18),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12, color=COLORS["slate"]),
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
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False, linecolor=COLORS["grid"])
    return fig


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_market_prices(tickers: Tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    try:
        raw = yf.download(
            list(tickers),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()

    out = pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        lvl1 = raw.columns.get_level_values(1)
        if "Close" in lvl0:
            block = raw["Close"]
            for ticker in tickers:
                if ticker in block.columns:
                    out[ticker] = pd.to_numeric(block[ticker], errors="coerce")
        elif "Close" in lvl1:
            for ticker in tickers:
                try:
                    out[ticker] = pd.to_numeric(raw[(ticker, "Close")], errors="coerce")
                except Exception:
                    try:
                        out[ticker] = pd.to_numeric(raw[("Close", ticker)], errors="coerce")
                    except Exception:
                        pass
    elif "Close" in raw.columns and len(tickers) == 1:
        out[tickers[0]] = pd.to_numeric(raw["Close"], errors="coerce")

    if out.empty:
        return out
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].sort_index().ffill().dropna(how="all")
    return out


def ratio_frame(market: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "HYG/LQD": ("HYG", "LQD"),
        "JNK/LQD": ("JNK", "LQD"),
        "BKLN/LQD": ("BKLN", "LQD"),
        "SRLN/LQD": ("SRLN", "LQD"),
        "EMB/LQD": ("EMB", "LQD"),
        "KRE/SPY": ("KRE", "SPY"),
        "XLF/SPY": ("XLF", "SPY"),
        "IWM/SPY": ("IWM", "SPY"),
    }
    out = pd.DataFrame(index=market.index)
    for label, (num, den) in definitions.items():
        if num in market.columns and den in market.columns:
            out[label] = market[num] / market[den].replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).dropna(how="all").ffill()


def _get_secret(name: str) -> Optional[str]:
    try:
        if name in st.secrets:
            value = str(st.secrets[name]).strip()
            if value:
                return value
    except Exception:
        pass
    value = os.getenv(name, "").strip()
    return value or None


def _parse_stooq_csv(text: str) -> pd.Series:
    if not text or "Date" not in text[:120]:
        return pd.Series(dtype=float)
    try:
        frame = pd.read_csv(StringIO(text))
    except Exception:
        return pd.Series(dtype=float)
    if "Date" not in frame.columns or "Close" not in frame.columns:
        return pd.Series(dtype=float)
    dates = pd.to_datetime(frame["Date"], errors="coerce")
    values = pd.to_numeric(frame["Close"], errors="coerce")
    series = pd.Series(values.values, index=dates).dropna().sort_index()
    return series[(series > -5.0) & (series < 100.0)]


def _fetch_stooq_symbol(symbol: str, start_date: date, end_date: date) -> pd.Series:
    d1 = start_date.strftime("%Y%m%d")
    d2 = end_date.strftime("%Y%m%d")
    symbol_q = symbol.lower()
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ADFM-Analytics/1.0)",
        "Accept": "text/csv,text/plain,*/*",
    }
    urls = (
        f"https://stooq.com/q/d/l/?s={symbol_q}&d1={d1}&d2={d2}&i=d",
        f"https://stooq.pl/q/d/l/?s={symbol_q}&d1={d1}&d2={d2}&i=d",
    )
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code != 200:
                continue
            series = _parse_stooq_csv(response.text)
            if len(series) >= 2:
                return series
        except Exception:
            continue
    return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stooq_sovereigns(start_date: date, end_date: date) -> Tuple[Dict[str, pd.Series], List[str]]:
    probe = _fetch_stooq_symbol("10YUSY.B", start_date, end_date)
    if probe.empty:
        return {}, ["Stooq daily sovereign-yield endpoint unavailable."]

    results: Dict[str, pd.Series] = {"United States": probe}
    errors: List[str] = []
    rows = [row for row in SOVEREIGN_UNIVERSE if row["country"] != "United States"]
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(_fetch_stooq_symbol, row["stooq"], start_date, end_date): row
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                series = future.result()
            except Exception:
                series = pd.Series(dtype=float)
            if series.empty:
                errors.append(str(row["country"]))
            else:
                results[str(row["country"])] = series
    return results, errors


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_oecd_sovereigns(start_date: date, end_date: date) -> Dict[str, pd.Series]:
    results: Dict[str, pd.Series] = {}
    for row in SOVEREIGN_UNIVERSE:
        series_id = row.get("fred")
        if not series_id:
            continue
        try:
            raw = web.DataReader(series_id, "fred", start_date, end_date)
            if raw is None or raw.empty or series_id not in raw.columns:
                continue
            series = clean_series(raw[series_id])
            if len(series) >= 2:
                results[str(row["country"])] = series
        except Exception:
            continue
    return results


def _fetch_te_sovereigns(api_key: str, start_date: date, end_date: date) -> Tuple[Dict[str, pd.Series], str]:
    try:
        snap = requests.get(
            "https://api.tradingeconomics.com/markets/bond",
            params={"c": api_key, "type": "10Y", "f": "json"},
            timeout=10,
        )
        snap.raise_for_status()
        payload = snap.json()
        if not isinstance(payload, list):
            return {}, "Trading Economics returned no 10Y snapshot."
    except Exception as exc:
        return {}, f"Trading Economics snapshot failed: {type(exc).__name__}"

    wanted = {str(row["country"]) for row in SOVEREIGN_UNIVERSE}
    symbol_to_country: Dict[str, str] = {}
    for item in payload:
        country = str(item.get("Country", "")).strip()
        symbol = str(item.get("Symbol", "")).strip()
        if country in wanted and symbol:
            symbol_to_country[symbol] = country
    if not symbol_to_country:
        return {}, "Trading Economics returned no matching sovereign symbols."

    out: Dict[str, pd.Series] = {}
    symbols = list(symbol_to_country.keys())
    for i in range(0, len(symbols), 12):
        batch = symbols[i : i + 12]
        try:
            response = requests.get(
                "https://api.tradingeconomics.com/markets/historical/" + ",".join(batch),
                params={
                    "c": api_key,
                    "d1": start_date.isoformat(),
                    "d2": end_date.isoformat(),
                    "f": "json",
                },
                timeout=15,
            )
            response.raise_for_status()
            history = response.json()
            if not isinstance(history, list):
                continue
        except Exception:
            continue

        grouped: Dict[str, List[tuple[pd.Timestamp, float]]] = {}
        for item in history:
            symbol = str(item.get("Symbol", "")).strip()
            if symbol not in symbol_to_country:
                continue
            dt = pd.to_datetime(item.get("Date"), dayfirst=True, errors="coerce")
            close = pd.to_numeric(item.get("Close"), errors="coerce")
            if pd.isna(dt) or pd.isna(close):
                continue
            grouped.setdefault(symbol, []).append((pd.Timestamp(dt), float(close)))

        for symbol, pairs in grouped.items():
            series = pd.Series(
                [value for _, value in pairs],
                index=[dt for dt, _ in pairs],
                dtype=float,
            ).sort_index()
            if len(series) >= 2:
                out[symbol_to_country[symbol]] = series
    return out, "" if out else "Trading Economics historical data unavailable."


def sovereign_move_rows(series_map: Dict[str, pd.Series], horizon: str, source: str) -> pd.DataFrame:
    meta = {str(row["country"]): row for row in SOVEREIGN_UNIVERSE}
    rows: List[dict] = []
    today = pd.Timestamp(date.today())

    for country, series in series_map.items():
        if country not in meta:
            continue
        s = clean_series(series)
        if len(s) < 2:
            continue
        latest_dt = pd.Timestamp(s.index[-1])
        age_days = (today - latest_dt.normalize()).days
        max_age = 7 if source != "OECD/FRED monthly" else 75
        if age_days > max_age:
            continue
        end_yield = float(s.iloc[-1])

        if horizon == "5D":
            if len(s) < 6:
                continue
            start_yield = float(s.iloc[-6])
            start_dt = pd.Timestamp(s.index[-6])
        elif horizon == "YTD":
            start_target = pd.Timestamp(latest_dt.year, 1, 1)
            eligible = s.loc[s.index >= start_target]
            if eligible.empty:
                continue
            start_yield = float(eligible.iloc[0])
            start_dt = pd.Timestamp(eligible.index[0])
        else:
            target = focus_target(horizon, latest_dt)
            eligible = s.loc[s.index <= target]
            if eligible.empty:
                continue
            start_yield = float(eligible.iloc[-1])
            start_dt = pd.Timestamp(eligible.index[-1])

        move_bp = (end_yield - start_yield) * 100.0
        if not np.isfinite(move_bp):
            continue
        rows.append(
            {
                "Country": country,
                "Label": meta[country]["label"],
                "Group": meta[country]["group"],
                "Move bp": float(move_bp),
                "Start Yield": start_yield,
                "End Yield": end_yield,
                "Start Date": start_dt,
                "End Date": latest_dt,
                "Source": source,
            }
        )
    return pd.DataFrame(rows)


def load_global_sovereign_moves(horizon: str) -> Tuple[pd.DataFrame, str, str]:
    end_date = date.today()
    years_needed = 6 if horizon == "5Y" else 4 if horizon == "3Y" else 2
    start_date = end_date - timedelta(days=365 * years_needed + 45)

    te_key = _get_secret("TRADING_ECONOMICS_API_KEY")
    if te_key:
        te_map, te_error = _fetch_te_sovereigns(te_key, start_date, end_date)
        te_rows = sovereign_move_rows(te_map, horizon, "Trading Economics")
        if len(te_rows) >= 8:
            return te_rows, "Trading Economics", te_error

    stooq_map, stooq_errors = fetch_stooq_sovereigns(start_date, end_date)
    stooq_rows = sovereign_move_rows(stooq_map, horizon, "Stooq daily")
    if len(stooq_rows) >= 8:
        note = f"{len(stooq_rows)} fresh countries loaded"
        if stooq_errors:
            note += f"; {len(stooq_errors)} unavailable"
        return stooq_rows, "Stooq daily", note

    if horizon in {"1Y", "3Y", "5Y"}:
        oecd_map = fetch_oecd_sovereigns(start_date, end_date)
        oecd_rows = sovereign_move_rows(oecd_map, horizon, "OECD/FRED monthly")
        if len(oecd_rows) >= 8:
            return (
                oecd_rows,
                "OECD/FRED monthly",
                "Daily global source unavailable; structural monthly fallback is shown.",
            )

    return (
        pd.DataFrame(),
        "Unavailable",
        "Global daily sovereign data did not pass coverage/freshness checks. No ETF or bond-price proxy was substituted.",
    )


def sovereign_bar_chart(frame: pd.DataFrame, group: str, x_limit: float) -> go.Figure:
    group_frame = (
        frame.loc[frame["Group"] == group]
        .sort_values("Move bp", ascending=True)
        .reset_index(drop=True)
    )
    median = float(group_frame["Move bp"].median()) if not group_frame.empty else 0.0
    colors = [
        COLORS["orange"] if value > 0 else COLORS["blue"] if value < 0 else COLORS["grey"]
        for value in group_frame["Move bp"]
    ]
    custom = (
        np.column_stack(
            [
                group_frame["Start Yield"],
                group_frame["End Yield"],
                group_frame["Start Date"].dt.strftime("%Y-%m-%d"),
                group_frame["End Date"].dt.strftime("%Y-%m-%d"),
                group_frame["Source"],
            ]
        )
        if not group_frame.empty
        else np.empty((0, 5), dtype=object)
    )

    fig = go.Figure()
    fig.add_vline(x=0, line_width=1, line_color="#8a8a8a")
    if not group_frame.empty:
        fig.add_vline(x=median, line_width=1.5, line_dash="dash", line_color=COLORS["amber"])
        fig.add_trace(
            go.Bar(
                x=group_frame["Move bp"],
                y=group_frame["Label"],
                orientation="h",
                marker_color=colors,
                customdata=custom,
                text=[
                    f"{value:+.0f} bp   {start:.2f}% → {end:.2f}%"
                    for value, start, end in zip(
                        group_frame["Move bp"],
                        group_frame["Start Yield"],
                        group_frame["End Yield"],
                    )
                ],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "%{y}<br>Move: %{x:+.0f} bp"
                    "<br>%{customdata[0]:.2f}% → %{customdata[1]:.2f}%"
                    "<br>%{customdata[2]} → %{customdata[3]}"
                    "<br>Source: %{customdata[4]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=max(360, 34 * max(len(group_frame), 7) + 90),
        margin=dict(l=18, r=130, t=46, b=34),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        title=dict(
            text=f"{group.upper()} · median {median:+.0f} bp",
            x=0,
            xanchor="left",
            font=dict(size=14, color=COLORS["dark"]),
        ),
        xaxis=dict(
            title="Change in benchmark 10Y yield (basis points)",
            range=[-x_limit, x_limit],
            zeroline=False,
            gridcolor="#eeeeee",
        ),
        yaxis=dict(showgrid=False),
    )
    return fig


render_page_header(
    PageHeader(
        title=TITLE,
        description=SUBTITLE,
        eyebrow="ADFM Credit Regimes",
    )
)

history_start = (date.today() - timedelta(days=365 * 10 + 45)).isoformat()
history_end = (date.today() + timedelta(days=1)).isoformat()

with st.spinner("Loading credit and rates data..."):
    fred, fred_status = fetch_fred_series(
        CREDIT_FRED_DEFINITIONS,
        start=history_start,
        end=date.today().isoformat(),
    )
    market = fetch_market_prices(MARKET_TICKERS, history_start, history_end)

proxy = ratio_frame(market) if not market.empty else pd.DataFrame()
hy_oas = clean_series(fred["hy_oas"]) if "hy_oas" in fred else pd.Series(dtype=float)
ig_oas = clean_series(fred["ig_oas"]) if "ig_oas" in fred else pd.Series(dtype=float)
bbb_oas = clean_series(fred["bbb_oas"]) if "bbb_oas" in fred else pd.Series(dtype=float)
dgs10 = clean_series(fred["dgs10"]) if "dgs10" in fred else pd.Series(dtype=float)
dgs30 = clean_series(fred["dgs30"]) if "dgs30" in fred else pd.Series(dtype=float)

if hy_oas.empty and market.empty:
    st.error("Neither primary credit spreads nor market confirmation data loaded.")
    st.stop()

spread_percentiles = {
    "HY OAS": trailing_percentile(hy_oas, 5),
    "IG OAS": trailing_percentile(ig_oas, 5),
    "BBB OAS": trailing_percentile(bbb_oas, 5),
}
spread_values = [value for value in spread_percentiles.values() if np.isfinite(value)]
spread_stress = float(np.mean(spread_values)) if spread_values else np.nan

rate_percentiles = {
    "10Y": trailing_percentile(dgs10, 10),
    "30Y": trailing_percentile(dgs30, 10),
}
rate_values = [value for value in rate_percentiles.values() if np.isfinite(value)]
funding_pressure = float(np.mean(rate_values)) if rate_values else np.nan

if np.isfinite(spread_stress) and np.isfinite(funding_pressure):
    if spread_stress >= 0.70 and funding_pressure >= 0.70:
        credit_state = "Broad credit stress"
    elif spread_stress <= 0.40 and funding_pressure >= 0.70:
        credit_state = "High-rate / tight-spread"
    elif spread_stress >= 0.70 and funding_pressure < 0.70:
        credit_state = "Credit-specific stress"
    elif spread_stress <= 0.40 and funding_pressure <= 0.40:
        credit_state = "Easy spreads / easy funding"
    else:
        credit_state = "Mixed credit conditions"
elif np.isfinite(spread_stress):
    credit_state = "Spread read only"
elif np.isfinite(funding_pressure):
    credit_state = "Funding-cost read only"
else:
    credit_state = "Insufficient data"

hyg_lqd_move = pct_move(proxy["HYG/LQD"], focus_window) if "HYG/LQD" in proxy else np.nan
kre_spy_move = pct_move(proxy["KRE/SPY"], focus_window) if "KRE/SPY" in proxy else np.nan
vix_level = latest(market["^VIX"]) if "^VIX" in market else np.nan
hy_oas_level = latest(hy_oas)
hy_oas_1m_bp = absolute_move(hy_oas, "1M", scale=100.0)
ig_oas_level = latest(ig_oas)
ten_y = latest(dgs10)
ten_y_pct = trailing_percentile(dgs10, 10)

render_kpi_cards(
    [
        (
            "Credit state",
            credit_state,
            f"Spread stress {fmt_percentile(spread_stress)} · funding pressure {fmt_percentile(funding_pressure)}",
        ),
        (
            "HY OAS",
            f"{hy_oas_level * 100:.0f} bp" if np.isfinite(hy_oas_level) else "N/A",
            f"1M {fmt_bp(hy_oas_1m_bp)} · 5Y {fmt_percentile(spread_percentiles['HY OAS'])}",
        ),
        (
            "IG OAS",
            f"{ig_oas_level * 100:.0f} bp" if np.isfinite(ig_oas_level) else "N/A",
            f"5Y {fmt_percentile(spread_percentiles['IG OAS'])}",
        ),
        (
            "10Y Treasury",
            fmt_yield(ten_y),
            f"10Y history {fmt_percentile(ten_y_pct)}",
        ),
        (
            "HY vs IG",
            fmt_pct(hyg_lqd_move),
            f"HYG/LQD · {focus_window}",
        ),
        (
            "Bank beta",
            fmt_pct(kre_spy_move),
            f"KRE/SPY · {focus_window} · VIX {vix_level:.1f}"
            if np.isfinite(vix_level)
            else f"KRE/SPY · {focus_window}",
        ),
    ]
)

if np.isfinite(spread_stress) and np.isfinite(funding_pressure):
    if credit_state == "High-rate / tight-spread":
        active_read = (
            "Outright borrowing costs are historically expensive while credit spreads remain compressed. "
            "That is a very different regime from low credit stress: the market is charging little incremental "
            "default/liquidity premium on top of a high risk-free base rate."
        )
    elif credit_state == "Broad credit stress":
        active_read = (
            "Both the risk-free base rate and credit risk premia are elevated. This is the cleanest broad "
            "tightening signal and the most hostile configuration for levered balance sheets."
        )
    elif credit_state == "Credit-specific stress":
        active_read = (
            "Credit risk premia are elevated even without unusually high sovereign funding costs. "
            "The stress is coming from credit transmission rather than the risk-free curve."
        )
    else:
        active_read = (
            "Funding costs and spread stress are giving a mixed signal. Treat them as separate dimensions "
            "and use bank, loan, and HY relative performance for confirmation."
        )
else:
    active_read = "Primary spread or rate data are incomplete; use the loaded series without forcing a composite."

render_selection_note("Active credit read", active_read)

display_start = pd.Timestamp(date.today() - timedelta(days=HISTORY_DAYS[history_label]))
left, right = st.columns([1.0, 1.0])

with left:
    st.markdown('<div class="section-title">Credit Spread Stress</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Primary-source ICE BofA option-adjusted spreads from FRED. '
        'This measures default/liquidity risk premium, not the outright Treasury yield level.</div>',
        unsafe_allow_html=True,
    )
    spread_frame = pd.DataFrame(index=fred.index)
    if not hy_oas.empty:
        spread_frame["HY OAS"] = hy_oas * 100.0
    if not bbb_oas.empty:
        spread_frame["BBB OAS"] = bbb_oas * 100.0
    if not ig_oas.empty:
        spread_frame["IG OAS"] = ig_oas * 100.0
    spread_frame = spread_frame.loc[spread_frame.index >= display_start].dropna(how="all")

    if spread_frame.empty:
        st.info("Primary credit spread data unavailable.")
    else:
        fig = go.Figure()
        spread_colors = {
            "HY OAS": COLORS["red"],
            "BBB OAS": COLORS["orange"],
            "IG OAS": COLORS["blue"],
        }
        for column in spread_frame.columns:
            fig.add_trace(
                go.Scatter(
                    x=spread_frame.index,
                    y=spread_frame[column],
                    mode="lines",
                    name=f"{column} {spread_frame[column].dropna().iloc[-1]:.0f} bp",
                    line=dict(color=spread_colors[column], width=2.3),
                )
            )
        fig.update_layout(**chart_layout(height=390, showlegend=True))
        fig.update_yaxes(title_text="Option-adjusted spread (bp)")
        apply_axis_style(fig)
        st.plotly_chart(fig, width="stretch")

with right:
    st.markdown('<div class="section-title">Funding Cost Pressure</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Outright U.S. Treasury yields. A tight spread regime can still '
        'coexist with historically expensive base rates, which is why this is kept separate.</div>',
        unsafe_allow_html=True,
    )
    rate_frame = pd.DataFrame(index=fred.index)
    if not dgs10.empty:
        rate_frame["10Y"] = dgs10
    if not dgs30.empty:
        rate_frame["30Y"] = dgs30
    rate_frame = rate_frame.loc[rate_frame.index >= display_start].dropna(how="all")

    if rate_frame.empty:
        st.info("Primary Treasury yield data unavailable.")
    else:
        fig = go.Figure()
        for column, color in [("10Y", COLORS["blue"]), ("30Y", COLORS["purple"])]:
            if column in rate_frame:
                fig.add_trace(
                    go.Scatter(
                        x=rate_frame.index,
                        y=rate_frame[column],
                        mode="lines",
                        name=f"{column} {rate_frame[column].dropna().iloc[-1]:.2f}%",
                        line=dict(color=color, width=2.4),
                    )
                )
        fig.update_layout(**chart_layout(height=390, showlegend=True))
        fig.update_yaxes(title_text="Yield (%)")
        apply_axis_style(fig)
        st.plotly_chart(fig, width="stretch")

st.markdown('<div class="section-title">Global 10Y Government Yield Moves</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-subtitle">Change in benchmark 10-year sovereign yields over <b>{global_window}</b>. '
    'Orange means yields rose; blue means yields fell. The group median is shown as a dashed line. '
    'Only observations that pass freshness and coverage checks are displayed.</div>',
    unsafe_allow_html=True,
)

with st.spinner("Loading global sovereign yields..."):
    sovereign_moves, sovereign_source, sovereign_note = load_global_sovereign_moves(global_window)

if sovereign_moves.empty:
    st.info(
        "Global sovereign-yield panel is unavailable for this horizon. "
        "The page will not substitute bond ETFs or stale values for benchmark yields."
    )
    st.markdown(f'<div class="source-line">{sovereign_note}</div>', unsafe_allow_html=True)
else:
    dm_count = int((sovereign_moves["Group"] == "Developed").sum())
    em_count = int((sovereign_moves["Group"] == "Emerging").sum())
    dm_median = sovereign_moves.loc[sovereign_moves["Group"] == "Developed", "Move bp"].median()
    em_median = sovereign_moves.loc[sovereign_moves["Group"] == "Emerging", "Move bp"].median()
    st.markdown(
        '<div class="quality-line">'
        f'<b>Source:</b> {sovereign_source} · '
        f'<b>Coverage:</b> {dm_count} developed / {em_count} emerging · '
        f'<b>Median move:</b> Developed {dm_median:+.0f} bp · Emerging {em_median:+.0f} bp'
        + (f' · {sovereign_note}' if sovereign_note else '')
        + '</div>',
        unsafe_allow_html=True,
    )
    max_abs = float(sovereign_moves["Move bp"].abs().max())
    x_limit = max(50.0, np.ceil((max_abs * 1.18) / 25.0) * 25.0)
    dm_col, em_col = st.columns(2)
    with dm_col:
        st.plotly_chart(sovereign_bar_chart(sovereign_moves, "Developed", x_limit), width="stretch")
    with em_col:
        st.plotly_chart(sovereign_bar_chart(sovereign_moves, "Emerging", x_limit), width="stretch")

st.markdown('<div class="section-title">Credit Risk Appetite</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Liquid market confirmation only. These ratios are not treated as '
    'substitutes for primary credit spreads or sovereign yields.</div>',
    unsafe_allow_html=True,
)
proxy_cols = [
    c
    for c in ["HYG/LQD", "BKLN/LQD", "SRLN/LQD", "EMB/LQD", "KRE/SPY", "XLF/SPY"]
    if c in proxy.columns
]

if not proxy_cols:
    st.info("Market confirmation ratios unavailable.")
else:
    proxy_view = proxy.loc[proxy.index >= display_start, proxy_cols].dropna(how="all")
    fig = go.Figure()
    for i, column in enumerate(proxy_view.columns):
        s = clean_series(proxy_view[column])
        if s.empty or s.iloc[0] == 0:
            continue
        rebased = s / s.iloc[0] * 100.0
        fig.add_trace(
            go.Scatter(
                x=rebased.index,
                y=rebased,
                mode="lines",
                name=column,
                line=dict(color=LINE_COLORS[i % len(LINE_COLORS)], width=2.0),
            )
        )
    fig.add_hline(y=100, line_width=1, line_color=COLORS["grid"])
    fig.update_layout(**chart_layout(height=390, showlegend=True))
    fig.update_yaxes(title_text="Rebased to 100")
    apply_axis_style(fig)
    st.plotly_chart(fig, width="stretch")

st.markdown('<div class="section-title">Credit Tape</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Current levels and recent moves. Spreads and Treasury yields are '
    'shown directly; ETF ratios remain confirmation signals.</div>',
    unsafe_allow_html=True,
)
rows: List[dict] = []


def add_spread_row(label: str, series: pd.Series) -> None:
    if series.empty:
        return
    rows.append(
        {
            "Signal": label,
            "Latest": f"{latest(series) * 100:.0f} bp",
            "5D": fmt_bp(absolute_move(series, "5D", 100.0)),
            "1M": fmt_bp(absolute_move(series, "1M", 100.0)),
            "YTD": fmt_bp(absolute_move(series, "YTD", 100.0)),
            "1Y": fmt_bp(absolute_move(series, "1Y", 100.0)),
            "Context": fmt_percentile(trailing_percentile(series, 5)),
            "Role": "Primary spread",
        }
    )


def add_yield_row(label: str, series: pd.Series) -> None:
    if series.empty:
        return
    rows.append(
        {
            "Signal": label,
            "Latest": fmt_yield(latest(series)),
            "5D": fmt_bp(absolute_move(series, "5D", 100.0)),
            "1M": fmt_bp(absolute_move(series, "1M", 100.0)),
            "YTD": fmt_bp(absolute_move(series, "YTD", 100.0)),
            "1Y": fmt_bp(absolute_move(series, "1Y", 100.0)),
            "Context": fmt_percentile(trailing_percentile(series, 10)),
            "Role": "Funding cost",
        }
    )


def add_ratio_row(label: str, series: pd.Series) -> None:
    if series.empty:
        return
    rows.append(
        {
            "Signal": label,
            "Latest": f"{latest(series):.3f}",
            "5D": fmt_pct(pct_move(series, "5D")),
            "1M": fmt_pct(pct_move(series, "1M")),
            "YTD": fmt_pct(pct_move(series, "YTD")),
            "1Y": fmt_pct(pct_move(series, "1Y")),
            "Context": "Higher = stronger banks" if label == "KRE/SPY" else "Higher = stronger",
            "Role": "Market confirmation",
        }
    )


add_spread_row("US HY OAS", hy_oas)
add_spread_row("US BBB OAS", bbb_oas)
add_spread_row("US IG OAS", ig_oas)
add_yield_row("US 10Y Treasury", dgs10)
add_yield_row("US 30Y Treasury", dgs30)

for label in ["HYG/LQD", "BKLN/LQD", "SRLN/LQD", "EMB/LQD", "KRE/SPY", "XLF/SPY"]:
    if label in proxy:
        add_ratio_row(label, proxy[label])

if "^VIX" in market:
    vix = clean_series(market["^VIX"])
    rows.append(
        {
            "Signal": "VIX",
            "Latest": f"{latest(vix):.1f}",
            "5D": f"{absolute_move(vix, '5D'):+.1f}",
            "1M": f"{absolute_move(vix, '1M'):+.1f}",
            "YTD": f"{absolute_move(vix, 'YTD'):+.1f}",
            "1Y": f"{absolute_move(vix, '1Y'):+.1f}",
            "Context": fmt_percentile(trailing_percentile(vix, 5)),
            "Role": "Volatility",
        }
    )

if rows:
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
else:
    st.info("Credit tape unavailable.")

if show_raw:
    st.markdown('<div class="section-title">Data Audit</div>', unsafe_allow_html=True)
    if not fred_status.empty:
        st.markdown(
            '<div class="section-subtitle">Primary FRED source status. Failed or stale series are '
            'left unavailable rather than replaced with a market proxy.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(fred_status, width="stretch", hide_index=True)
    if not market.empty:
        market_audit = pd.DataFrame(
            {
                "Ticker": market.columns,
                "Latest Observation": [
                    latest_timestamp(market[col]).date().isoformat()
                    if latest_timestamp(market[col]) is not None
                    else "N/A"
                    for col in market.columns
                ],
            }
        )
        st.dataframe(market_audit, width="stretch", hide_index=True)

render_footer(
    data_note=(
        "Primary credit spreads and U.S. Treasury yields: Federal Reserve FRED / ICE BofA. "
        "Market confirmation: Yahoo Finance. Global benchmark 10Y yields: Trading Economics when "
        "an API key is configured, otherwise Stooq daily with freshness checks; OECD monthly data "
        "via FRED is used only as a structural fallback for 1Y/3Y/5Y horizons. Missing or stale "
        "sovereign observations are not replaced with bond ETFs."
    )
)
