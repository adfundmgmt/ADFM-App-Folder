############################################################
# Cross-Asset RVOL Stress Composite
# AD Fund Management LP
#
# Pure market-data version
# - Yahoo Finance only
# - Separates raw prices from filled prices
# - Adds local last-good cache fallback
# - Builds class-level RVOL stress across equities, credit,
#   commodities, FX, and rates
# - Adds cross-asset RVOL breadth and dispersion
# - Adds factor change diagnostics and regime transition read
# - Keeps diagnostics, contribution table, and CSV export
############################################################

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---------------- App config ----------------
TITLE = "Market Stress Composite"
st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)

# ---------------- Constants ----------------
NY_TZ = "America/New_York"

CACHE_DIR = Path(".cache_cross_asset_rvol")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LOOKBACK = "3y"
DEFAULT_PCTL_WINDOW_YEARS = 3
DEFAULT_SMOOTH = 1

RV_WINDOW = 21
REGIME_HI = 70
REGIME_LO = 30

YF_TIMEOUT = 15
YF_RETRIES = 3

# ---------------- Universe ----------------
ASSET_CLASS_MAP: Dict[str, List[str]] = {
    "Equities": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
    "Credit": ["HYG", "JNK", "LQD", "EMB"],
    "Commodities": ["GLD", "SLV", "USO", "DBA", "CPER"],
    "FX": ["UUP", "FXE", "FXY", "FXB", "CEW"],
    "Rates": ["TLT", "IEF", "SHY", "TIP"],
}

ALL_TICKERS = [t for group in ASSET_CLASS_MAP.values() for t in group]

MARKET_CONTEXT_TICKERS = [
    "SPY",
    "^GSPC",
    "^VIX",
    "DX-Y.NYB",
]

LOAD_TICKERS = sorted(set(ALL_TICKERS + MARKET_CONTEXT_TICKERS))

# ---------------- Weights ----------------
BASE_WEIGHTS = {
    "Equities_p": 0.25,
    "Credit_p": 0.20,
    "Commodities_p": 0.15,
    "FX_p": 0.15,
    "Rates_p": 0.15,
    "Breadth_p": 0.07,
    "Dispersion_p": 0.03,
}

FACTOR_ORDER = list(BASE_WEIGHTS.keys())
WEIGHTS_VEC = np.array([BASE_WEIGHTS[k] for k in FACTOR_ORDER], dtype=float)
WEIGHTS_VEC = WEIGHTS_VEC / WEIGHTS_VEC.sum()

WEIGHTS_TEXT = ", ".join(
    [
        f"{k.replace('_p', '')} {v:.2f}"
        for k, v in BASE_WEIGHTS.items()
    ]
)

# stale limit in trading sessions
STALE_LIMITS = {
    "Equities": 2,
    "Credit": 2,
    "Commodities": 2,
    "FX": 2,
    "Rates": 2,
    "Breadth": 2,
    "Dispersion": 2,
    "SPX": 2,
    "DD": 2,
}

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")

    lookback = st.selectbox(
        "Lookback",
        ["1y", "2y", "3y", "5y", "10y"],
        index=["1y", "2y", "3y", "5y", "10y"].index(DEFAULT_LOOKBACK),
    )
    years = int(lookback[:-1])

    pctl_window_years = st.slider(
        "Percentile window, years",
        1,
        10,
        DEFAULT_PCTL_WINDOW_YEARS,
        1,
        help="RVOL percentiles are calculated versus this trailing history window.",
    )

    smooth_days = st.slider("Composite smoothing", 1, 10, DEFAULT_SMOOTH, 1)

    rv_window = st.slider(
        "RVOL window",
        10,
        63,
        RV_WINDOW,
        1,
        help="Trading-day window used to calculate annualized realized volatility.",
    )

    stress_cutoff = st.slider(
        "Breadth stress threshold",
        60,
        90,
        70,
        1,
        help="Ticker-level RVOL percentile threshold used to measure stress breadth.",
    )

    st.subheader("Weights")
    st.caption(WEIGHTS_TEXT)

    st.markdown("---")
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose:** Market-data cross-asset stress gauge built from realized volatility across major markets.

        **What this tab shows**
        - Daily, 5D, and 21D stress-regime readings.
        - Which sleeves are driving the composite stress score.
        - Whether realized-volatility pressure is concentrated or broadening.
        - Whether the signal is rising, fading, or sitting near peak stress.

        **Data source**
        - Yahoo Finance for ETFs and market proxies.
        - Local last-good cache is used when Yahoo fails for a ticker.
        """
    )

# ---------------- Helpers ----------------
def now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY_TZ)


def today_naive() -> pd.Timestamp:
    return pd.Timestamp(now_et().date())


def as_naive_date(ts_like) -> pd.Timestamp:
    t = pd.Timestamp(ts_like)
    if t.tz is not None:
        t = t.tz_convert(NY_TZ).tz_localize(None)
    return pd.Timestamp(t.date())


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def normalize_date_index(obj):
    if obj is None:
        return obj

    if isinstance(obj, (pd.Series, pd.DataFrame)):
        out = obj.copy()
        if out.empty:
            return out

        idx = pd.DatetimeIndex(pd.to_datetime(out.index))
        if idx.tz is not None:
            idx = idx.tz_convert(None)

        out.index = idx.normalize()
        out = out.sort_index()
        out = out.groupby(level=0).last()
        return out

    return obj


def clean_cache_name(ticker: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_")


def cache_path_for(ticker: str) -> Path:
    return CACHE_DIR / f"{clean_cache_name(ticker)}.csv"


def read_cached_series(ticker: str) -> pd.Series:
    path = cache_path_for(ticker)

    if not path.exists():
        return pd.Series(dtype=float, name=ticker)

    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        if df.empty or "price" not in df.columns:
            return pd.Series(dtype=float, name=ticker)

        s = pd.to_numeric(df["price"], errors="coerce")
        out = pd.Series(s.values, index=pd.to_datetime(df["Date"]), name=ticker).dropna()
        out = normalize_date_index(out)
        return out.sort_index()
    except Exception:
        return pd.Series(dtype=float, name=ticker)


def write_cached_series(ticker: str, series: pd.Series) -> None:
    if series is None or series.empty:
        return

    s = normalize_date_index(series).dropna()
    if s.empty:
        return

    path = cache_path_for(ticker)

    out = pd.DataFrame(
        {
            "Date": s.index,
            "price": pd.to_numeric(s, errors="coerce").values,
        }
    ).dropna()

    if out.empty:
        return

    try:
        out.to_csv(path, index=False)
    except Exception:
        pass


def _extract_preferred_price(df: pd.DataFrame, ticker: str) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float, name=ticker)

    data = df.copy()

    if isinstance(data.columns, pd.MultiIndex):
        level0 = [str(x) for x in data.columns.get_level_values(0)]
        level1 = [str(x) for x in data.columns.get_level_values(1)]

        if ticker in level1:
            sub = data.xs(ticker, axis=1, level=1, drop_level=True).copy()
        elif ticker in level0:
            sub = data[ticker].copy()
            if isinstance(sub, pd.Series):
                sub = sub.to_frame(name=ticker)
        else:
            return pd.Series(dtype=float, name=ticker)
    else:
        sub = data.copy()

    if isinstance(sub, pd.Series):
        s = pd.to_numeric(sub, errors="coerce").dropna().rename(ticker)
        s.index = pd.to_datetime(s.index)
        return normalize_date_index(s.sort_index())

    sub.columns = [str(c) for c in sub.columns]

    preferred_cols = [c for c in ["Adj Close", "Close", ticker] if c in sub.columns]

    if preferred_cols:
        preferred = preferred_cols[0]
    else:
        num_cols = sub.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return pd.Series(dtype=float, name=ticker)
        preferred = num_cols[0]

    s = pd.to_numeric(sub[preferred], errors="coerce").dropna().rename(ticker)
    s.index = pd.to_datetime(sub.index)
    return normalize_date_index(s.sort_index())


def yf_download_one(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    start = as_naive_date(start)
    end = as_naive_date(end)

    for attempt in range(YF_RETRIES):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end + pd.Timedelta(days=1),
                progress=False,
                auto_adjust=False,
                actions=False,
                threads=False,
                group_by="column",
                timeout=YF_TIMEOUT,
            )

            s = _extract_preferred_price(df, ticker)
            if not s.empty:
                s = s[~s.index.duplicated(keep="last")]
                s = s.loc[(s.index >= start) & (s.index <= end)]
                if not s.empty:
                    return s.sort_index()

        except Exception:
            pass

        if attempt < YF_RETRIES - 1:
            time.sleep(1.0 + attempt)

    return pd.Series(dtype=float, name=ticker)


@st.cache_data(ttl=1800, show_spinner=False)
def load_price_panel(
    tickers_tuple: Tuple[str, ...],
    start_iso: str,
    end_iso: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(start_iso)
    end = pd.Timestamp(end_iso)

    series_list = []
    meta_rows = []

    for ticker in tickers_tuple:
        cached = read_cached_series(ticker)

        if not cached.empty:
            fetch_start = max(start, cached.index.max() - pd.Timedelta(days=10))
        else:
            fetch_start = start

        downloaded = yf_download_one(ticker, fetch_start, end)

        if not downloaded.empty:
            if not cached.empty:
                combined = pd.concat([cached, downloaded], axis=0)
                combined = normalize_date_index(combined)
                combined = combined.loc[combined.index <= end]
            else:
                combined = normalize_date_index(downloaded)

            write_cached_series(ticker, combined)
            s = combined.loc[(combined.index >= start) & (combined.index <= end)].dropna()
            source = "Yahoo plus cache" if not cached.empty else "Yahoo"
        else:
            s = cached.loc[(cached.index >= start) & (cached.index <= end)].dropna()
            source = "Local cache fallback" if not s.empty else "Missing"

        if not s.empty:
            s = s.rename(ticker)
            series_list.append(s)

        last_date = s.index.max() if not s.empty else pd.NaT
        first_date = s.index.min() if not s.empty else pd.NaT

        meta_rows.append(
            {
                "Ticker": ticker,
                "Source": source,
                "Observations": int(s.dropna().shape[0]) if not s.empty else 0,
                "First Date": first_date.date().isoformat() if pd.notna(first_date) else "",
                "Last Date": last_date.date().isoformat() if pd.notna(last_date) else "",
                "Calendar Days Since Last": int((end - last_date).days) if pd.notna(last_date) else np.nan,
            }
        )

    if series_list:
        panel = pd.concat(series_list, axis=1).sort_index()
        panel = normalize_date_index(panel)
    else:
        panel = pd.DataFrame()

    meta = pd.DataFrame(meta_rows)
    return panel, meta


def build_trading_calendar_from_panel(
    raw_px: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Tuple[pd.DatetimeIndex, str]:
    candidates = ["SPY", "^GSPC", "QQQ", "IWM"]

    for ticker in candidates:
        if ticker in raw_px.columns:
            s = raw_px[ticker].dropna()
            if s.shape[0] > 100:
                return pd.DatetimeIndex(s.index.unique()).sort_values(), ticker

    return pd.DatetimeIndex(pd.bdate_range(start, end)), "Business-day fallback"


def reindex_raw(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    if s is None or s.empty:
        return pd.Series(index=idx, dtype=float)

    s = normalize_date_index(s).dropna()
    return s.reindex(idx)


def rolling_percentile_trading(
    values: pd.Series,
    idx: pd.DatetimeIndex,
    window_trading_days: int,
) -> pd.Series:
    s = normalize_date_index(values).reindex(idx)

    def _last_percentile(x):
        arr = pd.Series(x).dropna()
        if arr.empty:
            return np.nan
        return arr.rank(pct=True).iloc[-1]

    out = s.rolling(
        window_trading_days,
        min_periods=max(40, window_trading_days // 6),
    ).apply(_last_percentile, raw=False)

    return normalize_date_index(out)


def compute_stale_info(
    original_raw: pd.Series,
    idx: pd.DatetimeIndex,
    max_stale: int,
) -> Tuple[pd.Series, pd.Series]:
    original_raw = normalize_date_index(original_raw).reindex(idx)

    pos = np.arange(len(idx))
    has = original_raw.notna().values

    last_pos = pd.Series(np.where(has, pos, np.nan), index=idx).ffill().values
    age = pos - last_pos
    age = np.where(np.isnan(last_pos), np.nan, age)

    age_s = pd.Series(age, index=idx).astype(float)
    mask = pd.Series((~np.isnan(age)) & (age <= max_stale), index=idx)

    return mask, age_s


def regime_label(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    if x >= REGIME_HI:
        return "High stress"
    if x <= REGIME_LO:
        return "Low stress"
    return "Neutral"


def regime_bucket(x: float) -> str:
    if pd.isna(x):
        return "na"
    if x >= REGIME_HI:
        return "high"
    if x <= REGIME_LO:
        return "low"
    return "neutral"


def dd_bucket(dd: float) -> str:
    if pd.isna(dd):
        return "na"
    if dd < 5:
        return "shallow"
    if dd < 15:
        return "medium"
    return "deep"


def mean_last(series: pd.Series, n: int) -> float:
    ser = series.dropna()
    if ser.empty:
        return np.nan
    return float(ser.tail(n).mean())


def change_last(series: pd.Series, n: int) -> float:
    ser = series.dropna()
    if len(ser) <= n:
        return np.nan
    return float(ser.iloc[-1] - ser.iloc[-1 - n])


def latest_or_nan(series: pd.Series) -> float:
    ser = series.dropna()
    if ser.empty:
        return np.nan
    return float(ser.iloc[-1])


def stress_cell_style(v):
    try:
        x = float(v)
    except Exception:
        return ""

    if pd.isna(x):
        return ""

    if x >= REGIME_HI:
        return "background-color: #f8d7da; color: #842029;"
    if x <= REGIME_LO:
        return "background-color: #d1e7dd; color: #0f5132;"

    return "background-color: #f1f3f5; color: #343a40;"


def style_factor_table(df: pd.DataFrame):
    fmt = {
        "Current %ile": "{:.1f}",
        "1D Chg": "{:+.1f}",
        "5D Chg": "{:+.1f}",
        "21D Chg": "{:+.1f}",
        "Base Weight %": "{:.1f}",
        "Effective Weight %": "{:.1f}",
        "Contribution pts": "{:.1f}",
        "Contribution 5D Chg": "{:+.1f}",
    }

    styler = df.style.format(fmt, na_rep="N/A")

    cols_to_color = [
        c
        for c in ["Current %ile", "1D Chg", "5D Chg", "21D Chg"]
        if c in df.columns
    ]

    try:
        styler = styler.map(stress_cell_style, subset=["Current %ile"])
    except Exception:
        try:
            styler = styler.applymap(stress_cell_style, subset=["Current %ile"])
        except Exception:
            pass

    return styler


def generate_commentary(
    as_of_date: pd.Timestamp,
    daily_val: float,
    five_day_val: float,
    twenty_one_day_val: float,
    comp_5d_chg: float,
    spx_level: float,
    dd_val: float,
    latest_factor_table: pd.DataFrame,
    active_weight: float,
    active_factors: int,
    breadth_raw_latest: float,
    breadth_raw_5d_chg: float,
    days_since_peak: float,
) -> str:
    d_reg = regime_bucket(daily_val)
    f_reg = regime_bucket(five_day_val)
    m_reg = regime_bucket(twenty_one_day_val)
    dd_b = dd_bucket(dd_val)

    date_str = as_of_date.date().isoformat()
    dd_pct = f"{dd_val:.1f}" if not pd.isna(dd_val) else "N/A"
    spx_str = f"{spx_level:,.1f}" if not pd.isna(spx_level) else "N/A"

    top = latest_factor_table.sort_values("Contribution pts", ascending=False).head(3)
    drivers = ", ".join(
        [
            f"{row['Factor']} ({row['Contribution pts']:.1f} pts)"
            for _, row in top.iterrows()
            if pd.notna(row["Contribution pts"]) and row["Contribution pts"] > 0
        ]
    )

    if not drivers:
        drivers = "no single sleeve is dominating the tape"

    rising = latest_factor_table.sort_values("5D Chg", ascending=False).head(1)
    fading = latest_factor_table.sort_values("5D Chg", ascending=True).head(1)

    rising_text = ""
    fading_text = ""

    if not rising.empty and pd.notna(rising.iloc[0]["5D Chg"]):
        rising_text = f"The fastest rising sleeve over five sessions is {rising.iloc[0]['Factor']} at {rising.iloc[0]['5D Chg']:+.1f} points."

    if not fading.empty and pd.notna(fading.iloc[0]["5D Chg"]):
        fading_text = f"The fastest fading sleeve is {fading.iloc[0]['Factor']} at {fading.iloc[0]['5D Chg']:+.1f} points."

    if pd.isna(comp_5d_chg):
        direction_text = "The five-session direction is incomplete because the recent composite history is not long enough."
    elif comp_5d_chg >= 8:
        direction_text = f"The composite has risen {comp_5d_chg:+.1f} points over five sessions, so stress is actively building."
    elif comp_5d_chg <= -8:
        direction_text = f"The composite has fallen {comp_5d_chg:+.1f} points over five sessions, so realized-vol pressure is fading."
    else:
        direction_text = f"The composite is roughly stable over five sessions at {comp_5d_chg:+.1f} points."

    daily_map = {
        "low": "The daily tape is low stress, with realized volatility contained across the major sleeves.",
        "neutral": "The daily tape is neutral, which fits a market digesting movement without broad cross-asset destabilization.",
        "high": "The daily tape is high stress, with realized volatility elevated enough to matter across the broader risk stack.",
        "na": "The daily read is incomplete because too many market inputs are missing.",
    }

    five_day_map = {
        "low": "The 5D average still looks calm.",
        "neutral": "The 5D average is neutral and unstable rather than cleanly risk-on or risk-off.",
        "high": "The 5D average is elevated, which means the stress is persisting beyond a one-session impulse.",
        "na": "The 5D average is not fully clean, so the daily signal deserves more weight.",
    }

    twenty_one_day_map = {
        "low": "The 21D backdrop remains contained.",
        "neutral": "The 21D backdrop is neutral, which usually lines up with a choppier but still tradable market.",
        "high": "The 21D backdrop is elevated, which is where volatility regimes can start feeding on themselves.",
        "na": "The 21D state is blurred by missing data.",
    }

    dd_map = {
        "shallow": f"SPX is near {spx_str} with a shallow drawdown of about {dd_pct}%, so the price tape itself has not delivered a deep reset.",
        "medium": f"SPX is near {spx_str} with a drawdown of about {dd_pct}%, which is where investors start asking whether stress is cyclical or something larger.",
        "deep": f"SPX is near {spx_str} with a deep drawdown of about {dd_pct}%, which is where reflexive turns can emerge if realized vol begins to compress.",
        "na": "The drawdown context is unclear, so the composite should be read mainly as a realized-vol barometer.",
    }

    breadth_text = ""
    if pd.notna(breadth_raw_latest):
        breadth_text = f"Raw stress breadth is {breadth_raw_latest:.0%}"
        if pd.notna(breadth_raw_5d_chg):
            breadth_text += f", with a five-session change of {breadth_raw_5d_chg:+.0%}."
        else:
            breadth_text += "."

    peak_text = ""
    if pd.notna(days_since_peak):
        peak_text = f"The current lookback peak occurred {days_since_peak:.0f} trading sessions ago."

    confidence_text = (
        f"Composite confidence is moderate to high with {active_factors} active factors "
        f"and {active_weight:.0%} active weight."
        if pd.notna(active_weight)
        else "Composite confidence is incomplete because active factor weight is unavailable."
    )

    return (
        f"As of {date_str}, {daily_map[d_reg]} {five_day_map[f_reg]} {twenty_one_day_map[m_reg]} "
        f"{dd_map[dd_b]} The main drivers are {drivers}. {direction_text} "
        f"{rising_text} {fading_text} {breadth_text} {peak_text} {confidence_text}"
    )


# ---------------- Load history ----------------
status = st.status("Loading cross-asset market inputs...", expanded=False)

load_ts = now_et()
today = today_naive()
start_all = as_naive_date(today - pd.DateOffset(years=12))

status.write("Loading Yahoo market proxies with local cache fallback...")
raw_px, fetch_meta = load_price_panel(
    tuple(LOAD_TICKERS),
    start_all.date().isoformat(),
    today.date().isoformat(),
)

if raw_px.empty:
    status.update(label="No usable market data could be loaded", state="error")
    st.error("No usable market data could be loaded from Yahoo or local cache.")
    st.stop()

status.write("Building trading calendar...")
trade_idx_all, calendar_source = build_trading_calendar_from_panel(raw_px, start_all, today)

if len(trade_idx_all) == 0:
    status.update(label="Failed to build trading calendar", state="error")
    st.error("Unable to construct any usable trading calendar.")
    st.stop()

if calendar_source == "Business-day fallback":
    st.warning("Market session calendar could not be loaded from Yahoo. Using business-day fallback.")
else:
    st.caption(f"Calendar source: {calendar_source}")

st.caption(f"Data load timestamp: {load_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")

# ---------------- Build raw and filled panels ----------------
raw_panel = pd.DataFrame(index=trade_idx_all)

for ticker in sorted(set(ALL_TICKERS)):
    if ticker in raw_px.columns:
        raw_panel[ticker] = reindex_raw(raw_px[ticker], trade_idx_all)
    else:
        raw_panel[ticker] = np.nan

if "^GSPC" in raw_px.columns and not raw_px["^GSPC"].dropna().empty:
    raw_panel["SPX"] = reindex_raw(raw_px["^GSPC"], trade_idx_all)
elif "SPY" in raw_px.columns and not raw_px["SPY"].dropna().empty:
    spy_raw = reindex_raw(raw_px["SPY"], trade_idx_all)
    base_spy_raw = safe_float(spy_raw.dropna().iloc[0]) if not spy_raw.dropna().empty else np.nan
    raw_panel["SPX"] = (spy_raw / base_spy_raw) * 100.0 if pd.notna(base_spy_raw) and base_spy_raw != 0 else np.nan
else:
    raw_panel["SPX"] = np.nan

if "^VIX" in raw_px.columns:
    raw_panel["VIX"] = reindex_raw(raw_px["^VIX"], trade_idx_all)
else:
    raw_panel["VIX"] = np.nan

if "DX-Y.NYB" in raw_px.columns:
    raw_panel["DXY"] = reindex_raw(raw_px["DX-Y.NYB"], trade_idx_all)
else:
    raw_panel["DXY"] = np.nan

raw_panel = normalize_date_index(raw_panel)
filled_panel = raw_panel.ffill()

panel = filled_panel.copy()

# ---------------- RVOL construction ----------------
status.write("Computing realized volatility and rolling percentiles...")

ret = panel[sorted(set(ALL_TICKERS))].pct_change(fill_method=None)
rvol = np.sqrt(252.0) * ret.rolling(
    rv_window,
    min_periods=max(5, rv_window // 2),
).std()

rvol = normalize_date_index(rvol)

window_td = int(pctl_window_years * 252)
rvol_pct = pd.DataFrame(index=panel.index)

for ticker in rvol.columns:
    rvol_pct[f"{ticker}_p"] = rolling_percentile_trading(
        rvol[ticker],
        panel.index,
        window_td,
    )

rvol_pct = normalize_date_index(rvol_pct)

# ---------------- Asset-class factors ----------------
factor_panel = pd.DataFrame(index=panel.index)

for asset_class, tickers in ASSET_CLASS_MAP.items():
    cols = [f"{t}_p" for t in tickers if f"{t}_p" in rvol_pct.columns]
    if cols:
        factor_panel[f"{asset_class}_p"] = rvol_pct[cols].mean(axis=1, skipna=True)
    else:
        factor_panel[f"{asset_class}_p"] = np.nan

all_pct_cols = [c for c in rvol_pct.columns if c.endswith("_p")]

if all_pct_cols:
    pct_values = rvol_pct[all_pct_cols] * 100.0
    valid_values = rvol_pct[all_pct_cols].notna()
    stress_bool = pct_values.ge(stress_cutoff)

    valid_count = valid_values.sum(axis=1).replace(0, np.nan)
    breadth_raw = stress_bool.where(valid_values).sum(axis=1) / valid_count
    factor_panel["Breadth_p"] = rolling_percentile_trading(
        breadth_raw,
        panel.index,
        window_td,
    )
else:
    breadth_raw = pd.Series(index=panel.index, dtype=float)
    factor_panel["Breadth_p"] = np.nan

class_cols = [f"{k}_p" for k in ASSET_CLASS_MAP.keys()]
disp_raw = factor_panel[class_cols].std(axis=1, skipna=True)
factor_panel["Dispersion_p"] = rolling_percentile_trading(
    disp_raw,
    panel.index,
    window_td,
)

factor_panel = normalize_date_index(factor_panel)

# ---------------- Derived series ----------------
panel["DD_stress"] = -(
    100.0 * (panel["SPX"] / panel["SPX"].cummax() - 1.0).clip(upper=0)
)

# ---------------- Lookback slice ----------------
start_lb = as_naive_date(today - pd.DateOffset(years=years))

panel_lb = panel.loc[panel.index >= start_lb].copy()
raw_panel_lb = raw_panel.loc[raw_panel.index >= start_lb].copy()
scores = factor_panel.loc[factor_panel.index >= start_lb].copy()

if smooth_days > 1:
    scores = scores.rolling(smooth_days, min_periods=1).mean()

# ---------------- Freshness masks ----------------
mask_map = {}
age_map = {}

for asset_class, tickers in ASSET_CLASS_MAP.items():
    tickers_present = [t for t in tickers if t in raw_panel.columns]

    if tickers_present:
        raw_series = raw_panel[tickers_present].mean(axis=1, skipna=True)
    else:
        raw_series = pd.Series(index=raw_panel.index, dtype=float)

    m, age = compute_stale_info(
        raw_series,
        panel.index,
        STALE_LIMITS.get(asset_class, 2),
    )

    mask_map[asset_class] = m
    age_map[asset_class] = age

universe_raw_series = raw_panel[sorted(set(ALL_TICKERS))].mean(axis=1, skipna=True)

mask_map["Breadth"], age_map["Breadth"] = compute_stale_info(
    universe_raw_series,
    panel.index,
    STALE_LIMITS["Breadth"],
)

mask_map["Dispersion"], age_map["Dispersion"] = compute_stale_info(
    universe_raw_series,
    panel.index,
    STALE_LIMITS["Dispersion"],
)

mask_map["SPX"], age_map["SPX"] = compute_stale_info(
    raw_panel["SPX"],
    panel.index,
    STALE_LIMITS["SPX"],
)

mask_map["DD"] = mask_map["SPX"].copy()
age_map["DD"] = age_map["SPX"].copy()

# ---------------- Masks aligned ----------------
masks = pd.DataFrame(index=scores.index)

for factor_name in ["Equities", "Credit", "Commodities", "FX", "Rates", "Breadth", "Dispersion"]:
    masks[f"{factor_name}_m"] = (
        mask_map[factor_name]
        .reindex(scores.index)
        .fillna(False)
        .astype(float)
    )

score_cols = FACTOR_ORDER
mask_cols = [c.replace("_p", "_m") for c in score_cols]

missing_score_cols = [c for c in score_cols if c not in scores.columns]
if missing_score_cols:
    status.update(label="Composite has missing factor columns", state="error")
    st.error(f"Missing score columns: {missing_score_cols}")
    st.stop()

X = scores[score_cols].values
M = masks[mask_cols].values
W = WEIGHTS_VEC.reshape(1, -1)

X_masked = np.nan_to_num(X, nan=0.0) * M
weight_mask = W * M
active_w = weight_mask.sum(axis=1)

weighted_contrib = np.where(
    active_w[:, None] > 0,
    100.0 * (X_masked * W) / active_w[:, None],
    np.nan,
)

comp = np.where(
    active_w > 0,
    weighted_contrib.sum(axis=1),
    np.nan,
)

comp_s = pd.Series(comp, index=scores.index, name="Composite").dropna()
active_weight_s = pd.Series(active_w, index=scores.index, name="active_weight").reindex(comp_s.index)
active_factors_s = pd.Series(M.sum(axis=1), index=scores.index, name="active_factors").reindex(comp_s.index)

contrib_cols = [c.replace("_p", "_contribution") for c in score_cols]
contrib_panel = pd.DataFrame(
    weighted_contrib,
    index=scores.index,
    columns=contrib_cols,
)

if comp_s.empty:
    status.update(label="Composite has no valid points for the selected lookback", state="error")
    st.error("Composite has no valid points for the selected lookback.")
    st.stop()

status.update(label="Data loaded", state="complete")

# ---------------- Latest stats ----------------
latest_idx = comp_s.index[-1]
latest_val = float(comp_s.iloc[-1])
five_day_val = mean_last(comp_s, 5)
twenty_one_day_val = mean_last(comp_s, 21)

comp_1d_chg = change_last(comp_s, 1)
comp_5d_chg = change_last(comp_s, 5)
comp_21d_chg = change_last(comp_s, 21)

plot_idx = comp_s.index.copy()
panel_plot = panel_lb.reindex(plot_idx).ffill()
raw_panel_plot = raw_panel_lb.reindex(plot_idx)

spx = panel_plot["SPX"].dropna()
base = safe_float(spx.iloc[0]) if len(spx) else np.nan
spx_rebased = (
    (panel_plot["SPX"] / base) * 100.0
    if pd.notna(base) and base != 0
    else pd.Series(index=plot_idx, dtype=float)
)

dd_plot = (
    100.0 * (panel_plot["SPX"] / panel_plot["SPX"].cummax() - 1.0)
).clip(upper=0)

latest_spx_level = safe_float(panel_plot["SPX"].iloc[-1]) if len(panel_plot) else np.nan
latest_dd_val = safe_float(panel_plot["DD_stress"].iloc[-1]) if len(panel_plot) else np.nan
latest_active_weight = safe_float(active_weight_s.iloc[-1]) if not active_weight_s.empty else np.nan
latest_active_factors = (
    int(active_factors_s.iloc[-1])
    if not active_factors_s.empty and pd.notna(active_factors_s.iloc[-1])
    else 0
)

latest_pos = scores.index.get_loc(latest_idx)

current_scores = scores.loc[latest_idx, score_cols] * 100.0
score_1d_chg = scores[score_cols].diff(1).loc[latest_idx] * 100.0
score_5d_chg = scores[score_cols].diff(5).loc[latest_idx] * 100.0
score_21d_chg = scores[score_cols].diff(21).loc[latest_idx] * 100.0

current_contrib = contrib_panel.loc[latest_idx, contrib_cols]
contrib_5d_chg = contrib_panel[contrib_cols].diff(5).loc[latest_idx]

factor_names = [c.replace("_p", "") for c in score_cols]

latest_factor_table = pd.DataFrame(
    {
        "Factor": factor_names,
        "Current %ile": current_scores.values,
        "1D Chg": score_1d_chg.values,
        "5D Chg": score_5d_chg.values,
        "21D Chg": score_21d_chg.values,
        "Active": masks.loc[latest_idx, mask_cols].values.astype(bool),
        "Base Weight %": WEIGHTS_VEC * 100.0,
        "Effective Weight %": np.where(
            latest_active_weight > 0,
            (WEIGHTS_VEC * masks.loc[latest_idx, mask_cols].values) / latest_active_weight * 100.0,
            np.nan,
        ),
        "Contribution pts": current_contrib.values,
        "Contribution 5D Chg": contrib_5d_chg.values,
    }
)

latest_factor_table = latest_factor_table.sort_values("Contribution pts", ascending=False)

breadth_raw_plot = breadth_raw.reindex(plot_idx)
latest_breadth_raw = latest_or_nan(breadth_raw_plot)
breadth_raw_5d_chg = change_last(breadth_raw_plot, 5)

peak_idx = comp_s.idxmax()
days_since_peak = float(len(comp_s.loc[peak_idx:]) - 1) if pd.notna(peak_idx) else np.nan
peak_val = safe_float(comp_s.max())

# ---------------- Commentary ----------------
commentary_text = generate_commentary(
    as_of_date=latest_idx,
    daily_val=latest_val,
    five_day_val=five_day_val,
    twenty_one_day_val=twenty_one_day_val,
    comp_5d_chg=comp_5d_chg,
    spx_level=latest_spx_level,
    dd_val=latest_dd_val,
    latest_factor_table=latest_factor_table,
    active_weight=latest_active_weight,
    active_factors=latest_active_factors,
    breadth_raw_latest=latest_breadth_raw,
    breadth_raw_5d_chg=breadth_raw_5d_chg,
    days_since_peak=days_since_peak,
)

st.info(commentary_text)

# ---------------- Top metrics ----------------
c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric(
    "Daily regime",
    regime_label(latest_val),
    f"{latest_val:.0f}",
)

c2.metric(
    "5D regime",
    regime_label(five_day_val),
    f"{five_day_val:.0f}" if not pd.isna(five_day_val) else "N/A",
)

c3.metric(
    "21D regime",
    regime_label(twenty_one_day_val),
    f"{twenty_one_day_val:.0f}" if not pd.isna(twenty_one_day_val) else "N/A",
)

c4.metric(
    "5D change",
    f"{comp_5d_chg:+.1f}" if not pd.isna(comp_5d_chg) else "N/A",
)

c5.metric(
    "Raw stress breadth",
    f"{latest_breadth_raw:.0%}" if not pd.isna(latest_breadth_raw) else "N/A",
    f"{breadth_raw_5d_chg:+.0%}" if not pd.isna(breadth_raw_5d_chg) else None,
)

c6.metric(
    "Active weight",
    f"{latest_active_weight:.0%}" if pd.notna(latest_active_weight) else "N/A",
    f"{latest_active_factors}/7 factors",
)

# ---------------- Tabs ----------------
tab_composite, tab_factors, tab_diagnostics, tab_download = st.tabs(
    [
        "Composite",
        "Factors",
        "Diagnostics",
        "Download Data",
    ]
)

with tab_composite:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.62, 0.38],
        specs=[[{"secondary_y": True}], [{}]],
        subplot_titles=("SPX and Drawdown", "Cross-Asset RVOL Stress Composite"),
    )

    fig.add_trace(
        go.Scatter(
            x=plot_idx,
            y=spx_rebased.reindex(plot_idx),
            name="SPX rebased",
            line=dict(width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>SPX rebased: %{y:.1f}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=plot_idx,
            y=dd_plot.reindex(plot_idx),
            name="Drawdown",
            line=dict(width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=plot_idx,
            y=comp_s.reindex(plot_idx),
            name="Composite",
            line=dict(width=2.3),
            hovertemplate="%{x|%Y-%m-%d}<br>Composite: %{y:.1f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=plot_idx,
            y=comp_s.reindex(plot_idx).rolling(21, min_periods=1).mean(),
            name="Composite 21D avg",
            line=dict(width=1.4, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d}<br>21D avg: %{y:.1f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_hrect(
        y0=REGIME_HI,
        y1=100,
        line_width=0,
        fillcolor="rgba(214,39,40,0.10)",
        row=2,
        col=1,
    )

    fig.add_hrect(
        y0=0,
        y1=REGIME_LO,
        line_width=0,
        fillcolor="rgba(44,160,44,0.10)",
        row=2,
        col=1,
    )

    finite_dd = dd_plot.replace([np.inf, -np.inf], np.nan).dropna()
    dd_floor = -5.0 if finite_dd.empty else min(-5.0, float(finite_dd.min()) * 1.1)

    fig.update_yaxes(title_text="Rebased", row=1, col=1, secondary_y=False)

    fig.update_yaxes(
        title_text="Drawdown %",
        row=1,
        col=1,
        secondary_y=True,
        range=[dd_floor, 0],
        tickformat=".1f",
        zeroline=True,
        zerolinewidth=1,
    )

    fig.update_yaxes(title_text="Score", row=2, col=1, range=[0, 100])
    fig.update_xaxes(title_text="Date", row=2, col=1, tickformat="%b-%d-%y")

    fig.update_layout(
        template="plotly_white",
        height=780,
        margin=dict(l=50, r=30, t=80, b=50),
        legend=dict(orientation="h", x=0, y=1.08, xanchor="left"),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Regime transition")

    t1, t2, t3, t4, t5 = st.columns(5)

    t1.metric(
        "Peak score",
        f"{peak_val:.1f}" if pd.notna(peak_val) else "N/A",
    )

    t2.metric(
        "Days since peak",
        f"{days_since_peak:.0f}" if pd.notna(days_since_peak) else "N/A",
    )

    t3.metric(
        "1D composite change",
        f"{comp_1d_chg:+.1f}" if pd.notna(comp_1d_chg) else "N/A",
    )

    t4.metric(
        "21D composite change",
        f"{comp_21d_chg:+.1f}" if pd.notna(comp_21d_chg) else "N/A",
    )

    t5.metric(
        "SPX drawdown",
        f"{latest_dd_val:.1f}%" if pd.notna(latest_dd_val) else "N/A",
    )

    st.subheader("Current factor contributions")
    show_factor = latest_factor_table.copy()

    round_cols = [
        "Current %ile",
        "1D Chg",
        "5D Chg",
        "21D Chg",
        "Base Weight %",
        "Effective Weight %",
        "Contribution pts",
        "Contribution 5D Chg",
    ]

    for col in round_cols:
        show_factor[col] = pd.to_numeric(show_factor[col], errors="coerce").round(1)

    st.dataframe(
        style_factor_table(show_factor),
        use_container_width=True,
        hide_index=True,
    )

with tab_factors:
    st.subheader("Factor heatmap")

    factor_heat = (scores[score_cols] * 100.0).copy()
    factor_heat.columns = [c.replace("_p", "") for c in factor_heat.columns]

    if len(factor_heat) > 650:
        factor_heat_plot = factor_heat.resample("W-FRI").last().dropna(how="all")
        heat_caption = "Weekly sampled to keep the heatmap readable."
    else:
        factor_heat_plot = factor_heat.copy()
        heat_caption = "Daily factor percentiles."

    heat_fig = go.Figure(
        data=go.Heatmap(
            z=factor_heat_plot.T.values,
            x=factor_heat_plot.index,
            y=factor_heat_plot.columns,
            zmin=0,
            zmax=100,
            colorscale=[
                [0.00, "#d1e7dd"],
                [0.50, "#f1f3f5"],
                [1.00, "#f8d7da"],
            ],
            colorbar=dict(title="Percentile"),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y}: %{z:.1f}<extra></extra>",
        )
    )

    heat_fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=50, r=30, t=40, b=40),
    )

    st.caption(heat_caption)
    st.plotly_chart(heat_fig, use_container_width=True)

    st.subheader("Today versus five sessions ago")

    compare_df = latest_factor_table.copy()
    compare_df = compare_df.sort_values("Contribution pts", ascending=True)

    bar_fig = go.Figure()

    bar_fig.add_trace(
        go.Bar(
            x=compare_df["Contribution pts"],
            y=compare_df["Factor"],
            orientation="h",
            name="Contribution today",
            hovertemplate="%{y}<br>Contribution: %{x:.1f} pts<extra></extra>",
        )
    )

    bar_fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=70, r=30, t=40, b=40),
        xaxis_title="Composite points",
        yaxis_title="",
        showlegend=False,
    )

    st.plotly_chart(bar_fig, use_container_width=True)

    st.subheader("Underlying instrument RVOL snapshot")

    snap = []

    for ticker in sorted(set(ALL_TICKERS)):
        col = f"{ticker}_p"
        if col in rvol_pct.columns:
            asset_class = next(
                k
                for k, v in ASSET_CLASS_MAP.items()
                if ticker in v
            )

            last_raw_date = raw_panel[ticker].dropna().index.max() if ticker in raw_panel.columns else pd.NaT

            snap.append(
                {
                    "Ticker": ticker,
                    "Asset Class": asset_class,
                    "Last Raw Date": last_raw_date.date().isoformat() if pd.notna(last_raw_date) else "",
                    "RVOL Percentile": safe_float(rvol_pct.loc[latest_idx, col] * 100.0),
                    "RVOL": safe_float(rvol.loc[latest_idx, ticker]) if ticker in rvol.columns else np.nan,
                    "Latest Price": safe_float(panel.loc[latest_idx, ticker]) if ticker in panel.columns else np.nan,
                }
            )

    snap_df = pd.DataFrame(snap)

    if not snap_df.empty:
        snap_df = snap_df.sort_values(
            ["Asset Class", "RVOL Percentile"],
            ascending=[True, False],
        )
        snap_df["RVOL Percentile"] = pd.to_numeric(
            snap_df["RVOL Percentile"],
            errors="coerce",
        ).round(1)
        snap_df["RVOL"] = pd.to_numeric(
            snap_df["RVOL"],
            errors="coerce",
        ).round(3)
        snap_df["Latest Price"] = pd.to_numeric(
            snap_df["Latest Price"],
            errors="coerce",
        ).round(2)

        st.dataframe(
            snap_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No instrument RVOL snapshot is available.")

with tab_diagnostics:
    st.subheader("Fetch diagnostics")

    fetch_show = fetch_meta.copy()

    if not fetch_show.empty:
        fetch_show["Calendar Days Since Last"] = pd.to_numeric(
            fetch_show["Calendar Days Since Last"],
            errors="coerce",
        ).round(0)

        st.dataframe(
            fetch_show.sort_values(["Source", "Ticker"]),
            use_container_width=True,
            hide_index=True,
        )

        fallback_count = int((fetch_show["Source"] == "Local cache fallback").sum())
        missing_count = int((fetch_show["Source"] == "Missing").sum())

        if missing_count > 0:
            st.warning(f"{missing_count} ticker(s) are missing from both Yahoo and local cache.")

        if fallback_count > 0:
            st.warning(f"{fallback_count} ticker(s) used local cache fallback.")

    st.subheader("Factor freshness")

    latest = latest_idx

    diag = pd.DataFrame(
        {
            "Factor": ["Equities", "Credit", "Commodities", "FX", "Rates", "Breadth", "Dispersion"],
            "Raw Score": [
                safe_float(scores.loc[latest, "Equities_p"] * 100.0) if "Equities_p" in scores.columns else np.nan,
                safe_float(scores.loc[latest, "Credit_p"] * 100.0) if "Credit_p" in scores.columns else np.nan,
                safe_float(scores.loc[latest, "Commodities_p"] * 100.0) if "Commodities_p" in scores.columns else np.nan,
                safe_float(scores.loc[latest, "FX_p"] * 100.0) if "FX_p" in scores.columns else np.nan,
                safe_float(scores.loc[latest, "Rates_p"] * 100.0) if "Rates_p" in scores.columns else np.nan,
                safe_float(scores.loc[latest, "Breadth_p"] * 100.0) if "Breadth_p" in scores.columns else np.nan,
                safe_float(scores.loc[latest, "Dispersion_p"] * 100.0) if "Dispersion_p" in scores.columns else np.nan,
            ],
            "Active": [
                bool(masks.loc[latest, "Equities_m"]) if "Equities_m" in masks.columns else False,
                bool(masks.loc[latest, "Credit_m"]) if "Credit_m" in masks.columns else False,
                bool(masks.loc[latest, "Commodities_m"]) if "Commodities_m" in masks.columns else False,
                bool(masks.loc[latest, "FX_m"]) if "FX_m" in masks.columns else False,
                bool(masks.loc[latest, "Rates_m"]) if "Rates_m" in masks.columns else False,
                bool(masks.loc[latest, "Breadth_m"]) if "Breadth_m" in masks.columns else False,
                bool(masks.loc[latest, "Dispersion_m"]) if "Dispersion_m" in masks.columns else False,
            ],
            "Trading Sessions Since Raw Update": [
                age_map["Equities"].reindex(plot_idx).loc[latest] if "Equities" in age_map else np.nan,
                age_map["Credit"].reindex(plot_idx).loc[latest] if "Credit" in age_map else np.nan,
                age_map["Commodities"].reindex(plot_idx).loc[latest] if "Commodities" in age_map else np.nan,
                age_map["FX"].reindex(plot_idx).loc[latest] if "FX" in age_map else np.nan,
                age_map["Rates"].reindex(plot_idx).loc[latest] if "Rates" in age_map else np.nan,
                age_map["Breadth"].reindex(plot_idx).loc[latest] if "Breadth" in age_map else np.nan,
                age_map["Dispersion"].reindex(plot_idx).loc[latest] if "Dispersion" in age_map else np.nan,
            ],
        }
    )

    diag["Raw Score"] = pd.to_numeric(
        diag["Raw Score"],
        errors="coerce",
    ).round(1)

    diag["Trading Sessions Since Raw Update"] = pd.to_numeric(
        diag["Trading Sessions Since Raw Update"],
        errors="coerce",
    ).round(0)

    st.dataframe(
        diag,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Market context")

    context_rows = []

    for ticker, label in [
        ("SPX", "S&P 500"),
        ("VIX", "VIX"),
        ("DXY", "DXY"),
    ]:
        if ticker in panel.columns:
            s = panel[ticker].dropna()
            if not s.empty:
                context_rows.append(
                    {
                        "Series": label,
                        "Latest": safe_float(s.iloc[-1]),
                        "1D Chg %": safe_float(s.pct_change(fill_method=None).iloc[-1] * 100.0),
                        "5D Chg %": safe_float((s.iloc[-1] / s.iloc[-6] - 1.0) * 100.0) if len(s) > 5 else np.nan,
                        "21D Chg %": safe_float((s.iloc[-1] / s.iloc[-22] - 1.0) * 100.0) if len(s) > 21 else np.nan,
                    }
                )

    context_df = pd.DataFrame(context_rows)

    if not context_df.empty:
        for col in ["Latest", "1D Chg %", "5D Chg %", "21D Chg %"]:
            context_df[col] = pd.to_numeric(context_df[col], errors="coerce").round(2)

        st.dataframe(
            context_df,
            use_container_width=True,
            hide_index=True,
        )

with tab_download:
    st.subheader("Download composite data")

    export_idx = plot_idx

    export_panel_cols = [
        c
        for c in ["SPX", "VIX", "DXY", "DD_stress"] + sorted(set(ALL_TICKERS))
        if c in panel.columns
    ]

    export_filled_prices = panel.reindex(export_idx)[export_panel_cols].copy()
    export_filled_prices = export_filled_prices.rename(
        columns={c: f"{c}_filled" for c in export_filled_prices.columns}
    )

    export_raw_cols = [
        c
        for c in ["SPX", "VIX", "DXY"] + sorted(set(ALL_TICKERS))
        if c in raw_panel.columns
    ]

    export_raw_prices = raw_panel.reindex(export_idx)[export_raw_cols].copy()
    export_raw_prices = export_raw_prices.rename(
        columns={c: f"{c}_raw" for c in export_raw_prices.columns}
    )

    export_rvol = rvol.reindex(export_idx).rename(
        columns={c: f"{c}_rvol" for c in rvol.columns}
    )

    export_pct = rvol_pct.reindex(export_idx).rename(
        columns={c: f"{c}_pct" for c in rvol_pct.columns}
    )

    export_scores = (scores.reindex(export_idx) * 100.0).rename(
        columns={
            "Equities_p": "Equities_pct",
            "Credit_p": "Credit_pct",
            "Commodities_p": "Commodities_pct",
            "FX_p": "FX_pct",
            "Rates_p": "Rates_pct",
            "Breadth_p": "Breadth_pct",
            "Dispersion_p": "Dispersion_pct",
        }
    )

    export_masks = masks.reindex(export_idx).copy()

    export_contrib = contrib_panel.reindex(export_idx).copy()

    export_meta = pd.concat(
        [
            active_weight_s.reindex(export_idx),
            active_factors_s.reindex(export_idx),
            comp_s.reindex(export_idx),
            breadth_raw.reindex(export_idx).rename("raw_stress_breadth"),
            disp_raw.reindex(export_idx).rename("raw_factor_dispersion"),
        ],
        axis=1,
    )

    out = pd.concat(
        [
            export_raw_prices,
            export_filled_prices,
            export_rvol,
            export_pct,
            export_scores,
            export_masks,
            export_contrib,
            export_meta,
        ],
        axis=1,
    )

    out.index.name = "Date"

    st.download_button(
        "Download CSV",
        out.to_csv(),
        file_name="cross_asset_rvol_stress_composite.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "CSV includes raw prices, filled prices, RVOL, RVOL percentiles, factor scores, masks, contributions, breadth, dispersion, and composite metadata."
    )

st.caption("© 2026 AD Fund Management LP")
