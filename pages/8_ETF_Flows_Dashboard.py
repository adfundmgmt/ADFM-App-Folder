import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime, date
import pytz
import time
from typing import Dict, Tuple, List

st.set_page_config(page_title="ETF Net Flows (Estimated)", layout="wide")

# --------------------------- SIDEBAR ---------------------------
st.sidebar.title("ETF Net Flows (Estimated)")
st.sidebar.markdown("""
**Methodology priority**
1) True flows: ΔShares Outstanding × NAV (uses Close as proxy for NAV).
2) If historical shares are unavailable: show CMF turnover proxy as a fallback and label it.

Notes:
- yfinance historical shares (`get_shares_full`) is not available for every ETF.
- Close is used as a proxy for daily NAV.
""")

# Timezone-aware today for YTD and windows
TZ = pytz.timezone("US/Eastern")
now_et = datetime.now(TZ)

def as_naive_ts(dt: datetime) -> pd.Timestamp:
    """Return a tz-naive pandas Timestamp from a tz-aware datetime."""
    return pd.Timestamp(dt).tz_convert(None)

def ytd_days() -> int:
    start_ytd = TZ.localize(datetime(now_et.year, 1, 1))
    return (now_et - start_ytd).days

lookback_dict = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "12 Months": 365,
    "YTD": ytd_days(),
}
period_label = st.sidebar.radio("Select Lookback Period", list(lookback_dict.keys()), index=0)
period_days = int(lookback_dict[period_label])

etf_info: Dict[str, Tuple[str, str]] = {
    "MAGS": ("Mag 7", "Magnificent 7 stocks ETF"),
    "SMH": ("Semiconductors", "VanEck Semiconductor ETF"),
    "BOTZ": ("Robotics/AI", "Global robotics and AI leaders"),
    "ICLN": ("Clean Energy", "Global clean energy stocks"),
    "URNM": ("Uranium", "Uranium miners (Sprott)"),
    "ARKK": ("Innovation", "Disruptive growth stocks (ARK)"),
    "KWEB": ("China Internet", "China internet leaders (KraneShares)"),
    "FXI": ("China Large-Cap", "China mega-cap stocks"),
    "EWZ": ("Brazil", "Brazil large-cap equities"),
    "EEM": ("Emerging Markets", "EM equities (MSCI)"),
    "VWO": ("Emerging Markets", "EM equities (Vanguard)"),
    "VGK": ("Europe Large-Cap", "Developed Europe stocks (Vanguard)"),
    "FEZ": ("Eurozone", "Euro STOXX 50 ETF"),
    "ILF": ("Latin America", "Latin America 40 ETF"),
    "ARGT": ("Argentina", "Global X MSCI Argentina ETF"),
    "GLD": ("Gold", "SPDR Gold Trust ETF"),
    "SLV": ("Silver", "iShares Silver Trust ETF"),
    "DBC": ("Commodities", "Invesco DB Commodity Index ETF"),
    "HEDJ": ("Hedged Europe", "WisdomTree Europe Hedged Equity ETF"),
    "USMV": ("US Min Volatility", "iShares MSCI USA Min Volatility ETF"),
    "COWZ": ("US Free Cash Flow", "Pacer US Cash Cows 100 ETF"),
    "BITO": ("BTC Futures", "Bitcoin futures ETF"),
    "IBIT": ("Spot BTC", "BlackRock spot Bitcoin ETF"),
    "BIL": ("1-3mo T-Bills", "1-3 month U.S. Treasury bills"),
    "TLT": ("20+yr Treasuries", "20+ year U.S. Treasuries"),
    "SHV": ("0-1yr T-Bills", "Short-term Treasury bonds"),
}
etf_tickers: List[str] = list(etf_info.keys())

# --------------------------- HELPERS ---------------------------
def fmt_compact_cur(x) -> str:
    if x is None or pd.isna(x):
        return ""
    ax = abs(float(x))
    if ax >= 1e12: return f"${x/1e12:,.0f}T"
    if ax >= 1e9:  return f"${x/1e9:,.0f}B"
    if ax >= 1e6:  return f"${x/1e6:,.0f}M"
    if ax >= 1e3:  return f"${x/1e3:,.0f}K"
    return f"${x:,.0f}"

def axis_fmt(x, _pos=None) -> str:
    return fmt_compact_cur(x)

def retry(n=3, delay=0.6):
    def deco(fn):
        def wrap(*args, **kwargs):
            last = None
            for i in range(n):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last = e
                    time.sleep(delay * (i + 1))
            if last: raise last
        return wrap
    return deco

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    need = ["Open", "High", "Low", "Close", "Volume"]
    for c in need:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.index = pd.to_datetime(df.index, errors="coerce").tz_localize(None)
    return df.dropna(subset=["Close"])

# --------------------------- DATA ---------------------------
def _calc_start_date(days: int) -> date:
    padding = 7
    return (now_et - pd.Timedelta(days=days + padding)).date()

@retry()
@st.cache_data(show_spinner=False, ttl=300)
def fetch_prices(tickers: Tuple[str, ...], start_date: date) -> Dict[str, pd.DataFrame]:
    data = yf.download(
        tickers=list(tickers),
        start=start_date,
        end=now_et.date() + pd.Timedelta(days=1),
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    out = {}
    for tk in tickers:
        try:
            df = data[tk].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
        except Exception:
            df = pd.DataFrame()
        out[tk] = _normalize_ohlcv(df) if df is not None else pd.DataFrame()
    return out

@retry()
@st.cache_data(show_spinner=False, ttl=1800)
def fetch_shares_series(ticker: str, start_date: date) -> pd.Series:
    t = yf.Ticker(ticker)
    try:
        s = t.get_shares_full(start=start_date)
    except Exception:
        s = None
    if s is None or (isinstance(s, pd.DataFrame) and s.empty):
        return pd.Series(dtype="float64")
    if isinstance(s, pd.DataFrame):
        if s.shape[1] >= 1:
            s = s.iloc[:, 0]
        else:
            return pd.Series(dtype="float64")
    s = pd.to_numeric(s, errors="coerce")
    s.index = pd.to_datetime(s.index, errors="coerce").tz_localize(None)
    s = s.dropna()
    return s

def compute_daily_flows(close: pd.Series, shares: pd.Series) -> pd.Series:
    if close.empty or shares.empty:
        return pd.Series(dtype="float64")
    idx = pd.DatetimeIndex(close.index)
    sh_daily = shares.sort_index().resample("B").ffill().reindex(idx).ffill()
    delta_shares = sh_daily.diff().fillna(0.0)
    flows = delta_shares * close.astype(float)
    return flows

def compute_cmf_turnover_proxy(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    hl = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl
    mfm = mfm.fillna(0.0)
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return float((mfm * typical * vol).sum())

# --------------------------- BUILD TABLE ---------------------------
@st.cache_data(show_spinner=True, ttl=300)
def build_table(tickers: List[str], period_days: int) -> pd.DataFrame:
    start_date = _calc_start_date(period_days)
    price_map = fetch_prices(tuple(tickers), start_date)

    rows = []
    for tk in tickers:
        px = price_map.get(tk, pd.DataFrame())
        if not px.empty:
            cutoff = as_naive_ts(now_et) - pd.Timedelta(days=period_days)
            px = px[px.index >= cutoff]

        flow_usd_sum = np.nan
        method = "flows"
        if not px.empty:
            shares = fetch_shares_series(tk, start_date)
            daily_flows = compute_daily_flows(px["Close"], shares) if not shares.empty else pd.Series(dtype="float64")
            if not daily_flows.empty and daily_flows.abs().sum() > 0:
                flow_usd_sum = float(daily_flows.sum())
            else:
                method = "cmf_proxy"
                flow_usd_sum = compute_cmf_turnover_proxy(px)
        cat, desc = etf_info.get(tk, ("", ""))
        rows.append({
            "Ticker": tk,
            "Category": cat,
            "Flow ($)": flow_usd_sum,
            "Method": method,
            "Description": desc
        })
    df = pd.DataFrame(rows)
    df["Label"] = [f"{etf_info[t][0]} ({t})" for t in df["Ticker"]]
    return df

# --------------------------- MAIN ---------------------------
st.title("ETF Net Flows")
st.caption(f"Estimated net creations/redemptions over: {period_label}. "
           f"When shares history is unavailable, shows CMF turnover proxy and labels it as such.")

df = build_table(etf_tickers, period_days)
chart_df = df.sort_values("Flow ($)", ascending=False).copy()

vals = pd.to_numeric(chart_df["Flow ($)"], errors="coerce").fillna(0.0)
abs_max = float(vals.abs().max()) if len(vals) else 0.0

if not len(vals) or abs_max == 0.0 or np.isnan(abs_max):
    st.info("No valid values computed. Try a different period or different ETFs.")
else:
    colors = ["green" if v > 0 else "red" if v < 0 else "gray" for v in vals]

    fig, ax = plt.subplots(figsize=(15, max(6, len(chart_df) * 0.42)))
    bars = ax.barh(chart_df["Label"], vals, color=colors, alpha=0.9)

    ax.set_xlabel("Estimated Net Flow ($)")
    ax.set_title(f"ETF Net Flows - {period_label}")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(axis_fmt))

    buffer = 0.15 * abs_max if abs_max > 0 else 1.0
    ax.set_xlim([-abs_max - buffer, abs_max + buffer])

    for bar, raw in zip(bars, vals):
        label = fmt_compact_cur(raw) if pd.notna(raw) else "$0"
        x = bar.get_width()
        if raw > 0:
            x_text, ha = x + 0.01 * abs_max, "left"
        elif raw < 0:
            x_text, ha = x - 0.01 * abs_max, "right"
        else:
            x_text, ha = 0, "center"
        ax.text(
            x_text, bar.get_y() + bar.get_height() / 2, label,
            va="center", ha=ha, fontsize=10, color="black", clip_on=True
        )

    plt.tight_layout()
    st.pyplot(fig)

# --------------------------- TOP LISTS ---------------------------
st.markdown("#### Top Inflows and Outflows")
valid = df.dropna(subset=["Flow ($)"]).copy()

if valid.empty:
    st.write("No ETFs with computable values in the selected period.")
else:
    valid["Value"] = valid["Flow ($)"].apply(fmt_compact_cur)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Top Inflows**")
        st.table(valid[valid["Flow ($)"] > 0].nlargest(3, "Flow ($)").set_index("Label")[["Value", "Method"]])
    with col2:
        st.write("**Top Outflows**")
        st.table(valid[valid["Flow ($)"] < 0].nsmallest(3, "Flow ($)").set_index("Label")[["Value", "Method"]])

st.caption("© 2025 AD Fund Management LP")
