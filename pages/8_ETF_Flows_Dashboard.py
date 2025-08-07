import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
import concurrent.futures
import time
from typing import Tuple, Optional

st.set_page_config(page_title="ETF Flows Dashboard", layout="wide")

# --------------------------- SIDEBAR ---------------------------
st.sidebar.title("ETF Flows")
st.sidebar.markdown("""
A dashboard of **thematic and global ETF flows**.

- **Method options**
  - **SO-based (if available):** sum of daily changes in shares outstanding multiplied by close.
  - **Chaikin Money Flow ($ proxy):** money flow multiplier × volume × typical price, summed over the period.
  - **Dollar OBV proxy:** sign of daily return × volume × close, summed over the period.

These are **proxies**, not official flow figures.
""")

lookback_dict = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "12 Months": 365,
    "YTD": (datetime.now() - datetime(datetime.now().year, 1, 1)).days
}
period_label = st.sidebar.radio("Select Lookback Period", list(lookback_dict.keys()), index=1)
period_days = int(lookback_dict[period_label])

method = st.sidebar.radio(
    "Method",
    ["SO-based (if available)", "Chaikin Money Flow ($ proxy)", "Dollar OBV proxy"],
    index=1
)

etf_info = {
    "MAGS": ("Mag 7", "Magnificent 7 stocks ETF"),
    "SMH": ("Semiconductors", "Semiconductor stocks (VanEck)"),
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
etf_tickers = list(etf_info.keys())

# --------------------------- HELPERS ---------------------------
def fmt_compact_cur(x) -> str:
    if x is None or pd.isna(x):
        return ""
    ax = abs(x)
    if ax >= 1e9:
        return f"${x/1e9:,.0f}B"
    if ax >= 1e6:
        return f"${x/1e6:,.0f}M"
    if ax >= 1e3:
        return f"${x/1e3:,.0f}K"
    return f"${x:,.0f}"

def axis_fmt(x, _pos=None) -> str:
    ax = abs(x)
    if ax >= 1e9:
        return f"${x/1e9:,.0f}B"
    if ax >= 1e6:
        return f"${x/1e6:,.0f}M"
    if ax >= 1e3:
        return f"${x/1e3:,.0f}K"
    return f"${x:,.0f}"

def retry(n=3, delay=0.4):
    def deco(fn):
        def wrap(*args, **kwargs):
            last = None
            for i in range(n):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last = e
                    time.sleep(delay * (i + 1))
            raise last
        return wrap
    return deco

@st.cache_data(show_spinner=False, ttl=300)
@retry(n=3, delay=0.5)
def fetch_hist(ticker: str, period_days: int) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    hist = t.history(period=f"{period_days+10}d", interval="1d", auto_adjust=False)
    if hist is None or hist.empty:
        return pd.DataFrame()
    hist = hist.dropna()
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in cols:
        if c in hist.columns:
            hist[c] = pd.to_numeric(hist[c], errors="coerce")
    hist = hist.dropna(subset=["Close", "Volume", "High", "Low"])
    return hist

@st.cache_data(show_spinner=False, ttl=300)
def fetch_shares_series(ticker: str, hist_index: pd.DatetimeIndex) -> Optional[pd.Series]:
    try:
        t = yf.Ticker(ticker)
        so = None
        try:
            so = t.get_shares_full()
        except Exception:
            pass
        if so is None or (hasattr(so, "empty") and so.empty):
            try:
                so = t.get_shares()
            except Exception:
                so = None
        if so is None or (hasattr(so, "empty") and so.empty):
            return None
        so.index = pd.to_datetime(so.index).tz_localize(None)
        so = pd.to_numeric(so, errors="coerce").dropna().sort_index()
        if so.empty:
            return None
        so = so.loc[hist_index.min():hist_index.max()]
        if len(so) < 2:
            return None
        so_daily = so.reindex(hist_index, method="ffill")
        return so_daily
    except Exception:
        return None

def compute_so_flow(hist: pd.DataFrame, so_daily: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    if hist.empty or so_daily is None or so_daily.empty:
        return None, None
    close = hist["Close"]
    dso = so_daily.diff().fillna(0.0)
    flow = float((dso * close).sum())
    aum = float(so_daily.iloc[-1] * close.iloc[-1]) if pd.notna(so_daily.iloc[-1]) else None
    return flow, aum

def compute_cmf_dollar_proxy(hist: pd.DataFrame) -> float:
    # Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    hl_range = (hist["High"] - hist["Low"]).replace(0, np.nan)
    mfm = ((hist["Close"] - hist["Low"]) - (hist["High"] - hist["Close"])) / hl_range
    mfm = mfm.fillna(0.0)
    typical_price = (hist["High"] + hist["Low"] + hist["Close"]) / 3.0
    dollar_mf = mfm * hist["Volume"] * typical_price
    return float(dollar_mf.sum())

def compute_dollar_obv_proxy(hist: pd.DataFrame) -> float:
    ret_sign = np.sign(hist["Close"].diff().fillna(0.0))
    dollar_obv = ret_sign * hist["Volume"] * hist["Close"]
    return float(dollar_obv.sum())

# --------------------------- DATA AGG ---------------------------
@st.cache_data(show_spinner=True, ttl=300)
def build_table(tickers, period_days: int, method: str) -> pd.DataFrame:
    rows = []

    def worker(tk):
        hist = fetch_hist(tk, period_days)
        if hist.empty:
            return tk, np.nan, np.nan, np.nan
        if method == "SO-based (if available)":
            so_daily = fetch_shares_series(tk, hist.index)
            flow, aum = compute_so_flow(hist, so_daily)
            return tk, flow if flow is not None else np.nan, np.nan, aum if aum is not None else np.nan
        elif method == "Chaikin Money Flow ($ proxy)":
            proxy = compute_cmf_dollar_proxy(hist)
            return tk, proxy, np.nan, np.nan
        else:  # Dollar OBV proxy
            proxy = compute_dollar_obv_proxy(hist)
            return tk, proxy, np.nan, np.nan

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(worker, tickers))

    for tk, flow, flow_pct, aum in results:
        cat, desc = etf_info[tk]
        rows.append({
            "Ticker": tk,
            "Category": cat,
            "Flow ($)": float(flow) if pd.notna(flow) else np.nan,
            "Flow (%)": float(flow_pct) if pd.notna(flow_pct) else np.nan,
            "AUM ($)": float(aum) if pd.notna(aum) else np.nan,
            "Description": desc
        })
    df = pd.DataFrame(rows)
    df["Label"] = [f"{etf_info[t][0]} ({t})" for t in df["Ticker"]]
    return df

# --------------------------- MAIN ---------------------------
st.title("ETF Flows Dashboard")
st.caption(f"Flows are proxies, not official. Period: {period_label}. Method: {method}")

df = build_table(etf_tickers, period_days, method)
chart_df = df.sort_values("Flow ($)", ascending=False).copy()

# Guard when no values
max_val = pd.to_numeric(chart_df["Flow ($)"], errors="coerce").abs().max()
if pd.isna(max_val) or max_val == 0:
    if method == "SO-based (if available)":
        st.info("No valid SO-based data for the selected period. Switch to a proxy method for a robust approximation.")
    else:
        st.info("No valid values computed. Try a different period.")
else:
    flows_series = pd.to_numeric(chart_df["Flow ($)"], errors="coerce")
    colors = []
    for x in flows_series:
        if pd.isna(x) or abs(x) < 1e-9:
            colors.append("gray")
        elif x > 0:
            colors.append("green")
        else:
            colors.append("red")

    fig, ax = plt.subplots(figsize=(15, max(6, len(chart_df) * 0.42)))
    bars = ax.barh(chart_df["Label"], flows_series.fillna(0.0), color=colors, alpha=0.85)

    ax.set_xlabel("Estimated Flow ($)" if method == "SO-based (if available)" else f"{method}")
    ax.set_title(f"ETF {('Proxy Flows' if method == 'SO-based (if available)' else 'Demand Proxies')} - {period_label}")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(axis_fmt))

    # Axis limits
    min_flow = flows_series.min(skipna=True)
    max_flow = flows_series.max(skipna=True)
    abs_max = max(abs(min_flow if pd.notna(min_flow) else 0.0), abs(max_flow if pd.notna(max_flow) else 0.0))
    buffer = 0.15 * abs_max if abs_max > 0 else 1.0
    left_lim = -abs_max - buffer if pd.notna(min_flow) and min_flow < 0 else 0 - buffer * 0.15
    right_lim = abs_max + buffer
    ax.set_xlim([left_lim, right_lim])

    # Annotations
    x_range = right_lim - left_lim
    for bar, val in zip(bars, flows_series):
        if pd.notna(val):
            label = fmt_compact_cur(val)
            x_text = bar.get_width()
            align = "left" if val > 0 else "right" if val < 0 else "center"
            x_offset = 0.01 * x_range if val > 0 else -0.01 * x_range if val < 0 else 0.0
            ax.text(
                x_text + x_offset,
                bar.get_y() + bar.get_height() / 2,
                label if label else "$0",
                va="center",
                ha=align,
                fontsize=10,
                color="black",
                clip_on=True
            )

    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("*Green indicates positive, red negative, gray zero or missing*")

# --------------------------- TOP LISTS ---------------------------
st.markdown("#### Top Inflows and Outflows")
valid = df.dropna(subset=["Flow ($)"]).copy()

if valid.empty:
    if method == "SO-based (if available)":
        st.write("No ETFs with computable SO-based flows in the selected period.")
    else:
        st.write("No ETFs with computable values in the selected period.")
else:
    top_in = valid.nlargest(3, "Flow ($)")[["Label", "Flow ($)"]].copy()
    top_out = valid.nsmallest(3, "Flow ($)")[["Label", "Flow ($)"]].copy()
    top_in["Value"] = top_in["Flow ($)"].apply(fmt_compact_cur)
    top_out["Value"] = top_out["Flow ($)"].apply(fmt_compact_cur)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Top Inflows**" if method == "SO-based (if available)" else "**Top Positive**")
        st.table(top_in[["Value"]].set_index(top_in["Label"]))
    with col2:
        st.write("**Top Outflows**" if method == "SO-based (if available)" else "**Top Negative**")
        st.table(top_out[["Value"]].set_index(top_out["Label"]))

# --------------------------- STATUS ---------------------------
if method == "SO-based (if available)":
    if df["Flow ($)"].isna().any():
        st.warning("Some ETFs are missing SO-based flows due to unavailable shares outstanding history.")
    else:
        st.success("All SO-based flow proxies computed using available shares outstanding history.")
else:
    st.info("Proxy methods approximate money movement using price and volume. They are not official creation or redemption figures.")

st.caption("© 2025 AD Fund Management LP")
