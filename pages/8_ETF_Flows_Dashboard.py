import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
import concurrent.futures
import time

st.set_page_config(page_title="ETF Demand Proxies", layout="wide")

# --------------------------- SIDEBAR ---------------------------
st.sidebar.title("ETF Demand Proxies")
st.sidebar.markdown("""
**Chaikin Money Flow ($ proxy) only.**

Proxy formula per day:
- Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
- Dollar Flow Proxy = Multiplier × Volume × Typical Price
- We sum Dollar Flow Proxy over the selected period.

Bars show **absolute** magnitude to the right. Color shows sign.
""")

lookback_dict = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "12 Months": 365,
    "YTD": (datetime.now() - datetime(datetime.now().year, 1, 1)).days
}
period_label = st.sidebar.radio("Select Lookback Period", list(lookback_dict.keys()), index=0)
period_days = int(lookback_dict[period_label])

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
    # axis is absolute values
    ax = abs(x)
    if ax >= 1e9:
        return f"${ax/1e9:,.0f}B"
    if ax >= 1e6:
        return f"${ax/1e6:,.0f}M"
    if ax >= 1e3:
        return f"${ax/1e3:,.0f}K"
    return f"${ax:,.0f}"

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

def compute_cmf_dollar_proxy(hist: pd.DataFrame) -> float:
    # Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    hl_range = (hist["High"] - hist["Low"]).replace(0, np.nan)
    mfm = ((hist["Close"] - hist["Low"]) - (hist["High"] - hist["Close"])) / hl_range
    mfm = mfm.fillna(0.0)
    typical_price = (hist["High"] + hist["Low"] + hist["Close"]) / 3.0
    dollar_mf = mfm * hist["Volume"] * typical_price
    return float(dollar_mf.sum())

# --------------------------- DATA ---------------------------
@st.cache_data(show_spinner=True, ttl=300)
def build_table(tickers, period_days: int) -> pd.DataFrame:
    rows = []

    def worker(tk):
        hist = fetch_hist(tk, period_days)
        if hist.empty:
            return tk, np.nan
        proxy = compute_cmf_dollar_proxy(hist)
        return tk, proxy

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(worker, tickers))

    for tk, value in results:
        cat, desc = etf_info[tk]
        rows.append({
            "Ticker": tk,
            "Category": cat,
            "CMF Proxy ($)": float(value) if pd.notna(value) else np.nan,
            "Description": desc
        })
    df = pd.DataFrame(rows)
    df["Label"] = [f"{etf_info[t][0]} ({t})" for t in df["Ticker"]]
    # Add sign and absolute magnitude for scaling and sorting
    df["Sign"] = np.sign(df["CMF Proxy ($)"]).replace({-1.0: -1, 0.0: 0, 1.0: 1})
    df["Abs Proxy ($)"] = df["CMF Proxy ($)"].abs()
    return df

# --------------------------- MAIN ---------------------------
st.title("ETF Demand Proxies")
st.caption(f"Chaikin Money Flow ($ proxy). Period: {period_label}. Bars show absolute magnitude. Color shows sign.")

df = build_table(etf_tickers, period_days)
chart_df = df.sort_values("Abs Proxy ($)", ascending=False).copy()

# Guard when no values
max_val = pd.to_numeric(chart_df["Abs Proxy ($)"], errors="coerce").max()
if pd.isna(max_val) or max_val == 0:
    st.info("No valid values computed. Try a different period.")
else:
   # Replace this part in the chart section
values_signed = pd.to_numeric(chart_df["CMF Proxy ($)"], errors="coerce").fillna(0.0)
colors = ["green" if v > 0 else "red" if v < 0 else "gray" for v in values_signed]

fig, ax = plt.subplots(figsize=(15, max(6, len(chart_df) * 0.42)))
bars = ax.barh(chart_df["Label"], values_signed, color=colors, alpha=0.85)

ax.set_xlabel("Chaikin Money Flow ($ proxy)")
ax.set_title(f"ETF Demand Proxies - {period_label}")
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mticker.FuncFormatter(axis_fmt))

# Axis limits with buffer
abs_max = values_signed.abs().max()
buffer = 0.15 * abs_max if abs_max > 0 else 1.0
ax.set_xlim([-abs_max - buffer, abs_max + buffer])

# Annotate: show signed label next to each bar
x_range = (abs_max + buffer) * 2
for bar, raw in zip(bars, values_signed):
    label = fmt_compact_cur(raw) if pd.notna(raw) else ""
    x_text = bar.get_width()
    align = "left" if raw > 0 else "right" if raw < 0 else "center"
    x_offset = 0.01 * abs_max if raw > 0 else -0.01 * abs_max if raw < 0 else 0.0
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


# --------------------------- TOP LISTS ---------------------------
st.markdown("#### Top Positive and Negative (by absolute magnitude)")
valid = df.dropna(subset=["CMF Proxy ($)"]).copy()

if valid.empty:
    st.write("No ETFs with computable values in the selected period.")
else:
    valid["Label"] = [f"{etf_info[t][0]} ({t})" for t in valid["Ticker"]]
    top_pos = valid[valid["CMF Proxy ($)"] > 0].nlargest(3, "Abs Proxy ($)")[["Label", "CMF Proxy ($)"]].copy()
    top_neg = valid[valid["CMF Proxy ($)"] < 0].nlargest(3, "Abs Proxy ($)")[["Label", "CMF Proxy ($)"]].copy()
    top_pos["Value"] = top_pos["CMF Proxy ($)"].apply(fmt_compact_cur)
    top_neg["Value"] = top_neg["CMF Proxy ($)"].apply(fmt_compact_cur)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Top Positive**")
        st.table(top_pos[["Value"]].set_index(top_pos["Label"]))
    with col2:
        st.write("**Top Negative**")
        st.table(top_neg[["Value"]].set_index(top_neg["Label"]))

st.caption("© 2025 AD Fund Management LP")
