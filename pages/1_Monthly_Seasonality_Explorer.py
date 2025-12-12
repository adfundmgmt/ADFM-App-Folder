import datetime as dt
import io
import time
import warnings
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter, MaxNLocator
from matplotlib import gridspec

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#222222",
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "font.size": 10,
})

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

try:
    from pandas_datareader import data as pdr
except ImportError:
    pdr = None

FALLBACK_MAP = {"^GSPC": "SP500", "^DJI": "DJIA", "^IXIC": "NASDAQCOM"}
MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ============================== STREAMLIT UI ============================== #
st.set_page_config(page_title="Seasonality Dashboard", layout="wide")
st.markdown("## Monthly Seasonality Explorer")
st.caption("Seasonality decomposed by intra-month contribution and trading-day path")

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        """
        This tool decomposes monthly returns into first-half and second-half contributions
        and visualizes the average intra-month trading-day path.

        **Design principles**
        • Contribution clarity over raw averages  
        • Signal first, annotation second  
        • All curves anchored to prior month-end  
        • No lookahead, no smoothing distortion
        """
    )

# ============================== DATA HELPERS ============================== #
def _yf_download(symbol: str, start: str, end: str, retries: int = 3) -> Optional[pd.Series]:
    for n in range(retries):
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if not df.empty and "Close" in df:
            return df["Close"]
        time.sleep(1.5 * (n + 1))
    return None

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices(symbol: str, start: str, end: str) -> Optional[pd.Series]:
    symbol = symbol.upper().strip()
    end = min(pd.Timestamp(end), pd.Timestamp.today()).strftime("%Y-%m-%d")
    start_pad = (pd.Timestamp(start) - pd.DateOffset(days=45)).strftime("%Y-%m-%d")

    px = _yf_download(symbol, start_pad, end)
    if px is not None:
        return px

    if symbol == "SPY":
        px = _yf_download("^GSPC", start_pad, end)
        if px is not None:
            return px

    if pdr and symbol in FALLBACK_MAP:
        try:
            df = pdr.DataReader(FALLBACK_MAP[symbol], "fred", start_pad, end)
            if not df.empty:
                return df.iloc[:, 0].rename("Close")
        except Exception:
            pass

    return None

# ============================== SEASONALITY CORE ============================== #
def _intra_month_halves(prices: pd.Series) -> pd.DataFrame:
    rows = []
    months = pd.period_range(prices.index.min(), prices.index.max(), freq="M")

    for m in months:
        cur = prices[prices.index.to_period("M") == m]
        prev = prices[prices.index.to_period("M") == (m - 1)]
        if len(cur) < 3 or prev.empty:
            continue

        prev_eom = prev.iloc[-1]
        mid = cur.iloc[len(cur) // 2 - 1]
        last = cur.iloc[-1]

        total = (last / prev_eom - 1) * 100
        h1 = (mid / prev_eom - 1) * 100
        h2 = total - h1

        rows.append({
            "month": m.month,
            "year": m.year,
            "mean_h1": h1,
            "mean_h2": h2,
            "mean_total": total,
        })

    return pd.DataFrame(rows)

def seasonal_stats(prices: pd.Series, start_year: int, end_year: int) -> pd.DataFrame:
    df = _intra_month_halves(prices)
    df = df[(df.year >= start_year) & (df.year <= end_year)]

    stats = df.groupby("month").agg(
        mean_h1=("mean_h1", "mean"),
        mean_h2=("mean_h2", "mean"),
        mean_total=("mean_total", "mean"),
        min_ret=("mean_total", "min"),
        max_ret=("mean_total", "max"),
        hit_rate=("mean_total", lambda x: (x > 0).mean() * 100),
        years=("year", "nunique"),
    )

    stats["label"] = [MONTH_LABELS[m-1] for m in stats.index]
    return stats

# ============================== MAIN BAR CHART ============================== #
def plot_seasonality(stats: pd.DataFrame, title: str) -> io.BytesIO:
    fig = plt.figure(figsize=(13, 8), dpi=200)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.5, 1], hspace=0.22)
    ax = fig.add_subplot(gs[0])

    x = np.arange(len(stats))
    h1 = stats.mean_h1.values
    h2 = stats.mean_h2.values
    total = stats.mean_total.values

    ax.bar(x, h1, width=0.75, color="#4c9f70", label="First half")
    ax.bar(x, h2, width=0.75, bottom=h1, color="#a3c9a8", hatch="//", label="Second half")

    ax.errorbar(
        x,
        total,
        yerr=[total - stats.min_ret, stats.max_ret - total],
        fmt="none",
        ecolor="#666666",
        capsize=5,
        lw=1,
    )

    ax.axhline(0, color="#999999", lw=1)
    ax.set_xticks(x, stats.label)
    ax.set_ylabel("Average return (%)", weight="bold")
    ax.yaxis.set_major_formatter(PercentFormatter())

    ax2 = ax.twinx()
    ax2.scatter(x, stats.hit_rate, s=55, color="black", zorder=5)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Hit rate (%)", weight="bold")

    ax.set_title(title, fontsize=16, weight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Table
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")

    tbl = ax_tbl.table(
        cellText=[
            [f"{v:+.1f}%" for v in h1],
            [f"{v:+.1f}%" for v in h2],
            [f"{v:+.1f}%" for v in total],
        ],
        rowLabels=["1H", "2H", "Total"],
        colLabels=stats.label.tolist(),
        cellLoc="center",
        loc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf

# ============================== CONTROLS ============================== #
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    symbol = st.text_input("Ticker", "^GSPC")
with c2:
    start_year = st.number_input("Start year", 1900, dt.datetime.today().year, 2020)
with c3:
    end_year = st.number_input("End year", start_year, dt.datetime.today().year, dt.datetime.today().year)

prices = fetch_prices(symbol, f"{start_year}-01-01", f"{end_year}-12-31")
if prices is None or prices.empty:
    st.error("No data available.")
    st.stop()

stats = seasonal_stats(prices, start_year, end_year)

best = stats.loc[stats.mean_total.idxmax()]
worst = stats.loc[stats.mean_total.idxmin()]

st.markdown(
    f"""
    **Best month:** {best.label} ({best.mean_total:.2f}% avg)  
    **Worst month:** {worst.label} ({worst.mean_total:.2f}% avg)
    """
)

buf = plot_seasonality(stats, f"{symbol} Seasonality {start_year}–{end_year}")
st.image(buf, use_container_width=True)

st.caption("© 2025 AD Fund Management LP")
