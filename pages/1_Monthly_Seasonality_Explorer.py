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

plt.style.use("default")
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ================= Streamlit ================= #
st.set_page_config(page_title="Seasonality Dashboard", layout="wide")
st.title("Monthly Seasonality Explorer")

# ================= Data ================= #
def _safe_close(df) -> Optional[pd.Series]:
    if df is None or len(df) == 0:
        return None

    if isinstance(df, pd.Series):
        s = df.dropna()
        return s.rename("Close") if not s.empty else None

    if isinstance(df, pd.DataFrame):
        if "Close" not in df.columns:
            return None
        c = df["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        c = c.dropna()
        return c.rename("Close") if not c.empty else None

    return None

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices(symbol: str, start: str, end: str) -> Optional[pd.Series]:
    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    return _safe_close(df)

# ================= Monthly stats ================= #
def _intra_month_halves(prices: pd.Series) -> pd.DataFrame:
    rows = []
    months = pd.period_range(prices.index.min(), prices.index.max(), freq="M")

    for m in months:
        m_px = prices[prices.index.to_period("M") == m]
        if len(m_px) < 3:
            continue

        prev_px = prices[prices.index.to_period("M") == (m - 1)]
        if prev_px.empty:
            continue

        prev_eom = float(prev_px.iloc[-1])
        n = len(m_px)
        mid = int(np.ceil(n / 2)) - 1

        total = (m_px.iloc[-1] / prev_eom - 1) * 100
        h1 = (m_px.iloc[mid] / prev_eom - 1) * 100
        h2 = total - h1

        rows.append(
            dict(
                year=m.year,
                month=m.month,
                total_ret=total,
                h1_ret=h1,
                h2_ret=h2,
            )
        )

    return pd.DataFrame(rows)

def seasonal_stats(prices: pd.Series, start_year: int, end_year: int) -> pd.DataFrame:
    df = _intra_month_halves(prices)
    df = df[(df.year >= start_year) & (df.year <= end_year)]

    g = df.groupby("month")
    out = pd.DataFrame(index=range(1,13))
    out["label"] = MONTH_LABELS
    out["mean_h1"] = g.h1_ret.mean()
    out["mean_h2"] = g.h2_ret.mean()
    out["mean_total"] = g.total_ret.mean()
    out["min_ret"] = g.total_ret.min()
    out["max_ret"] = g.total_ret.max()
    out["hit_rate"] = g.total_ret.apply(lambda x: (x > 0).mean() * 100)
    return out

# ================= Intra-month paths (FIXED) ================= #
def _month_paths_rebased(
    prices: pd.Series, month: int, start_year: int, end_year: int
) -> Tuple[pd.DataFrame, pd.Series]:

    px = prices[(prices.index.year >= start_year) & (prices.index.year <= end_year)]
    paths = {}

    for y in sorted(px.index.year.unique()):
        m = px[(px.index.year == y) & (px.index.month == month)]
        if len(m) < 3:
            continue

        first = float(m.iloc[0])
        norm = (m / first - 1) * 100
        norm.index = pd.RangeIndex(1, len(norm)+1)
        paths[y] = norm

    if not paths:
        return pd.DataFrame(), pd.Series(dtype=float)

    max_len = max(len(v) for v in paths.values())
    df = pd.DataFrame(index=pd.RangeIndex(1, max_len+1))
    for y, s in paths.items():
        df[y] = s

    return df, df.mean(axis=1)

def _trading_day_ordinal(month_index: pd.DatetimeIndex, when: pd.Timestamp) -> int:
    idx = month_index.sort_values()
    d = pd.Timestamp(when).normalize()
    ord_ = int(np.searchsorted(idx.values, np.datetime64(d), side="right"))
    return max(1, min(ord_, len(idx)))

# ================= Intra-month plot ================= #
def plot_intra_month_curve(prices, month, start_year, end_year, symbol):
    df, avg = _month_paths_rebased(prices, month, start_year, end_year)

    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=200)

    if avg.empty:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
        ax.axis("off")
    else:
        std = df.std(axis=1)
        ax.axhline(0, color="gray", linestyle=":")
        ax.fill_between(avg.index, avg - std, avg + std, alpha=0.25, label="±1σ")
        ax.plot(avg.index, avg, color="black", linewidth=2.6, label="Avg path")

        today = pd.Timestamp.today()
        if today.month == month:
            m = prices[(prices.index.year == today.year) & (prices.index.month == month)]
            if not m.empty:
                t_ord = _trading_day_ordinal(m.index, today)
                ax.axvline(t_ord, linestyle=":", color="gray")

        ax.set_title(
            f"{symbol} {MONTH_LABELS[month-1]}: Intra-Month Performance (Day 1 = 0%)",
            fontsize=16,
            weight="bold",
        )
        ax.set_xlabel("Trading day of month")
        ax.set_ylabel("Return since first trading day (%)")
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend(frameon=False)

    buf = io.BytesIO()
    fig.savefig(buf, dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# ================= UI ================= #
col1, col2, col3 = st.columns([2,1,1])
symbol = col1.text_input("Ticker", "^GSPC").upper()
start_year = col2.number_input("Start year", 1900, dt.datetime.today().year, 2020)
end_year = col3.number_input("End year", start_year, dt.datetime.today().year, dt.datetime.today().year)

prices = fetch_prices(symbol, f"{start_year}-01-01", f"{end_year}-12-31")
if prices is None:
    st.error("No data")
    st.stop()

stats = seasonal_stats(prices, start_year, end_year)

st.image(plot_intra_month_curve(prices, dt.datetime.today().month, start_year, end_year, symbol),
         use_container_width=True)

st.caption("© 2025 AD Fund Management LP")
