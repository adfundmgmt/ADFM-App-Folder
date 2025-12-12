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

try:
    from pandas_datareader import data as pdr
except ImportError:
    pdr = None

FALLBACK_MAP = {"^GSPC": "SP500", "^DJI": "DJIA", "^IXIC": "NASDAQCOM"}
MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ========================== Streamlit UI ========================== #
st.set_page_config(page_title="Seasonality Dashboard", layout="wide")
st.title("Monthly Seasonality Explorer")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        • Monthly seasonality decomposed into 1H + 2H contribution  
        • Error bars show historical min / max outcomes  
        • Hit rate shows frequency of positive full-month returns  
        • Intra-month curve is **rebased to Day 1 = 0%** and shows
          average cumulative performance trading-day by trading-day
        """,
        unsafe_allow_html=True,
    )

# ========================== Data helpers ========================== #
def _extract_close_series(df) -> Optional[pd.Series]:
    if df is None:
        return None

    if isinstance(df, pd.Series):
        s = df.dropna()
        return s.rename("Close") if not s.empty else None

    if isinstance(df, pd.DataFrame):
        if df.empty:
            return None

        if "Close" in df.columns:
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            s = close.dropna()
            return s.rename("Close") if not s.empty else None

        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = df.columns.get_level_values(0)
            if "Close" in set(lvl0):
                close_df = df.loc[:, lvl0 == "Close"]
                if close_df.shape[1] >= 1:
                    s = close_df.iloc[:, 0].dropna()
                    return s.rename("Close") if not s.empty else None

    return None

def _yf_download(symbol: str, start: str, end: str) -> Optional[pd.Series]:
    try:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        return _extract_close_series(df)
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices(symbol: str, start: str, end: str) -> Optional[pd.Series]:
    symbol = symbol.strip().upper()
    end = min(pd.Timestamp(end), pd.Timestamp.today()).strftime("%Y-%m-%d")
    start_pad = (pd.Timestamp(start) - pd.DateOffset(days=45)).strftime("%Y-%m-%d")

    s = _yf_download(symbol, start_pad, end)
    if s is not None:
        return s

    if symbol == "SPY":
        s = _yf_download("^GSPC", start_pad, end)
        if s is not None:
            return s

    if pdr and symbol in FALLBACK_MAP:
        try:
            fred = FALLBACK_MAP[symbol]
            df = pdr.DataReader(fred, "fred", start_pad, end)
            if fred in df.columns:
                s = df[fred].dropna()
                return s.rename("Close") if not s.empty else None
        except Exception:
            pass

    return None

# ========================== Monthly decomposition ========================== #
def _intra_month_halves(prices: pd.Series) -> pd.DataFrame:
    rows = []

    months = pd.period_range(
        prices.index.min().to_period("M"),
        prices.index.max().to_period("M"),
        freq="M",
    )

    for m in months:
        m_px = prices.loc[prices.index.to_period("M") == m]
        if len(m_px) < 3:
            continue

        prev_px = prices.loc[prices.index.to_period("M") == (m - 1)]
        if prev_px.empty:
            continue

        prev_eom = float(prev_px.iloc[-1])
        if not np.isfinite(prev_eom) or prev_eom <= 0:
            continue

        n = len(m_px)
        mid_idx = int(np.ceil(n / 2)) - 1

        total = (float(m_px.iloc[-1]) / prev_eom - 1.0) * 100.0
        h1 = (float(m_px.iloc[mid_idx]) / prev_eom - 1.0) * 100.0
        h2 = total - h1

        rows.append(
            dict(
                year=m.year,
                month=m.month,
                total_ret=total,
                h1_ret=h1,
                h2_contrib=h2,
            )
        )

    return pd.DataFrame(rows)

def seasonal_stats(prices: pd.Series, start_year: int, end_year: int) -> pd.DataFrame:
    df = _intra_month_halves(prices)
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    stats = pd.DataFrame(index=range(1, 13))
    stats["label"] = MONTH_LABELS

    g = df.groupby("month")
    stats["mean_h1"] = g["h1_ret"].mean()
    stats["mean_h2"] = g["h2_contrib"].mean()
    stats["mean_total"] = g["total_ret"].mean()
    stats["min_ret"] = g["total_ret"].min()
    stats["max_ret"] = g["total_ret"].max()
    stats["hit_rate"] = g["total_ret"].apply(lambda x: (x > 0).mean() * 100)

    return stats

# ========================== Seasonality plot ========================== #
def plot_seasonality(stats: pd.DataFrame, title: str) -> io.BytesIO:
    df = stats.dropna(subset=["mean_h1", "mean_h2", "mean_total"])

    labels = df["label"].tolist()
    h1 = df["mean_h1"].values
    h2 = df["mean_h2"].values
    tot = df["mean_total"].values
    min_r = df["min_ret"].values
    max_r = df["max_ret"].values
    hit = df["hit_rate"].values

    x = np.arange(len(labels))

    fig = plt.figure(figsize=(12.5, 7.8), dpi=200)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.2, 1.1])
    ax = fig.add_subplot(gs[0])

    ax.bar(x, h1, color=np.where(h1 >= 0, "#62c38e", "#e07a73"), edgecolor="black")

    for i in range(len(x)):
        base = h1[i] if np.sign(h1[i]) == np.sign(h2[i]) else 0.0
        ax.bar(
            x[i],
            h2[i],
            bottom=base,
            color="#bbbbbb",
            edgecolor="black",
            hatch="///",
        )

    yerr = np.abs(np.vstack([tot - min_r, max_r - tot]))
    ax.errorbar(x, tot, yerr=yerr, fmt="none", ecolor="gray", capsize=6)

    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean return (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    ax2 = ax.twinx()
    ax2.scatter(x, hit, marker="D", color="black")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Hit rate (%)")

    ax.legend(
        handles=[
            Patch(facecolor="#62c38e", edgecolor="black", label="First half"),
            Patch(facecolor="#bbbbbb", edgecolor="black", hatch="///", label="Second half contribution"),
        ],
        frameon=False,
    )

    fig.suptitle(title, fontsize=17, weight="bold")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# ========================== REBASED intra-month paths ========================== #
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
        if not np.isfinite(first) or first <= 0:
            continue

        norm = (m / first - 1.0) * 100.0
        norm.index = pd.RangeIndex(1, len(norm) + 1)
        paths[y] = norm

    if not paths:
        return pd.DataFrame(), pd.Series(dtype=float)

    max_len = max(len(v) for v in paths.values())
    df = pd.DataFrame(index=pd.RangeIndex(1, max_len + 1))
    for y, s in paths.items():
        df[y] = s

    return df, df.mean(axis=1)

# ========================== Intra-month plot ========================== #
def plot_intra_month_curve(prices, month, start_year, end_year, symbol):
    df, avg = _month_paths_rebased(prices, month, start_year, end_year)

    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=200)

    if avg.empty:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
        ax.axis("off")
    else:
        std = df.std(axis=1)

        ax.axhline(0, color="gray", linestyle=":")
        ax.fill_between(avg.index, avg - std, avg + std, alpha=0.25)

        ax.plot(avg.index, avg, color="black", linewidth=2.6)
        ax.set_title(
            f"{symbol} {MONTH_LABELS[month-1]}: Intra-Month Performance (Day 1 = 0%)",
            fontsize=16,
            weight="bold",
        )
        ax.set_xlabel("Trading day of month")
        ax.set_ylabel("Return since first trading day (%)")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.yaxis.set_major_formatter(PercentFormatter(100))

    buf = io.BytesIO()
    fig.savefig(buf, dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# ========================== Controls ========================== #
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    symbol = st.text_input("Ticker", value="^GSPC").upper()
with col2:
    start_year = st.number_input("Start year", value=2020, min_value=1900)
with col3:
    end_year = st.number_input("End year", value=dt.datetime.today().year, min_value=start_year)

start = f"{start_year}-01-01"
end = f"{end_year}-12-31"

prices = fetch_prices(symbol, start, end)
if prices is None:
    st.error("No data available")
    st.stop()

stats = seasonal_stats(prices, start_year, end_year)
st.image(plot_seasonality(stats, f"{symbol} Seasonality ({start_year}-{end_year})"), use_container_width=True)

st.subheader("Intra-Month Seasonality (Rebased)")
month = st.selectbox("Month", list(range(1,13)), format_func=lambda m: MONTH_LABELS[m-1])
st.image(plot_intra_month_curve(prices, month, start_year, end_year, symbol), use_container_width=True)

st.caption("© 2025 AD Fund Management LP")
