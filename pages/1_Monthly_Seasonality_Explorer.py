import datetime as dt
import io
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter, MaxNLocator
from matplotlib import gridspec

plt.style.use("default")
warnings.filterwarnings("ignore", category=FutureWarning)

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ========================== Streamlit UI ========================== #
st.set_page_config(page_title="Monthly Seasonality Explorer", layout="wide")
st.title("Monthly Seasonality Explorer")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Monthly seasonality analysis using trading-day aligned data.

        • Returns anchored to prior month-end  
        • First half = first ceil(N/2) trading days  
        • Second half shown as contribution (totals reconcile exactly)  
        • Error bars = historical min / max outcomes  
        """,
        unsafe_allow_html=True,
    )

# ========================== Data layer ========================== #
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices(symbol: str, start_year: int, end_year: int) -> Optional[pd.Series]:
    symbol = symbol.strip().upper()

    start = f"{start_year}-01-01"
    end = min(
        pd.Timestamp(f"{end_year}-12-31"),
        pd.Timestamp.today(),
    ).strftime("%Y-%m-%d")

    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df is None or len(df) == 0:
        return None

    # yfinance can return Series or DataFrame
    if isinstance(df, pd.Series):
        s = df.dropna()
        return s.rename("Close") if not s.empty else None

    if isinstance(df, pd.DataFrame):
        if "Close" not in df.columns:
            return None

        close = df["Close"]

        # Handle MultiIndex columns safely
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close = close.dropna()
        return close.rename("Close") if not close.empty else None

    return None

# ========================== Monthly decomposition ========================== #
def intra_month_decomposition(prices: pd.Series) -> pd.DataFrame:
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

        last = float(m_px.iloc[-1])
        mid = float(m_px.iloc[mid_idx])

        total = (last / prev_eom - 1.0) * 100.0
        h1 = (mid / prev_eom - 1.0) * 100.0
        h2_contrib = total - h1

        rows.append(
            {
                "year": m.year,
                "month": m.month,
                "total": total,
                "h1": h1,
                "h2_contrib": h2_contrib,
            }
        )

    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def seasonal_stats(prices: pd.Series, start_year: int, end_year: int) -> pd.DataFrame:
    df = intra_month_decomposition(prices)
    if df.empty:
        return pd.DataFrame(index=range(1, 13))

    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    grouped = df.groupby("month")

    stats = pd.DataFrame(index=range(1, 13))
    stats["label"] = MONTH_LABELS
    stats["mean_h1"] = grouped["h1"].mean()
    stats["mean_h2_contrib"] = grouped["h2_contrib"].mean()
    stats["mean_total"] = grouped["total"].mean()
    stats["min_ret"] = grouped["total"].min()
    stats["max_ret"] = grouped["total"].max()
    stats["hit_rate"] = grouped["total"].apply(lambda x: (x > 0).mean() * 100)
    stats["years"] = grouped["year"].nunique()

    return stats

# ========================== Plot ========================== #
def plot_seasonality(stats: pd.DataFrame, title: str) -> io.BytesIO:
    df = stats.dropna(subset=["mean_h1", "mean_h2_contrib", "mean_total"])
    labels = df["label"].tolist()

    h1 = df["mean_h1"].values
    h2 = df["mean_h2_contrib"].values
    tot = df["mean_total"].values
    min_r = df["min_ret"].values
    max_r = df["max_ret"].values
    hit = df["hit_rate"].values

    x = np.arange(len(labels))

    fig = plt.figure(figsize=(12.5, 7.8), dpi=200)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.2, 1.1], hspace=0.25)
    ax = fig.add_subplot(gs[0])

    ax.bar(
        x,
        h1,
        color=np.where(h1 >= 0, "#62c38e", "#e07a73"),
        edgecolor="black",
        linewidth=1,
    )

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
    ax.set_ylabel("Mean return (%)", weight="bold")
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.yaxis.set_major_locator(MaxNLocator(8))
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    ax2 = ax.twinx()
    ax2.scatter(x, hit, marker="D", color="black")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Hit rate", weight="bold")
    ax2.yaxis.set_major_formatter(PercentFormatter(100))

    ax.legend(
        handles=[
            Patch(facecolor="#62c38e", edgecolor="black", label="First half"),
            Patch(facecolor="#bbbbbb", edgecolor="black", hatch="///", label="Second half contribution"),
        ],
        frameon=False,
        loc="upper left",
    )

    fig.suptitle(title, fontsize=17, weight="bold")
    fig.tight_layout(pad=1.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
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
    end_year = st.number_input(
        "End year",
        value=dt.datetime.today().year,
        min_value=int(start_year),
        max_value=dt.datetime.today().year,
    )

with st.spinner("Loading data..."):
    prices = fetch_prices(symbol, int(start_year), int(end_year))

if prices is None or prices.empty:
    st.error("No data available for selected inputs.")
    st.stop()

prices = prices.loc[prices.first_valid_index():]

stats = seasonal_stats(prices, int(start_year), int(end_year))
buf = plot_seasonality(stats, f"{symbol} Seasonality ({start_year}–{end_year})")
st.image(buf, use_container_width=True)

st.caption("© 2025 AD Fund Management LP")
