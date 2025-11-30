import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------- Config ----------------
st.set_page_config(page_title="Factor Momentum and Leadership", layout="wide")
plt.style.use("default")

PASTELS = [
    "#A8DADC", "#F4A261", "#90BE6D", "#FFD6A5", "#BDE0FE",
    "#CDB4DB", "#E2F0CB", "#F1C0E8", "#B9FBC0", "#F7EDE2"
]

TEXT = "#222222"
GRID = "#e6e6e6"


# ---------------- Helpers ----------------
def load_prices(tickers, start):
    data = yf.download(tickers, start=start, progress=False, auto_adjust=True)
    # yfinance returns:
    # - MultiIndex columns for multiple tickers
    # - Simple columns for single ticker
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            data = data["Close"]
    else:
        if "Adj Close" in data.columns:
            data = data["Adj Close"]
        elif "Close" in data.columns:
            data = data["Close"]
    return data.dropna(how="all")


def pct_change(series, days):
    # last vs N sessions ago
    if len(series) <= days:
        return np.nan
    return (series.iloc[-1] / series.iloc[-(days + 1)] - 1.0)


def momentum(series, win=20):
    r = series.pct_change().dropna()
    if len(r) < win:
        return np.nan
    return r.rolling(win).mean().iloc[-1]


def rs(series_a, series_b):
    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    return aligned.iloc[:, 0] / aligned.iloc[:, 1] if not aligned.empty else pd.Series(dtype=float)


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def trend_class(series):
    """Uses EMA 10, 20, 40 to classify trend."""
    if len(series) < 50:
        return "Neutral"
    e1 = ema(series, 10).iloc[-1]
    e2 = ema(series, 20).iloc[-1]
    e3 = ema(series, 40).iloc[-1]
    if e1 > e2 > e3:
        return "Up"
    if e1 < e2 < e3:
        return "Down"
    return "Neutral"


def inflection(short_mom, long_mom):
    if pd.isna(short_mom) or pd.isna(long_mom):
        return "Neutral"
    if short_mom > 0 and long_mom < 0:
        return "Turning Up"
    if short_mom < 0 and long_mom > 0:
        return "Turning Down"
    if abs(short_mom) > abs(long_mom):
        return "Strengthening"
    return "Weakening"


def card_box(inner_html: str):
    st.markdown(
        f"""
        <div style="border:1px solid #e0e0e0; border-radius:10px; padding:14px; background:#fafafa; color:{TEXT}; font-size:14px; line-height:1.35;">
          {inner_html}
        </div>
        """.strip(),
        unsafe_allow_html=True,
    )


# ---------------- Factors ----------------
FACTOR_ETFS = {
    "Growth vs Value": ("VUG", "VTV"),
    "Quality vs Junk": ("QUAL", "JNK"),
    "High Beta vs Low Vol": ("SPHB", "SPLV"),
    "Small vs Large": ("IWM", "SPY"),
    "Tech vs Broad": ("XLK", "SPY"),
    "Cyclicals vs Defensives": ("XLY", "XLP"),
    "US vs World": ("SPY", "VEA"),
    "Momentum": ("MTUM", None),
    "Equal Weight vs Cap": ("RSP", "SPY"),
}

ALL_TICKERS = sorted({t for pair in FACTOR_ETFS.values() for t in pair if t is not None})

# ---------------- Layout and sidebar ----------------
st.title("Factor Momentum and Leadership Dashboard")
st.caption("Relative performance view of major equity factor and style spreads. Data: Yahoo Finance.")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose**

        Track which equity factors are leading or lagging and whether leadership
        is broad or narrow.

        **What it does**

        - Builds factor series as ratios of ETF pairs (growth vs value, small vs large, etc.)  
        - Computes short and long horizon relative returns for each factor  
        - Classifies trend using 10, 20, 40 day EMAs  
        - Flags inflection points where short momentum flips against long momentum  
        - Aggregates into a simple breadth and regime read

        **How to use**

        - Use the snapshot table to see which factor spreads are in strong up or down trends  
        - Use the breadth and regime scores to judge how concentrated leadership is  
        - Use the leadership matrix and correlation panel to see where factor bets diversify or overlap
        """
    )

    st.header("Settings")
    start_date = st.date_input("History start", datetime(2015, 1, 1))
    lookback_short = st.slider("Short momentum window (days)", 10, 60, 20)
    lookback_long = st.slider("Long momentum window (days)", 30, 180, 60)
    st.caption("Factors are built on daily closes. Returns are not annualized.")

# ---------------- Load data ----------------
prices = load_prices(ALL_TICKERS, start=str(start_date))
if prices.empty:
    st.error("No price data returned for the selected range.")
    st.stop()

# ---------------- Compute factor level series ----------------
factor_levels = {}
for name, (up, down) in FACTOR_ETFS.items():
    if down is None:
        if up in prices.columns:
            factor_levels[name] = prices[up].dropna()
        continue
    if up in prices.columns and down in prices.columns:
        factor_levels[name] = rs(prices[up], prices[down])

factor_df = pd.DataFrame(factor_levels).dropna(how="all")
if factor_df.empty:
    st.error("No valid factor series could be constructed.")
    st.stop()

# ---------------- Momentum snapshots ----------------
rows = []
for f in factor_df.columns:
    s = factor_df[f].dropna()
    if len(s) < lookback_long + 10:
        continue
    r5 = pct_change(s, 5)
    r_short = pct_change(s, lookback_short)
    r_long = pct_change(s, lookback_long)
    mom_val = momentum(s, win=lookback_short)
    tclass = trend_class(s)
    infl = inflection(r_short, r_long)
    rows.append([f, r5, r_short, r_long, mom_val, tclass, infl])

mom_df = pd.DataFrame(
    rows,
    columns=["Factor", "%5D", "Short", "Long", "Momentum", "Trend", "Inflection"],
).set_index("Factor")

if mom_df.empty:
    st.error("No factors have enough history for the chosen windows.")
    st.stop()

mom_df = mom_df.sort_values("Short", ascending=False)

st.subheader("Factor Momentum Snapshot")
st.dataframe(mom_df.round(3), use_container_width=True)

# ---------------- Breadth and regime ----------------
st.subheader("Breadth and Regime")

trend_vals = mom_df["Trend"].value_counts()
num_up = int(trend_vals.get("Up", 0))
num_down = int(trend_vals.get("Down", 0))
breadth = num_up / len(mom_df) * 100.0 if len(mom_df) else 0.0

# Regime score: 0 to 100
raw_score = (
    40 * mom_df["Short"].mean()
    + 30 * ((mom_df["Inflection"] == "Turning Up").mean() - (mom_df["Inflection"] == "Turning Down").mean())
    + 30 * ((mom_df["Trend"] == "Up").mean() - (mom_df["Trend"] == "Down").mean())
)

regime_score = 50 + 50 * (raw_score / 5.0)
regime_score = max(0.0, min(100.0, regime_score))

c1, c2, c3 = st.columns(3)
c1.metric("Breadth Index (Up trends)", f"{breadth:.1f}%")
c2.metric("Up factors", f"{num_up}")
c3.metric("Factor Regime Score", f"{regime_score:.1f}")

# Quick commentary card
if len(mom_df) > 0:
    if regime_score >= 65 and breadth >= 60:
        conclusion = "Leadership is broad and supportive of risk. Factor tape is friendly to trend following."
    elif regime_score <= 35 and num_down > num_up:
        conclusion = "Leadership is weak and skewed to the downside. Factor tape argues for defense or relative shorts."
    else:
        conclusion = "Leadership is mixed. Factor tape favors selective expressions rather than high conviction index level bets."

    card_box(
        f"""
        <b>Conclusion</b><br>
        {conclusion}<br><br>
        <b>Why it matters</b><br>
        Breadth and regime score tell you if the equity tape is being driven by a narrow group of factors
        or by broad participation across styles. That should shape how aggressive you are with gross and how much you lean
        on single factor trades versus idiosyncratic stock picking.
        """
    )

# ---------------- Leadership matrix (Short vs Long momentum) ----------------
st.subheader("Leadership Matrix (Short vs Long momentum)")

mat = mom_df[["Short", "Long"]].copy()
if not mat.empty:
    data = mat.values
    fig_h, ax_h = plt.subplots(figsize=(6, 5))
    im = ax_h.imshow(data, aspect="auto", cmap="RdYlGn")

    ax_h.set_xticks(range(data.shape[1]))
    ax_h.set_xticklabels(["Short", "Long"])
    ax_h.set_yticks(range(data.shape[0]))
    ax_h.set_yticklabels(mat.index)

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax_h.text(
                j,
                i,
                f"{val:.2f}" if not np.isnan(val) else "",
                ha="center",
                va="center",
                fontsize=8,
                color=TEXT,
            )

    ax_h.set_title("Factor Leadership Heatmap", color=TEXT)
    ax_h.grid(False)
    fig_h.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
    plt.tight_layout()
    st.pyplot(fig_h)
else:
    st.info("No data for leadership matrix.")

# ---------------- Time series panels ----------------
st.subheader("Factor Time Series")

ncols = 3
cols = st.columns(ncols)
i = 0

for f in factor_df.columns:
    s = factor_df[f].dropna()
    if s.empty:
        continue
    with cols[i % ncols]:
        fig_ts, ax_ts = plt.subplots(figsize=(5, 3))
        ax_ts.plot(s.index, s.values, color=PASTELS[i % len(PASTELS)], linewidth=2)
        ax_ts.set_title(f, color=TEXT)
        ax_ts.grid(color=GRID)
        st.pyplot(fig_ts)
    i += 1

# ---------------- Cross factor correlation ----------------
st.subheader("Cross Factor Correlation Matrix")

corr = factor_df.pct_change().dropna(how="all").corr()
if not corr.empty:
    data_c = corr.values
    fig_c, ax_c = plt.subplots(figsize=(8, 6))
    im_c = ax_c.imshow(data_c, cmap="PuBuGn", vmin=-1, vmax=1)

    ax_c.set_xticks(range(data_c.shape[1]))
    ax_c.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax_c.set_yticks(range(data_c.shape[0]))
    ax_c.set_yticklabels(corr.index)

    for i in range(data_c.shape[0]):
        for j in range(data_c.shape[1]):
            val = data_c[i, j]
            ax_c.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=TEXT,
            )

    ax_c.set_title("Correlation Matrix", color=TEXT)
    ax_c.grid(False)
    fig_c.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
    plt.tight_layout()
    st.pyplot(fig_c)
else:
    st.info("Not enough data to compute correlations.")

# ---------------- Relative Strength vs SPY ----------------
st.subheader("Relative Strength vs SPY")

if "SPY" in prices.columns:
    spy = prices["SPY"].dropna()
    rs_series = {}
    for name, (up, _) in FACTOR_ETFS.items():
        if up in prices.columns:
            rel = rs(prices[up], spy)
            if not rel.empty:
                rs_series[name] = rel

    rs_df = pd.DataFrame(rs_series).dropna(how="all")
    if not rs_df.empty:
        fig_rs, ax_rs = plt.subplots(figsize=(10, 4))
        for i, f in enumerate(rs_df.columns):
            ax_rs.plot(
                rs_df.index,
                rs_df[f],
                label=f,
                color=PASTELS[i % len(PASTELS)],
                linewidth=1.6,
            )
        ax_rs.grid(color=GRID)
        ax_rs.legend(fontsize=8)
        ax_rs.set_title("Factor Relative Strength vs SPY (up leg vs SPY)", color=TEXT)
        st.pyplot(fig_rs)
    else:
        st.info("No relative strength series could be computed versus SPY.")
else:
    st.info("SPY prices missing. Cannot compute relative strength vs SPY.")

st.caption("ADFM Factor Momentum and Leadership Dashboard")
