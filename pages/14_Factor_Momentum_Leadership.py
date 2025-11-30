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

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
    h1, h2, h3 {font-weight: 600; letter-spacing: 0.15px;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def card_box(inner_html: str):
    st.markdown(
        f"""
        <div style="border:1px solid #e0e0e0; border-radius:10px;
                    padding:14px; background:#fafafa; color:{TEXT};
                    font-size:14px; line-height:1.35;">
          {inner_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- Helpers ----------------
def load_prices(tickers, start):
    data = yf.download(tickers, start=start, progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    return data.dropna(how="all")


def pct_change_window(series: pd.Series, days: int) -> float:
    if len(series) <= days:
        return np.nan
    return float(series.iloc[-1] / series.iloc[-days] - 1.0)


def momentum(series: pd.Series, win: int = 20) -> float:
    r = series.pct_change().dropna()
    if len(r) < win:
        return np.nan
    return float(r.rolling(win).mean().iloc[-1])


def rs(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    return aligned.iloc[:, 0] / aligned.iloc[:, 1]


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def trend_class(series: pd.Series) -> str:
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


def inflection(short_mom: float, long_mom: float) -> str:
    if pd.isna(short_mom) or pd.isna(long_mom):
        return "Neutral"
    if short_mom > 0 and long_mom < 0:
        return "Turning Up"
    if short_mom < 0 and long_mom > 0:
        return "Turning Down"
    if abs(short_mom) > abs(long_mom):
        return "Strengthening"
    return "Weakening"


def factor_commentary(breadth_up, num_up, num_down, regime_score):
    if breadth_up >= 60 and regime_score >= 60:
        headline = "Leadership is broad and skewed to the upside."
        tail = "Tape supports leaning into risk and pressing winners."
    elif breadth_up <= 30 and regime_score <= 40:
        headline = "Leadership is weak and tilted to the downside."
        tail = "Tape argues for defense, relative shorts, or lower gross."
    else:
        headline = "Factor tape is mixed and rotational."
        tail = "Edge favors stock selection over big factor bets."

    body = f"""
    <b>Conclusion</b><br>
    {headline}<br><br>
    <b>Why it matters</b><br>
    Breadth and factor regime tell you if the equity tape is being driven by a narrow group of factors
    or by broad participation across styles. That should shape how aggressive you are with gross,
    how much you lean on factor expressions versus idiosyncratic stock picking,
    and whether you fade or ride the current leadership.<br><br>
    <b>Key drivers</b><br>
    Up trends: {num_up} factors, Down trends: {num_down}. Breadth index {breadth_up:.1f}%.<br>
    Regime score {regime_score:.1f} on a 0–100 scale, where 50 is neutral.
    """
    return body

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

# ---------------- Sidebar ----------------
st.title("Factor Momentum and Leadership Dashboard")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        "Tracks style and macro factor leadership using ETF pairs. "
        "Each factor is built as a relative strength ratio and scored on short and long momentum, "
        "trend structure, and inflection. Top section summarizes breadth and overall regime."
    )
    st.markdown(
        "- Breadth Index: share of factors in up trends.\n"
        "- Regime Score: composite of momentum, trend, and turning points.\n"
        "- Scatter plot: Short vs Long momentum quadrants for quick leadership read."
    )
    st.divider()
    st.header("Settings")
    start_date = st.date_input("History start", datetime(2015, 1, 1))
    lookback_short = st.slider("Short momentum window (days)", 10, 60, 20)
    lookback_long = st.slider("Long momentum window (days)", 30, 180, 60)
    st.caption("Data source: Yahoo Finance. Calculations are for internal research only.")

# ---------------- Load data ----------------
prices = load_prices(ALL_TICKERS, start=str(start_date))
if prices.empty:
    st.error("No data returned.")
    st.stop()

# ---------------- Compute factor levels ----------------
factor_levels = {}
for name, (up, down) in FACTOR_ETFS.items():
    if down is None:
        if up in prices:
            factor_levels[name] = prices[up]
        continue
    if up in prices and down in prices:
        factor_levels[name] = rs(prices[up], prices[down])

factor_df = pd.DataFrame(factor_levels).dropna(how="all")
if factor_df.empty:
    st.error("No factor series could be constructed.")
    st.stop()

# ---------------- Momentum snapshot table ----------------
rows = []
for f in factor_df.columns:
    s = factor_df[f].dropna()
    if len(s) < lookback_long + 5:
        continue
    r5 = pct_change_window(s, 5)
    r_short = pct_change_window(s, lookback_short)
    r_long = pct_change_window(s, lookback_long)
    mom_val = momentum(s, win=lookback_short)
    tclass = trend_class(s)
    infl = inflection(r_short, r_long)
    rows.append([f, r5, r_short, r_long, mom_val, tclass, infl])

mom_df = pd.DataFrame(
    rows,
    columns=["Factor", "%5D", "Short", "Long", "Momentum", "Trend", "Inflection"],
).set_index("Factor")

if mom_df.empty:
    st.error("No factors passed data length checks.")
    st.stop()

mom_df = mom_df.sort_values("Short", ascending=False)

# ---------------- Breadth and regime (top section) ----------------
st.subheader("Breadth and Regime")

trend_counts = mom_df["Trend"].value_counts()
num_up = int(trend_counts.get("Up", 0))
num_down = int(trend_counts.get("Down", 0))
breadth = num_up / len(mom_df) * 100.0

raw_score = (
    0.4 * mom_df["Short"].mean()
    + 0.3 * ((mom_df["Inflection"] == "Turning Up").mean() - (mom_df["Inflection"] == "Turning Down").mean())
    + 0.3 * ((mom_df["Trend"] == "Up").mean() - (mom_df["Trend"] == "Down").mean())
)
regime_score = max(0.0, min(100.0, 50.0 + 50.0 * (raw_score / 5.0)))

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Breadth Index (Up trends)", f"{breadth:.1f}%")
with c2:
    st.metric("Up factors", f"{num_up}")
with c3:
    st.metric("Factor Regime Score", f"{regime_score:.1f}")

commentary_html = factor_commentary(breadth, num_up, num_down, regime_score)
card_box(commentary_html)

# ---------------- Factor time series (2nd block) ----------------
st.subheader("Factor Time Series")

n_factors = len(factor_df.columns)
ncols = 3
nrows = int(np.ceil(n_factors / ncols))

fig_ts, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), squeeze=False)
axes = axes.ravel()

for i, f in enumerate(factor_df.columns):
    ax = axes[i]
    s = factor_df[f].dropna()
    ax.plot(s.index, s.values, color=PASTELS[i % len(PASTELS)], linewidth=2)
    ax.set_title(f, color=TEXT)
    ax.grid(color=GRID, linewidth=0.5)

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

fig_ts.tight_layout()
st.pyplot(fig_ts, clear_figure=True)

# ---------------- Factor momentum snapshot (table) ----------------
st.subheader("Factor Momentum Snapshot")

display_df = mom_df.copy()
for col in ["%5D", "Short", "Long", "Momentum"]:
    display_df[col] = display_df[col] * 100.0

display_df = display_df[
    ["%5D", "Short", "Long", "Momentum", "Trend", "Inflection"]
]

st.dataframe(
    display_df.style.format(
        {
            "%5D": "{:.1f}%",
            "Short": "{:.1f}%",
            "Long": "{:.1f}%",
            "Momentum": "{:.2f}%",
        }
    ),
    use_container_width=True,
)

# ---------------- Leadership map (Short vs Long scatter) ----------------
st.subheader("Leadership Map (Short vs Long Momentum)")

fig_lead, ax_lead = plt.subplots(figsize=(7, 5))

short_vals = mom_df["Short"] * 100.0
long_vals = mom_df["Long"] * 100.0

ax_lead.axvline(0, color="#bbbbbb", linewidth=1)
ax_lead.axhline(0, color="#bbbbbb", linewidth=1)

for i, factor in enumerate(mom_df.index):
    x = short_vals.loc[factor]
    y = long_vals.loc[factor]
    ax_lead.scatter(x, y, s=60, color=PASTELS[i % len(PASTELS)], edgecolor="#444444", linewidth=0.5)
    ax_lead.text(x, y, " " + factor, fontsize=9, va="center")

ax_lead.set_xlabel("Short window return %", color=TEXT)
ax_lead.set_ylabel("Long window return %", color=TEXT)
ax_lead.set_title("Factors by Short vs Long Momentum", color=TEXT)
ax_lead.grid(color=GRID, linewidth=0.6)
st.pyplot(fig_lead, clear_figure=True)

# ---------------- Cross factor correlation matrix ----------------
st.subheader("Cross Factor Correlation Matrix")

corr = factor_df.pct_change().dropna(how="all").corr()
labels = corr.columns.tolist()

fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
im = ax_corr.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)

ax_corr.set_xticks(range(len(labels)))
ax_corr.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax_corr.set_yticks(range(len(labels)))
ax_corr.set_yticklabels(labels, fontsize=9)
ax_corr.set_title("Correlation of Daily Returns", color=TEXT, pad=10)

for i in range(len(labels)):
    for j in range(len(labels)):
        val = corr.values[i, j]
        if abs(val) < 0.3:
            txt = ""
        else:
            txt = f"{val:.2f}"
        ax_corr.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

cbar = fig_corr.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel("Corr", rotation=270, labelpad=12)

fig_corr.tight_layout()
st.pyplot(fig_corr, clear_figure=True)

# ---------------- Relative strength vs SPY (up leg vs SPY) ----------------
st.subheader("Relative Strength vs SPY (Up Leg ETFs)")

if "SPY" in prices.columns:
    spy = prices["SPY"]
    rs_series = {}


    rs_df = pd.DataFrame(rs_series).dropna(how="all")

    if not rs_df.empty:
        # focus on last 5 years to keep it readable
        max_date = rs_df.index.max()
        cutoff = max_date - pd.DateOffset(years=5)
        rs_df = rs_df[rs_df.index >= cutoff]

        fig_rs, ax_rs = plt.subplots(figsize=(10, 4))
        for i, f in enumerate(rs_df.columns):
            ax_rs.plot(
                rs_df.index,
                rs_df[f],
                label=f,
                color=PASTELS[i % len(PASTELS)],
                linewidth=1.8,
            )

        ax_rs.axhline(1.0, color="#444444", linewidth=1.2, linestyle="--", label="SPY baseline")

        ax_rs.set_title("Factor Up-Leg Relative Strength vs SPY", color=TEXT)
        ax_rs.grid(color=GRID, linewidth=0.6)
        ax_rs.legend(
            fontsize=8,
            ncol=3,
            frameon=False,
            loc="upper left",
        )
        st.pyplot(fig_rs, clear_figure=True)

st.caption("ADFM Factor Momentum and Leadership Dashboard")
