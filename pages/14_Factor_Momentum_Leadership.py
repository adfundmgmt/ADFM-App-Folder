import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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

# --------- Timeframe config ---------
TIMEFRAME_OPTIONS = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "10Y"]


def get_timeframe_config(label: str):
    """
    Returns (short_window_days, long_window_days, start_date_for_download, cutoff_date_for_display)
    Short/long are in trading days; cutoff is calendar date for charts / returns.
    """
    today = datetime.today().date()

    if label == "1M":
        short_win = 10
        long_win = 21
        display_days = 30
        cutoff = today - timedelta(days=display_days)
    elif label == "3M":
        short_win = 21
        long_win = 63
        display_days = 90
        cutoff = today - timedelta(days=display_days)
    elif label == "6M":
        short_win = 63
        long_win = 126
        display_days = 180
        cutoff = today - timedelta(days=display_days)
    elif label == "YTD":
        year_start = datetime(today.year, 1, 1).date()
        display_days = (today - year_start).days
        short_win = 21
        long_win = max(60, display_days)  # effectively YTD long lookback
        cutoff = year_start
    elif label == "1Y":
        short_win = 63
        long_win = 252
        display_days = 365
        cutoff = today - timedelta(days=display_days)
    elif label == "3Y":
        short_win = 126
        long_win = 756
        display_days = 365 * 3
        cutoff = today - timedelta(days=display_days)
    elif label == "5Y":
        short_win = 252
        long_win = 1260
        display_days = 365 * 5
        cutoff = today - timedelta(days=display_days)
    else:  # "10Y"
        short_win = 252
        long_win = 2520
        display_days = 365 * 10
        cutoff = today - timedelta(days=display_days)

    # download a bit more than long window so rolling stats have room
    start_date = cutoff - timedelta(days=long_win // 2 + 30)
    return short_win, long_win, datetime.combine(start_date, datetime.min.time()), datetime.combine(cutoff, datetime.min.time())

# ---------------- Helpers ----------------
def load_prices(tickers, start):
    data = yf.download(tickers, start=start, progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    return data.dropna(how="all")


def pct_change_window(series: pd.Series, days: int) -> float:
    # allow len == days (we will use the first observation)
    if len(series) < days or days <= 0:
        return np.nan
    return float(series.iloc[-1] / series.iloc[-days] - 1.0)


def momentum(series: pd.Series, win: int = 20) -> float:
    r = series.pct_change().dropna()
    if len(r) < win or win <= 0:
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


def bucket_breadth(breadth: float) -> str:
    if breadth < 10:
        return "extremely narrow"
    if breadth < 25:
        return "narrow and selective"
    if breadth < 40:
        return "tilted to a small group of styles"
    if breadth < 60:
        return "balanced across factors"
    if breadth < 75:
        return "broadening out across styles"
    return "very broad and inclusive"


def bucket_regime(regime_score: float) -> str:
    if regime_score < 25:
        return "deeply defensive and stress driven"
    if regime_score < 40:
        return "defensive and risk averse"
    if regime_score < 55:
        return "roughly neutral with a mild defensive lean"
    if regime_score < 70:
        return "constructive and risk friendly"
    return "high beta, late-cycle risk on"


def factor_tilt_phrase(leaders: list, laggards: list) -> str:
    lead_str = ", ".join(leaders) if leaders else "no clear leaders"
    lag_str = ", ".join(laggards) if laggards else "no obvious laggards"
    return (
        f"Leadership today is coming from {lead_str}, "
        f"while pressure is most visible in {lag_str}."
    )


def positioning_hint(breadth: float, regime_score: float, up_count: int, down_count: int) -> str:
    if breadth < 25 and regime_score < 40:
        return (
            "Keep gross light, lean on index or factor hedges, and size single name longs off the "
            "few factors that are actually in up trends. This is a good environment to express "
            "views through relative value and pair trades rather than outright beta."
        )
    if breadth < 40 and regime_score >= 55:
        return (
            "Tape is friendly but leadership is narrow. Concentrate capital in the leading styles, "
            "avoid shorting them mechanically, and use losers as funding shorts. Be careful with "
            "broad risk-on assumptions because only a subset of factors is really working."
        )
    if breadth >= 60 and regime_score >= 60:
        return (
            "This is a broad, constructive regime. It supports running higher gross and letting "
            "winners compound, but you should already be thinking about where to hide once "
            "breadth and momentum start to roll over."
        )
    if breadth >= 60 and regime_score < 45:
        return (
            "Breadth is okay but the quality of leadership is questionable. Rotate toward higher "
            "quality expressions inside each factor and be quicker to cut when momentum rolls."
        )
    if abs(up_count - down_count) <= 2:
        return (
            "Up and down trends are roughly balanced. Stock selection matters more than big "
            "top-down factor tilts here."
        )
    return (
        "Treat factors as a map, not a trade list. Use them to sanity-check your book: make sure "
        "your largest positions rhyme with the leadership you actually see in the tape."
    )


def build_commentary(mom_df: pd.DataFrame, breadth: float, regime_score: float) -> str:
    trend_counts = mom_df["Trend"].value_counts()
    up_count = int(trend_counts.get("Up", 0))
    down_count = int(trend_counts.get("Down", 0))

    short_sorted = mom_df["Short"].sort_values(ascending=False)
    leaders = [f for f in short_sorted.index[:3]]
    laggards = [f for f in short_sorted.index[-3:]]

    breadth_desc = bucket_breadth(breadth)
    regime_desc = bucket_regime(regime_score)
    tilt_text = factor_tilt_phrase(leaders, laggards)
    position_text = positioning_hint(breadth, regime_score, up_count, down_count)

    conclusion = (
        f"Factor tape is {breadth_desc} and currently {regime_desc}. "
        f"{tilt_text}"
    )

    why_matters = (
        "Factor structure tells you whether the equity market is being driven by style and macro "
        "buckets or by idiosyncratic stories. It should anchor how much you trust breakout moves, "
        "how aggressive you are with gross, and whether you express views through index, factor, "
        "or single stock risk."
    )

    key_stats = (
        f"Up trends: {up_count} factors, Down trends: {down_count}. "
        f"Breadth index {breadth:.1f}%. "
        f"Regime score {regime_score:.1f} on a 0–100 scale, where 50 is neutral."
    )

    return (
        f"<b>Conclusion</b><br>{conclusion}<br><br>"
        f"<b>Why it matters</b><br>{why_matters}<br><br>"
        f"<b>Positioning cues</b><br>{position_text}<br><br>"
        f"<b>Key stats</b><br>{key_stats}"
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

# ---------------- Sidebar ----------------
st.title("Factor Momentum and Leadership Dashboard")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        "Tracks style and macro factor leadership using ETF pairs. "
        "Each factor is built as a relative strength ratio and scored on short and long momentum, "
        "trend structure, and inflection."
    )
    st.markdown(
        "- Snapshot table shows short and long window performance, trend, and inflection.\n"
        "- Scatter map puts factors into quadrants by short vs long momentum.\n"
        "- Correlation heatmap helps you see clustering and crowding."
    )
    st.divider()
    st.header("Settings")
    timeframe_label = st.selectbox("Analysis window", TIMEFRAME_OPTIONS, index=4)
    short_window, long_window, start_date, cutoff_date = get_timeframe_config(timeframe_label)
    st.caption(
        f"Short window: {short_window} trading days. "
        f"Long window: {long_window} trading days.\n"
        f"Displayed history: {timeframe_label}."
    )
    st.caption("Data source: Yahoo Finance. Internal use only.")

# ---------------- Load data ----------------
prices = load_prices(ALL_TICKERS, start=start_date)
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

factor_df_all = pd.DataFrame(factor_levels).dropna(how="all")
if factor_df_all.empty:
    st.error("No factor series could be constructed.")
    st.stop()

# restrict to selected display window
cutoff_ts = pd.Timestamp(cutoff_date)
factor_df = factor_df_all[factor_df_all.index >= cutoff_ts]
if factor_df.empty:
    st.error("No data in the selected window.")
    st.stop()

# ---------------- Momentum snapshot base data ----------------
rows = []
min_len_required = max(long_window, short_window, 5)

for f in factor_df.columns:
    s = factor_df[f].dropna()
    if len(s) < min_len_required:
        continue
    r5 = pct_change_window(s, min(5, len(s) - 1))
    r_short = pct_change_window(s, short_window)
    r_long = pct_change_window(s, long_window)
    mom_val = momentum(s, win=short_window)
    tclass = trend_class(s)
    infl = inflection(r_short, r_long)
    rows.append([f, r5, r_short, r_long, mom_val, tclass, infl])

mom_df = pd.DataFrame(
    rows,
    columns=["Factor", "%5D", "Short", "Long", "Momentum", "Trend", "Inflection"],
).set_index("Factor")

if mom_df.empty:
    st.error("No factors passed data length checks for this window.")
    st.stop()

mom_df = mom_df.sort_values("Short", ascending=False)

# ---------------- Compute breadth and regime (for commentary only) ----------------
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

# ---------------- Top: Factor tape summary ----------------
st.subheader("Factor Tape Summary")
summary_html = build_commentary(mom_df, breadth, regime_score)
card_box(summary_html)

# ---------------- Factor time series ----------------
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
st.caption(
    f"Short window: {short_window} trading days. "
    f"Long window: {long_window} trading days. "
    f"Returns shown for the selected {timeframe_label} window."
)

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
        txt = f"{val:.2f}" if abs(val) >= 0.3 else ""
        ax_corr.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

cbar = fig_corr.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel("Corr", rotation=270, labelpad=12)

fig_corr.tight_layout()
st.pyplot(fig_corr, clear_figure=True)

st.caption("ADFM Factor Momentum and Leadership Dashboard")
