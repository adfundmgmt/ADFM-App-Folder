import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, date

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
    if len(series) <= 1:
        return np.nan
    days = int(min(days, len(series) - 1))
    return float(series.iloc[-1] / series.iloc[-days] - 1.0)

def momentum(series: pd.Series, win: int = 20) -> float:
    r = series.pct_change().dropna()
    if len(r) < 2:
        return np.nan
    win = int(min(win, len(r)))
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

ALL_TICKERS = sorted({t for pair in FACTOR_ETFS.values() for t in pair if t})

WINDOW_MAP_DAYS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
    "3Y": 252 * 3,
    "5Y": 252 * 5,
    "10Y": 252 * 10,
}

# ---------------- Sidebar ----------------
st.title("Factor Momentum and Leadership Dashboard")

with st.sidebar:
    st.header("Settings")
    history_start = st.date_input("History start", datetime(2015, 1, 1))
    window_choice = st.selectbox(
        "Analysis window",
        ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "10Y"],
        index=3,
    )
    lookback_short = st.slider("Short momentum window (days)", 10, 60, 20)
    lookback_long = st.slider("Long momentum window (days)", 30, 180, 60)

# ---------------- Load data ----------------
prices_full = load_prices(ALL_TICKERS, start=str(history_start))
if prices_full.empty:
    st.error("No data returned.")
    st.stop()

# ---------------- Build factor series ----------------
factor_levels_full = {}
for name, (up, down) in FACTOR_ETFS.items():
    if down is None:
        factor_levels_full[name] = prices_full[up]
    else:
        factor_levels_full[name] = rs(prices_full[up], prices_full[down])

factor_df_full = pd.DataFrame(factor_levels_full).dropna(how="all")

# ---------------- Window selection ----------------
if window_choice == "YTD":
    factor_df = factor_df_full[factor_df_full.index >= pd.to_datetime(date(datetime.now().year, 1, 1))]
else:
    days = WINDOW_MAP_DAYS[window_choice]
    factor_df = factor_df_full.iloc[-days:].copy()

# ---------------- Momentum snapshot ----------------
rows = []
for f in factor_df.columns:
    s = factor_df[f].dropna()
    if len(s) < 10:
        continue
    rows.append([
        f,
        pct_change_window(s, lookback_short),
        pct_change_window(s, lookback_long),
        trend_class(s),
        inflection(
            pct_change_window(s, lookback_short),
            pct_change_window(s, lookback_long),
        )
    ])

mom_df = pd.DataFrame(
    rows,
    columns=["Factor", "Short", "Long", "Trend", "Inflection"]
).set_index("Factor").sort_values("Short", ascending=False)

# ---------------- Leadership map ----------------
st.subheader("Leadership Map (Short vs Long Momentum)")
fig, ax = plt.subplots(figsize=(8, 6))
x = mom_df["Short"] * 100
y = mom_df["Long"] * 100
ax.axhline(0, color="#888", lw=1)
ax.axvline(0, color="#888", lw=1)
for i, f in enumerate(mom_df.index):
    ax.scatter(x[f], y[f], s=70, color=PASTELS[i % len(PASTELS)], edgecolor="#444")
    ax.annotate(f, (x[f], y[f]), xytext=(4, 4), textcoords="offset points", fontsize=9)
ax.set_xlabel("Short %")
ax.set_ylabel("Long %")
ax.grid(color=GRID)
st.pyplot(fig, clear_figure=True)

# ================== REPLACEMENT SECTION ==================

# ---------------- Basket-driven factor rotation ----------------
st.subheader("Factor Rotation Engine (Basket-Based)")

# Example minimal basket universe hook
# You already have ALL_BASKETS and all_basket_rets in your basket app
# Here we assume you load or import them before this block
# all_basket_rets must be a DataFrame of daily returns by basket

@st.cache_data(show_spinner=False)
def load_example_baskets():
    # placeholder hook – replace with shared basket returns dataframe
    return pd.DataFrame()

all_basket_rets = load_example_baskets()

if all_basket_rets.empty:
    st.info("Basket return series not loaded. Connect basket engine to enable rotation view.")
    st.stop()

basket_short = {}
basket_long = {}

for b in all_basket_rets.columns:
    s = (1 + all_basket_rets[b]).cumprod().dropna()
    if len(s) < lookback_long + 5:
        continue
    basket_short[b] = pct_change_window(s, lookback_short)
    basket_long[b] = pct_change_window(s, lookback_long)

basket_mom = pd.DataFrame({"Short": basket_short, "Long": basket_long}).dropna()
basket_mom["Rank"] = basket_mom["Short"].rank(ascending=False)

top_q = basket_mom[basket_mom["Rank"] <= basket_mom["Rank"].quantile(0.25)]
bot_q = basket_mom[basket_mom["Rank"] >= basket_mom["Rank"].quantile(0.75)]

FACTOR_TO_BASKETS = {
    "Growth vs Value": {"pro": top_q, "anti": bot_q},
    "Cyclicals vs Defensives": {"pro": top_q, "anti": bot_q},
    "High Beta vs Low Vol": {"pro": top_q, "anti": bot_q},
    "Small vs Large": {"pro": top_q, "anti": bot_q},
    "Momentum": {"pro": top_q, "anti": bot_q},
}

rows = []
for f, sides in FACTOR_TO_BASKETS.items():
    if sides["pro"].empty or sides["anti"].empty:
        continue
    rs = sides["pro"]["Short"].mean() - sides["anti"]["Short"].mean()
    rl = sides["pro"]["Long"].mean() - sides["anti"]["Long"].mean()
    rows.append({
        "Factor": f,
        "Short Rotation %": rs * 100,
        "Long Rotation %": rl * 100,
        "Acceleration": (rs - rl) * 100,
    })

rot_df = pd.DataFrame(rows).set_index("Factor").sort_values("Acceleration", ascending=False)

st.dataframe(
    rot_df.style.format("{:.1f}%"),
    use_container_width=True
)

leaders = rot_df[rot_df["Acceleration"] > 0].index.tolist()
laggards = rot_df[rot_df["Acceleration"] < 0].index.tolist()

card_box(
    f"<b>Rotation Summary</b><br>"
    f"Acceleration into {', '.join(leaders[:3]) if leaders else 'none'}, "
    f"with deceleration in {', '.join(laggards[:3]) if laggards else 'none'}."
)

st.caption("© 2026 AD Fund Management LP")
