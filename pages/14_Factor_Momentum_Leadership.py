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


def build_commentary(
    mom_df: pd.DataFrame,
    breadth: float,
    regime_score: float,
    corr: pd.DataFrame | None = None,
) -> str:
    trend_counts = mom_df["Trend"].value_counts()
    up_count = int(trend_counts.get("Up", 0))
    down_count = int(trend_counts.get("Down", 0))

    established_leaders = mom_df[
        (mom_df["Short"] > 0) & (mom_df["Long"] > 0)
    ].sort_values("Short", ascending=False).index.tolist()

    new_rotations = mom_df[mom_df["Inflection"] == "Turning Up"].index.tolist()
    fading_leaders = mom_df[mom_df["Inflection"] == "Turning Down"].index.tolist()

    leaders_text = ", ".join(established_leaders[:4]) if established_leaders else "no factor pair in a clean dual-horizon uptrend"
    rotations_text = ", ".join(new_rotations[:4]) if new_rotations else "no factor is clearly turning up yet"
    fading_text = ", ".join(fading_leaders[:4]) if fading_leaders else "no obvious factor is rolling over from strength"

    breadth_desc = bucket_breadth(breadth)
    regime_desc = bucket_regime(regime_score)

    crowd_line = "Correlation picture is mixed and does not add a strong crowding signal."
    if corr is not None and not corr.empty:
        avg_abs = {}
        for f in corr.columns:
            vals = corr.loc[f].drop(f)
            if vals.empty:
                continue
            avg_abs[f] = float(vals.abs().mean())
        if avg_abs:
            crowded_factor = max(avg_abs, key=avg_abs.get)
            diversifier = min(avg_abs, key=avg_abs.get)
            crowd_line = (
                f"Most crowded style on this window is {crowded_factor} with average |corr| around "
                f"{avg_abs[crowded_factor]:.2f} to peers, while the cleanest diversifier is "
                f"{diversifier} with average |corr| near {avg_abs[diversifier]:.2f}."
            )

    conclusion = (
        f"Factor tape is {breadth_desc} and currently {regime_desc}. "
        f"Leadership is anchored in {leaders_text}, with rotations starting to show up in "
        f"{rotations_text}, and pressure building in {fading_text}."
    )

    why_matters = (
        "This grid is the style map for the equity tape. It tells you which buckets the market is "
        "paying for right now, how persistent that preference is across short and long windows, "
        "and whether you should lean into existing trends or hunt for rotations."
    )

    drivers = []

    drivers.append(
        f"{up_count} factors are in up trends and {down_count} are in down trends based on the "
        "10/20/40-day moving average stack, with the rest stuck in noisy ranges."
    )

    drivers.append(
        f"Short horizon strength is concentrated in "
        f"{', '.join(mom_df.sort_values('Short', ascending=False).index.tolist()[:5])}, "
        "while the weakest short-term tape sits in "
        f"{', '.join(mom_df.sort_values('Short', ascending=True).index.tolist()[:3])}."
    )

    if new_rotations:
        drivers.append(
            f"Inflection signals flag {', '.join(new_rotations)} as turning up from weaker long-term trends, "
            "which is where new leaders usually emerge if the regime stays constructive."
        )
    if fading_leaders:
        drivers.append(
            f"On the other side, {', '.join(fading_leaders)} are turning down against still-positive long windows, "
            "a typical pattern near the end of a leadership run."
        )

    drivers.append(crowd_line)

    key_stats = (
        f"Breadth index {breadth:.1f}%. "
        f"Regime score {regime_score:.1f} on a 0-100 scale, where 50 is neutral."
    )

    body = (
        '<div style="font-weight:700; margin-bottom:6px;">Conclusion</div>'
        f'<div>{conclusion}</div>'
        '<div style="font-weight:700; margin:10px 0 6px;">Why it matters</div>'
        f'<div>{why_matters}</div>'
        '<div style="font-weight:700; margin:10px 0 6px;">Key drivers</div>'
        '<ul style="margin-top:4px; margin-bottom:4px;">'
        + "".join(f"<li>{d}</li>" for d in drivers)
        + "</ul>"
        '<div style="font-weight:700; margin:10px 0 6px;">Key stats</div>'
        f'<div>{key_stats}</div>'
    )

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
    st.header("About This Tool")
    st.markdown(
        """
        Tracks style and macro factor leadership using ETF pairs and scores each factor on
        short and long momentum, trend structure, and inflection. Use it as a style map
        for how your book lines up with what the tape is rewarding.
        """
    )
    st.markdown(
        """
        - Time series grid: context for each factor ratio  
        - Leadership map: short vs long momentum quadrants  
        - Crowding view: which styles are most correlated and where diversifiers sit
        """
    )
    st.divider()
    st.header("Settings")
    history_start = st.date_input("History start", datetime(2015, 1, 1))
    window_choice = st.selectbox(
        "Analysis window",
        ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "10Y"],
        index=3,
    )
    lookback_short = st.slider("Short momentum window (days)", 10, 60, 20)
    lookback_long = st.slider("Long momentum window (days)", 30, 180, 60)
    st.caption("Data source: Yahoo Finance. Internal use only.")

# ---------------- Load data ----------------
prices_full = load_prices(ALL_TICKERS, start=str(history_start))
if prices_full.empty:
    st.error("No data returned.")
    st.stop()

# ---------------- Build factor series (full history) ----------------
factor_levels_full = {}
for name, (up, down) in FACTOR_ETFS.items():
    if down is None:
        if up in prices_full:
            factor_levels_full[name] = prices_full[up]
        continue
    if up in prices_full and down in prices_full:
        factor_levels_full[name] = rs(prices_full[up], prices_full[down])

factor_df_full = pd.DataFrame(factor_levels_full).dropna(how="all")
if factor_df_full.empty:
    st.error("No factor series could be constructed.")
    st.stop()

# ---------------- Window selection ----------------
if window_choice == "YTD":
    current_year_start = date(datetime.now().year, 1, 1)
    factor_df = factor_df_full[factor_df_full.index >= pd.to_datetime(current_year_start)]
else:
    days = WINDOW_MAP_DAYS[window_choice]
    if len(factor_df_full) <= days:
        factor_df = factor_df_full.copy()
    else:
        factor_df = factor_df_full.iloc[-days:].copy()

if factor_df.empty:
    st.error("No data available for the selected window.")
    st.stop()

# ---------------- Momentum snapshot data ----------------
rows = []
for f in factor_df.columns:
    s = factor_df[f].dropna()
    if len(s) < 5:
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
    st.error("No factors passed data checks for this window.")
    st.stop()

mom_df = mom_df.sort_values("Short", ascending=False)

# ---------------- Breadth & regime + correlation for commentary ----------------
trend_counts = mom_df["Trend"].value_counts()
num_up = int(trend_counts.get("Up", 0))
breadth = num_up / len(mom_df) * 100.0

raw_score = (
    0.4 * mom_df["Short"].mean()
    + 0.3 * ((mom_df["Inflection"] == "Turning Up").mean() - (mom_df["Inflection"] == "Turning Down").mean())
    + 0.3 * ((mom_df["Trend"] == "Up").mean() - (mom_df["Trend"] == "Down").mean())
)
regime_score = max(0.0, min(100.0, 50.0 + 50.0 * (raw_score / 5.0)))

corr_matrix = factor_df.pct_change().dropna(how="all").corr()

# ---------------- Factor tape summary ----------------
st.subheader(f"Factor Tape Summary ({window_choice})")
summary_html = build_commentary(mom_df, breadth, regime_score, corr_matrix)
card_box(summary_html)

# ---------------- Factor time series ----------------
st.subheader(f"Factor Time Series ({window_choice})")

n_factors = len(factor_df.columns)
ncols = 3
nrows = int(np.ceil(n_factors / ncols))

fig_ts, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), squeeze=False)
axes = axes.ravel()

if len(factor_df.index) > 1:
    span_days = (factor_df.index[-1] - factor_df.index[0]).days
else:
    span_days = 0

if span_days <= 370:
    locator = mdates.MonthLocator()
    formatter = mdates.DateFormatter("%b")
else:
    locator = mdates.YearLocator()
    formatter = mdates.DateFormatter("%Y")

for i, f in enumerate(factor_df.columns):
    ax = axes[i]
    s = factor_df[f].dropna()
    ax.plot(s.index, s.values, color=PASTELS[i % len(PASTELS)], linewidth=2)
    ax.set_title(f, color=TEXT)
    ax.grid(color=GRID, linewidth=0.5)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_fontsize(8)

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

fig_lead, ax_lead = plt.subplots(figsize=(8, 6))

short_vals = mom_df["Short"] * 100.0
long_vals = mom_df["Long"] * 100.0

x_max = max(abs(short_vals.min()), abs(short_vals.max()))
y_max = max(abs(long_vals.min()), abs(long_vals.max()))
pad_x = x_max * 0.15 if x_max > 0 else 1.0
pad_y = y_max * 0.15 if y_max > 0 else 1.0

ax_lead.set_xlim(-x_max - pad_x, x_max + pad_x)
ax_lead.set_ylim(-y_max - pad_y, y_max + pad_y)

x_min, x_max_lim = ax_lead.get_xlim()
y_min, y_max_lim = ax_lead.get_ylim()

# Red-yellow-green spectrum:
#   bottom-left (laggards)  -> red
#   left-top & right-bottom -> yellow / amber (neutral / rotational)
#   top-right (leaders)     -> green
ax_lead.fill_between([0, x_max_lim], 0, y_max_lim, color="#e1f5e0", alpha=0.55)   # green
ax_lead.fill_between([x_min, 0], 0, y_max_lim, color="#fff9c4", alpha=0.55)       # yellow
ax_lead.fill_between([x_min, 0], y_min, 0, color="#fde0dc", alpha=0.55)           # red
ax_lead.fill_between([0, x_max_lim], y_min, 0, color="#ffe9b3", alpha=0.55)       # amber

ax_lead.axvline(0, color="#888888", linewidth=1)
ax_lead.axhline(0, color="#888888", linewidth=1)

ax_lead.text(
    x_max_lim * 0.65,
    y_max_lim * 0.75,
    "Short ↑ / Long ↑\nEstablished leaders",
    fontsize=9,
    ha="center",
    va="center",
    color="#333333",
)
ax_lead.text(
    x_min * 0.65,
    y_max_lim * 0.75,
    "Short ↓ / Long ↑\nMean reversion",
    fontsize=9,
    ha="center",
    va="center",
    color="#333333",
)
ax_lead.text(
    x_min * 0.65,
    y_min * 0.75,
    "Short ↓ / Long ↓\nPersistent laggards",
    fontsize=9,
    ha="center",
    va="center",
    color="#333333",
)
ax_lead.text(
    x_max_lim * 0.65,
    y_min * 0.75,
    "Short ↑ / Long ↓\nNew rotations",
    fontsize=9,
    ha="center",
    va="center",
    color="#333333",
)

for i, factor in enumerate(mom_df.index):
    x = short_vals.loc[factor]
    y = long_vals.loc[factor]
    ax_lead.scatter(
        x,
        y,
        s=70,
        color=PASTELS[i % len(PASTELS)],
        edgecolor="#444444",
        linewidth=0.6,
        zorder=3,
    )
    ax_lead.annotate(
        factor,
        xy=(x, y),
        xytext=(4, 3),
        textcoords="offset points",
        fontsize=9,
        va="center",
        color="#111111",
    )

ax_lead.set_xlabel("Short window return %", color=TEXT)
ax_lead.set_ylabel("Long window return %", color=TEXT)
ax_lead.set_title("Factors by Short vs Long Momentum", color=TEXT, pad=10)
ax_lead.grid(color=GRID, linewidth=0.6, alpha=0.6)
fig_lead.tight_layout()
st.pyplot(fig_lead, clear_figure=True)

# ---------------- Factor crowding & correlation ----------------
st.subheader("Factor Crowding and Diversifiers")

if corr_matrix.empty or corr_matrix.shape[0] < 2:
    st.info("Not enough data to compute factor correlations.")
else:
    crowd_rows = []
    for f in corr_matrix.columns:
        peers = corr_matrix.loc[f].drop(f)
        if peers.empty:
            continue
        avg_abs = float(peers.abs().mean())
        max_corr = float(peers.max())
        min_corr = float(peers.min())
        top_partner = peers.abs().idxmax()
        crowd_rows.append(
            {
                "Factor": f,
                "Avg |corr| to others": avg_abs,
                "Max corr partner": top_partner,
                "Max corr": max_corr,
                "Min corr": min_corr,
            }
        )

    crowd_df = pd.DataFrame(crowd_rows)
    if not crowd_df.empty:
        crowd_df = crowd_df.sort_values("Avg |corr| to others", ascending=False)

        fig_crowd, ax_crowd = plt.subplots(figsize=(8, 4.5))
        ax_crowd.barh(
            crowd_df["Factor"],
            crowd_df["Avg |corr| to others"],
            color="#A8DADC",
        )
        ax_crowd.invert_yaxis()
        ax_crowd.set_xlabel("Average |correlation| vs other factors", color=TEXT)
        ax_crowd.set_title("Which styles are most crowded?", color=TEXT, pad=8)
        ax_crowd.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.7)
        fig_crowd.tight_layout()
        st.pyplot(fig_crowd, clear_figure=True)

        display_crowd = crowd_df.copy()
        display_crowd["Avg |corr| to others"] = display_crowd["Avg |corr| to others"].map(
            lambda x: f"{x:.2f}"
        )
        display_crowd["Max corr"] = display_crowd["Max corr"].map(lambda x: f"{x:.2f}")
        display_crowd["Min corr"] = display_crowd["Min corr"].map(lambda x: f"{x:.2f}")

        st.dataframe(
            display_crowd,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No valid correlation pairs to show.")

st.caption("© 2025 AD Fund Management LP")
