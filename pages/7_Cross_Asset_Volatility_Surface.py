import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

# ---------------- Config ----------------
st.set_page_config(page_title="Cross Asset Volatility Surface", layout="wide")
plt.style.use("default")

PASTEL = [
    "#A8DADC", "#F4A261", "#90BE6D", "#FFD6A5", "#BDE0FE",
    "#CDB4DB", "#E2F0CB", "#F1C0E8", "#B9FBC0", "#F7EDE2"
]

GRID = "#e8e8e8"
TEXT = "#222222"


# ---------------- Helpers ----------------
def zscore_series(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s * np.nan
    mu = s.mean()
    sig = s.std(ddof=0)
    if sig == 0 or np.isnan(sig):
        return s * np.nan
    return (s - mu) / sig


def last_z(s: pd.Series) -> float:
    if s is None or s.dropna().empty:
        return np.nan
    z = zscore_series(s)
    return float(z.iloc[-1]) if not z.empty else np.nan


def percentile_rank_last(s: pd.Series) -> float:
    s = s.dropna()
    if s.empty:
        return np.nan
    return float(s.rank(pct=True).iloc[-1] * 100.0)


def realized_vol(series, window=21, ann=252):
    r = series.pct_change().dropna()
    return r.rolling(window).std() * np.sqrt(ann)


@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_series(tickers, start="2010-01-01"):
    data = yf.download(
        tickers, start=start, progress=False, auto_adjust=False, group_by="ticker"
    )
    out = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data.columns.get_level_values(0):
                df_t = data[t]
                if "Close" in df_t.columns:
                    out[t] = df_t["Close"].dropna()
    else:
        if "Close" in data.columns and len(tickers) == 1:
            out[tickers[0]] = data["Close"].dropna()
    return out


def pastel_cmap():
    return LinearSegmentedColormap.from_list(
        "pastel_scale",
        ["#F1FAEE", "#A8DADC", "#457B9D"],
        N=256
    )


def card_box(inner_html: str):
    st.markdown(
        f"""
        <div style="border:1px solid #e0e0e0; border-radius:10px; padding:14px; background:#fafafa; color:{TEXT}; font-size:14px; line-height:1.35;">
          {inner_html}
        </div>
        """.strip(),
        unsafe_allow_html=True,
    )


def fmt_pct(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.0f}%"


def fmt_z(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.2f}"


# ---------------- Sidebar ----------------
st.title("Cross Asset Volatility Surface Monitor")
st.caption("Regime view for equity, rates, and commodity volatility. Data: Yahoo Finance.")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose**

        Track cross asset implied and realized volatility in one place
        and map it to a simple regime read that can steer gross and hedging.

        **What it does**

        - Pulls daily history for key vol indices and underlyings  
        - Computes Z scores and percentile ranks vs the selected history window  
        - Flags when equity, rates, and commodity vol are simultaneously hot or cold  
        - Gives a compact time series grid for context

        **How to use**

        - Use the regime header to decide if vol is a tailwind or headwind for risk  
        - Use the levels table to spot where hedges are rich or cheap vs history  
        - Use the Z bar chart to see which sleeve is driving the regime
        """
    )

    st.header("Controls")
    start_date = st.date_input("History start", datetime(2010, 1, 1))
    lookback = st.slider("Realized vol window (days)", 15, 90, 21)
    show_z = st.checkbox("Overlay Z scores on charts", value=True)
    st.caption("Research tool only. No guarantee of data completeness.")

# ---------------- Data universe ----------------
VOL_IDX = [
    "^VIX",      # S&P implied vol
    "^VIX3M",    # 3M implied vol
    "^VVIX",     # Vol of vol
    "^MOVE",     # Treasury vol
    "^GVZ",      # Gold vol
    "^OVX",      # Oil vol
]

PRICE_UNDERLYING = [
    "^GSPC",
    "TLT",
    "GLD",
    "CL=F"
]

ALL = VOL_IDX + PRICE_UNDERLYING

# ---------------- Load data ----------------
series = load_series(ALL, start=str(start_date))
if not series:
    st.error("No data returned from Yahoo Finance for the selected window.")
    st.stop()

# ---------------- Realized vol ----------------
realized = {}
for p in PRICE_UNDERLYING:
    if p in series:
        realized[p] = realized_vol(series[p], window=lookback)

# ---------------- Build table ----------------
rows = []

for idx in VOL_IDX:
    if idx not in series:
        continue
    s = series[idx]
    if s.dropna().empty:
        continue
    lvl = float(s.iloc[-1])
    z = last_z(s)
    pct_rank = percentile_rank_last(s)
    ch_5d = np.nan
    if len(s) > 5:
        ch_5d = float((s.iloc[-1] / s.iloc[-6] - 1.0) * 100.0)

    rows.append(
        {
            "Series": idx,
            "Level": round(lvl, 2),
            "5D %": round(ch_5d, 1) if not np.isnan(ch_5d) else np.nan,
            "Z": round(z, 2) if not np.isnan(z) else np.nan,
            "Percentile": round(pct_rank, 0) if not np.isnan(pct_rank) else np.nan,
            "Type": "Implied",
        }
    )

for p in PRICE_UNDERLYING:
    if p not in realized:
        continue
    rv = realized[p]
    if rv.dropna().empty:
        continue
    lvl = float(rv.iloc[-1])
    z = last_z(rv)
    pct_rank = percentile_rank_last(rv)
    ch_5d = np.nan
    if len(rv) > 5:
        ch_5d = float((rv.iloc[-1] / rv.iloc[-6] - 1.0) * 100.0)

    rows.append(
        {
            "Series": f"{p} 1M RV",
            "Level": round(lvl, 2),
            "5D %": round(ch_5d, 1) if not np.isnan(ch_5d) else np.nan,
            "Z": round(z, 2) if not np.isnan(z) else np.nan,
            "Percentile": round(pct_rank, 0) if not np.isnan(pct_rank) else np.nan,
            "Type": "Realized",
        }
    )

df = pd.DataFrame(rows)
df = df.sort_values(["Type", "Series"]).reset_index(drop=True)

if df.empty:
    st.error("No valid volatility series for the selected window.")
    st.stop()

# ---------------- Regime header ----------------
vix_row = df[df["Series"] == "^VIX"]
move_row = df[df["Series"] == "^MOVE"]

vix_z = float(vix_row["Z"].iloc[0]) if not vix_row.empty and not pd.isna(vix_row["Z"].iloc[0]) else np.nan
move_z = float(move_row["Z"].iloc[0]) if not move_row.empty and not pd.isna(move_row["Z"].iloc[0]) else np.nan

stress_score = 0
for z in [vix_z, move_z]:
    if not np.isnan(z) and z >= 1.0:
        stress_score += 1
    if not np.isnan(z) and z >= 2.0:
        stress_score += 1

if stress_score >= 3:
    regime = "Risk off"
elif stress_score == 2:
    regime = "Cautious"
else:
    regime = "Neutral to constructive"

hot_count = df[(df["Z"].abs() >= 1.0)].shape[0]

c1, c2, c3 = st.columns(3)
c1.metric("Regime", regime)
c2.metric("VIX Z", fmt_z(vix_z))
c3.metric("Series |Z| ≥ 1", int(hot_count))

# ---------------- Dynamic Commentary ----------------
def sleeve_stats(df_in: pd.DataFrame, names):
    sub = df_in[df_in["Series"].isin(names)]
    if sub.empty:
        return np.nan, np.nan, 0
    med_z = sub["Z"].median(skipna=True)
    med_pct = sub["Percentile"].median(skipna=True)
    hot = sub[sub["Z"].abs() >= 1.0].shape[0]
    return med_z, med_pct, hot


eq_names = ["^VIX", "^VIX3M", "^VVIX"]
rt_names = ["^MOVE", "TLT 1M RV"]
cmd_names = ["^GVZ", "^OVX", "GLD 1M RV", "CL=F 1M RV"]

eq_z, eq_pct, eq_hot = sleeve_stats(df, eq_names)
rt_z, rt_pct, rt_hot = sleeve_stats(df, rt_names)
cmd_z, cmd_pct, cmd_hot = sleeve_stats(df, cmd_names)

top_outliers = (
    df.dropna(subset=["Z"])
    .assign(abs_z=lambda x: x["Z"].abs())
    .sort_values("abs_z", ascending=False)
    .head(3)
)

if not df.empty:
    if regime == "Risk off":
        conclusion = (
            "Vol regime is hostile to net risk. Equity and rates volatility sit at elevated Z scores, "
            "which argues for tighter gross and cleaner hedges rather than adding exposure."
        )
    elif regime == "Cautious":
        conclusion = (
            "Vol regime is cautious. One sleeve is stretched while the other is closer to its median range, "
            "so sizing and hedge expression matter more than direction."
        )
    else:
        conclusion = (
            "Vol regime is neutral to constructive. Neither equity nor rates volatility is screaming, "
            "which keeps optionality cheap enough to add selectively instead of defensively."
        )

    why = (
        "The table and Z bar tell you where optionality is rich or cheap versus its own history across equity, "
        "rates, and commodity sleeves. That should inform where to source hedges and where to lean on outright risk."
    )

    drivers = []

    drivers.append(
        f"Equity vol sleeve (VIX, VIX3M, VVIX) sits at median Z {fmt_z(eq_z)} and roughly {fmt_pct(eq_pct)} percentile, "
        f"with {eq_hot} series showing |Z| ≥ 1."
    )

    drivers.append(
        f"Rates vol sleeve (MOVE, TLT realized) shows median Z {fmt_z(rt_z)} and about {fmt_pct(rt_pct)} percentile, "
        f"with {rt_hot} hot series. This is the anchor for how aggressive you want to be with duration and curve trades."
    )

    drivers.append(
        f"Commodity vol sleeve (gold and crude implied and realized) has median Z {fmt_z(cmd_z)} and "
        f"around {fmt_pct(cmd_pct)} percentile, with {cmd_hot} names stretched. This flags whether macro hedges "
        f"via gold or energy are rich or still reasonably priced."
    )

    if not top_outliers.empty:
        bullet_lines = []
        for _, r in top_outliers.iterrows():
            bullet_lines.append(
                f"{r['Series']} at level {r['Level']}, Z {fmt_z(r['Z'])}, about {fmt_pct(r['Percentile'])} percentile."
            )
        drivers.append(
            "Top outliers right now: " + " ".join(bullet_lines)
        )

    body = (
        '<div style="font-weight:700; margin-bottom:6px;">Conclusion</div>'
        f'<div>{conclusion}</div>'
        '<div style="font-weight:700; margin:10px 0 6px;">Why it matters</div>'
        f'<div>{why}</div>'
        '<div style="font-weight:700; margin:10px 0 6px;">Key drivers</div>'
        '<ul style="margin-top:4px; margin-bottom:4px;">'
        + "".join(f"<li>{d}</li>" for d in drivers)
        + "</ul>"
    )

    st.subheader("Dynamic Commentary")
    card_box(body)

# ---------------- Levels table ----------------
st.subheader("Current Volatility Levels")

st.dataframe(
    df[["Series", "Type", "Level", "5D %", "Z", "Percentile"]],
    use_container_width=True,
    hide_index=True,
)

# ---------------- Z score bar chart ----------------
st.subheader("Cross Asset Volatility Z Scores")

z_df = df.dropna(subset=["Z"]).copy()
if z_df.empty:
    st.info("No valid Z scores to plot.")
else:
    fig_z, ax_z = plt.subplots(figsize=(8, 4))
    ax_z.barh(z_df["Series"], z_df["Z"], color="#A8DADC")
    ax_z.axvline(0, color="#666666", linewidth=1)
    ax_z.axvline(1, color="#bbbbbb", linewidth=0.8, linestyle=":")
    ax_z.axvline(-1, color="#bbbbbb", linewidth=0.8, linestyle=":")
    ax_z.set_xlabel("Z score", color=TEXT)
    ax_z.grid(axis="x", color=GRID, linewidth=0.6)
    plt.tight_layout()
    st.pyplot(fig_z)

# ---------------- Volatility Time Series grid ----------------
st.subheader("Volatility Time Series")

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.ravel()

targets = [
    "^VIX", "^VVIX", "^MOVE",
    "^GVZ", "^OVX", "^VIX3M",
]

for i, t in enumerate(targets):
    ax = axes[i]
    if t not in series or series[t].dropna().empty:
        ax.set_title(f"{t} missing", color=TEXT)
        ax.axis("off")
        continue

    s = series[t].dropna()
    ax.plot(s.index, s.values, color=PASTEL[i % len(PASTEL)], linewidth=1.6)
    ax.set_title(t, color=TEXT)
    ax.grid(color=GRID, linewidth=0.5)

    if show_z:
        zs = zscore_series(s)
        ax2 = ax.twinx()
        ax2.plot(zs.index, zs.values, color="#777777", linewidth=0.7, alpha=0.6)
        ax2.axhline(0, color="#999999", linewidth=0.8)
        ax2.set_yticks([])

plt.tight_layout()
st.pyplot(fig)

st.caption("ADFM Volatility Surface Monitor")
