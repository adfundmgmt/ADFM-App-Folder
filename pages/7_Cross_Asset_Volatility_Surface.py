# cross_asset_vol_surface.py
# ADFM Analytics Platform | Cross Asset Volatility Surface Monitor

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
import streamlit as st

# ---------------- Config ----------------
st.set_page_config(page_title="Cross Asset Volatility Surface", layout="wide")
plt.style.use("default")

PASTEL = [
    "#A8DADC", "#F4A261", "#90BE6D", "#FFD6A5", "#BDE0FE",
    "#CDB4DB", "#E2F0CB", "#F1C0E8", "#B9FBC0", "#F7EDE2"
]
GRID = "#e8e8e8"
TEXT = "#222222"

def card_box(html_body: str):
    st.markdown(
        f"""
        <div style="
            border:1px solid #e0e0e0;
            border-radius:10px;
            padding:14px;
            background:#fafafa;
            color:{TEXT};
            font-size:14px;
            line-height:1.35;
        ">
        {html_body}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- Helpers ----------------
def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return s
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd

def realized_vol(series: pd.Series, window=21, ann=252) -> pd.Series:
    r = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return pd.Series(dtype=float, index=series.index)
    rv = r.rolling(window).std(ddof=0) * np.sqrt(ann)
    return rv

def fetch_history(tickers, start="2010-01-01"):
    raw = yf.download(
        tickers=tickers,
        start=start,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    out = {}
    if isinstance(raw, pd.DataFrame) and not isinstance(raw.columns, pd.MultiIndex):
        # single ticker case
        col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        out[tickers[0]] = raw[col].dropna()
        return out

    for t in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if t not in raw.columns.get_level_values(0):
                continue
            df = raw[t]
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            out[t] = df[col].dropna()
        else:
            if t not in raw.columns:
                continue
            col = "Adj Close" if "Adj Close" in raw.columns else "Close"
            out[t] = raw[col].dropna()
    return out

def percentile_rank(val: float, s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty or pd.isna(val):
        return np.nan
    return float((s <= val).mean() * 100.0)

def pastel_cmap():
    return LinearSegmentedColormap.from_list(
        "pastel_scale",
        ["#F1FAEE", "#A8DADC", "#457B9D"],
        N=256
    )

def classify_z(z: float) -> str:
    if pd.isna(z):
        return "NA"
    if z >= 2.0:
        return "Extreme"
    if z >= 1.0:
        return "Hot"
    if z <= -0.7:
        return "Cheap"
    return "Normal"

# ---------------- Sidebar ----------------
st.title("Cross Asset Volatility Surface Monitor")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Objective**

        Cross asset volatility dashboard that tells you:
        - Where vol is rich or cheap versus its own history  
        - How equity, rates, and commodity vol composites look right now  
        - Whether the VIX term structure is flashing stress or calm  

        **Design**

        - Daily closes from Yahoo Finance  
        - 1M rolling window for realized vol on the underlyings  
        - Z scores and percentile ranks versus full history since your start date  
        - Simple traffic light regime tags for each vol series and asset class  

        **Use cases**

        - Decide how much gross and net to run  
        - Choose between buying convexity or overwriting  
        - Sanity check whether moves are local noise or real regime shifts
        """
    )
    st.divider()
    st.header("Settings")
    start_date = st.date_input("History start", datetime(2010, 1, 1))
    lookback = st.slider("Realized vol window (days)", 15, 90, 21)
    show_z = st.checkbox("Overlay Z score lines on charts", value=True)
    st.caption("Data: Yahoo Finance. Signals are historical, not forward looking.")

# ---------------- Data universe ----------------
VOL_IDX = [
    "^VIX",      # S&P implied vol
    "^VIX3M",    # 3M implied vol
    "^VVIX",     # Vol of vol
    "^MOVE",     # Treasury vol
    "^GVZ",      # Gold vol
    "^OVX",      # Oil vol
]
PRICE_UNDERLYING = ["^GSPC", "TLT", "GLD", "CL=F"]

# ---------------- Load data ----------------
all_ticks = list(dict.fromkeys(VOL_IDX + PRICE_UNDERLYING))
series = fetch_history(all_ticks, start=str(start_date))

# realized vol on underlyings
realized = {}
for p in PRICE_UNDERLYING:
    s = series.get(p)
    if s is None or s.empty:
        continue
    realized[p] = realized_vol(s, window=lookback)

# ---------------- Build current levels table ----------------
rows = []
for idx in VOL_IDX:
    s = series.get(idx)
    if s is None or s.empty:
        continue
    lvl = float(s.iloc[-1])
    zs = zscore_series(s)
    z_now = float(zs.iloc[-1]) if not zs.empty else np.nan
    pct = percentile_rank(lvl, s)
    regime = classify_z(z_now)
    rows.append([
        idx,
        round(lvl, 2),
        round(z_now, 2) if show_z and not pd.isna(z_now) else "",
        pct,
        regime,
        "Index"
    ])

for p in PRICE_UNDERLYING:
    rv = realized.get(p)
    if rv is None or rv.empty:
        continue
    lvl = float(rv.iloc[-1])
    zs = zscore_series(rv)
    z_now = float(zs.iloc[-1]) if not zs.empty else np.nan
    pct = percentile_rank(lvl, rv)
    regime = classify_z(z_now)
    rows.append([
        f"{p} 1M RV",
        round(lvl, 2),
        round(z_now, 2) if show_z and not pd.isna(z_now) else "",
        pct,
        regime,
        "Realized"
    ])

df = pd.DataFrame(
    rows,
    columns=["Series", "Level", "Z", "Percentile", "Regime", "Type"]
).sort_values(["Type", "Series"]).reset_index(drop=True)

# ---------------- Composite asset class scores ----------------
def composite_z(names):
    vals = []
    for n in names:
        if "RV" in n:
            base = n.replace(" 1M RV", "")
            s = realized.get(base)
        else:
            s = series.get(n)
        if s is None or s.empty:
            continue
        zs = zscore_series(s)
        if zs.empty:
            continue
        vals.append(float(zs.iloc[-1]))
    if not vals:
        return np.nan
    return float(np.mean(vals))

equity_z = composite_z(["^VIX", "^VIX3M", "^GSPC 1M RV"])
rates_z = composite_z(["^MOVE", "TLT 1M RV"])
commod_z = composite_z(["^GVZ", "^OVX", "GLD 1M RV", "CL=F 1M RV"])

def composite_label(z):
    r = classify_z(z)
    if r == "Extreme":
        return "High stress"
    if r == "Hot":
        return "Elevated"
    if r == "Cheap":
        return "Vol cheap"
    if r == "Normal":
        return "Normal"
    return "NA"

comp_labels = {
    "Equity vol": composite_label(equity_z),
    "Rates vol": composite_label(rates_z),
    "Commodity vol": composite_label(commod_z),
}

# VIX term structure
v = series.get("^VIX")
v3 = series.get("^VIX3M")
ts_lvl = None
ts_tag = "NA"
if v is not None and not v.empty and v3 is not None and not v3.empty:
    last_v = float(v.iloc[-1])
    last_v3 = float(v3.iloc[-1])
    if last_v > 0:
        ts_lvl = last_v3 / last_v - 1.0
        if ts_lvl < 0:
            ts_tag = "Backwardation"
        elif ts_lvl > 0.1:
            ts_tag = "Steep contango"
        else:
            ts_tag = "Mild contango"

# ---------------- Decision box ----------------
st.subheader("Volatility Regime Read")

lines = []
lines.append(
    f"Equity vol composite z {equity_z:+.2f} ({comp_labels['Equity vol']}). "
    "Think about gross and hedging in SPX and single names through this lens."
    if not pd.isna(equity_z) else
    "Equity vol composite unavailable under current settings."
)
lines.append(
    f"Rates vol composite z {rates_z:+.2f} ({comp_labels['Rates vol']}). "
    "Higher readings argue for tighter duration risk and cleaner hedges in futures or options."
    if not pd.isna(rates_z) else
    "Rates vol composite unavailable under current settings."
)
lines.append(
    f"Commodity vol composite z {commod_z:+.2f} ({comp_labels['Commodity vol']}). "
    "Use this as a sanity check on energy and gold exposure sizing."
    if not pd.isna(commod_z) else
    "Commodity vol composite unavailable under current settings."
)

if ts_lvl is not None:
    lines.append(
        f"VIX term structure {ts_tag.lower()}, 3M vs 1M at {ts_lvl:+.2%}. "
        "Backwardation flags gap risk and usually argues for lower gross and more convexity. "
        "Steep contango makes systematic short vol carry more attractive but still needs tape confirmation."
    )

body = (
    '<div style="font-weight:700; margin-bottom:6px;">Conclusion</div>'
    f'<div>{" ".join(lines)}</div>'
)

drivers = []
if not pd.isna(equity_z):
    if equity_z >= 1.0:
        drivers.append("Equity vol rich to history. Respect drawdowns, prefer spreads and defined risk rather than outright naked short options.")
    elif equity_z <= -0.7:
        drivers.append("Equity vol cheap. Overwriting and call spreads against core longs make sense if macro tape is stable.")
if not pd.isna(rates_z):
    if rates_z >= 1.0:
        drivers.append("Rates vol hot. Duration moves can dominate equity factor behavior. Keep an eye on SPX vs 10Y correlation from the Cross Asset Compass.")
    elif rates_z <= -0.7:
        drivers.append("Rates vol muted. Macro shock probability priced as low, which often coincides with grindy trend markets.")
if not pd.isna(commod_z):
    if commod_z >= 1.0:
        drivers.append("Commodity vol elevated. Reduce leverage in energy or gold beta trades and lean more on options.")
    elif commod_z <= -0.7:
        drivers.append("Commodity vol subdued. Directional futures or ETF positions have less convexity cost but can lull you into ignoring supply shocks.")

if ts_lvl is not None:
    if ts_lvl < 0:
        drivers.append("Backwardation in VIX curve historically clusters around stress windows. Do not run maximum gross here without strong idiosyncratic edges.")
    elif ts_lvl > 0.1:
        drivers.append("Contango and low VIX z scores usually mean carry trades in vol work until a catalyst shows up.")

if drivers:
    body += (
        '<div style="font-weight:700; margin:10px 0 6px;">Why it matters</div>'
        '<ul style="margin-top:4px; margin-bottom:4px;">'
        + "".join(f"<li>{d}</li>" for d in drivers) +
        "</ul>"
    )

card_box(body)

# ---------------- Current levels table ----------------
st.subheader("Current Volatility Levels")
if df.empty:
    st.info("No volatility data available for the selected history window.")
else:
    show = df.copy()
    show["Percentile"] = show["Percentile"].apply(lambda x: "" if pd.isna(x) else f"{x:.0f}%")
    st.dataframe(show, use_container_width=True)

# ---------------- VIX term structure figure ----------------
st.subheader("VIX Term Structure")
def plot_term_structure(last_v, last_v3):
    fig, ax = plt.subplots(figsize=(7, 3))
    points = [0.0, 0.25]
    vals = [last_v, last_v3]
    ax.plot(points, vals, marker="o", linewidth=2, color=PASTEL[1])
    ax.set_xticks(points)
    ax.set_xticklabels(["1M", "3M"])
    ax.set_title("VIX Term Structure", color=TEXT)
    ax.grid(color=GRID)
    st.pyplot(fig, clear_figure=True)

if v is not None and not v.empty and v3 is not None and not v3.empty:
    plot_term_structure(float(v.iloc[-1]), float(v3.iloc[-1]))
else:
    st.info("VIX or VIX3M history missing for this window.")

# ---------------- Time series panels ----------------
st.subheader("Volatility Time Series")

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.ravel()

targets = ["^VIX", "^VVIX", "^MOVE", "^GVZ", "^OVX", "^VIX3M"]

for i, t in enumerate(targets):
    ax = axes[i]
    s = series.get(t)
    if s is None or s.empty:
        ax.set_title(f"{t} missing")
        ax.axis("off")
        continue

    ax.plot(s.index, s.values, color=PASTEL[i % len(PASTEL)], linewidth=2)
    ax.set_title(t, color=TEXT)
    ax.grid(color=GRID, linewidth=0.6)

    if show_z:
        zs = zscore_series(s)
        if not zs.empty:
            ax2 = ax.twinx()
            ax2.plot(zs.index, zs.values, color="#999999", linewidth=1, alpha=0.5)
            ax2.axhline(0, color="#999999", linewidth=0.8)
            ax2.set_yticks([])

plt.tight_layout()
st.pyplot(fig, clear_figure=True)

# ---------------- Shock map ----------------
st.subheader("Cross Asset Vol Shock Map")

def daily_change_map(series_dict):
    names = []
    vals = []
    for k, v in series_dict.items():
        if len(v) < 2 or k not in VOL_IDX:
            continue
        ch = (v.iloc[-1] - v.iloc[-2]) / v.iloc[-2]
        names.append(k)
        vals.append(ch)
    return pd.DataFrame({"Series": names, "DailyChange": vals})

heat = daily_change_map(series).dropna()
if heat.empty:
    st.info("Not enough history to compute daily shock map.")
else:
    heat = heat.set_index("Series")
    fig_h, ax_h = plt.subplots(figsize=(6, 3))
    cmap = pastel_cmap()
    im = ax_h.imshow(heat.values, cmap=cmap, aspect="auto")
    ax_h.set_yticks(range(len(heat.index)))
    ax_h.set_yticklabels(heat.index)
    ax_h.set_xticks([0])
    ax_h.set_xticklabels(["Daily %"])
    plt.colorbar(im, ax=ax_h)
    st.pyplot(fig_h, clear_figure=True)

    # Text summary of largest shock
    max_row = heat["DailyChange"].abs().idxmax()
    val = float(heat.loc[max_row, "DailyChange"])
    s_full = series.get(max_row)
    pct_rank = percentile_rank(val, s_full.pct_change()) if s_full is not None else np.nan
    shock_text = (
        f"{max_row} had the largest one day move at {val:+.1%}, roughly {pct_rank:.0f}th percentile "
        "of its own daily history. Treat that as a real information event rather than noise."
        if not pd.isna(pct_rank) else
        f"{max_row} had the largest one day move at {val:+.1%}."
    )
    card_box(f"<b>Shock read</b><br>{shock_text}")

st.caption("ADFM Volatility Surface Monitor")
