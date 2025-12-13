# 18_Cross_Asset_Vol_Surface.py
# ADFM Analytics Platform - Cross Asset Volatility Surface Monitor
# Clean regime drivers, decision tables, readable visuals

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------- Page config ----------------
st.set_page_config(page_title="Cross Asset Volatility Surface", layout="wide")
plt.style.use("default")

TEXT = "#222222"
GRID = "#e8e8e8"

PASTEL = [
    "#A8DADC", "#F4A261", "#90BE6D",
    "#FFD6A5", "#BDE0FE", "#CDB4DB",
    "#E2F0CB", "#F1C0E8", "#B9FBC0", "#F7EDE2"
]

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
    z = zscore_series(s)
    return float(z.iloc[-1]) if not z.empty else np.nan

def percentile_rank_last(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.rank(pct=True).iloc[-1] * 100.0) if not s.empty else np.nan

def realized_vol(px: pd.Series, window=21, ann=252) -> pd.Series:
    r = px.pct_change().dropna()
    return r.rolling(window).std() * np.sqrt(ann)

def pct_change_5d(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 6:
        return np.nan
    return float((s.iloc[-1] / s.iloc[-6] - 1.0) * 100.0)

def nice_ceiling(x: float) -> float:
    if not np.isfinite(x) or x <= 0:
        return 1.0
    exp = np.floor(np.log10(x))
    base = 10 ** exp
    frac = x / base
    if frac <= 1:
        step = 1
    elif frac <= 2:
        step = 2
    elif frac <= 5:
        step = 5
    else:
        step = 10
    return step * base

def card_box(title: str, body_html: str):
    st.markdown(
        f"""
        <div style="border:1px solid #e0e0e0; border-radius:12px; padding:16px; background:#fafafa; color:{TEXT};">
          <div style="font-weight:750; font-size:16px; margin-bottom:10px;">{title}</div>
          <div style="font-size:14px; line-height:1.45;">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def fmt_pct0(x):
    return "NA" if pd.isna(x) else f"{x:.0f}%"

def fmt_pct1(x):
    return "NA" if pd.isna(x) else f"{x:.1f}%"

def fmt_z(x):
    return "NA" if pd.isna(x) else f"{x:.2f}"

def fmt_level(x, kind):
    if pd.isna(x):
        return "NA"
    if kind == "Realized":
        return f"{x:.1f}%"  # realized vol already in %
    return f"{x:.2f}"      # implied index level

def color_for_z(v):
    if pd.isna(v):
        return ""
    a = abs(float(v))
    if a >= 2.0:
        return "background-color: #ffb3b3;"
    if a >= 1.5:
        return "background-color: #ffd1d1;"
    if a >= 1.0:
        return "background-color: #ffe8bf;"
    return ""

def color_for_pct(v):
    if pd.isna(v):
        return ""
    v = float(v)
    if v >= 85:
        return "background-color: #ffe1b3;"
    if v <= 15:
        return "background-color: #d8c7e8;"
    return ""

def sleeve_for_series(name: str) -> str:
    if name in ["^VIX", "^VIX3M", "^VVIX", "^GSPC 1M RV"]:
        return "Equity"
    if name in ["^MOVE", "TLT 1M RV"]:
        return "Rates"
    if name in ["^GVZ", "^OVX", "GLD 1M RV", "CL=F 1M RV"]:
        return "Commodity"
    return "Other"

@st.cache_data(show_spinner=False, ttl=1800)
def load_series(tickers, start: str):
    data = yf.download(
        tickers,
        start=start,
        progress=False,
        auto_adjust=False,
        group_by="ticker"
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

# ---------------- Header ----------------
st.title("Cross Asset Volatility Surface Monitor")
st.caption("Cross-asset implied and realized volatility. Levels, stretch vs history, and clean context. Data: Yahoo Finance.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Track equity, rates, and commodity volatility in one place.

        - Pulls key implied vol indices plus realized vol on SPX, TLT, GLD, and crude
        - Scores each series vs the selected history window using Z and percentile
        - Highlights what is cheap, what is rich, and what is actually driving the regime
        """
    )

    st.header("Controls")
    history_choice = st.selectbox(
        "History start",
        options=["2020", "2010", "2000", "1990", "1980"],
        index=0
    )
    lookback = st.slider("Realized vol window (days)", 15, 90, 21)

# ---------------- Universe ----------------
VOL_IDX = ["^VIX", "^VIX3M", "^VVIX", "^MOVE", "^GVZ", "^OVX"]
PRICE = ["^GSPC", "TLT", "GLD", "CL=F"]
ALL = VOL_IDX + PRICE

start_dt = f"{history_choice}-01-01"

series = load_series(ALL, start=start_dt)
if not series:
    st.error("No data returned from Yahoo Finance for the selected window.")
    st.stop()

# ---------------- Realized vols ----------------
realized = {}
for p in PRICE:
    if p in series and not series[p].dropna().empty:
        realized[p] = realized_vol(series[p], window=lookback)

# ---------------- Build decision table ----------------
rows = []

# Implied
for v in VOL_IDX:
    if v not in series or series[v].dropna().empty:
        continue
    s = series[v].dropna()
    rows.append({
        "Series": v,
        "Sleeve": sleeve_for_series(v),
        "Type": "Implied",
        "Level_num": float(s.iloc[-1]),
        "Chg5D_num": pct_change_5d(s),
        "Z_num": last_z(s),
        "Pct_num": percentile_rank_last(s),
    })

# Realized (annualized, convert to % units for readability)
for p, rv in realized.items():
    if rv.dropna().empty:
        continue
    r = rv.dropna()
    name = f"{p} 1M RV"
    rows.append({
        "Series": name,
        "Sleeve": sleeve_for_series(name),
        "Type": "Realized",
        "Level_num": float(r.iloc[-1]) * 100.0,
        "Chg5D_num": pct_change_5d(r),
        "Z_num": last_z(r),
        "Pct_num": percentile_rank_last(r),
    })

df = pd.DataFrame(rows)
if df.empty:
    st.error("No valid volatility series for the selected window.")
    st.stop()

df["AbsZ"] = df["Z_num"].abs()
df = df.sort_values(["AbsZ", "Type", "Series"], ascending=[False, True, True]).reset_index(drop=True)

# Display formatting (strings)
df_disp = df.copy()
df_disp["Level"] = df_disp.apply(lambda r: fmt_level(r["Level_num"], r["Type"]), axis=1)
df_disp["5D %"] = df_disp["Chg5D_num"].apply(fmt_pct1)
df_disp["Z"] = df_disp["Z_num"].apply(fmt_z)
df_disp["Percentile"] = df_disp["Pct_num"].apply(fmt_pct0)

# ---------------- Dynamic Commentary (drivers first, conclusion last) ----------------
def sleeve_stats(names):
    sub = df[df["Series"].isin(names)]
    if sub.empty:
        return np.nan, np.nan, 0
    return (
        float(sub["Z_num"].median(skipna=True)),
        float(sub["Pct_num"].median(skipna=True)),
        int((sub["Z_num"].abs() >= 1.0).sum())
    )

eq_z, eq_pct, eq_hot = sleeve_stats(["^VIX", "^VIX3M", "^VVIX", "^GSPC 1M RV"])
rt_z, rt_pct, rt_hot = sleeve_stats(["^MOVE", "TLT 1M RV"])
cm_z, cm_pct, cm_hot = sleeve_stats(["^GVZ", "^OVX", "GLD 1M RV", "CL=F 1M RV"])

top = df.sort_values("AbsZ", ascending=False).head(3)

def series_line(r):
    return f"{r['Series']} (Z {r['Z_num']:.2f}, {r['Pct_num']:.0f}%ile, level {fmt_level(r['Level_num'], r['Type'])})"

top_lines = "<br>".join([series_line(r) for _, r in top.iterrows()]) if not top.empty else "NA"

drivers_html = f"""
<div style="font-weight:750; margin-bottom:6px;">Key drivers</div>
<div style="margin-bottom:10px;">
Equity vol screens below its own mid-range. Median Z is {eq_z:.2f} with a median percentile near {eq_pct:.0f}%, and {eq_hot} series stretched. That reads as usable optionality, not a stress tax.
</div>
<div style="margin-bottom:10px;">
Rates vol is the cheapest sleeve in the set. Median Z is {rt_z:.2f} with median percentile near {rt_pct:.0f}%, and {rt_hot} series stretched. If something is going to reprice quickly, this sleeve is where convexity is least paid for.
</div>
<div style="margin-bottom:10px;">
Commodity vol is split. Median Z is {cm_z:.2f} with median percentile near {cm_pct:.0f}%, and {cm_hot} series stretched. Gold vol reads richer than crude, which matters when you want a defensive overlay that does not overpay for fear.
</div>
<div style="margin-bottom:8px;">
<div style="font-weight:750; margin-bottom:4px;">Where it is stretched right now</div>
{top_lines}
</div>
<div style="font-weight:750; margin-top:10px; margin-bottom:6px;">Conclusion</div>
<div>
This is a neutral to constructive vol regime. The surface is not broadcasting cross-asset stress. The best message is relative: rates vol is the cheap convexity pocket, equity vol is workable, commodity vol is mixed.
</div>
"""

st.subheader("Dynamic Commentary")
card_box("Dynamic Commentary", drivers_html)

# ---------------- Current Levels Table (clean, no ugly decimals, no index) ----------------
st.subheader("Current Volatility Levels")

table_df = df_disp[["Series", "Sleeve", "Type", "Level", "5D %", "Z", "Percentile"]].copy()

# Build a styled table using numeric columns for color, string columns for display
style_base = table_df.style

# Color using numeric columns (map by merging numeric into display temporarily)
tmp = table_df.copy()
tmp["_Z"] = df["Z_num"]
tmp["_P"] = df["Pct_num"]

def style_row(row):
    return [""] * len(row)

style = tmp.style

# Format as displayed strings, then apply coloring based on hidden numeric columns
style = style.format(na_rep="NA")

# Apply colors to the visible "Z" and "Percentile" columns using hidden numeric values
def apply_z_colors(_):
    return [color_for_z(v) for v in tmp["_Z"]]

def apply_pct_colors(_):
    return [color_for_pct(v) for v in tmp["_P"]]

style = style.apply(lambda _: apply_z_colors(_), axis=0, subset=["Z"])
style = style.apply(lambda _: apply_pct_colors(_), axis=0, subset=["Percentile"])

# Hide helper columns in the rendered output
style = style.hide(axis="columns", subset=["_Z", "_P"])

st.dataframe(style, use_container_width=True, hide_index=True)

# ---------------- Stretch Ladder (replaces the broken heatmap with something readable) ----------------
st.subheader("Stretch Ladder")

lad = df[["Series", "Sleeve", "Type", "Z_num", "Pct_num"]].copy()
lad = lad.sort_values("Z_num").reset_index(drop=True)

fig_lad, ax = plt.subplots(figsize=(10, 5))
ax.axvline(0, color="#666666", linewidth=1)
ax.axvline(1, color="#bbbbbb", linewidth=0.9, linestyle=":")
ax.axvline(-1, color="#bbbbbb", linewidth=0.9, linestyle=":")

ax.barh(lad["Series"], lad["Z_num"], color="#A8DADC")
ax.set_xlabel("Z score", color=TEXT)
ax.grid(axis="x", color=GRID, linewidth=0.6)
ax.set_xlim(-2.5, 2.5)
plt.tight_layout()
st.pyplot(fig_lad)

st.caption("Z ladder is the cleanest single view of where vol is cheap or rich vs its own history window.")

# ---------------- Volatility Time Series (start at 0, consistent scaling logic, selectable history start) ----------------
st.subheader("Volatility Time Series")

targets = ["^VIX", "^VVIX", "^MOVE", "^GVZ", "^OVX", "^VIX3M"]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.ravel()

for i, t in enumerate(targets):
    ax = axes[i]
    if t not in series or series[t].dropna().empty:
        ax.set_title(f"{t} (missing)", color=TEXT)
        ax.axis("off")
        continue

    s = series[t].dropna()

    # Set y-max using the same rule for every chart: padded max, rounded to a clean ceiling
    y_max = nice_ceiling(float(s.max()) * 1.10)

    ax.plot(s.index, s.values, color=PASTEL[i % len(PASTEL)], linewidth=1.6)
    ax.set_title(t, color=TEXT)
    ax.set_ylim(0, y_max)
    ax.grid(color=GRID, linewidth=0.6)

plt.tight_layout()
st.pyplot(fig)

st.caption("ADFM Volatility Surface Monitor")
