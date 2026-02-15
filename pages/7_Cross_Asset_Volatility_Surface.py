import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# ---------------- Config ----------------
st.set_page_config(page_title="Cross Asset Volatility Surface", layout="wide")

TEXT = "#222222"
SUBTLE = "#666666"

PASTEL = [
    "#A8DADC", "#F4A261", "#90BE6D", "#FFD6A5", "#BDE0FE",
    "#CDB4DB", "#E2F0CB", "#F1C0E8", "#B9FBC0", "#F7EDE2"
]

# Red / Yellow / Green scale (low -> mid -> high)
RYG = [
    [0.0, "#1a9641"],   # green
    [0.5, "#ffffbf"],   # yellow
    [1.0, "#d7191c"],   # red
]

VOL_IDX = [
    "^VIX",
    "^VIX3M",
    "^VVIX",
    "^MOVE",
    "^GVZ",
    "^OVX",
]

PRICE_UNDERLYING = [
    "^GSPC",
    "TLT",
    "GLD",
    "CL=F"
]

IMPLIED_LABELS = {
    "^VIX": "Equity | VIX",
    "^VIX3M": "Equity | VIX3M",
    "^VVIX": "Equity | VVIX",
    "^MOVE": "Rates | MOVE",
    "^GVZ": "Commodity | GVZ",
    "^OVX": "Commodity | OVX",
}

UNDERLYING_LABELS = {
    "^GSPC": "Equity | SPX",
    "TLT": "Rates | TLT",
    "GLD": "Commodity | GLD",
    "CL=F": "Commodity | WTI",
}

SLEEVE_ORDER = ["Equity", "Rates", "Commodity"]

# ---------------- Helpers ----------------
def realized_vol(price: pd.Series, window: int, ann: int = 252) -> pd.Series:
    r = price.pct_change()
    return r.rolling(window).std() * np.sqrt(ann)

def trailing_window(s: pd.Series, window_days: int) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    cutoff = s.index.max() - pd.Timedelta(days=window_days)
    return s[s.index >= cutoff]

def zscore_last(s: pd.Series) -> float:
    s = s.dropna()
    if s.empty:
        return np.nan
    mu = s.mean()
    sig = s.std(ddof=0)
    if sig == 0 or np.isnan(sig):
        return np.nan
    return float((s.iloc[-1] - mu) / sig)

def pct_rank_last(s: pd.Series) -> float:
    s = s.dropna()
    if s.empty:
        return np.nan
    return float(s.rank(pct=True).iloc[-1] * 100.0)

def safe_5d_change(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 6:
        return np.nan
    return float((s.iloc[-1] / s.iloc[-6] - 1.0) * 100.0)

def parse_sleeve(name: str) -> str:
    return name.split("|")[0].strip() if "|" in name else "Other"

def normalize_spark(arr: np.ndarray) -> np.ndarray:
    """Row-wise normalization for sparklines so variance is visible."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return arr
    s = pd.Series(arr).dropna()
    if s.empty:
        return np.full_like(arr, np.nan)
    mu = s.mean()
    sig = s.std(ddof=0)
    if sig == 0 or np.isnan(sig):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    return (arr - mu) / sig

def make_card(title: str, value: str, subtitle: str):
    st.markdown(
        f"""
        <div style="border:1px solid #e0e0e0; border-radius:12px; padding:14px; background:#fbfbfb;">
          <div style="font-size:12px; color:{SUBTLE}; font-weight:700; letter-spacing:0.2px;">{title}</div>
          <div style="font-size:22px; color:{TEXT}; font-weight:800; margin-top:4px;">{value}</div>
          <div style="font-size:12px; color:{SUBTLE}; margin-top:6px; line-height:1.3;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def fmt_pct(x, d=2):
    if pd.isna(x):
        return "NA"
    return f"{x:.{d}f}%"

def fmt_z(x, d=2):
    if pd.isna(x):
        return "NA"
    return f"{x:.{d}f}"

def sleeve_color(sleeve: str) -> str:
    return {
        "Equity": PASTEL[0],
        "Rates": PASTEL[4],
        "Commodity": PASTEL[2],
    }.get(sleeve, PASTEL[1])

@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_close_series(tickers, start="2010-01-01"):
    data = yf.download(tickers, start=start, progress=False, auto_adjust=False, group_by="ticker")
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

# ---------------- Header + Sidebar ----------------
st.title("Cross Asset Volatility Surface Monitor")
st.caption("Cross-asset implied and realized volatility in one regime view. Data: Yahoo Finance.")

with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        **Purpose**

        Monitor cross-asset volatility so you can quickly judge whether vol is a tailwind or headwind for risk
        and where hedges are rich or cheap versus history.

        **What it does**

        - Pulls daily history for key implied vol indices and underlyings  
        - Computes realized vol (selected horizons) for SPX, duration, gold, and oil proxies  
        - Normalizes via a selectable trailing window and reports Z-scores + percentile ranks  
        - Produces a sleeve-level composite so the regime read is not hostage to one ticker  
        - Surfaces current outliers and heatmaps for fast scanning

        **How to use**

        - Start on *Overview* for the regime and top drivers  
        - Use *Heatmap* as the fast scan across sleeves and horizons  
        - Use *Details* when you need exact prints plus recent shape context  
        - Use *Time Series* to sanity-check whether a move is a one-off spike or a persistent regime shift
        """
    )

    st.divider()
    st.header("Controls")
    start_date = st.date_input("History start", datetime(2010, 1, 1))
    rv_windows = st.multiselect("Realized vol horizons (days)", [5, 21, 63], default=[21])
    norm_window = st.selectbox("Normalization window", ["1Y", "3Y", "5Y", "Max"], index=1)
    st.caption("Z-score and percentile are computed on the selected normalization window.")

norm_days = {"1Y": 365, "3Y": 365 * 3, "5Y": 365 * 5, "Max": 10_000}.get(norm_window, 365 * 3)

# ---------------- Load data ----------------
ALL = VOL_IDX + PRICE_UNDERLYING
series = load_close_series(ALL, start=str(start_date))

available = set(series.keys())
missing = [t for t in ALL if t not in available or series[t].dropna().empty]

if len(available) == 0:
    st.error("No data returned from Yahoo Finance for the selected window.")
    st.stop()

# ---------------- Build feature rows ----------------
rows = []

# Implied indices
for t in VOL_IDX:
    if t not in series or series[t].dropna().empty:
        continue

    s_full = series[t].dropna()
    s_norm = trailing_window(s_full, norm_days)

    label = IMPLIED_LABELS.get(t, t)
    sleeve = parse_sleeve(label)

    rows.append({
        "Series": label,
        "Sleeve": sleeve,
        "Kind": "Implied",
        "Horizon": "Index",
        "Level": float(s_full.iloc[-1]),
        "LevelDisplay": float(s_full.iloc[-1]),  # already in vol points read as %
        "LevelUnit": "%",
        "5D %": safe_5d_change(s_full),
        "Z": zscore_last(s_norm),
        "Percentile": pct_rank_last(s_norm),
        "SparkRaw": s_full.tail(90).values,
    })

# Realized vols on underlyings
for u in PRICE_UNDERLYING:
    if u not in series or series[u].dropna().empty:
        continue

    px_s = series[u].dropna()
    base_label = UNDERLYING_LABELS.get(u, u)
    sleeve = parse_sleeve(base_label)

    for w in rv_windows:
        rv = realized_vol(px_s, window=w).dropna()
        if rv.empty:
            continue

        rv_norm = trailing_window(rv, norm_days)

        rows.append({
            "Series": f"{base_label} | RV{w}",
            "Sleeve": sleeve,
            "Kind": "Realized",
            "Horizon": f"RV{w}",
            "Level": float(rv.iloc[-1]),
            "LevelDisplay": float(rv.iloc[-1]) * 100.0,  # convert decimal -> %
            "LevelUnit": "%",
            "5D %": safe_5d_change(rv),
            "Z": zscore_last(rv_norm),
            "Percentile": pct_rank_last(rv_norm),
            "SparkRaw": (rv.tail(90).values * 100.0),
        })

df = pd.DataFrame(rows)
if df.empty:
    st.error("No valid volatility series for the selected window.")
    st.stop()

df["AbsZ"] = df["Z"].abs()
df["Sleeve"] = pd.Categorical(df["Sleeve"], categories=SLEEVE_ORDER, ordered=True)
df = df.sort_values(["Sleeve", "Kind", "Series"]).reset_index(drop=True)

# Normalize sparklines per row so variance is visible
df["Spark"] = [normalize_spark(r["SparkRaw"]) for _, r in df.iterrows()]

# ---------------- Sleeve composites + Regime ----------------
def sleeve_composite(frame: pd.DataFrame, sleeve: str):
    sub = frame[frame["Sleeve"] == sleeve].copy()
    if sub.empty:
        return {"med_z": np.nan, "med_pct": np.nan, "hot": 0, "count": 0}
    med_z = float(sub["Z"].median(skipna=True))
    med_pct = float(sub["Percentile"].median(skipna=True))
    hot = int((sub["AbsZ"] >= 1.0).sum())
    return {"med_z": med_z, "med_pct": med_pct, "hot": hot, "count": len(sub)}

eq = sleeve_composite(df, "Equity")
rt = sleeve_composite(df, "Rates")
cmd = sleeve_composite(df, "Commodity")

def pos(x):
    return 0.0 if (pd.isna(x) or x < 0) else float(x)

eq_stress = pos(eq["med_z"]) + 0.5 * (eq["hot"] >= max(1, eq["count"] // 2))
rt_stress = pos(rt["med_z"]) + 0.5 * (rt["hot"] >= max(1, rt["count"] // 2))
cmd_stress = 0.25 * pos(cmd["med_z"]) + 0.25 * (cmd["hot"] >= max(1, cmd["count"] // 2))

stress = eq_stress + rt_stress + cmd_stress

if stress >= 2.5:
    regime = "Risk off"
elif stress >= 1.2:
    regime = "Cautious"
else:
    regime = "Neutral to constructive"

top_outliers = df.dropna(subset=["Z"]).sort_values("AbsZ", ascending=False).head(5)

# ---------------- Tabs ----------------
tab_overview, tab_heatmap, tab_details, tab_ts = st.tabs(["Overview", "Heatmap", "Details", "Time Series"])

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        make_card("Regime", regime, f"Normalization window: {norm_window}")

    with c2:
        make_card(
            "Equity sleeve",
            f"Z {fmt_z(eq['med_z'], 2)}",
            f"Median pct {fmt_pct(eq['med_pct'], 0)} | Hot {eq['hot']}/{eq['count']}"
        )

    with c3:
        make_card(
            "Rates sleeve",
            f"Z {fmt_z(rt['med_z'], 2)}",
            f"Median pct {fmt_pct(rt['med_pct'], 0)} | Hot {rt['hot']}/{rt['count']}"
        )

    with c4:
        make_card(
            "Commodity sleeve",
            f"Z {fmt_z(cmd['med_z'], 2)}",
            f"Median pct {fmt_pct(cmd['med_pct'], 0)} | Hot {cmd['hot']}/{cmd['count']}"
        )

    st.subheader("What is driving the tape")
    if top_outliers.empty:
        st.info("No outliers available.")
    else:
        out = top_outliers[["Series", "Kind", "Horizon", "LevelDisplay", "5D %", "Z", "Percentile"]].copy()
        out = out.rename(columns={"LevelDisplay": "Level"})
        st.dataframe(
            out,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Level": st.column_config.NumberColumn(format="%.2f%%"),
                "5D %": st.column_config.NumberColumn(format="%.1f%%"),
                "Z": st.column_config.NumberColumn(format="%.2f"),
                "Percentile": st.column_config.NumberColumn(format="%.0f%%"),
            }
        )

    st.subheader("Cross-asset Z snapshot")
    z_df = df.dropna(subset=["Z"]).copy()
    z_df = z_df.sort_values(["Sleeve", "AbsZ"], ascending=[True, False])

    fig_bar = go.Figure()
    for sleeve in SLEEVE_ORDER:
        sub = z_df[z_df["Sleeve"] == sleeve]
        if sub.empty:
            continue
        fig_bar.add_trace(
            go.Bar(
                x=sub["Z"],
                y=sub["Series"],
                orientation="h",
                name=sleeve,
                marker_color=sleeve_color(sleeve),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Z: %{x:.2f}<br>"
                    "Kind: %{customdata[0]}<br>"
                    "Horizon: %{customdata[1]}<br>"
                    "Level: %{customdata[2]:.2f}%<br>"
                    "Pct: %{customdata[3]:.0f}%<extra></extra>"
                ),
                customdata=np.stack(
                    [
                        sub["Kind"].values,
                        sub["Horizon"].values,
                        sub["LevelDisplay"].values,
                        sub["Percentile"].values,
                    ],
                    axis=1
                )
            )
        )

    fig_bar.add_vline(x=0, line_width=1, line_color="#777")
    fig_bar.add_vline(x=1, line_width=1, line_dash="dot", line_color="#bbb")
    fig_bar.add_vline(x=-1, line_width=1, line_dash="dot", line_color="#bbb")
    fig_bar.update_layout(
        height=520,
        barmode="group",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Z-score",
        yaxis_title="",
        legend_title_text="Sleeve",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with tab_heatmap:
    st.subheader("Heatmap")
    st.caption("Red = high / stretched. Green = low / cheap. Z heatmap is centered at 0, percentile at 50.")

    heat = df.copy()
    heat["Row"] = heat["Series"]
    heat["Col"] = heat.apply(lambda r: f"{r['Kind']} {r['Horizon']}", axis=1)

    z_pivot = heat.pivot_table(index="Row", columns="Col", values="Z", aggfunc="last")
    z_pivot = z_pivot.loc[z_pivot.abs().max(axis=1).sort_values(ascending=False).index]

    zmax = float(np.nanmax(np.abs(z_pivot.values))) if np.isfinite(np.nanmax(np.abs(z_pivot.values))) else 2.0
    zmax = max(2.0, min(6.0, zmax))

    fig_hm_z = px.imshow(
        z_pivot,
        aspect="auto",
        color_continuous_scale=RYG,
        zmin=-zmax,
        zmax=zmax,
        height=min(900, 28 * len(z_pivot) + 160),
        labels=dict(color="Z"),
    )
    fig_hm_z.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_hm_z, use_container_width=True)

    pct_pivot = heat.pivot_table(index="Row", columns="Col", values="Percentile", aggfunc="last")
    pct_pivot = pct_pivot.loc[pct_pivot.max(axis=1).sort_values(ascending=False).index]

    fig_hm_pct = px.imshow(
        pct_pivot,
        aspect="auto",
        color_continuous_scale=RYG,
        zmin=0,
        zmax=100,
        height=min(900, 28 * len(pct_pivot) + 160),
        labels=dict(color="Pct"),
    )
    fig_hm_pct.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_hm_pct, use_container_width=True)

with tab_details:
    st.subheader("Details")
    st.caption("Vol levels are formatted as % for digestibility. Sparklines are normalized per row to make variance visible.")

    view = df[["Series", "Sleeve", "Kind", "Horizon", "LevelDisplay", "5D %", "Z", "Percentile", "Spark"]].copy()
    view = view.rename(columns={"LevelDisplay": "Level", "Spark": "90D"})

    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Level": st.column_config.NumberColumn(format="%.2f%%"),
            "5D %": st.column_config.NumberColumn(format="%.1f%%"),
            "Z": st.column_config.NumberColumn(format="%.2f"),
            "Percentile": st.column_config.NumberColumn(format="%.0f%%"),
            "90D": st.column_config.LineChartColumn(
                "90D",
                help="Last 90 observations, normalized per row.",
                width="small",
                y_min=-3.0,
                y_max=3.0,
            ),
        }
    )

    if missing:
        st.warning("Missing series: " + ", ".join(missing))

with tab_ts:
    st.subheader("Time series")
    st.caption("All series on one page, arranged as small multiples (3 per row). Pastel sleeve colors, consistent styling.")

    plot_items = []

    for t in VOL_IDX:
        if t in series and not series[t].dropna().empty:
            s = series[t].dropna()
            label = IMPLIED_LABELS.get(t, t)
            plot_items.append((label, "Implied", s))

    for u in PRICE_UNDERLYING:
        if u not in series or series[u].dropna().empty:
            continue
        px_s = series[u].dropna()
        base_label = UNDERLYING_LABELS.get(u, u)
        for w in rv_windows:
            rv = realized_vol(px_s, window=w).dropna()
            if rv.empty:
                continue
            rv_disp = rv * 100.0
            plot_items.append((f"{base_label} | RV{w}", "Realized", rv_disp))

    if not plot_items:
        st.info("No series available to plot.")
    else:
        cols_per_row = 3
        n = len(plot_items)
        for start in range(0, n, cols_per_row):
            row = plot_items[start:start + cols_per_row]
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                with cols[j]:
                    if j >= len(row):
                        st.empty()
                        continue

                    name, kind, s = row[j]
                    sleeve = parse_sleeve(name)
                    color = sleeve_color(sleeve)
                    s_norm = trailing_window(s, norm_days)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name="Level",
                                             line=dict(color=color, width=2)))
                    fig.add_trace(go.Scatter(x=s_norm.index, y=s_norm.values, mode="lines", name=f"Norm ({norm_window})",
                                             line=dict(color=color, width=1, dash="dot")))

                    fig.update_layout(
                        height=320,
                        margin=dict(l=10, r=10, t=35, b=10),
                        title=dict(text=name, x=0.02, xanchor="left", font=dict(size=14, color=TEXT)),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, font=dict(size=10)),
                        xaxis=dict(title="", showgrid=False),
                        yaxis=dict(title="%", showgrid=True, gridcolor="#f0f0f0"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

st.caption("© 2026 AD Fund Management LP")
