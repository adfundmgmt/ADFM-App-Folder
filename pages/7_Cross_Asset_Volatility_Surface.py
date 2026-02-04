import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# ---------------- Config ----------------
st.set_page_config(page_title="Cross Asset Volatility Surface", layout="wide")

TEXT = "#222222"
GRID = "#e8e8e8"

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

def fmt(x, d=2, suffix=""):
    if pd.isna(x):
        return "NA"
    return f"{x:.{d}f}{suffix}"

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

def parse_sleeve(name: str) -> str:
    return name.split("|")[0].strip() if "|" in name else "Other"

def make_card(title: str, value: str, subtitle: str):
    st.markdown(
        f"""
        <div style="border:1px solid #e0e0e0; border-radius:12px; padding:14px; background:#fbfbfb;">
          <div style="font-size:12px; color:#666; font-weight:700; letter-spacing:0.2px;">{title}</div>
          <div style="font-size:22px; color:{TEXT}; font-weight:800; margin-top:4px;">{value}</div>
          <div style="font-size:12px; color:#666; margin-top:6px; line-height:1.3;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- Sidebar ----------------
st.title("Cross Asset Volatility Surface Monitor")
st.caption("Cross-asset implied and realized vol in one regime view. Data: Yahoo Finance.")

with st.sidebar:
    st.header("Controls")
    start_date = st.date_input("History start", datetime(2010, 1, 1))
    rv_windows = st.multiselect("Realized vol horizons (days)", [5, 21, 63], default=[21])
    norm_window = st.selectbox("Normalization window", ["1Y", "3Y", "5Y", "Max"], index=1)
    show_sparklines = st.checkbox("Show sparklines in table", value=True)
    st.caption("Z-score and percentile are computed on the selected normalization window.")

norm_days = {"1Y": 365, "3Y": 365*3, "5Y": 365*5, "Max": 10_000}.get(norm_window, 365*3)

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
        "5D %": safe_5d_change(s_full),
        "Z": zscore_last(s_norm),
        "Percentile": pct_rank_last(s_norm),
        "Spark": s_full.tail(90).values
    })

# Realized vols on underlyings for selected horizons
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
            "5D %": safe_5d_change(rv),
            "Z": zscore_last(rv_norm),
            "Percentile": pct_rank_last(rv_norm),
            "Spark": rv.tail(90).values
        })

df = pd.DataFrame(rows)
if df.empty:
    st.error("No valid volatility series for the selected window.")
    st.stop()

df["AbsZ"] = df["Z"].abs()
df["Sleeve"] = pd.Categorical(df["Sleeve"], categories=SLEEVE_ORDER, ordered=True)
df = df.sort_values(["Sleeve", "Kind", "Series"]).reset_index(drop=True)

# ---------------- Sleeve composites ----------------
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

# Composite stress score
stress = 0
for sleeve in [eq, rt, cmd]:
    if not np.isnan(sleeve["med_z"]) and sleeve["med_z"] >= 1.0:
        stress += 1
    if not np.isnan(sleeve["med_z"]) and sleeve["med_z"] >= 2.0:
        stress += 1
    stress += int(sleeve["hot"] >= max(1, sleeve["count"] // 2))

if stress >= 5:
    regime = "Risk off"
elif stress >= 3:
    regime = "Cautious"
else:
    regime = "Neutral to constructive"

top_outliers = df.dropna(subset=["Z"]).sort_values("AbsZ", ascending=False).head(5)

# ---------------- Layout ----------------
tab_overview, tab_heatmap, tab_details, tab_ts = st.tabs(["Overview", "Heatmap", "Details", "Time Series"])

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("###")
    with c1:
        make_card("Regime", regime, f"Normalization window: {norm_window}")
    with c2:
        make_card("Equity sleeve", f"Z {fmt(eq['med_z'], 2)}", f"Median pct {fmt(eq['med_pct'], 0, '%')} | Hot {eq['hot']}/{eq['count']}")
    with c3:
        make_card("Rates sleeve", f"Z {fmt(rt['med_z'], 2)}", f"Median pct {fmt(rt['med_pct'], 0, '%')} | Hot {rt['hot']}/{rt['count']}")
    with c4:
        make_card("Commodity sleeve", f"Z {fmt(cmd['med_z'], 2)}", f"Median pct {fmt(cmd['med_pct'], 0, '%')} | Hot {cmd['hot']}/{cmd['count']}")

    st.subheader("What is driving the tape")
    if top_outliers.empty:
        st.info("No outliers available.")
    else:
        out = top_outliers[["Series", "Kind", "Horizon", "Level", "5D %", "Z", "Percentile"]].copy()
        out["Level"] = out["Level"].map(lambda x: float(x) if pd.notna(x) else np.nan)
        st.dataframe(out, use_container_width=True, hide_index=True)

    st.subheader("Cross-asset Z snapshot")
    z_df = df.dropna(subset=["Z"]).copy()
    z_df = z_df.sort_values(["Sleeve", "AbsZ"], ascending=[True, False])

    fig_bar = px.bar(
        z_df,
        x="Z",
        y="Series",
        orientation="h",
        color="Sleeve",
        hover_data={"Kind": True, "Horizon": True, "Level": ":.2f", "Percentile": ":.0f"},
        height=520
    )
    fig_bar.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Z-score",
        yaxis_title="",
        legend_title_text="Sleeve",
    )
    fig_bar.add_vline(x=0, line_width=1, line_color="#777")
    fig_bar.add_vline(x=1, line_width=1, line_dash="dot", line_color="#bbb")
    fig_bar.add_vline(x=-1, line_width=1, line_dash="dot", line_color="#bbb")
    st.plotly_chart(fig_bar, use_container_width=True)

with tab_heatmap:
    st.subheader("Heatmap")
    st.caption("Scan percentiles and Z-scores across series and horizons. Use this as the first pass.")

    # Heatmap matrix: rows = Series, cols = Metric
    heat = df.copy()
    heat["Row"] = heat["Series"]
    heat["Col"] = heat.apply(lambda r: f"{r['Kind']} {r['Horizon']}", axis=1)

    # Z heatmap
    z_pivot = heat.pivot_table(index="Row", columns="Col", values="Z", aggfunc="last")
    z_pivot = z_pivot.loc[z_pivot.abs().max(axis=1).sort_values(ascending=False).index]

    fig_hm_z = px.imshow(
        z_pivot,
        aspect="auto",
        height=min(900, 28 * len(z_pivot) + 160),
        labels=dict(color="Z"),
    )
    fig_hm_z.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_hm_z, use_container_width=True)

    # Percentile heatmap
    pct_pivot = heat.pivot_table(index="Row", columns="Col", values="Percentile", aggfunc="last")
    pct_pivot = pct_pivot.loc[pct_pivot.max(axis=1).sort_values(ascending=False).index]

    fig_hm_pct = px.imshow(
        pct_pivot,
        aspect="auto",
        height=min(900, 28 * len(pct_pivot) + 160),
        labels=dict(color="Pct"),
    )
    fig_hm_pct.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_hm_pct, use_container_width=True)

with tab_details:
    st.subheader("Details")
    st.caption("Use this for exact prints and quick shape context.")

    view = df[["Series", "Sleeve", "Kind", "Horizon", "Level", "5D %", "Z", "Percentile", "Spark"]].copy()

    if show_sparklines:
        st.dataframe(
            view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Spark": st.column_config.LineChartColumn(
                    "90D",
                    help="Last 90 observations",
                    width="small",
                    y_min=float(np.nanmin(np.concatenate(view["Spark"].values))) if len(view) else None,
                    y_max=float(np.nanmax(np.concatenate(view["Spark"].values))) if len(view) else None,
                ),
                "Level": st.column_config.NumberColumn(format="%.2f"),
                "5D %": st.column_config.NumberColumn(format="%.1f"),
                "Z": st.column_config.NumberColumn(format="%.2f"),
                "Percentile": st.column_config.NumberColumn(format="%.0f"),
            }
        )
    else:
        st.dataframe(
            view.drop(columns=["Spark"]),
            use_container_width=True,
            hide_index=True
        )

    if missing:
        st.warning("Missing series: " + ", ".join(missing))

with tab_ts:
    st.subheader("Time series")
    st.caption("Pick any series and view the full history you requested, plus the normalization window overlay.")

    options = df["Series"].unique().tolist()
    pick = st.selectbox("Series", options)

    # Map back to underlying ticker logic
    # For implied: find original ticker by reverse label map
    reverse_implied = {v: k for k, v in IMPLIED_LABELS.items()}

    if pick in reverse_implied:
        tkr = reverse_implied[pick]
        s = series.get(tkr, pd.Series(dtype=float)).dropna()
        if s.empty:
            st.info("No data for selected series.")
        else:
            s_norm = trailing_window(s, norm_days)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name="Level"))
            fig.add_trace(go.Scatter(x=s_norm.index, y=s_norm.values, name=f"Norm window ({norm_window})"))
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=500, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Realized series: reconstruct from label
        # Example: "Equity | SPX | RV21"
        try:
            parts = [p.strip() for p in pick.split("|")]
            base = "|".join(parts[:2])  # "Equity | SPX"
            horizon = parts[-1]         # "RV21"
            w = int(horizon.replace("RV", ""))

            # find underlying ticker
            rev_under = {v: k for k, v in UNDERLYING_LABELS.items()}
            u_tkr = rev_under.get(base, None)

            if u_tkr is None or u_tkr not in series:
                st.info("Could not map this realized series back to an underlying ticker.")
            else:
                px_s = series[u_tkr].dropna()
                rv = realized_vol(px_s, window=w).dropna()
                rv_norm = trailing_window(rv, norm_days)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rv.index, y=rv.values, name="Realized vol"))
                fig.add_trace(go.Scatter(x=rv_norm.index, y=rv_norm.values, name=f"Norm window ({norm_window})"))
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=500, xaxis_title="", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Could not parse this series.")

st.caption("ADFM Volatility Surface Monitor")
st.caption("© 2026 AD Fund Management LP")
