# sp_sector_rotation_clean_v2.py
# ADFM — S&P 500 Sector Breadth and Rotation Monitor
# Clean rotation views: quadrant-occupancy heatmap, rank-flow slopegraph, minimal snapshot.
# Plotly, Streamlit, yfinance.

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------- Page config -------------------------------
st.set_page_config(page_title="S&P 500 Sector Breadth & Rotation Monitor", layout="wide")
st.title("S&P 500 Sector Breadth & Rotation Monitor")

# ------------------------------- Constants --------------------------------
SECTORS = {
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLK": "Technology",
    "XLU": "Utilities",
}
LOOKBACK_PERIOD = "1y"
INTERVAL = "1d"
DAYS_1M = 21
DAYS_3M = 63
MIN_OBS_FOR_RETURNS = DAYS_3M

# ------------------------------- Data fetch --------------------------------
@st.cache_data(ttl=3600)
def robust_fetch_batch(tickers, period="1y", interval="1d") -> pd.DataFrame:
    """
    Batch download Adjusted Close for all tickers.
    Keep only Adj Close to preserve total return semantics.
    Returns a DateIndex DataFrame with columns per ticker.
    """
    try:
        data = yf.download(
            tickers,
            period=period,
            interval=interval,
            progress=False,
            group_by="ticker",
            auto_adjust=False,
        )
    except Exception:
        return pd.DataFrame()

    out = {}
    # yfinance returns MultiIndex columns when group_by="ticker"
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            try:
                s = data[(t, "Adj Close")].dropna()
                if not s.empty:
                    out[t] = s
            except Exception:
                continue
    else:
        if "Adj Close" in data.columns:
            name = tickers[0] if isinstance(tickers, list) and len(tickers) == 1 else "TICKER"
            out[name] = data["Adj Close"].dropna()

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out).sort_index()
    df = df.dropna(how="all")
    return df

# ------------------------------- Load prices -------------------------------
tickers = list(SECTORS.keys()) + ["SPY"]
prices = robust_fetch_batch(tickers, period=LOOKBACK_PERIOD, interval=INTERVAL)

if prices.empty:
    st.error("Downloaded data is empty. Yahoo Finance might be rate limited or unavailable.")
    st.stop()

if "SPY" not in prices.columns:
    st.error("SPY data unavailable. Cannot compute relative measures.")
    st.stop()

# Clean SPY and enforce minimum history
prices["SPY"] = prices["SPY"].ffill()
prices = prices[prices["SPY"].notna()]
enough_history_cols = [c for c in prices.columns if prices[c].dropna().shape[0] >= MIN_OBS_FOR_RETURNS]
prices = prices[enough_history_cols]

price_cols = [str(c).strip() for c in prices.columns]
available_sector_tickers = [t for t in SECTORS.keys() if t in price_cols]
missing_sectors = [SECTORS[t] for t in SECTORS if t not in price_cols]

if not available_sector_tickers:
    st.error("No sector tickers with sufficient data.")
    st.stop()

# ------------------------------- Sidebar -----------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This dashboard monitors sector relative strength and rotation.

        Views
        • Relative Strength vs SPY  
        • Quadrant-occupancy heatmap  
        • Rank-flow slopegraph  
        • Minimal latest snapshot  
        • Downloadable returns table

        Data: Yahoo Finance via yfinance, refreshed hourly.
        """
    )
    st.markdown("---")
    if available_sector_tickers:
        st.write("Sectors available: " + ", ".join([SECTORS[t] for t in available_sector_tickers]))
    if missing_sectors:
        st.warning("Missing or insufficient data for: " + ", ".join(missing_sectors))

# ------------------------------- Relative strength -------------------------
relative_strength = prices[available_sector_tickers].div(prices["SPY"], axis=0)

default_choice = st.session_state.get("sel_sector", available_sector_tickers[0])
if default_choice not in available_sector_tickers:
    default_choice = available_sector_tickers[0]

selected_sector = st.selectbox(
    "Select sector to compare with S&P 500 (SPY):",
    options=available_sector_tickers,
    index=available_sector_tickers.index(default_choice),
    format_func=lambda x: SECTORS[x]
)
st.session_state["sel_sector"] = selected_sector

fig_rs = px.line(
    relative_strength[selected_sector].dropna(),
    title=f"Relative Strength: {SECTORS[selected_sector]} vs SPY",
    labels={"value": "Ratio (Sector Price / SPY Price)", "index": "Date"},
)
fig_rs.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=30))
st.plotly_chart(fig_rs, use_container_width=True)

# ------------------------------- Rolling returns ---------------------------
rets_1m_df = prices[available_sector_tickers].pct_change(DAYS_1M)
rets_3m_df = prices[available_sector_tickers].pct_change(DAYS_3M)

# Latest snapshot table data
returns_1m_latest = rets_1m_df.iloc[-1]
returns_3m_latest = rets_3m_df.iloc[-1]

# Palette
palette = px.colors.qualitative.Dark24
sector_names = [SECTORS[t] for t in available_sector_tickers]
color_map = {SECTORS[t]: palette[i % len(palette)] for i, t in enumerate(available_sector_tickers)}

# ------------------------------- Helpers -----------------------------------
def weekly_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return weekly sampling on Fridays or last business day present in idx."""
    s = pd.Series(1, index=idx)
    wk = s.resample("W-FRI").last().index
    wk = pd.DatetimeIndex([d for d in wk if d in idx])
    return wk if len(wk) else idx

# ------------------------------- Rotation summary --------------------------
st.subheader("Rotation summary, clutter-free")

c1, c2, _ = st.columns([2, 2, 3])
weeks_back = c1.slider("Lookback weeks", 8, 52, 26, step=1, help="Weekly sampling window for heatmap")
focus_topn = c2.slider("Top N for rank-flow", 3, len(available_sector_tickers), 8, step=1)

# ---------- 1) Quadrant occupancy heatmap ----------
wk_idx = weekly_index(rets_1m_df.index)
wk_idx = wk_idx[-weeks_back:] if len(wk_idx) > weeks_back else wk_idx

quad_cols = ["Q1 ++", "Q2 -+", "Q3 --", "Q4 +-"]  # Q1: 3M>0,1M>0 etc.
rows = []
for t in available_sector_tickers:
    s1 = rets_1m_df[t].reindex(wk_idx)
    s3 = rets_3m_df[t].reindex(wk_idx)
    mask = s1.notna() & s3.notna()
    if mask.sum() == 0:
        rows.append([SECTORS[t], 0, 0, 0, 0])
        continue
    s1 = s1[mask]; s3 = s3[mask]
    q1 = ((s3 > 0) & (s1 > 0)).mean()
    q2 = ((s3 < 0) & (s1 > 0)).mean()
    q3 = ((s3 < 0) & (s1 < 0)).mean()
    q4 = ((s3 > 0) & (s1 < 0)).mean()
    rows.append([SECTORS[t], q1, q2, q3, q4])

occ = pd.DataFrame(rows, columns=["Sector"] + quad_cols).set_index("Sector").loc[sector_names]

fig_occ = px.imshow(
    occ[quad_cols].T,
    aspect="auto",
    color_continuous_scale="Blues",
    labels=dict(x="Sector", y="Quadrant", color="Share of weeks"),
    origin="lower",
)
fig_occ.update_coloraxes(cmin=0, cmax=1)
fig_occ.update_layout(
    title=f"Quadrant occupancy, last {weeks_back} weeks (weekly sampling)",
    height=380, margin=dict(l=30, r=30, t=60, b=30)
)
st.plotly_chart(fig_occ, use_container_width=True)

# ---------- 2) Rank-flow slopegraph (monthly checkpoints, 3M returns) ----------
# Choose four recent month-end checkpoints aligned to available dates
all_idx = rets_3m_df.index.dropna()
if len(all_idx) >= 1:
    end = all_idx[-1]
    month_ends = pd.date_range(end=end, periods=4, freq="M")
    marks = []
    for d in month_ends:
        if (all_idx <= d).any():
            marks.append(max(all_idx[all_idx <= d]))
        else:
            marks.append(all_idx[0])
    marks = sorted(list(dict.fromkeys(marks)))
else:
    marks = list(rets_3m_df.index[-4:])

tidy_rank = []
for d in marks:
    row = rets_3m_df.loc[d].dropna()
    if row.empty:
        continue
    r = (-row).rank(method="min")  # descending rank, 1 is best
    for t, val in row.items():
        tidy_rank.append({
            "date": d.strftime("%Y-%m-%d"),
            "Sector": SECTORS[t],
            "rank": int(r[t]),
            "ret3m": val
        })
rank_df = pd.DataFrame(tidy_rank)

if not rank_df.empty:
    latest_date = rank_df["date"].iloc[-1]
    keep = rank_df[rank_df["date"] == latest_date].nsmallest(focus_topn, "rank")["Sector"].tolist()
    plot_df = rank_df[rank_df["Sector"].isin(keep)].copy()

    x_positions = {d: i for i, d in enumerate(sorted(plot_df["date"].unique()))}
    fig_slope = go.Figure()
    for sec in keep:
        df_s = plot_df[plot_df["Sector"] == sec].sort_values("date")
        fig_slope.add_trace(go.Scatter(
            x=[x_positions[d] for d in df_s["date"]],
            y=df_s["rank"],
            mode="lines+markers+text",
            line=dict(width=2, color=color_map[sec]),
            marker=dict(size=7, color=color_map[sec]),
            text=[sec] + [""] * (len(df_s) - 1),
            textposition="top left",
            hovertemplate="<b>"+sec+"</b><br>Rank %{y} at %{x}<extra></extra>",
            showlegend=False
        ))
    fig_slope.update_yaxes(autorange="reversed", dtick=1, title="Rank (1 best)")
    fig_slope.update_xaxes(
        tickmode="array",
        tickvals=list(x_positions.values()),
        ticktext=sorted(plot_df["date"].unique()),
        title="Checkpoint"
    )
    fig_slope.update_layout(
        title=f"Rank flow of 3M returns, monthly checkpoints, top {focus_topn}",
        height=420, margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig_slope, use_container_width=True)
else:
    st.info("Not enough monthly checkpoints to draw rank flow.")

# ---------- 3) Minimal latest snapshot ----------
snap = pd.DataFrame({
    "Sector": [SECTORS[t] for t in available_sector_tickers],
    "1M": [returns_1m_latest.get(t, np.nan) for t in available_sector_tickers],
    "3M": [returns_3m_latest.get(t, np.nan) for t in available_sector_tickers],
})
fig_snap = px.scatter(
    snap, x="3M", y="1M", text="Sector",
    color="Sector", color_discrete_map=color_map
)
fig_snap.update_traces(
    textposition="top center",
    marker=dict(size=9, line=dict(width=0.5, color="rgba(0,0,0,0.4)"))
)
fig_snap.add_hline(y=0, line_width=1, opacity=0.4)
fig_snap.add_vline(x=0, line_width=1, opacity=0.4)
fig_snap.update_layout(
    title="Current snapshot, 1M vs 3M total return",
    showlegend=False, height=440, margin=dict(l=40, r=40, t=60, b=40)
)
st.plotly_chart(fig_snap, use_container_width=True)

# Optional compact table sorted by speed
snap["Angle (deg)"] = np.degrees(np.arctan2(snap["1M"], snap["3M"]))
snap["Speed"] = np.sqrt(snap["1M"]**2 + snap["3M"]**2)
st.dataframe(
    snap.set_index("Sector")
        .sort_values("Speed", ascending=False)
        .style.format({"1M": "{:.2%}", "3M": "{:.2%}", "Speed": "{:.2%}"}),
    height=360
)

# ------------------------------- Returns table -----------------------------
df_table = pd.DataFrame({
    "Sector": [SECTORS[t] for t in available_sector_tickers],
    "1M Return": [returns_1m_latest.get(t, pd.NA) for t in available_sector_tickers],
    "3M Return": [returns_3m_latest.get(t, pd.NA) for t in available_sector_tickers],
}).set_index("Sector")

st.subheader("Sector Total Returns Table")
styled = df_table.style.format("{:.2%}").bar(subset=["1M Return", "3M Return"], align="mid")
st.dataframe(styled, height=400)

csv = df_table.to_csv().encode()
st.download_button(
    label="Download returns table as CSV",
    data=csv,
    file_name="sector_returns.csv",
    mime="text/csv"
)

# ------------------------------- Footer ------------------------------------
if not prices.empty:
    st.caption(f"Data through: {prices.index.max().date()}")
st.caption("© 2025 AD Fund Management LP")
