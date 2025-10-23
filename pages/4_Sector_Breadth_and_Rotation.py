# sp_sector_rotation_clean.py
# ADFM — S&P 500 Sector Breadth and Rotation Monitor
# Plotly, Streamlit, yfinance. Clean rotation views.

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
            t = tickers[0] if isinstance(tickers, list) and len(tickers) == 1 else "TICKER"
            out[t] = data["Adj Close"].dropna()

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
        Views include:
        • Relative Strength vs SPY  
        • Focus paths with weekly sampling  
        • Small multiples grid  
        • Angle and speed time series  
        • Downloadable returns table

        Data source: Yahoo Finance via yfinance, refreshed hourly.
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

# Latest snapshot table
returns_1m = rets_1m_df.iloc[-1]
returns_3m = rets_3m_df.iloc[-1]
rotation_df = pd.DataFrame({
    "1M Return": returns_1m,
    "3M Return": returns_3m,
    "Ticker": returns_1m.index
})
rotation_df = rotation_df[rotation_df["Ticker"].isin(available_sector_tickers)]
rotation_df["Sector"] = rotation_df["Ticker"].map(SECTORS)
rotation_df = rotation_df.sort_values("Sector")

# Palette
palette = px.colors.qualitative.Dark24
sector_list = list(rotation_df["Sector"])
color_map = {sec: palette[i % len(palette)] for i, sec in enumerate(sector_list)}

# ------------------------------- Helpers -----------------------------------
def weekly_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return weekly sampling on Fridays or last business day."""
    df_idx = pd.Series(1, index=idx).to_frame("x")
    wk = df_idx.resample("W-FRI").last().index
    wk = pd.DatetimeIndex([d for d in wk if d in idx])
    return wk if len(wk) else idx

# ------------------------------- Rotation tabs -----------------------------
st.subheader("Sector Rotation, cleaner views")
tab1, tab2, tab3 = st.tabs(["Focus paths", "Small multiples", "Angle and speed"])

# -------- Tab 1: Focus paths (1–3 sectors, weekly points, monthly knots) ---
with tab1:
    left, right = st.columns([2, 3])
    focus = left.multiselect(
        "Focus sectors",
        options=available_sector_tickers,
        default=[t for t in ["XLK", "XLU"] if t in available_sector_tickers],
        format_func=lambda t: SECTORS[t],
        max_selections=3
    )
    weeks_back = right.slider("Lookback weeks", 8, 52, 26, step=1)

    fig = go.Figure()
    fig.add_hline(y=0, line_width=1, opacity=0.4)
    fig.add_vline(x=0, line_width=1, opacity=0.4)

    # faint context: all sectors latest point
    fig.add_trace(go.Scatter(
        x=rotation_df["3M Return"], y=rotation_df["1M Return"],
        mode="markers+text",
        text=rotation_df["Sector"],
        textposition="top center",
        marker=dict(size=7, color="rgba(120,120,120,0.35)"),
        textfont=dict(color="rgba(80,80,80,0.6)"),
        hovertemplate="<b>%{text}</b><br>3M: %{x:.2%}<br>1M: %{y:.2%}<extra></extra>",
        showlegend=False
    ))

    if focus:
        wk_idx = weekly_index(rets_1m_df.index)
        wk_idx = wk_idx[-weeks_back:] if len(wk_idx) > weeks_back else wk_idx

        for t in focus:
            s1 = rets_1m_df[t].reindex(wk_idx)
            s3 = rets_3m_df[t].reindex(wk_idx)
            m = s1.notna() & s3.notna()
            if m.sum() < 2:
                continue

            sec = SECTORS[t]
            clr = color_map[sec]

            # path
            fig.add_trace(go.Scatter(
                x=s3[m], y=s1[m],
                mode="lines+markers",
                line=dict(width=2, color=clr),
                marker=dict(size=5, color=clr),
                name=sec,
                hovertemplate="<b>"+sec+"</b><br>%{customdata}<br>3M: %{x:.2%}<br>1M: %{y:.2%}<extra></extra>",
                customdata=[d.strftime("%Y-%m-%d") for d in s1[m].index]
            ))

            # monthly knots
            monthly = pd.Series(1, index=s1[m].index).resample("MS").first()
            month_knots = [d for d in monthly.index if d in s1[m].index]
            if month_knots:
                mk1 = s1.loc[month_knots]; mk3 = s3.loc[month_knots]
                fig.add_trace(go.Scatter(
                    x=mk3, y=mk1,
                    mode="markers+text",
                    marker=dict(size=9, symbol="circle-open", line=dict(width=2, color=clr)),
                    text=[d.strftime("%b") for d in mk1.index],
                    textposition="bottom center",
                    showlegend=False,
                    hovertemplate="<b>"+sec+"</b><br>%{text} knot<br>3M: %{x:.2%}<br>1M: %{y:.2%}<extra></extra>"
                ))

            # start to end arrow
            x0, y0 = s3[m].iloc[0], s1[m].iloc[0]
            x1, y1 = s3[m].iloc[-1], s1[m].iloc[-1]
            fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0,
                               xref="x", yref="y", axref="x", ayref="y",
                               arrowhead=3, arrowsize=1, arrowwidth=1.5, arrowcolor=clr,
                               opacity=0.9)

    fig.update_layout(
        title="Focus paths, weekly sampled",
        xaxis_title="3-Month Return",
        yaxis_title="1-Month Return",
        height=560,
        margin=dict(l=40, r=40, t=60, b=40),
        legend_title=""
    )
    st.plotly_chart(fig, use_container_width=True)

# -------- Tab 2: Small multiples grid -------------------------------------
with tab2:
    weeks_back_2 = st.slider("Lookback weeks for grid", 8, 52, 26, key="grid_weeks")
    wk_idx = weekly_index(rets_1m_df.index)
    wk_idx = wk_idx[-weeks_back_2:] if len(wk_idx) > weeks_back_2 else wk_idx

    # tidy frame
    tidy = []
    for t in available_sector_tickers:
        s1 = rets_1m_df[t].reindex(wk_idx)
        s3 = rets_3m_df[t].reindex(wk_idx)
        m = s1.notna() & s3.notna()
        if m.sum() < 2:
            continue
        df_t = pd.DataFrame({
            "date": s1[m].index,
            "one_m": s1[m].values,
            "three_m": s3[m].values,
            "Sector": SECTORS[t]
        })
        tidy.append(df_t)

    if tidy:
        tidy = pd.concat(tidy, ignore_index=True)

        fig_grid = px.line(
            tidy, x="three_m", y="one_m",
            facet_col="Sector", facet_col_wrap=4,
            color="Sector", color_discrete_map=color_map,
        )
        fig_grid.update_traces(line=dict(width=2), selector=dict(mode="lines"))
        fig_grid.update_layout(
            title="Small multiples, weekly sampled",
            height=920, margin=dict(l=30, r=30, t=60, b=30), showlegend=False
        )
        # quadrant axes
        fig_grid.add_hline(y=0, line_width=1, opacity=0.3)
        fig_grid.add_vline(x=0, line_width=1, opacity=0.3)
        st.plotly_chart(fig_grid, use_container_width=True)
    else:
        st.info("Insufficient data for grid.")

# -------- Tab 3: Angle and speed decomposition ----------------------------
with tab3:
    look_weeks = st.slider("Lookback weeks for angle and speed", 8, 104, 52, key="ang_weeks")
    wk_idx = weekly_index(rets_1m_df.index)
    wk_idx = wk_idx[-look_weeks:] if len(wk_idx) > look_weeks else wk_idx

    sec_pick = st.selectbox(
        "Sector",
        options=available_sector_tickers,
        format_func=lambda t: SECTORS[t],
        index=available_sector_tickers.index("XLK") if "XLK" in available_sector_tickers else 0
    )
    s1 = rets_1m_df[sec_pick].reindex(wk_idx)
    s3 = rets_3m_df[sec_pick].reindex(wk_idx)
    m = s1.notna() & s3.notna()
    a = np.degrees(np.arctan2(s1[m].values, s3[m].values))    # angle
    v = np.sqrt(s1[m].values**2 + s3[m].values**2)            # speed
    sec_name = SECTORS[sec_pick]
    clr = color_map[sec_name]

    fig_ang = go.Figure()
    fig_ang.add_trace(go.Scatter(x=s1[m].index, y=a, mode="lines+markers",
                                 line=dict(width=2, color=clr)))
    fig_ang.update_layout(title=f"{sec_name} rotation angle, +y is 1M, +x is 3M",
                          yaxis_title="Angle (degrees)", xaxis_title="Date",
                          height=320, margin=dict(l=40, r=40, t=50, b=40))
    st.plotly_chart(fig_ang, use_container_width=True)

    fig_spd = go.Figure()
    fig_spd.add_trace(go.Scatter(x=s1[m].index, y=v, mode="lines+markers",
                                 line=dict(width=2, color=clr)))
    fig_spd.update_layout(title=f"{sec_name} rotation speed, norm of 1M and 3M",
                          yaxis_title="Speed", xaxis_title="Date",
                          height=300, margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig_spd, use_container_width=True)

# ------------------------------- Returns table -----------------------------
df = pd.DataFrame({
    "Sector": [SECTORS[t] for t in available_sector_tickers],
    "1M Return": [returns_1m.get(t, pd.NA) for t in available_sector_tickers],
    "3M Return": [returns_3m.get(t, pd.NA) for t in available_sector_tickers],
}).set_index("Sector")

st.subheader("Sector Total Returns Table")
styled = df.style.format("{:.2%}").bar(subset=["1M Return", "3M Return"], align="mid")
st.dataframe(styled, height=400)

csv = df.to_csv().encode()
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
