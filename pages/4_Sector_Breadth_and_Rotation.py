import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="S&P 500 Sector Breadth & Rotation Monitor", layout="wide")
st.title("S&P 500 Sector Breadth & Rotation Monitor")

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

# Parameters
LOOKBACK_PERIOD = "1y"
INTERVAL = "1d"
MIN_OBS_FOR_RETURNS = 63  # need at least 63 daily obs for 3M returns

@st.cache_data(ttl=3600)
def robust_fetch_batch(tickers, period="1y", interval="1d") -> pd.DataFrame:
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

tickers = list(SECTORS.keys()) + ["SPY"]
prices = robust_fetch_batch(tickers, period=LOOKBACK_PERIOD, interval=INTERVAL)

if prices.empty:
    st.error("Downloaded data is empty. Yahoo Finance API might be rate-limited or unavailable.")
    st.stop()

if "SPY" not in prices.columns:
    st.error("SPY data unavailable. Cannot compute relative measures.")
    st.stop()

prices["SPY"] = prices["SPY"].ffill()
prices = prices[prices["SPY"].notna()]

enough_history_cols = [c for c in prices.columns if prices[c].dropna().shape[0] >= MIN_OBS_FOR_RETURNS]
prices = prices[enough_history_cols]

price_cols = [str(c).strip() for c in prices.columns]
available_sector_tickers = [t for t in SECTORS.keys() if t in price_cols]
missing_sectors = [SECTORS[t] for t in SECTORS if t not in price_cols]

if not available_sector_tickers:
    st.error("No sector tickers with sufficient data. Try again later.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        This dashboard monitors the relative performance and rotation of S&P 500 sectors.

        Features:
        • Sector Relative Strength vs. S&P 500  
        • Sector Rotation Quadrant with optional line trails  
        • Downloadable performance table

        Data Source: Yahoo Finance via yfinance, refreshed hourly.
        """
    )
    st.markdown("---")
    if available_sector_tickers:
        st.write(f"Sectors available: {', '.join([SECTORS[t] for t in available_sector_tickers])}")
    if missing_sectors:
        st.warning(f"Missing or insufficient data for: {', '.join(missing_sectors)}")

# --- Relative Strength vs SPY Chart ---
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
    title=f"Relative Strength: {SECTORS[selected_sector]} vs. S&P 500 (SPY)",
    labels={"value": "Ratio (Sector Price / SPY Price)", "index": "Date"},
)
fig_rs.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=40))
st.plotly_chart(fig_rs, use_container_width=True)

# --- Compute rolling returns for quadrant and trails ---
# Daily rolling total returns: 1M ~ 21 trading days, 3M ~ 63 trading days
rets_1m_df = prices[available_sector_tickers].pct_change(21)
rets_3m_df = prices[available_sector_tickers].pct_change(63)

# latest point
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

# --- Quadrant with optional line trails ---
st.subheader("Sector Rotation Quadrant (1M vs 3M Total Return)")
col_trail, col_days = st.columns([1, 2])
show_trails = col_trail.checkbox("Show trails", value=True, help="Draw day-by-day paths over the last N trading days.")
trail_days = col_days.slider("Trail length (trading days)", 10, 126, 63, step=1)

# base scatter for current points
fig_rot = go.Figure()

# color map using Plotly qualitative palette
palette = px.colors.qualitative.Dark24
sector_list = list(rotation_df["Sector"])
color_map = {sec: palette[i % len(palette)] for i, sec in enumerate(sector_list)}

if show_trails:
    # For each sector, draw the last N valid points of (3M, 1M) with date hover
    end_idx = rets_1m_df.index[-1]
    start_idx = rets_1m_df.index[-trail_days] if len(rets_1m_df.index) >= trail_days else rets_1m_df.index[0]
    trail_index = rets_1m_df.loc[start_idx:end_idx].index

    for t in available_sector_tickers:
        s1 = rets_1m_df[t].reindex(trail_index)
        s3 = rets_3m_df[t].reindex(trail_index)
        mask = s1.notna() & s3.notna()
        if mask.sum() < 2:
            continue

        fig_rot.add_trace(
            go.Scatter(
                x=s3[mask],
                y=s1[mask],
                mode="lines",
                name=f"{SECTORS[t]} trail",
                line=dict(width=1.5, color=color_map[SECTORS[t]]),
                hovertemplate="<b>%{customdata}</b><br>3M: %{x:.2%}<br>1M: %{y:.2%}<extra></extra>",
                customdata=[d.strftime("%Y-%m-%d") for d in s1[mask].index],
                showlegend=False,
            )
        )

# Add current points with labels
for _, row in rotation_df.iterrows():
    fig_rot.add_trace(
        go.Scatter(
            x=[row["3M Return"]],
            y=[row["1M Return"]],
            mode="markers+text",
            name=row["Sector"],
            marker=dict(size=9, color=color_map[row["Sector"]], line=dict(width=0.5, color="rgba(0,0,0,0.5)")),
            text=[row["Sector"]],
            textposition="top center",
            hovertemplate="<b>%{text}</b><br>3M: %{x:.2%}<br>1M: %{y:.2%}<extra></extra>",
            showlegend=False,
        )
    )

fig_rot.update_layout(
    title="Sector Rotation Quadrant (1M vs 3M Total Return)",
    xaxis_title="3-Month Return",
    yaxis_title="1-Month Return",
    height=520,
    margin=dict(l=40, r=40, t=60, b=40),
)

fig_rot.add_hline(y=0, line_width=1, opacity=0.4)
fig_rot.add_vline(x=0, line_width=1, opacity=0.4)
st.plotly_chart(fig_rot, use_container_width=True)

# --- Sortable Table with percentage formatting ---
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

if not prices.empty:
    st.caption(f"Data through: {prices.index.max().date()}")

st.caption("© 2025 AD Fund Management LP")
