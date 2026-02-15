import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Market Dashboard", layout="wide")

st.title("Market Dashboard")
st.caption(
    "Auto-updating market monitor covering broad risk, volatility, rates, dollar, breadth, sectors, and correlation structure."
)

with st.sidebar:
    st.header("Controls")
    lookback_days = st.slider("Lookback window (days)", min_value=90, max_value=1500, value=365, step=30)
    corr_window = st.selectbox("Rolling correlation window", [20, 30, 63, 126], index=2)
    auto_refresh = st.checkbox("Auto-refresh every 5 minutes", value=True)
    if auto_refresh:
        st.caption("Data cache TTL is 5 minutes. Refresh button bypasses cache.")

    force_refresh = st.button("Refresh now")

if auto_refresh:
    st.query_params["t"] = datetime.utcnow().strftime("%Y%m%d%H%M")

RISK_ON = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Russell 2000": "^RUT",
    "ACWI": "ACWI",
    "High Yield": "HYG",
    "Copper": "HG=F",
    "WTI Oil": "CL=F",
}

DEFENSIVE = {
    "US 10Y": "^TNX",
    "US 2Y": "^IRX",
    "DXY": "DX-Y.NYB",
    "Gold": "GC=F",
    "Long Treasuries": "TLT",
    "VIX": "^VIX",
}

SECTORS = {
    "XLB": "Materials",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Staples",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLC": "Communication Services",
}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(tickers, start_date):
    raw = yf.download(
        tickers,
        start=start_date,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    close_df = pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if (t, "Close") in raw.columns:
                close_df[t] = raw[(t, "Close")]
    else:
        if "Close" in raw.columns and len(tickers) == 1:
            close_df[tickers[0]] = raw["Close"]

    return close_df.dropna(how="all")


def safe_pct_change(series: pd.Series, periods: int = 1):
    series = series.dropna()
    if len(series) <= periods:
        return np.nan
    return (series.iloc[-1] / series.iloc[-(periods + 1)] - 1) * 100


def rolling_zscore(series: pd.Series, window: int = 126):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    z = (series - mean) / std
    return z

start = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
all_tickers = list(RISK_ON.values()) + list(DEFENSIVE.values()) + list(SECTORS.keys())

if force_refresh:
    st.cache_data.clear()

prices = fetch_prices(all_tickers, start)
if prices.empty:
    st.error("No market data returned from Yahoo Finance.")
    st.stop()

returns = prices.pct_change()
latest_row = []

for name, ticker in {**RISK_ON, **DEFENSIVE}.items():
    s = prices[ticker].dropna() if ticker in prices else pd.Series(dtype=float)
    if s.empty:
        continue
    latest_row.append(
        {
            "Asset": name,
            "Ticker": ticker,
            "Last": float(s.iloc[-1]),
            "1D %": safe_pct_change(s, 1),
            "1W %": safe_pct_change(s, 5),
            "1M %": safe_pct_change(s, 21),
            "YTD %": (s.iloc[-1] / s[s.index.year == s.index[-1].year].iloc[0] - 1) * 100
            if (s.index.year == s.index[-1].year).any()
            else np.nan,
            "Z-Score (6M)": rolling_zscore(s, 126).iloc[-1],
        }
    )

snapshot = pd.DataFrame(latest_row)

# Headline regime cards
spx = prices.get("^GSPC", pd.Series(dtype=float)).dropna()
vix = prices.get("^VIX", pd.Series(dtype=float)).dropna()
tnx = prices.get("^TNX", pd.Series(dtype=float)).dropna()
dxy = prices.get("DX-Y.NYB", pd.Series(dtype=float)).dropna()

col1, col2, col3, col4 = st.columns(4)
col1.metric("S&P 500 (1D)", f"{spx.iloc[-1]:,.1f}" if not spx.empty else "N/A", f"{safe_pct_change(spx,1):.2f}%" if not spx.empty else "N/A")
col2.metric("VIX Level", f"{vix.iloc[-1]:.2f}" if not vix.empty else "N/A", f"{safe_pct_change(vix,1):.2f}%" if not vix.empty else "N/A")
col3.metric("US 10Y Yield", f"{tnx.iloc[-1]:.2f}%" if not tnx.empty else "N/A", f"{safe_pct_change(tnx,1):.2f}%" if not tnx.empty else "N/A")
col4.metric("DXY Dollar Index", f"{dxy.iloc[-1]:.2f}" if not dxy.empty else "N/A", f"{safe_pct_change(dxy,1):.2f}%" if not dxy.empty else "N/A")

st.markdown("---")

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Cross-Asset Performance Snapshot")
    if not snapshot.empty:
        styled = snapshot.copy()
        st.dataframe(
            styled.style.format(
                {
                    "Last": "{:.2f}",
                    "1D %": "{:.2f}",
                    "1W %": "{:.2f}",
                    "1M %": "{:.2f}",
                    "YTD %": "{:.2f}",
                    "Z-Score (6M)": "{:.2f}",
                }
            ),
            use_container_width=True,
            height=420,
        )

with right:
    st.subheader("Risk/Defense Composite")
    risk_assets = [t for t in RISK_ON.values() if t in returns.columns]
    defense_assets = [t for t in ["DX-Y.NYB", "^VIX", "TLT", "GC=F"] if t in returns.columns]

    risk_composite = returns[risk_assets].mean(axis=1).cumsum()
    defense_composite = returns[defense_assets].mean(axis=1).cumsum()

    comp_df = pd.DataFrame(
        {
            "Risk-On Composite": risk_composite,
            "Defensive Composite": defense_composite,
        }
    ).dropna()

    fig_comp = px.line(comp_df, x=comp_df.index, y=comp_df.columns)
    fig_comp.update_layout(margin=dict(t=20, b=20, l=20, r=20), legend_title_text="")
    st.plotly_chart(fig_comp, use_container_width=True)

st.subheader("Sector Rotation Heatmap (1W / 1M / YTD)")
sector_rows = []
for ticker, sector_name in SECTORS.items():
    if ticker not in prices:
        continue
    s = prices[ticker].dropna()
    if s.empty:
        continue
    sector_rows.append(
        {
            "Sector": sector_name,
            "1W": safe_pct_change(s, 5),
            "1M": safe_pct_change(s, 21),
            "YTD": (s.iloc[-1] / s[s.index.year == s.index[-1].year].iloc[0] - 1) * 100
            if (s.index.year == s.index[-1].year).any()
            else np.nan,
        }
    )

sector_df = pd.DataFrame(sector_rows).set_index("Sector") if sector_rows else pd.DataFrame()
if not sector_df.empty:
    heat = go.Figure(
        data=go.Heatmap(
            z=sector_df.values,
            x=sector_df.columns,
            y=sector_df.index,
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(sector_df.values, 2),
            texttemplate="%{text}%",
            hovertemplate="%{y}<br>%{x}: %{z:.2f}%<extra></extra>",
        )
    )
    heat.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=460)
    st.plotly_chart(heat, use_container_width=True)

st.subheader("Advanced Diagnostics")
adv1, adv2 = st.columns(2)

with adv1:
    st.markdown("**Rolling Correlation: SPX vs 10Y Yield**")
    if "^GSPC" in returns and "^TNX" in returns:
        corr = returns["^GSPC"].rolling(corr_window).corr(returns["^TNX"])
        fig_corr = px.line(corr.dropna(), title=None)
        fig_corr.update_layout(
            yaxis_title="Correlation",
            xaxis_title="",
            margin=dict(t=20, b=20, l=20, r=20),
        )
        fig_corr.add_hline(y=0, line_dash="dot")
        st.plotly_chart(fig_corr, use_container_width=True)

with adv2:
    st.markdown("**Breadth Proxy: Equal-Weight Sector Advance Ratio**")
    sector_tickers = [t for t in SECTORS if t in returns.columns]
    if sector_tickers:
        daily_adv = (returns[sector_tickers] > 0).mean(axis=1)
        breadth = daily_adv.rolling(20).mean()
        fig_breadth = px.line(breadth.dropna(), title=None)
        fig_breadth.update_layout(
            yaxis_title="20D Avg % Sectors Up",
            xaxis_title="",
            margin=dict(t=20, b=20, l=20, r=20),
        )
        fig_breadth.add_hline(y=0.5, line_dash="dot")
        st.plotly_chart(fig_breadth, use_container_width=True)

with st.expander("Data and update notes"):
    st.markdown(
        """
        - **Primary source:** Yahoo Finance via `yfinance` (publicly available).
        - **Auto-updates:** Cached for 5 minutes and can be manually refreshed from the sidebar.
        - **Included basics:** broad indexes, volatility, rates, dollar, key commodities, and sector performance.
        - **Included advanced tools:** risk/defense composite, rolling correlation regime check, and sector-breadth proxy.
        """
    )
