from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
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
    "Finviz-style market cockpit with cross-asset tape, internals, leaders/laggards, sector map, and headline flow."
    "Auto-updating market monitor covering broad risk, volatility, rates, dollar, breadth, sectors, and correlation structure."
)

with st.sidebar:
    st.header("Controls")
    lookback_days = st.slider("Lookback window (days)", min_value=90, max_value=1500, value=365, step=30)
    corr_window = st.selectbox("Rolling correlation window", [20, 30, 63, 126], index=2)
    auto_refresh = st.checkbox("Auto-refresh every 5 minutes", value=True)
    show_sp500_internals = st.checkbox("Compute S&P 500 internals (slower)", value=True)
    if auto_refresh:
        st.caption("Cache TTL is 5 minutes. Use refresh to bypass cache.")
    if auto_refresh:
        st.caption("Data cache TTL is 5 minutes. Refresh button bypasses cache.")

    force_refresh = st.button("Refresh now")

if auto_refresh:
    st.query_params["t"] = datetime.utcnow().strftime("%Y%m%d%H%M")

if force_refresh:
    st.cache_data.clear()

RISK_ON = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Russell 2000": "^RUT",
    "Dow": "^DJI",
    "ACWI": "ACWI",
    "High Yield": "HYG",
}

DEFENSIVE = {
    "VIX": "^VIX",
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
    "WTI": "CL=F",
    "TLT": "TLT",
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

MEGA_CAPS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "LLY"]


def safe_pct_change(series: pd.Series, periods: int = 1) -> float:
    s = series.dropna()
    if len(s) <= periods:
        return np.nan
    return (s.iloc[-1] / s.iloc[-(periods + 1)] - 1) * 100


def ytd_return(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    curr_year = s.index[-1].year
    ytd = s[s.index.year == curr_year]
    if ytd.empty:
        return np.nan
    return (ytd.iloc[-1] / ytd.iloc[0] - 1) * 100


@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(tickers: list[str], start_date: str) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        start=start_date,
        progress=False,
        auto_adjust=False,
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
    vol_df = pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if (t, "Close") in raw.columns:
                close_df[t] = raw[(t, "Close")]
            if (t, "Volume") in raw.columns:
                vol_df[t] = raw[(t, "Volume")]
    else:
        if "Close" in raw.columns and len(tickers) == 1:
            close_df[tickers[0]] = raw["Close"]
        if "Volume" in raw.columns and len(tickers) == 1:
            vol_df[tickers[0]] = raw["Volume"]

    return close_df.dropna(how="all"), vol_df.dropna(how="all")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_sp500_symbols() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    symbols = table["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    return symbols


@st.cache_data(ttl=300, show_spinner=False)
def fetch_intraday(ticker: str) -> pd.Series:
    data = yf.download(ticker, period="1d", interval="5m", progress=False, auto_adjust=False)
    if data.empty or "Close" not in data:
        return pd.Series(dtype=float)
    return data["Close"].dropna()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_rss_headlines() -> list[dict]:
    feeds = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EVIX&region=US&lang=en-US",
    ]
    headlines = []
    for feed in feeds:
        try:
            r = requests.get(feed, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "xml")
            for item in soup.find_all("item")[:8]:
                headlines.append(
                    {
                        "title": item.title.text.strip() if item.title else "",
                        "source": "Yahoo Finance",
                        "published": item.pubDate.text.strip() if item.pubDate else "",
                        "link": item.link.text.strip() if item.link else "",
                    }
                )
        except Exception:
            continue
    return headlines[:12]


def mini_chart(series: pd.Series, title: str, color: str = "#33cc66") -> go.Figure:
    fig = go.Figure()
    if not series.empty:
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", line=dict(color=color, width=2)))
        fig.update_layout(
            title=title,
            margin=dict(l=10, r=10, t=30, b=10),
            height=170,
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(120,120,120,0.2)", tickformat=".2f"),
        )
    else:
        fig.update_layout(title=f"{title} (no intraday data)", height=170, margin=dict(l=10, r=10, t=30, b=10))
    return fig


start = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
core_tickers = list(RISK_ON.values()) + list(DEFENSIVE.values()) + list(SECTORS.keys()) + MEGA_CAPS
prices, volumes = fetch_prices(core_tickers, start)

if prices.empty:
    st.error("No market data returned. Yahoo may be temporarily blocked in this environment.")
    st.stop()

returns = prices.pct_change()

# ===== Top: index mini charts (Finviz-style tape) =====
st.subheader("Index Tape")
c1, c2, c3, c4 = st.columns(4)
for col, (name, ticker) in zip([c1, c2, c3, c4], list(RISK_ON.items())[:4]):
    intraday = fetch_intraday(ticker)
    delta = safe_pct_change(prices[ticker], 1) if ticker in prices else np.nan
    col.metric(name, f"{prices[ticker].dropna().iloc[-1]:,.2f}" if ticker in prices and not prices[ticker].dropna().empty else "N/A", f"{delta:.2f}%" if pd.notna(delta) else "N/A")
    col.plotly_chart(mini_chart(intraday, " "), use_container_width=True)

# ===== Internals bars =====
st.subheader("Market Internals")
ibar1, ibar2, ibar3, ibar4 = st.columns(4)

if show_sp500_internals:
    spx_symbols = fetch_sp500_symbols()
    sample_symbols = spx_symbols[:220]
    spx_prices, _ = fetch_prices(sample_symbols, start)
    spx_ret = spx_prices.pct_change().iloc[-1] * 100 if not spx_prices.empty else pd.Series(dtype=float)

    adv = int((spx_ret > 0).sum()) if not spx_ret.empty else 0
    dec = int((spx_ret <= 0).sum()) if not spx_ret.empty else 0

    sma50 = (spx_prices.iloc[-1] > spx_prices.rolling(50).mean().iloc[-1]).sum() if not spx_prices.empty else 0
    sma200 = (spx_prices.iloc[-1] > spx_prices.rolling(200).mean().iloc[-1]).sum() if not spx_prices.empty else 0
    total = int(spx_prices.iloc[-1].count()) if not spx_prices.empty else 1

    ibar1.metric("Advancing", f"{adv}", f"{(adv / max(total, 1)) * 100:.1f}%")
    ibar2.metric("Declining", f"{dec}", f"{(dec / max(total, 1)) * 100:.1f}%")
    ibar3.metric("Above SMA50", f"{int(sma50)}", f"{(sma50 / max(total, 1)) * 100:.1f}%")
    ibar4.metric("Above SMA200", f"{int(sma200)}", f"{(sma200 / max(total, 1)) * 100:.1f}%")
else:
    ibar1.info("Enable S&P internals in sidebar")

# ===== Movers + heatmap =====
left, right = st.columns([1.25, 1])

with left:
    st.subheader("Top Gainers / Losers / Most Active")
    universe = [t for t in MEGA_CAPS + list(SECTORS.keys()) + ["SPY", "QQQ", "IWM", "DIA", "SMH", "XBI", "ARKK"] if t in prices.columns]
    rows = []
    for t in universe:
        s = prices[t].dropna()
        v = volumes[t].dropna() if t in volumes else pd.Series(dtype=float)
        if s.empty:
            continue
        rows.append(
            {
                "Ticker": t,
                "Last": float(s.iloc[-1]),
                "1D %": safe_pct_change(s, 1),
                "1W %": safe_pct_change(s, 5),
                "YTD %": ytd_return(s),
                "Volume": int(v.iloc[-1]) if not v.empty else np.nan,
            }
        )

    tape = pd.DataFrame(rows)
    if not tape.empty:
        g, l, a = st.tabs(["Top Gainers", "Top Losers", "Most Active"])
        with g:
            st.dataframe(tape.sort_values("1D %", ascending=False).head(15), use_container_width=True)
        with l:
            st.dataframe(tape.sort_values("1D %", ascending=True).head(15), use_container_width=True)
        with a:
            st.dataframe(tape.sort_values("Volume", ascending=False).head(15), use_container_width=True)

with right:
    st.subheader("Sector Heatmap")
    sector_rows = []
    for t, name in SECTORS.items():
        if t not in prices:
            continue
        s = prices[t].dropna()
        if s.empty:
            continue
        sector_rows.append({"Sector": name, "Ticker": t, "1D %": safe_pct_change(s, 1), "1M %": safe_pct_change(s, 21)})

    sector_df = pd.DataFrame(sector_rows)
    if not sector_df.empty:
        fig_tree = px.treemap(
            sector_df,
            path=["Sector"],
            values=np.abs(sector_df["1D %"].fillna(0)) + 0.01,
            color="1D %",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            custom_data=["Ticker", "1M %"],
        )
        fig_tree.update_traces(
            hovertemplate="%{label}<br>1D: %{color:.2f}%<br>Ticker: %{customdata[0]}<br>1M: %{customdata[1]:.2f}%<extra></extra>"
        )
        fig_tree.update_layout(margin=dict(t=20, b=10, l=10, r=10), height=450)
        st.plotly_chart(fig_tree, use_container_width=True)

# ===== Advanced diagnostics =====
st.subheader("Advanced Diagnostics")
a1, a2, a3 = st.columns(3)

with a1:
    st.markdown("**Risk-On vs Defensive Composite**")
    risk_assets = [t for t in RISK_ON.values() if t in returns.columns]
    defense_assets = [t for t in ["DX-Y.NYB", "^VIX", "TLT", "GC=F"] if t in returns.columns]
    if risk_assets and defense_assets:
        comp = pd.DataFrame(
            {
                "Risk-On": returns[risk_assets].mean(axis=1).cumsum(),
                "Defensive": returns[defense_assets].mean(axis=1).cumsum(),
            }
        ).dropna()
        st.plotly_chart(px.line(comp), use_container_width=True)

with a2:
    st.markdown("**Rolling Correlation: SPX vs 10Y**")
    if "^GSPC" in returns and "^TNX" in returns:
        corr = returns["^GSPC"].rolling(corr_window).corr(returns["^TNX"])
        fig = px.line(corr.dropna())
        fig.add_hline(y=0, line_dash="dot")
        fig.update_layout(margin=dict(t=20, b=10, l=10, r=10), yaxis_title="Corr")
        st.plotly_chart(fig, use_container_width=True)

with a3:
    st.markdown("**Breadth Proxy: 20D Avg % Sectors Up**")
    sector_tickers = [t for t in SECTORS if t in returns.columns]
    if sector_tickers:
        breadth = (returns[sector_tickers] > 0).mean(axis=1).rolling(20).mean()
        fig = px.line(breadth.dropna())
        fig.add_hline(y=0.5, line_dash="dot")
        fig.update_layout(margin=dict(t=20, b=10, l=10, r=10), yaxis_title="% Up")
        st.plotly_chart(fig, use_container_width=True)

# ===== News and macro blocks =====
st.subheader("Headlines & Macro Tape")
n1, n2 = st.columns([1.8, 1])

with n1:
    st.markdown("**Market Headlines**")
    headlines = fetch_rss_headlines()
    if headlines:
        for h in headlines[:10]:
            st.markdown(f"- [{h['title']}]({h['link']})  ")
    else:
        st.info("No headlines available right now.")

with n2:
    st.markdown("**Futures / FX / Rates Snapshot**")
    macro_tickers = {
        "S&P 500": "^GSPC",
        "Nasdaq 100": "^NDX",
        "WTI": "CL=F",
        "Gold": "GC=F",
        "EUR/USD": "EURUSD=X",
        "USD/JPY": "JPY=X",
        "10Y": "^TNX",
        "30Y": "^TYX",
    }
    macro_prices, _ = fetch_prices(list(macro_tickers.values()), start)
    macro_rows = []
    for label, t in macro_tickers.items():
        if t not in macro_prices:
            continue
        s = macro_prices[t].dropna()
        if s.empty:
            continue
        macro_rows.append({"Asset": label, "Last": s.iloc[-1], "1D %": safe_pct_change(s, 1)})

    if macro_rows:
        st.dataframe(pd.DataFrame(macro_rows), use_container_width=True, hide_index=True)

with st.expander("Data and caveats"):
    st.markdown(
        """
        - **Primary feed:** Yahoo Finance via `yfinance` (public, no key).
        - **Headlines:** Yahoo Finance RSS feeds.
        - **S&P internals:** Constituents from Wikipedia + market data from Yahoo.
        - **Auto-update:** 5-minute cache TTL + manual refresh button.
        - **Environment note:** If this runtime blocks outbound quote requests, sections may show partial data.
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
