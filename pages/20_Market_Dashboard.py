from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup


st.set_page_config(page_title="Market Dashboard", layout="wide")

st.title("Market Dashboard")
st.caption(
    "Finviz-style market cockpit with cross-asset tape, internals, leaders/laggards, sector map, and headline flow. "
    "Auto-updating market monitor covering broad risk, volatility, rates, dollar, breadth, sectors, and correlation structure."
)

with st.sidebar:
    st.header("Controls")
    lookback_days = st.slider("Lookback window (days)", min_value=90, max_value=1500, value=365, step=30)
    corr_window = st.selectbox("Rolling correlation window", [20, 30, 63, 126], index=2)
    auto_refresh = st.checkbox("Auto-refresh every 5 minutes", value=True)
    show_sp500_internals = st.checkbox("Compute S&P 500 internals (slower)", value=True)

# Auto-refresh without extra dependencies
if auto_refresh:
    st.markdown("<meta http-equiv='refresh' content='300'>", unsafe_allow_html=True)

RISK_ON = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Russell 2000": "^RUT",
    "Dow": "^DJI",
    "ACWI": "ACWI",
    "High Yield": "HYG",
}

MACRO_DEFENSIVE = {
    "US 10Y": "^TNX",
    "US 30Y": "^TYX",
    "DXY": "DX-Y.NYB",
    "Gold": "GC=F",
    "WTI": "CL=F",
    "Copper": "HG=F",
    "TLT": "TLT",
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
    return (s.iloc[-1] / s.iloc[-(periods + 1)] - 1.0) * 100.0


def ytd_return(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    curr_year = s.index[-1].year
    ytd = s[s.index.year == curr_year]
    if ytd.empty:
        return np.nan
    return (ytd.iloc[-1] / ytd.iloc[0] - 1.0) * 100.0


def rolling_zscore(series: pd.Series, window: int = 126) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std


@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(tickers: list[str], start_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = yf.download(
        tickers=tickers,
        start=start_date,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    close_df = pd.DataFrame()
    vol_df = pd.DataFrame()

    if raw is None or raw.empty:
        return close_df, vol_df

    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if (t, "Close") in raw.columns:
                close_df[t] = raw[(t, "Close")]
            if (t, "Volume") in raw.columns:
                vol_df[t] = raw[(t, "Volume")]
    else:
        # Single ticker case: columns are not MultiIndex
        if "Close" in raw.columns and len(tickers) == 1:
            close_df[tickers[0]] = raw["Close"]
        if "Volume" in raw.columns and len(tickers) == 1:
            vol_df[tickers[0]] = raw["Volume"]

    return close_df.dropna(how="all"), vol_df.dropna(how="all")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_sp500_symbols() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return table["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_intraday(ticker: str) -> pd.Series:
    data = yf.download(ticker, period="1d", interval="5m", progress=False, auto_adjust=False)
    if data is None or data.empty or "Close" not in data:
        return pd.Series(dtype=float)
    return data["Close"].dropna()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_rss_headlines() -> list[dict]:
    feeds = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EVIX&region=US&lang=en-US",
    ]
    out: list[dict] = []
    for feed in feeds:
        try:
            r = requests.get(feed, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "xml")
            for item in soup.find_all("item")[:8]:
                out.append(
                    {
                        "title": item.title.text.strip() if item.title else "",
                        "source": "Yahoo Finance",
                        "published": item.pubDate.text.strip() if item.pubDate else "",
                        "link": item.link.text.strip() if item.link else "",
                    }
                )
        except Exception:
            continue
    return out[:12]


def mini_chart(series: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    if not series.empty:
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", line=dict(width=2)))
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
core_tickers = (
    list(dict.fromkeys(list(RISK_ON.values()) + list(MACRO_DEFENSIVE.values()))) + list(SECTORS.keys()) + MEGA_CAPS
)
prices, volumes = fetch_prices(core_tickers, start)

if prices.empty:
    st.error("No market data returned. Yahoo may be temporarily blocked in this environment.")
    st.stop()

returns = prices.pct_change()

# ===== Index tape =====
st.subheader("Index Tape")
c1, c2, c3, c4 = st.columns(4)
for col, (name, ticker) in zip([c1, c2, c3, c4], list(RISK_ON.items())[:4]):
    intraday = fetch_intraday(ticker)
    s = prices.get(ticker, pd.Series(dtype=float)).dropna()
    last = s.iloc[-1] if not s.empty else np.nan
    delta = safe_pct_change(s, 1) if not s.empty else np.nan
    col.metric(name, f"{last:,.2f}" if pd.notna(last) else "N/A", f"{delta:.2f}%" if pd.notna(delta) else "N/A")
    col.plotly_chart(mini_chart(intraday, " "), use_container_width=True)

# ===== Internals =====
st.subheader("Market Internals")
ibar1, ibar2, ibar3, ibar4 = st.columns(4)

if show_sp500_internals:
    spx_symbols = fetch_sp500_symbols()
    sample_symbols = spx_symbols[:220]
    spx_prices, _ = fetch_prices(sample_symbols, start)

    if spx_prices.empty:
        ibar1.info("Internals unavailable (no constituent data).")
    else:
        spx_ret = spx_prices.pct_change().iloc[-1] * 100.0

        adv = int((spx_ret > 0).sum())
        dec = int((spx_ret <= 0).sum())

        last_row = spx_prices.iloc[-1]
        sma50_row = spx_prices.rolling(50).mean().iloc[-1]
        sma200_row = spx_prices.rolling(200).mean().iloc[-1]

        sma50 = int((last_row > sma50_row).sum())
        sma200 = int((last_row > sma200_row).sum())
        total = int(last_row.count()) if int(last_row.count()) > 0 else 1

        ibar1.metric("Advancing", f"{adv}", f"{(adv / total) * 100.0:.1f}%")
        ibar2.metric("Declining", f"{dec}", f"{(dec / total) * 100.0:.1f}%")
        ibar3.metric("Above SMA50", f"{sma50}", f"{(sma50 / total) * 100.0:.1f}%")
        ibar4.metric("Above SMA200", f"{sma200}", f"{(sma200 / total) * 100.0:.1f}%")
else:
    ibar1.info("Enable S&P internals in sidebar")

# ===== Movers + heatmap =====
left, right = st.columns([1.25, 1])

with left:
    st.subheader("Top Gainers / Losers / Most Active")
    universe = [
        t
        for t in (MEGA_CAPS + list(SECTORS.keys()) + ["SPY", "QQQ", "IWM", "DIA", "SMH", "XBI", "ARKK"])
        if t in prices.columns
    ]
    rows: list[dict] = []
    for t in universe:
        s = prices[t].dropna()
        v = volumes[t].dropna() if t in volumes.columns else pd.Series(dtype=float)
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
    if tape.empty:
        st.info("No tape data available.")
    else:
        g, l, a = st.tabs(["Top Gainers", "Top Losers", "Most Active"])
        with g:
            st.dataframe(tape.sort_values("1D %", ascending=False).head(15), use_container_width=True, hide_index=True)
        with l:
            st.dataframe(tape.sort_values("1D %", ascending=True).head(15), use_container_width=True, hide_index=True)
        with a:
            st.dataframe(
                tape.sort_values("Volume", ascending=False).head(15), use_container_width=True, hide_index=True
            )

with right:
    st.subheader("Sector Heatmap")
    sector_rows: list[dict] = []
    for t, name in SECTORS.items():
        if t not in prices.columns:
            continue
        s = prices[t].dropna()
        if s.empty:
            continue
        sector_rows.append(
            {
                "Sector": name,
                "Ticker": t,
                "1D %": safe_pct_change(s, 1),
                "1M %": safe_pct_change(s, 21),
            }
        )

    sector_df = pd.DataFrame(sector_rows)
    if sector_df.empty:
        st.info("Sector heatmap unavailable.")
    else:
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
                "Risk-On": returns[risk_assets].mean(axis=1).fillna(0).cumsum(),
                "Defensive": returns[defense_assets].mean(axis=1).fillna(0).cumsum(),
            }
        )
        st.plotly_chart(px.line(comp), use_container_width=True)
    else:
        st.info("Composite unavailable (missing tickers).")

with a2:
    st.markdown("**Rolling Correlation: SPX vs 10Y**")
    if "^GSPC" in returns.columns and "^TNX" in returns.columns:
        corr = returns["^GSPC"].rolling(corr_window).corr(returns["^TNX"])
        fig = px.line(corr.dropna())
        fig.add_hline(y=0, line_dash="dot")
        fig.update_layout(margin=dict(t=20, b=10, l=10, r=10), yaxis_title="Corr")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Correlation unavailable (missing SPX or 10Y).")

with a3:
    st.markdown("**Breadth Proxy: 20D Avg % Sectors Up**")
    sector_tickers = [t for t in SECTORS if t in returns.columns]
    if sector_tickers:
        breadth = (returns[sector_tickers] > 0).mean(axis=1).rolling(20).mean()
        fig = px.line(breadth.dropna())
        fig.add_hline(y=0.5, line_dash="dot")
        fig.update_layout(margin=dict(t=20, b=10, l=10, r=10), yaxis_title="% Up")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Breadth proxy unavailable (missing sector data).")

# ===== Headlines & macro table =====
st.subheader("Headlines & Macro Tape")
n1, n2 = st.columns([1.8, 1])

with n1:
    st.markdown("**Market Headlines**")
    headlines = fetch_rss_headlines()
    if headlines:
        for h in headlines[:10]:
            title = h.get("title", "").strip()
            link = h.get("link", "").strip()
            if title and link:
                st.markdown(f"- [{title}]({link})")
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
    macro_rows: list[dict] = []
    for label, t in macro_tickers.items():
        s = macro_prices.get(t, pd.Series(dtype=float)).dropna()
        if s.empty:
            continue
        macro_rows.append({"Asset": label, "Last": float(s.iloc[-1]), "1D %": safe_pct_change(s, 1)})

    if macro_rows:
        st.dataframe(pd.DataFrame(macro_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Macro snapshot unavailable.")

with st.expander("Data and update notes"):
    st.markdown(
        """
- Primary feed: Yahoo Finance via `yfinance` (public, no key).
- Headlines: Yahoo Finance RSS feeds.
- S&P internals: Constituents from Wikipedia plus market data from Yahoo.
- Auto-update: browser refresh every 5 minutes when enabled.
- Environment note: if this runtime blocks outbound quote requests, sections may show partial data.
        """.strip()
    )
