# -------------------------------------------------------------
#  Market Drawdown Dashboard — v3.1  |  AD Fund Management LP
# -------------------------------------------------------------
#  Changelog (2025‑05‑13)
#    • Renamed column header → Max Drawdown (was Max DD)
#    • Breadth snapshot now tied to the selected headline benchmark
#    • Dropped theme toggle (default = light)
#    • Hides indices with missing data from the drawdown table
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from functools import lru_cache

TODAY = datetime.today().date()
YEAR_START = datetime(TODAY.year, 1, 1)

st.set_page_config(
    page_title="Market Drawdown Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Simple CSS polish
st.markdown(
    """
    <style>
    body { font-family: 'Helvetica Neue', sans-serif; }
    .css-1ht1j8u .stMetric label { font-size: .9rem; }
    .css-1ht1j8u .stMetric div { font-size: 2.2rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# Sidebar – explainer & controls
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/1f4c8.svg", width=40)
    st.markdown(
        """### Year‑to‑Date Market Stress Dashboard  
**What you get:**
* **Drawdown matrix** — instant view of how major global equity benchmarks have performed YTD, their bounces off the low, and max pain from the high.
* **Breadth gauges** — real‑time internals for the **selected benchmark**: trend (% above 50‑/200‑day MAs), momentum (% near 52‑week highs) and relative strength (constituents beating the index YTD).

Select a benchmark below — all panels update in sync."""
    )
    st.markdown("---")
    benchmark = st.selectbox(
        "Headline benchmark",
        (
            "S&P 500",
            "Nasdaq 100",
            "Russell 2000",
            "Dow 30",
            "Euro Stoxx 50",
            "FTSE 100",
            "Nikkei 225",
            "TOPIX",
            "FXI (China Large‑Cap)",
        ),
        index=0,
    )

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_prices(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, progress=False, group_by="ticker", auto_adjust=False)
    def adj(df):
        if isinstance(df.columns, pd.MultiIndex):
            lvl = "Adj Close" if "Adj Close" in df.columns.get_level_values(1) else "Close"
            return df.swaplevel(axis=1)[lvl]
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        return df[[col]].rename(columns={col: tickers[0]})
    return adj(raw).sort_index()

@lru_cache(maxsize=32)
def get_members(index_name: str):
    """Safely retrieve constituent tickers by scraping Wikipedia or ETF holdings.
    Falls back gracefully if the expected table layout changes."""
    import pandas as pd

    def first_table_with(col_name: str, url: str):
        try:
            dfs = pd.read_html(url, header=0)
            for d in dfs:
                if col_name in d.columns:
                    return d[col_name].tolist()
        except Exception:
            pass
        return []  # empty → handled upstream

    if index_name == "S&P 500":
        return first_table_with("Symbol", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")

    if index_name == "Nasdaq 100":
        out = first_table_with("Ticker", "https://en.wikipedia.org/wiki/Nasdaq-100")
        return [t.strip() for t in out]

    if index_name == "Russell 2000":
        return yf.Ticker("IWM").fund_holdings.get("symbol", []).tolist()

    if index_name == "Dow 30":
        return first_table_with("Symbol", "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")

    if index_name == "Euro Stoxx 50":
        tickers = first_table_with("Ticker symbol", "https://en.wikipedia.org/wiki/EURO_STOXX_50")
        # adjust Swiss listing suffixes if needed
        return [t.replace(".ST", ".SW") for t in tickers]

    if index_name == "FTSE 100":
        return first_table_with("EPIC", "https://en.wikipedia.org/wiki/FTSE_100_Index")

    if index_name == "Nikkei 225":
        return first_table_with("Ticker", "https://en.wikipedia.org/wiki/Nikkei_225")

    if index_name == "TOPIX":
        return yf.Ticker("1306.T").fund_holdings.get("symbol", []).tolist()

    if index_name == "FXI (China Large‑Cap)":
        return yf.Ticker("FXI").fund_holdings.get("symbol", []).tolist()

    return []

# Map display name → yfinance ticker
INDEX_TICKERS = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Russell 2000": "^RUT",
    "Dow 30": "^DJI",
    "Euro Stoxx 50": "^STOXX50E",
    "FTSE 100": "^FTSE",
    "Nikkei 225": "^N225",
    "TOPIX": "^TOPX",
    "FXI (China Large‑Cap)": "FXI",
}

# -------------------------------------------------------------
# Drawdown matrix (filter out missing)
# -------------------------------------------------------------
all_px = load_prices(list(INDEX_TICKERS.values()), YEAR_START, TODAY + timedelta(days=1))
rows = []
for name, tk in INDEX_TICKERS.items():
    ser = all_px[tk]
    if ser.isna().all():
        continue  # no data
    rows.append(
        {
            "Index": name,
            "YTD": ser.iloc[-1] / ser.iloc[0] - 1,
            "From YTD Low": ser.iloc[-1] / ser.min() - 1,
            "Max Drawdown": ser.iloc[-1] / ser.max() - 1,
        }
    )

dd = pd.DataFrame(rows)
fmt = {c: "{:+.1%}" for c in dd.columns if c != "Index"}
color = lambda v: f"color:{'green' if v > 0 else 'red'}" if isinstance(v, (int, float)) else ""

st.markdown("## YTD Drawdowns (Global Benchmarks)")
st.dataframe(
    dd.style.format(fmt).applymap(color, subset=["YTD", "From YTD Low", "Max Drawdown"]),
    use_container_width=True,
    height=min(360, 60 + 30 * len(dd)),
)

# -------------------------------------------------------------
# KPI strip for selected benchmark
# -------------------------------------------------------------
sel_ser = all_px[INDEX_TICKERS[benchmark]]
kp1, kp2, kp3 = st.columns(3)
kp1.metric("YTD Return", f"{sel_ser.iloc[-1] / sel_ser.iloc[0] - 1:+.1%}")
kp2.metric("Bounce from YTD Low", f"{sel_ser.iloc[-1] / sel_ser.min() - 1:+.1%}")
kp3.metric("Max Drawdown", f"{sel_ser.iloc[-1] / sel_ser.max() - 1:.1%}")

# -------------------------------------------------------------
# Breadth snapshot tied to benchmark
# -------------------------------------------------------------
constituents = get_members(benchmark)
if constituents:
    # Limit to first 800 to keep download light
    tickers_subset = constituents[:800]
    hist = load_prices(tickers_subset, TODAY - timedelta(days=400), TODAY + timedelta(days=1))
    latest = hist.iloc[-1]
    ma50 = hist.rolling(50).mean().iloc[-1]
    ma200 = hist.rolling(200).mean().iloc[-1]
    high52 = hist.rolling(252).max().iloc[-1]

    pct_above50 = (latest > ma50).mean()
    pct_above200 = (latest > ma200).mean()
    pct_near_high = (latest >= high52 * 0.98).mean()

    idx_ytd = sel_ser.iloc[-1] / sel_ser.iloc[0] - 1
    member_ytd = hist.apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)
    pct_beat_idx = (member_ytd > idx_ytd).mean()

    st.markdown(f"### {benchmark} Breadth Snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Above 50‑d MA", f"{pct_above50:.0%}")
    c2.metric("Above 200‑d MA", f"{pct_above200:.0%}")
    c3.metric("Near 52‑w High (<2%)", f"{pct_near_high:.0%}")
    c4.metric("Beating Index YTD", f"{pct_beat_idx:.0%}")
else:
    st.info("Breadth metrics unavailable for this benchmark (no constituent list)")

# -------------------------------------------------------------
# Price chart
# -------------------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=sel_ser.index, y=sel_ser, mode="lines", name=benchmark))
fig.add_trace(go.Scatter(x=[sel_ser.idxmax()], y=[sel_ser.max()], mode="markers", marker_symbol="triangle-up", marker_size=10, name="YTD High"))
fig.add_trace(go.Scatter(x=[sel_ser.idxmin()], y=[sel_ser.min()], mode="markers", marker_symbol="triangle-down", marker_size=10, name="YTD Low"))
fig.update_layout(title=f"{benchmark} — Price YTD", hovermode="x unified", height=420, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# Downloads
# -------------------------------------------------------------
col1, col2 = st.columns(2)
col1.download_button("Download Drawdowns CSV", dd.to_csv(index=False).encode(), "global_drawdowns.csv", "text/csv")
if constituents:
    breadth_df = pd.DataFrame(
        {
            "Metric": ["Above50d", "Above200d", "NearHigh", "BeatYTD"],
            "Value": [pct_above50, pct_above200, pct_near_high, pct_beat_idx],
        }
    )
    col2.download_button("Download Breadth CSV", breadth_df.to_csv(index=False).encode(), f"{benchmark.replace(' ', '')}_breadth.csv", "text/csv")

st.markdown(
    """<div style='text-align:center;font-size:0.75rem;margin-top:2rem;'>Data: Yahoo Finance • Calculations: AD Fund Management LP • Past performance is no guarantee of future results</div>""",
    unsafe_allow_html=True,
)
