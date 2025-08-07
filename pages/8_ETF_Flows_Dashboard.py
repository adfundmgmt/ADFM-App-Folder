import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
import concurrent.futures

st.set_page_config(page_title="ETF Flows Dashboard", layout="wide")

# --------------------------- SIDEBAR ---------------------------
st.sidebar.title("ETF Flows")
st.sidebar.markdown("""
A dashboard of **thematic and global ETF flows** to see where money is moving among major macro and innovation trades.

- **Flows are proxies** computed as the sum of daily changes in shares outstanding multiplied by daily close.
""")

lookback_dict = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "12 Months": 365,
    "YTD": (datetime.now() - datetime(datetime.now().year, 1, 1)).days
}
period_label = st.sidebar.radio("Select Lookback Period", list(lookback_dict.keys()), index=1)
period_days = int(lookback_dict[period_label])

etf_info = {
    "MAGS": ("Mag 7", "Magnificent 7 stocks ETF"),
    "SMH": ("Semiconductors", "Semiconductor stocks (VanEck)"),

}
etf_tickers = list(etf_info.keys())

# --------------------------- HELPERS ---------------------------
def _fmt_compact_cur(x):
    if x is None or pd.isna(x):
        return ""
    ax = abs(x)
    sign = "-" if x < 0 else "+" if x > 0 else ""
    if ax >= 1e9:
        return f"{sign}${int(round(ax/1e9))}B"
    if ax >= 1e6:
        return f"{sign}${int(round(ax/1e6))}M"
    if ax >= 1e3:
        return f"{sign}${int(round(ax/1e3))}K"
    if ax < 1:
        return "$0"
    return f"{sign}${int(round(ax))}"

def _axis_fmt(x, _pos=None):
    ax = abs(x)
    if ax >= 1e9:
        return f"${x/1e9:,.0f}B"
    if ax >= 1e6:
        return f"${x/1e6:,.0f}M"
    if ax >= 1e3:
        return f"${x/1e3:,.0f}K"
    return f"${x:,.0f}"

# --------------------------- DATA FUNCTIONS ---------------------------
@st.cache_data(show_spinner=True, ttl=300)
def robust_flow_estimate(ticker: str, period_days: int):
    """
    Returns (flow_usd, flow_pct_of_aum, aum_usd) or (None, None, None) if not computable.
    Flow proxy is sum over the period of delta(shares_outstanding) * close_price.
    """
    try:
        t = yf.Ticker(ticker)
        # Fetch a little extra on both ends for forward fill and diff stability
        hist = t.history(period=f"{period_days+10}d", interval="1d", auto_adjust=False)
        hist = hist.dropna()
        if hist.empty or len(hist) < 2:
            return None, None, None

        # Try to get shares outstanding history
        so = None
        try:
            so = t.get_shares_full()
        except Exception:
            pass
        if so is None or (hasattr(so, "empty") and so.empty):
            try:
                so = t.get_shares()
            except Exception:
                so = None

        if so is None or (hasattr(so, "empty") and so.empty):
            return None, None, None

        # Align indices and coerce types
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        so.index = pd.to_datetime(so.index).tz_localize(None)
        so = pd.to_numeric(so, errors="coerce").dropna()
        if so.empty:
            return None, None, None

        # Restrict to hist range and forward fill to trading days
        so = so.sort_index().loc[hist.index.min():hist.index.max()]
        if len(so) < 2:
            return None, None, None

        so_daily = so.reindex(hist.index, method="ffill")
        dso = so_daily.diff().fillna(0.0)

        close = pd.to_numeric(hist["Close"], errors="coerce").fillna(method="ffill")
        if close.isna().all():
            return None, None, None

        # Flow proxy and AUM
        flow = float((dso * close).sum())
        aum = float(so_daily.iloc[-1] * close.iloc[-1]) if pd.notna(so_daily.iloc[-1]) and pd.notna(close.iloc[-1]) else None
        flow_pct = (flow / aum) if (aum is not None and aum != 0) else None

        return flow, flow_pct, aum

    except Exception:
        return None, None, None

@st.cache_data(show_spinner=True, ttl=300)
def get_all_flows(etf_tickers, period_days: int) -> pd.DataFrame:
    results = []

    # Cap concurrency to avoid rate limits
    def _runner(tk):
        return robust_flow_estimate(tk, period_days)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        flows = list(executor.map(_runner, etf_tickers))

    for i, ticker in enumerate(etf_tickers):
        cat, desc = etf_info[ticker]
        flow, flow_pct, aum = flows[i]
        results.append({
            "Ticker": ticker,
            "Category": cat,
            "Flow ($)": float(flow) if flow is not None else np.nan,
            "Flow (%)": float(flow_pct * 100) if flow_pct is not None else np.nan,
            "AUM ($)": float(aum) if aum is not None else np.nan,
            "Description": desc
        })
    df = pd.DataFrame(results)
    return df

# --------------------------- MAIN ---------------------------
st.title("ETF Flows Dashboard")
st.caption(f"Flows are proxies, not official. Period: {period_label}")

df = get_all_flows(etf_tickers, period_days)

# Sort by Flow ($) desc for chart rendering but keep a clean copy for tables
chart_df = df.sort_values("Flow ($)", ascending=False).copy()
chart_df["Label"] = [f"{etf_info[t][0]} ({t})" for t in chart_df["Ticker"]]

# Guard when no valid flow data
max_val = pd.to_numeric(chart_df["Flow ($)"], errors="coerce").abs().max()
if pd.isna(max_val) or max_val == 0:
    st.info("No valid flow data available for the selected period. This likely means shares outstanding history is not provided by the data source for these ETFs.")
else:
    # Build colors: green for positive, red for negative, gray for zero or NaN
    flows_series = pd.to_numeric(chart_df["Flow ($)"], errors="coerce")
    colors = []
    for x in flows_series:
        if pd.isna(x) or abs(x) < 1e-9:
            colors.append("gray")
        elif x > 0:
            colors.append("green")
        else:
            colors.append("red")

    fig, ax = plt.subplots(figsize=(15, max(6, len(chart_df) * 0.42)))
    bars = ax.barh(chart_df["Label"], flows_series.fillna(0.0), color=colors, alpha=0.85)

    ax.set_xlabel("Estimated Flow ($)")
    ax.set_title(f"ETF Proxy Flows - {period_label}")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_axis_fmt))

    # Axis limits with buffer
    min_flow = flows_series.min(skipna=True)
    max_flow = flows_series.max(skipna=True)
    abs_max = max(abs(min_flow if pd.notna(min_flow) else 0.0), abs(max_flow if pd.notna(max_flow) else 0.0))
    buffer = 0.15 * abs_max if abs_max > 0 else 1.0
    left_lim = -abs_max - buffer if pd.notna(min_flow) and min_flow < 0 else 0 - buffer * 0.15
    right_lim = abs_max + buffer
    ax.set_xlim([left_lim, right_lim])

    # Annotations with scale aware offset
    x_range = right_lim - left_lim
    for bar, val in zip(bars, flows_series):
        if pd.notna(val):
            label = _fmt_compact_cur(val)
            x_text = bar.get_width()
            align = "left" if val > 0 else "right" if val < 0 else "center"
            if val > 0:
                x_offset = 0.01 * x_range
            elif val < 0:
                x_offset = -0.01 * x_range
            else:
                x_offset = 0.0
            ax.text(
                x_text + x_offset,
                bar.get_y() + bar.get_height() / 2,
                label if label else "$0",
                va="center",
                ha=align,
                fontsize=10,
                color="black",
                clip_on=True
            )

    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("*Green indicates inflow, red indicates outflow, gray indicates missing or zero flow*")

# --------------------------- TOP FLOWS AND OUTFLOWS ---------------------------
st.markdown("#### Top Inflows and Outflows")
valid = df.dropna(subset=["Flow ($)"]).copy()

if valid.empty:
    st.write("No ETFs with computable flows in the selected period.")
else:
    valid["Label"] = [f"{etf_info[t][0]} ({t})" for t in valid["Ticker"]]
    top_in = valid.nlargest(3, "Flow ($)")[["Label", "Flow ($)"]].copy()
    top_out = valid.nsmallest(3, "Flow ($)")[["Label", "Flow ($)"]].copy()
    top_in["Flow"] = top_in["Flow ($)"].apply(_fmt_compact_cur)
    top_out["Flow"] = top_out["Flow ($)"].apply(_fmt_compact_cur)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Top Inflows**")
        st.table(top_in[["Flow"]].set_index(top_in["Label"]))
    with col2:
        st.write("**Top Outflows**")
        st.table(top_out[["Flow"]].set_index(top_out["Label"]))

# --------------------------- STATUS ---------------------------
if df["Flow ($)"].isna().any():
    st.warning("Some ETFs are missing flow data due to unavailable shares outstanding history.")
else:
    st.success("All flow proxies computed using available shares outstanding history.")

st.caption("Methodology: Flow proxy is the sum over the selected period of daily changes in shares outstanding multiplied by the daily close. AUM is shares outstanding at the end of the period multiplied by the close on that date.")
st.caption("© 2025 AD Fund Management LP")
