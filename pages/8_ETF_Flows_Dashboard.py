import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
import concurrent.futures

st.set_page_config(page_title="ETF Flows Dashboard", layout="wide")

st.sidebar.title("ETF Flows")
st.sidebar.markdown("""
A dashboard of **thematic and global ETF flows** — see where money is moving among major macro and innovation trades.

- **Flows are proxies** using daily shares outstanding × price.
""")

lookback_dict = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "12 Months": 365,
    "YTD": (datetime.now() - datetime(datetime.now().year, 1, 1)).days
}
period_label = st.sidebar.radio("Select Lookback Period", list(lookback_dict.keys()), index=1)
period_days = lookback_dict[period_label]

etf_info = {
    "MAGS": ("Mag 7", "Magnificent 7 stocks ETF"),
    "SMH": ("Semiconductors", "Semiconductor stocks (VanEck)"),
    "BOTZ": ("Robotics/AI", "Global robotics and AI leaders"),
    "ICLN": ("Clean Energy", "Global clean energy stocks"),
    "URNM": ("Uranium", "Uranium miners (Sprott)"),
    "ARKK": ("Innovation", "Disruptive growth stocks (ARK)"),
    "KWEB": ("China Internet", "China internet leaders (KraneShares)"),
    "FXI": ("China Large-Cap", "China mega-cap stocks"),
    "EWZ": ("Brazil", "Brazil large-cap equities"),
    "EEM": ("Emerging Markets", "EM equities (MSCI)"),
    "VWO": ("Emerging Markets", "EM equities (Vanguard)"),
    "VGK": ("Europe Large-Cap", "Developed Europe stocks (Vanguard)"),
    "FEZ": ("Eurozone", "Euro STOXX 50 ETF"),
    "ILF": ("Latin America", "Latin America 40 ETF"),
    "ARGT": ("Argentina", "Global X MSCI Argentina ETF"),
    "GLD": ("Gold", "SPDR Gold Trust ETF"),
    "SLV": ("Silver", "iShares Silver Trust ETF"),
    "DBC": ("Commodities", "Invesco DB Commodity Index ETF"),
    "HEDJ": ("Hedged Europe", "WisdomTree Europe Hedged Equity ETF"),
    "USMV": ("US Min Volatility", "iShares MSCI USA Min Volatility ETF"),
    "COWZ": ("US Free Cash Flow", "Pacer US Cash Cows 100 ETF"),
    "BITO": ("BTC Futures", "Bitcoin futures ETF"),
    "IBIT": ("Spot BTC", "BlackRock spot Bitcoin ETF"),
    "BIL": ("1-3mo T-Bills", "1-3 month U.S. Treasury bills"),
    "TLT": ("20+yr Treasuries", "20+ year U.S. Treasuries"),
    "SHV": ("0-1yr T-Bills", "Short-term Treasury bonds"),
}
etf_tickers = list(etf_info.keys())

@st.cache_data(show_spinner=True)
def robust_flow_estimate(ticker, period_days):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=f"{period_days+10}d")
        hist = hist.dropna()
        if hist.empty or len(hist) < 2:
            return None, None, None
        try:
            so = t.get_shares_full()
            if so is not None and not so.empty:
                so = so.dropna()
                so.index = pd.to_datetime(so.index)
                so = so.loc[hist.index[0]:hist.index[-1]]
                if len(so) >= 2:
                    so = so.reindex(hist.index, method='ffill')
                    start, end = so.iloc[0], so.iloc[-1]
                    close = hist['Close']
                    flow = (end - start) * close.iloc[-1]
                    aum = end * close.iloc[-1]
                    flow_pct = flow / aum if aum != 0 else None
                    return flow, flow_pct, aum
        except Exception:
            pass
        flow = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) * hist['Volume'].mean()
        aum = hist['Close'].iloc[-1] * 1e6  # fallback guess: $1B AUM proxy
        flow_pct = flow / aum if aum != 0 else None
        return flow, flow_pct, aum
    except Exception:
        return None, None, None

@st.cache_data(show_spinner=True)
def get_all_flows(etf_tickers, period_days):
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        flows = list(executor.map(lambda t: robust_flow_estimate(t, period_days), etf_tickers))
    for i, ticker in enumerate(etf_tickers):
        cat, desc = etf_info[ticker]
        flow, flow_pct, aum = flows[i]
        results.append({
            "Ticker": ticker,
            "Category": cat,
            "Flow ($)": flow,
            "Flow (%)": flow_pct * 100 if flow_pct is not None else None,
            "AUM ($)": aum,
            "Description": desc
        })
    return pd.DataFrame(results)

df = get_all_flows(etf_tickers, period_days)
df = df.sort_values("Flow ($)", ascending=False)
df['Label'] = [f"{etf_info[t][0]} ({t})" for t in df['Ticker']]

def flow_label(x):
    if x is None:
        return ""
    abs_x = abs(x)
    if abs_x >= 1e9:
        return f"{'-' if x < 0 else '+'}{int(round(abs_x / 1e9))}B"
    elif abs_x >= 1e6:
        return f"{'-' if x < 0 else '+'}{int(round(abs_x / 1e6))}M"
    elif abs_x >= 1e3:
        return f"{'-' if x < 0 else '+'}{int(round(abs_x / 1e3))}K"
    else:
        return f"{x:,.0f}"

# ------ MAIN CONTENT ------
st.title("ETF Flows Dashboard")
st.caption(f"Flows are proxies (not official). Period: **{period_label}**")

chart_df = df
max_val = chart_df['Flow ($)'].dropna().abs().max()
buffer = max_val * 0.15

fig, ax = plt.subplots(figsize=(15, max(6, len(chart_df) * 0.42)))
bars = ax.barh(
    chart_df['Label'],
    chart_df['Flow ($)'].fillna(0),
    color=[
        'green' if (x is not None and x > 0) else ('red' if (x is not None and x < 0) else 'gray')
        for x in chart_df['Flow ($)']
    ],
    alpha=0.8
)
ax.set_xlabel('Estimated Flow ($)')
ax.set_title(f'ETF Proxy Flows – {period_label}')
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e9:,.0f}B' if abs(x)>=1e9 else f'${x/1e6:,.0f}M'))

# Set axis limits for better scaling/label fit
ax.set_xlim([-buffer if min(chart_df['Flow ($)'].fillna(0)) < 0 else 0, max_val + buffer])

# Annotate: show only amount, right-aligned if positive, left-aligned if negative
for bar, val in zip(bars, chart_df['Flow ($)']):
    if val is not None:
        label = flow_label(val)
        x_text = bar.get_width()
        align = 'left' if val > 0 else 'right'
        x_offset = 1e7 if (val > 0) else -1e7
        # keep annotation inside the chart, clip if too wide
        text_x = x_text + x_offset if (abs(x_text) + buffer*0.5 < ax.get_xlim()[1]) else x_text - (buffer*0.05 if val > 0 else 0)
        ax.text(
            text_x,
            bar.get_y() + bar.get_height() / 2,
            label,
            va='center',
            ha=align,
            fontsize=10,
            color='black',
            clip_on=True
        )
plt.tight_layout()
st.pyplot(fig)
st.markdown("*Green: inflow, Red: outflow, Gray: missing or flat*")

# ------ TOP FLOWS / OUTFLOWS SUMMARY ------
st.markdown("#### Top Inflows & Outflows")
top_in = df.head(3)[["Label", "Flow ($)"]].copy()
top_in['Flow'] = top_in['Flow ($)'].apply(flow_label)
top_out = df.sort_values("Flow ($)").head(3)[["Label", "Flow ($)"]].copy()
top_out['Flow'] = top_out['Flow ($)'].apply(flow_label)
col1, col2 = st.columns(2)
with col1:
    st.write("**Top Inflows**")
    st.table(top_in[["Flow"]].set_index(top_in["Label"]))
with col2:
    st.write("**Top Outflows**")
    st.table(top_out[["Flow"]].set_index(top_out["Label"]))

if df['Flow ($)'].isnull().any():
    st.warning("Some ETFs are missing flow data (no shares outstanding history or price). Gray bars indicate incomplete flow data.")
else:
    st.success("All flow proxies calculated using latest data.")

st.caption("© 2025 AD Fund Management LP")
