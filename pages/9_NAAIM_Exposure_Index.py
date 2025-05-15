import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

st.set_page_config(page_title="U.S. Equity Sentiment Dashboard", layout="wide")
st.title("🧭 U.S. Equity Sentiment/Exposure Proxies")

st.sidebar.header("About This Dashboard")
st.sidebar.markdown("""
This dashboard pulls working, public U.S. equity sentiment proxies from FRED:

- **AAII Bullish/Bearish Sentiment** — U.S. retail survey (% Bulls/Bears).
- **Put/Call Ratio** — CBOE equity put/call ratio (higher = more hedging/fear).

All data: FRED (Federal Reserve Economic Data).
""")

# -- Only include working FRED codes
PROXIES = {
    "AAII Bullish Sentiment (% Bulls)": {"code": "AAIIBULL", "label": "% Bulls", "hl": 38},
    "AAII Bearish Sentiment (% Bears)": {"code": "AAIIBEAR", "label": "% Bears", "hl": 30},
    "CBOE Equity Put/Call Ratio": {"code": "PUTCALL", "label": "Put/Call", "hl": 0.7}
}

choice = st.sidebar.selectbox("Choose a Sentiment/Exposure Proxy:", list(PROXIES.keys()))
proxy = PROXIES[choice]
code = proxy["code"]
ylab = proxy["label"]
hl = proxy["hl"]

@st.cache_data(ttl=86400)
def load_fred(code):
    try:
        df = pdr.DataReader(code, "fred")
        df = df.rename(columns={code: "Value"}).dropna()
        return df
    except Exception as e:
        return None

df = load_fred(code)
if df is None or df.empty:
    st.error(f"FRED series `{code}` is not available. Try another proxy.")
    st.stop()

# Chart
st.subheader(f"{choice} (Weekly)")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df["Value"], color="black", linewidth=2.2, label=ylab)
ax.axhline(hl, linestyle="--", color="gray", linewidth=1.2, label=f"Historical Reference ({hl})")
ax.set_title(choice, fontsize=18, weight="bold")
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel(ylab, fontsize=12)
ax.grid(alpha=0.3)
ax.legend(loc="best", frameon=False, fontsize=10)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)

# Latest value
last_val = df["Value"].iloc[-1]
last_date = df.index[-1].strftime("%b %d, %Y")
st.metric(
    label=f"Latest ({last_date})",
    value=f"{last_val:,.2f}" + ("" if 'Put/Call' in choice else "%")
)

with st.expander("See/download underlying data", expanded=False):
    st.dataframe(df.tail(156), use_container_width=True)
    st.download_button(
        "Download full series (CSV)",
        df.to_csv(index=True),
        file_name=f"{code}_FRED.csv"
    )
