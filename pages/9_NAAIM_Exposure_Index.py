import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

st.set_page_config(page_title="U.S. Sentiment & Exposure Dashboard", layout="wide")
st.title("🧭 U.S. Equity Sentiment & Exposure Proxies")

st.sidebar.header("About This Dashboard")
st.sidebar.markdown("""
While the official NAAIM Exposure Index is no longer available via public API, these widely-used sentiment/exposure proxies are reliable for monitoring U.S. equity positioning:

- **AAII Bullish Sentiment (% Bulls)** — U.S. retail investor survey, updated weekly.
- **CFTC S&P 500 Net Non-Commercial Positions** — Professional trader/fund futures net exposure, weekly.

**Data source:** FRED (Federal Reserve Economic Data)
""")

# --- Data Source Selection ---
dataset = st.sidebar.radio(
    "Select a Sentiment/Exposure Proxy:",
    [
        "AAII Bullish Sentiment (% Bulls)",
        "CFTC S&P 500 Net Non-Commercial (Contracts)"
    ]
)

if dataset == "AAII Bullish Sentiment (% Bulls)":
    FRED_CODE = "AAIIBULL"
    TITLE = "AAII Investor Sentiment: % Bulls"
    YLABEL = "Percent (%)"
    HL = 38  # Historical median
elif dataset == "CFTC S&P 500 Net Non-Commercial (Contracts)":
    FRED_CODE = "CFTC131741_F_L_NET"
    TITLE = "CFTC S&P 500 Futures: Net Non-Commercial Positions"
    YLABEL = "Contracts (Net Long/Short)"
    HL = 0

# --- Load Data from FRED ---
@st.cache_data(ttl=86400)
def load_fred_series(code):
    df = pdr.DataReader(code, "fred")
    df = df.rename(columns={code: "Value"})
    df = df.dropna()
    return df

try:
    df = load_fred_series(FRED_CODE)
    st.subheader(f"{TITLE} (Weekly)")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Value"], color="black", linewidth=2.2, label=TITLE)

    # Reference lines
    if dataset == "AAII Bullish Sentiment (% Bulls)":
        ax.axhline(HL, linestyle="--", color="gray", linewidth=1.1, label=f"Historical Median ({HL}%)")
        ax.set_ylim(0, 75)
    else:
        ax.axhline(0, linestyle="--", color="gray", linewidth=1.1, label="Net Flat (0 Contracts)")
        # Dynamic y-limits for clarity
        ymin = df["Value"].min() * 1.2 if df["Value"].min() < 0 else df["Value"].min() * 0.8
        ymax = df["Value"].max() * 1.2 if df["Value"].max() > 0 else df["Value"].max() * 0.8
        ax.set_ylim(ymin, ymax)

    ax.set_title(TITLE, fontsize=18, weight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(YLABEL, fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=False, fontsize=10)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Show latest value
    last_val = df["Value"].iloc[-1]
    last_date = df.index[-1].strftime("%b %d, %Y")
    st.metric(
        label=f"Latest ({last_date})",
        value=f"{last_val:,.1f}" + ("%" if dataset == "AAII Bullish Sentiment (% Bulls)" else "")
    )

    with st.expander("See underlying data (downloadable)", expanded=False):
        st.dataframe(df.tail(156), use_container_width=True)  # ~3 years
        st.download_button(
            "Download full series (CSV)",
            df.to_csv(index=True),
            file_name=f"{FRED_CODE}_FRED.csv"
        )

except Exception as e:
    st.error(f"Failed to fetch data from FRED: {e}")
