# pages/6_Google_Trends_Explorer.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# — Gracefully handle missing pytrends
try:
    from pytrends.request import TrendReq
except ModuleNotFoundError:
    st.error(
        """
        🚨 **Missing dependency**  
        The `pytrends` library isn’t installed in this environment.

        • **If you’re running locally**, do:
          ```
          pip install pytrends
          ```
        • **If you’re on Streamlit Cloud**, add `pytrends` to your `requirements.txt` (or `packages.txt`) and re‑deploy.
        """
    )
    st.stop()

# 1) Sidebar dropdown
TERMS = [
    "Recession", "Inflation", "Unemployment", "Layoffs",
    "Credit Crunch", "Rate Hike", "Bond Market Crash",
    "Stock Market Crash", "Hard Landing", "Stagflation",
    "Bank Run", "Yield Curve Inversion", "Debt Ceiling",
    "Hyperinflation", "Soft Landing"
]
st.sidebar.header("Google Trends Explorer")
selected_term = st.sidebar.selectbox("Choose a term:", TERMS)

# 2) Cache the pytrends fetch
@st.cache_data(ttl=3600, show_spinner=False)
def load_trends(term: str) -> pd.DataFrame:
    py = TrendReq(hl="en-US", tz=360)
    today = datetime.today().strftime("%Y-%m-%d")
    timeframe = f"2020-03-01 {today}"
    py.build_payload([term], timeframe=timeframe)
    df = py.interest_over_time()
    if term in df:
        return df[[term]]
    else:
        return pd.DataFrame()

# 3) Load & plot
data = load_trends(selected_term)

if data.empty:
    st.warning(f"No Google Trends data for **{selected_term}**.")
else:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data[selected_term], linewidth=1.5)
    ax.set_title(f'Search Interest: "{selected_term}"', pad=12)
    ax.set_ylabel("Score (0–100)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.2)

    # annotate top‑3 spikes
    spikes = data[selected_term].nlargest(3)
    for dt, val in spikes.items():
        ax.annotate(
            dt.strftime("%b %Y"),
            xy=(dt, val),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            arrowprops=dict(color="red", arrowstyle="->", lw=1),
        )

    st.pyplot(fig)

    if st.sidebar.checkbox("Show raw data"):
        st.dataframe(data.rename(columns={selected_term: "Trend Score"}))
