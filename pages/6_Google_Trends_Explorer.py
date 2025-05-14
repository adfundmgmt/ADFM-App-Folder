# pages/trends.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pytrends.request import TrendReq
from datetime import datetime

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
@st.cache_data(show_spinner=False, ttl=3600)
def load_trends(term: str) -> pd.DataFrame:
    pytrends = TrendReq(hl='en-US', tz=360)
    # timeframe from 2020-03-01 to today
    today = datetime.today().strftime("%Y-%m-%d")
    timeframe = f"2020-03-01 {today}"
    pytrends.build_payload([term], timeframe=timeframe)
    df = pytrends.interest_over_time()
    if df.empty:
        st.error("No data returned from Google Trends.")
    return df

# 3) Load & display
data = load_trends(selected_term)

if not data.empty:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data[selected_term], linewidth=1.5)
    ax.set_title(f'Search Interest Over Time: "{selected_term}"', pad=12)
    ax.set_ylabel("Google Trends Score (0–100)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.2)

    # 4) Annotate the top 3 spikes
    top3 = data[selected_term].nlargest(3)
    for date, value in top3.items():
        ax.annotate(
            date.strftime("%b %Y"),
            xy=(date, value),
            xytext=(date, value + 7),
            ha='center',
            arrowprops=dict(color='red', arrowstyle='->', lw=1)
        )

    st.pyplot(fig)

    # 5) Show raw data toggle
    if st.sidebar.checkbox("Show raw data", False):
        st.dataframe(data[selected_term].rename(selected_term).to_frame())

else:
    st.warning("Unable to load trend data for this term.")
