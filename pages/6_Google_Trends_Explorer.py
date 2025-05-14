# pages/6_Google_Trends_Explorer.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random

# -- Optional styling
plt.style.use("seaborn-v0_8-darkgrid")

# -- User agents for anti-bot evasion
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",
    "Mozilla/5.0 (iPad; CPU OS 13_6 like Mac OS X)",
]

# -- Search terms
TERMS = [
    "Recession", "Inflation", "Unemployment", "Layoffs",
    "Credit Crunch", "Rate Hike", "Bond Market Crash",
    "Stock Market Crash", "Hard Landing", "Stagflation",
    "Bank Run", "Yield Curve Inversion", "Debt Ceiling",
    "Hyperinflation", "Soft Landing"
]

# -- Sidebar UI
st.sidebar.header("Google Trends Explorer")
selected_term = st.sidebar.selectbox("Choose a term:", TERMS)

# -- Safe pytrends import
try:
    from pytrends.request import TrendReq
except ImportError:
    st.error("`pytrends` is not installed. Run `pip install pytrends` or add it to `requirements.txt`.")
    st.stop()

# -- Data fetcher with rotating headers + caching
@st.cache_data(ttl=86400, show_spinner=False)
def load_trends(term: str) -> pd.DataFrame:
    try:
        user_agent = random.choice(USER_AGENTS)
        py = TrendReq(requests_args={"headers": {"User-Agent": user_agent}})
        today = datetime.today().strftime("%Y-%m-%d")
        timeframe = f"2020-03-01 {today}"
        py.build_payload([term], timeframe=timeframe)
        df = py.interest_over_time()
        return df[[term]] if term in df else pd.DataFrame()
    except Exception as e:
        raise RuntimeError(f"Google Trends request failed: {e}")

# -- Load data
try:
    data = load_trends(selected_term)
except RuntimeError as e:
    st.error(str(e))
    st.stop()

if data.empty:
    st.warning(f"No data available for **{selected_term}**.")
    st.stop()

# -- Plot chart
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data.index, data[selected_term], color='black', linewidth=1.8)
ax.set_title(f'Search Interest Over Time: "{selected_term}"', pad=12)
ax.set_ylabel("Google Trend Score (0–100)")
ax.set_xlabel("Date")
ax.grid(alpha=0.25)

# -- Annotate top 3 spikes
spikes = data[selected_term].nlargest(3)
for dt, val in spikes.items():
    ax.annotate(
        dt.strftime("%b %Y"),
        xy=(dt, val),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        arrowprops=dict(facecolor="red", arrowstyle="->", lw=1),
    )

st.pyplot(fig)

# -- Optional raw data
if st.sidebar.checkbox("Show raw data"):
    st.dataframe(data.rename(columns={selected_term: "Google Trend Score"}))
