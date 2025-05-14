# pages/6_Google_Trends_Explorer.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Graceful import check for pytrends
try:
    from pytrends.request import TrendReq
    pytrends_available = True
except ImportError:
    pytrends_available = False

# --- Search terms
TERMS = [
    "Recession", "Inflation", "Unemployment", "Layoffs",
    "Credit Crunch", "Rate Hike", "Bond Market Crash",
    "Stock Market Crash", "Hard Landing", "Stagflation",
    "Bank Run", "Yield Curve Inversion", "Debt Ceiling",
    "Hyperinflation", "Soft Landing"
]

# --- Sidebar UI
st.sidebar.header("Google Trends Explorer")
selected_term = st.sidebar.selectbox("Choose a term:", TERMS)

# --- Data loader
@st.cache_data(ttl=3600, show_spinner=False)
def load_trends(term: str) -> pd.DataFrame:
    if not pytrends_available:
        raise RuntimeError("pytrends is not installed.")

    try:
        py = TrendReq(hl="en-US", tz=360)
        today = datetime.today().strftime("%Y-%m-%d")
        timeframe = f"2020-03-01 {today}"
        py.build_payload([term], timeframe=timeframe)
        df = py.interest_over_time()
        return df[[term]] if term in df else pd.DataFrame()
    except Exception as e:
        raise RuntimeError(f"Google Trends request failed: {e}")

# --- Main logic
if not pytrends_available:
    st.error("`pytrends` library is not installed. Run `pip install pytrends` or add it to `requirements.txt`.")
    st.stop()

try:
    data = load_trends(selected_term)
except RuntimeError as e:
    st.error(str(e))
    st.stop()

if data.empty:
    st.warning(f"No data available for **{selected_term}**.")
    st.stop()

# --- Plotting
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data.index, data[selected_term], linewidth=1.5, color="black")
ax.set_title(f'Search Interest Over Time: "{selected_term}"', pad=12)
ax.set_ylabel("Trend Score (0–100)")
ax.set_xlabel("Date")
ax.grid(alpha=0.3)

# --- Annotate top 3 spikes
spikes = data[selected_term].nlargest(3)
for dt, val in spikes.items():
    ax.annotate(
        dt.strftime("%b %Y"),
        xy=(dt, val),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        arrowprops=dict(color="red", arrowstyle="->", lw=1),
    )

st.pyplot(fig)

# --- Raw data toggle
if st.sidebar.checkbox("Show raw data"):
    st.dataframe(data.rename(columns={selected_term: "Google Trend Score"}))
