# pages/6_Google_Trends_Explorer.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ---- Search terms
TERMS = [
    "Recession", "Inflation", "Unemployment", "Layoffs",
    "Credit Crunch", "Rate Hike", "Bond Market Crash",
    "Stock Market Crash", "Hard Landing", "Stagflation",
    "Bank Run", "Yield Curve Inversion", "Debt Ceiling",
    "Hyperinflation", "Soft Landing"
]

# ---- Sidebar UI
st.sidebar.header("Google Trends Explorer")
selected_term = st.sidebar.selectbox("Choose a term:", TERMS)

# ---- Load CSV instead of using pytrends
@st.cache_data(ttl=3600, show_spinner=False)
def load_csv(term: str) -> pd.DataFrame:
    filepath = f"data/{term}.csv"
    if not os.path.exists(filepath):
        st.error(f"No data file found for {term}. Please upload `{term}.csv` to the `data/` folder.")
        return pd.DataFrame()
    df = pd.read_csv(filepath, parse_dates=["date"])
    df.set_index("date", inplace=True)
    return df

# ---- Load and visualize
data = load_csv(selected_term)

if data.empty:
    st.stop()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data.index, data[selected_term], linewidth=1.5)
ax.set_title(f'Search Interest: "{selected_term}" (Google Trends)', pad=12)
ax.set_ylabel("Trend Score (0–100)")
ax.set_xlabel("Date")
ax.grid(alpha=0.2)

# ---- Annotate top 3 spikes
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

# ---- Optional raw data
if st.sidebar.checkbox("Show raw data"):
    st.dataframe(data.rename(columns={selected_term: "Google Trend Score"}))
