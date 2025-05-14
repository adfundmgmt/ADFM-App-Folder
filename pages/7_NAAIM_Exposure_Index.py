import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Page config
st.set_page_config(page_title="NAAIM Exposure Index", layout="wide")
st.title("📊 NAAIM Exposure Index Dashboard")

# Sidebar
st.sidebar.header("About the Index")
st.sidebar.markdown("""
The **NAAIM Exposure Index** reflects how actively managed portfolios are positioned in U.S. equities — 
ranging from 200% short to 200% long. It’s published weekly by the National Association of Active Investment Managers.

**Source**: Nasdaq Data Link  
**Updated Weekly**
""")

# ✅ Your API key from Nasdaq Data Link
API_KEY = "jP7MhQWA7jBxms_3FYzp"
DATASET_CODE = "NAAIM/NAAIM-Exposure-Index"
API_URL = f"https://data.nasdaq.com/api/v3/datasets/{DATASET_CODE}/data.json?api_key={API_KEY}"

# Load data
@st.cache_data(ttl=86400)
def load_naaim():
    r = requests.get(API_URL)
    if r.status_code != 200:
        raise RuntimeError(f"API error: {r.status_code}")
    raw = r.json()["dataset_data"]
    df = pd.DataFrame(raw["data"], columns=raw["column_names"])
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df.sort_index()

# Try to load and display chart
try:
    df = load_naaim()

    st.subheader("Exposure Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Value"], label="NAAIM Exposure Index", color="black", linewidth=2.25)
    ax.axhline(100, linestyle="--", color="gray", linewidth=1, label="Fully Long")
    ax.axhline(0, linestyle="--", color="red", linewidth=1, label="Neutral")
    ax.axhline(-100, linestyle="--", color="blue", linewidth=1, label="Fully Short")
    ax.set_title("NAAIM Active Manager Equity Exposure", fontsize=18, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Exposure Level")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

except Exception as e:
    st.error(f"❌ Failed to fetch NAAIM data.\n\n{e}")
