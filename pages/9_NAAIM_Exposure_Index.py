import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

# --- Page Config ---
st.set_page_config(page_title="NAAIM Exposure Index", layout="wide")
st.title("📊 NAAIM Exposure Index Dashboard")

# --- Sidebar Info ---
st.sidebar.header("About the Index")
st.sidebar.markdown("""
The **NAAIM Exposure Index** reflects how actively managed portfolios are positioned in U.S. equities — 
ranging from 200% short to 200% long. It’s published weekly by the National Association of Active Investment Managers.

**Source**: Nasdaq Data Link  
**Updated Weekly**
""")

# --- API Config ---
API_KEY = "jP7MhQWA7jBxms_3FYzp"
DATASET_CODE = "NAAIM/NAAIM-Exposure-Index"
API_URL = f"https://data.nasdaq.com/api/v3/datasets/{DATASET_CODE}/data.json?api_key={API_KEY}"

# --- Data Loader ---
@st.cache_data(ttl=86400)
def load_naaim():
    r = requests.get(API_URL, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"API error: {r.status_code} — {r.text}")
    raw = r.json().get("dataset_data", {})
    cols = raw.get("column_names", [])
    data = raw.get("data", [])
    if not data or "Date" not in cols:
        raise RuntimeError("No data available or column names missing.")
    df = pd.DataFrame(data, columns=cols)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    # Find exposure column by flexible naming
    value_col = next((c for c in df.columns if "Value" in c or "Exposure" in c), df.columns[-1])
    df = df.sort_index()
    return df[[value_col]].rename(columns={value_col: "Exposure"})

# --- Chart & UI ---
try:
    df = load_naaim()
    st.subheader("Exposure Over Time")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Exposure"], label="NAAIM Exposure Index", color="black", linewidth=2.3)
    ax.axhline(100, linestyle="--", color="gray", linewidth=1, label="Fully Long (+100)")
    ax.axhline(0, linestyle="--", color="red", linewidth=1, label="Neutral (0)")
    ax.axhline(-100, linestyle="--", color="blue", linewidth=1, label="Fully Short (–100)")
    ax.set_title("NAAIM Active Manager Equity Exposure", fontsize=18, weight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Exposure Level", fontsize=12)
    ax.set_ylim(-200, 200)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Optional: show most recent value as metric
    recent = df["Exposure"].dropna().iloc[-1]
    st.metric(label="Latest Weekly Exposure", value=f"{recent:.1f}%")

except Exception as e:
    st.error(f"❌ Failed to fetch NAAIM data.\n\n{e}")
