# pages/6_Google_Trends_Explorer.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random
import time

# -- Matplotlib styling
plt.style.use("seaborn-v0_8-darkgrid")

# -- User agents to reduce bot detection
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",
    "Mozilla/5.0 (iPad; CPU OS 13_6 like Mac OS X)",
]

# -- Macro-relevant search terms
TERMS = [
    "Recession", "Inflation", "Unemployment", "Layoffs",
    "Credit Crunch", "Rate Hike", "Bond Market Crash",
    "Stock Market Crash", "Hard Landing", "Stagflation",
    "Bank Run", "Yield Curve Inversion", "Debt Ceiling",
    "Hyperinflation", "Soft Landing"
]

# -- Sidebar UI
st.sidebar.header("About This Tool")
st.sidebar.markdown(
    """
Track macro sentiment shifts by visualizing live Google search interest from 2020 to today.

⚠️ **Note**: Google may temporarily block multiple requests.  
Wait **~45–60 seconds between queries** to avoid being rate-limited (error 429).  
Data is cached for 24 hours per term.
"""
)

selected_term = st.sidebar.selectbox("Choose a term:", TERMS)

# -- Check pytrends
try:
    from pytrends.request import TrendReq
except ImportError:
    st.error("`pytrends` is not installed. Run `pip install pytrends` or add it to requirements.txt.")
    st.stop()

# -- Google Trends fetcher with rate-limit retry logic
@st.cache_data(ttl=86400, show_spinner=False)
def load_trends(term: str) -> pd.DataFrame:
    def fetch():
        user_agent = random.choice(USER_AGENTS)
        py = TrendReq(requests_args={"headers": {"User-Agent": user_agent}})
        today = datetime.today().strftime("%Y-%m-%d")
        timeframe = f"2020-03-01 {today}"
        py.build_payload([term], timeframe=timeframe)
        df = py.interest_over_time()
        return df[[term]] if term in df else pd.DataFrame()

    try:
        return fetch()
    except Exception as e:
        if "429" in str(e).lower() or "too many requests" in str(e).lower():
            st.warning("🚧 Rate-limited by Google. Waiting 45 seconds before retrying...")
            time.sleep(45)
            try:
                return fetch()
            except Exception as retry_e:
                st.error("❌ Google blocked the request again. Please wait a few minutes and try again.")
                return pd.DataFrame()
        else:
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

# -- Plot the chart
fig, ax = plt.subplots(figsize=(11, 5.5))  # Full-width

ax.plot(data.index, data[selected_term], color='black', linewidth=2.25)
ax.set_title(f'Search Interest Over Time: "{selected_term}"', fontsize=18, pad=15, weight='bold')
ax.set_ylabel("Google Trend Score (0–100)", fontsize=13)
ax.set_xlabel("Date", fontsize=13)
ax.tick_params(axis='both', which='major', labelsize=11)
ax.grid(alpha=0.25, linestyle='--')

# -- Annotate top 3 spikes
spikes = data[selected_term].nlargest(3)
for dt, val in spikes.items():
    ax.annotate(
        dt.strftime("%b %Y"),
        xy=(dt, val),
        xytext=(0, 15),
        textcoords="offset points",
        ha="center",
        fontsize=11,
        arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.8)
    )

fig.tight_layout()
st.pyplot(fig)
