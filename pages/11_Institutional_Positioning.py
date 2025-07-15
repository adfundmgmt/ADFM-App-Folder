import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="CFTC Positioning Dashboard", layout="wide")
st.title("CFTC Net Non-Commercial Positioning Dashboard (Direct CSV, 2024)")

st.markdown("""
#### About This Tool

Live COT positioning for **S&P 500, Nasdaq 100, Russell 2000, Gold, Oil, and Bitcoin** — always up-to-date with 2024 data, directly from CFTC, no uploads or extra dependencies.
""")

ASSET_CONFIG = {
    'SPY (S&P 500)':    {'code': 13874,   'label': 'S&P 500 E-mini'},
    'QQQ (Nasdaq 100)': {'code': 20974,   'label': 'Nasdaq 100 E-mini'},
    'IWM (Russell 2000)': {'code': 23974, 'label': 'Russell 2000 E-mini'},
    'Gold (COMEX)':     {'code': 8836,    'label': 'COMEX Gold'},
    'Oil (WTI Crude)':  {'code': 335,     'label': 'WTI Crude Oil'},
    'Bitcoin (CME)':    {'code': 133741,  'label': 'CME Bitcoin'},
}

@st.cache_data(show_spinner=True)
def load_cot_2024():
    url = "https://www.cftc.gov/files/dea/history/fut_fin_2024.csv"
    df = pd.read_csv(url)
    df['Report_Date_as_YYYY-MM-DD'] = pd.to_datetime(df['Report_Date_as_YYYY-MM-DD'])
    return df

cot = load_cot_2024()

def render_asset_chart(asset_name, conf):
    cftc_code = conf['code']
    label = conf['label']
    df = cot[cot['CFTC_Contract_Market_Code'] == cftc_code].copy()
    if df.empty:
        st.error(f"No data found for {asset_name}.")
        return
    df = df.sort_values('Report_Date_as_YYYY-MM-DD')
    # Net non-commercial = NonComm_Long_All - NonComm_Short_All
    df['net_noncom'] = df['NonComm_Positions_Long_All'] - df['NonComm_Positions_Short_All']
    df['net_noncom_pct'] = 100 * df['net_noncom'] / df['Open_Interest_All']
    x = df['Report_Date_as_YYYY-MM-DD']
    y = df['net_noncom_pct']
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.fill_between(x, y, 0, where=(y >= 0), color='green', alpha=0.25)
    ax.fill_between(x, y, 0, where=(y < 0), color='red', alpha=0.25)
    ax.plot(x, y, color='black', linewidth=1.2)
    ax.axhline(0, color='grey', linewidth=1)
    ax.set_title(f"{asset_name} — Net Non-Commercial Positioning (% of Open Interest)", fontsize=13)
    ax.set_ylabel("% of Open Interest", fontsize=11)
    ax.set_xlabel("Date", fontsize=10)
    ax.grid(True, alpha=0.18)
    ax.set_xlim(x.min(), x.max())
    y_buffer = max(5, int((y.max() - y.min()) * 0.1))
    ax.set_ylim(y.min() - y_buffer, y.max() + y_buffer)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown(
        f"**Latest ({x.iloc[-1].date()}):** "
        f"{y.iloc[-1]:+.2f}% of open interest ({'Net Long' if y.iloc[-1] > 0 else 'Net Short'})"
    )
    st.markdown("---")

for asset, conf in ASSET_CONFIG.items():
    st.subheader(asset)
    render_asset_chart(asset, conf)

st.info(
    "Green = net speculative long, Red = net speculative short. "
    "Extreme readings may mark crowding or reversal risk. "
    "Data updated weekly from CFTC. For multi-year history, just extend the CSV URLs."
)
