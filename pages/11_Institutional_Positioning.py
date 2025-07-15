import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import zipfile
import io
from datetime import datetime

# --- Asset CFTC Codes ---
ASSET_CONFIG = {
    'SPY (S&P 500 E-mini)':      {'code': 13874,  'label': 'S&P 500 E-mini'},
    'QQQ (Nasdaq 100 E-mini)':   {'code': 20974,  'label': 'Nasdaq 100 E-mini'},
    'IWM (Russell 2000 E-mini)': {'code': 23974,  'label': 'Russell 2000 E-mini'},
    'Gold (COMEX)':              {'code': 8836,   'label': 'COMEX Gold'},
    'Oil (WTI Crude, NYMEX)':    {'code': 335,    'label': 'WTI Crude Oil'},
    'Bitcoin (CME)':             {'code': 133741, 'label': 'CME Bitcoin'},
}

st.set_page_config(page_title="CFTC Positioning Dashboard", layout="wide")
st.title("CFTC Non-Commercial Positioning Dashboard (Live Data)")

st.markdown("""
#### About This Tool

Institutional COT dashboard, automatically pulling weekly net non-commercial futures positioning (% of open interest) from CFTC.gov—no uploads, always current.  
Tracks US indices, gold, oil, and bitcoin for crowding and extreme sentiment.
""")

@st.cache_data(show_spinner=True)
def download_and_load_cot(years=7):
    """Download and concatenate the latest N years of Financial Futures COT data."""
    dfs = []
    base_url = "https://www.cftc.gov/files/dea/history/fut_fin_{year}.zip"
    current_year = datetime.now().year
    for yr in range(current_year, current_year - years, -1):
        url = base_url.format(year=yr)
        try:
            r = requests.get(url, timeout=20)
            if r.ok:
                with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                    for fn in zf.namelist():
                        if fn.lower().endswith('.txt') or fn.lower().endswith('.csv'):
                            df = pd.read_csv(zf.open(fn))
                            dfs.append(df)
        except Exception as e:
            st.warning(f"Could not load COT data for {yr}: {e}")
    if not dfs:
        st.error("Failed to download any COT data from CFTC. Try again later.")
        st.stop()
    df_all = pd.concat(dfs, ignore_index=True)
    df_all['Report_Date_as_YYYY-MM-DD'] = pd.to_datetime(df_all['Report_Date_as_YYYY-MM-DD'])
    df_all = df_all.sort_values('Report_Date_as_YYYY-MM-DD')
    return df_all

cot = download_and_load_cot(years=7)

def render_asset_chart(asset_name, cot, asset_conf):
    cftc_code = asset_conf['code']
    label = asset_conf['label']
    df = cot[cot['CFTC_Contract_Market_Code'] == cftc_code].copy()
    if df.empty:
        st.error(f"No data found for {asset_name}.")
        return
    df = df.sort_values('Report_Date_as_YYYY-MM-DD')
    df['net_noncom'] = df['NonComm_Positions_Long_All'] - df['NonComm_Positions_Short_All']
    df['net_noncom_pct'] = 100 * df['net_noncom'] / df['Open_Interest_All']
    fig, ax = plt.subplots(figsize=(11, 3))
    x = df['Report_Date_as_YYYY-MM-DD']
    y = df['net_noncom_pct']
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
    render_asset_chart(asset, cot, conf)

st.info(
    "Green = net speculative long, Red = net speculative short. "
    "Extreme readings may mark crowding or reversal risk. "
    "All data updated weekly from [CFTC.gov](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm)."
)
