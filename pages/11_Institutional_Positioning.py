import streamlit as st
import pandas as pd

st.set_page_config(page_title="CFTC Positioning Dashboard", layout="wide")
st.title("CFTC Net Non-Commercial Positioning — Latest Weekly (No Uploads)")

st.markdown("""
#### About This Tool

Shows the most recent CFTC non-commercial futures positioning (% of open interest) for **S&P 500, Nasdaq 100, Russell 2000, Gold, Oil, and Bitcoin**. Data is always live, direct from [CFTC.gov](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm).
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
def load_latest_cot():
    url = "https://www.cftc.gov/files/dea/cotfinxf.csv"
    df = pd.read_csv(url)
    # Standardize columns for consistent access
    df.columns = [c.strip() for c in df.columns]
    return df

cot = load_latest_cot()

def render_asset_stat(asset_name, conf):
    cftc_code = conf['code']
    label = conf['label']
    df = cot[cot['CFTC_Contract_Market_Code'] == cftc_code]
    if df.empty:
        st.error(f"No data found for {asset_name}.")
        return
    # Use only the most recent row (latest week)
    row = df.iloc[-1]
    net_noncom = row['NonComm_Positions_Long_All'] - row['NonComm_Positions_Short_All']
    net_noncom_pct = 100 * net_noncom / row['Open_Interest_All']
    st.subheader(asset_name)
    st.markdown(
        f"**As of {row['Report_Date_as_YYYY-MM-DD']}:**  "
        f"**{net_noncom_pct:+.2f}%** net non-commercial ({'Net Long' if net_noncom_pct > 0 else 'Net Short'})"
    )
    st.markdown(
        f"- Open Interest: {row['Open_Interest_All']:,}"
        f"\n- Non-Comm Long: {row['NonComm_Positions_Long_All']:,}"
        f"\n- Non-Comm Short: {row['NonComm_Positions_Short_All']:,}"
    )
    st.markdown("---")

for asset, conf in ASSET_CONFIG.items():
    render_asset_stat(asset, conf)

st.info(
    "Live COT data — latest weekly. For multi-year positioning trends, a manual download/merge of historical COT CSVs is required due to CFTC website limitations."
)
