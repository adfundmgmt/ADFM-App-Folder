import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="CFTC Positioning Dashboard", layout="wide")
st.title("CFTC Non-Commercial Positioning Dashboard (Live Data)")

st.markdown("""
#### About This Tool

Institutional COT dashboard for **S&P 500, Nasdaq 100, Russell 2000, Gold, Oil, and Bitcoin** — pulling weekly net non-commercial futures positioning (% of open interest) directly from CFTC, always current.
""")

ASSET_CONFIG = {
    'SPY (S&P 500)': {
        'market_and_exchange_names': 'E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE',
        'label': 'S&P 500 E-mini'
    },
    'QQQ (Nasdaq 100)': {
        'market_and_exchange_names': 'E-MINI NASDAQ 100 - CHICAGO MERCANTILE EXCHANGE',
        'label': 'Nasdaq 100 E-mini'
    },
    'IWM (Russell 2000)': {
        'market_and_exchange_names': 'E-MINI RUSSELL 2000 - CHICAGO MERCANTILE EXCHANGE',
        'label': 'Russell 2000 E-mini'
    },
    'Gold (COMEX)': {
        'market_and_exchange_names': 'GOLD - COMMODITY EXCHANGE INC.',
        'label': 'COMEX Gold'
    },
    'Oil (WTI Crude, NYMEX)': {
        'market_and_exchange_names': 'WTI CRUDE OIL - NEW YORK MERCANTILE EXCHANGE',
        'label': 'NYMEX WTI Crude'
    },
    'Bitcoin (CME)': {
        'market_and_exchange_names': 'BITCOIN - CHICAGO MERCANTILE EXCHANGE',
        'label': 'CME Bitcoin'
    },
}

@st.cache_data(show_spinner=True)
def get_cot_data(lookback_years=7):
    """
    Download and parse the latest Disaggregated Futures Only report from the CFTC.
    Returns historical data for several years (weekly frequency).
    """
    base_url = 'https://www.cftc.gov/files/dea/history/deafut_disagg_xls_{}.zip'
    years = []
    now = datetime.now()
    for y in range(now.year, now.year - lookback_years, -1):
        years.append(str(y))
    df_list = []
    for year in years:
        url = base_url.format(year)
        try:
            r = requests.get(url, timeout=20)
            if r.ok:
                from zipfile import ZipFile
                import io
                zf = ZipFile(io.BytesIO(r.content))
                # Find the CSV/XLS file
                for fn in zf.namelist():
                    if fn.lower().endswith('.csv'):
                        with zf.open(fn) as f:
                            dfx = pd.read_csv(f)
                            df_list.append(dfx)
                    elif fn.lower().endswith('.xls'):
                        with zf.open(fn) as f:
                            dfx = pd.read_excel(f)
                            df_list.append(dfx)
        except Exception as e:
            continue
    if not df_list:
        st.error("Failed to download COT data from CFTC (disaggregated). Try again later.")
        st.stop()
    cot = pd.concat(df_list, ignore_index=True)
    cot['Report_Date_as_YYYY-MM-DD'] = pd.to_datetime(cot['Report_Date_as_YYYY-MM-DD'])
    cot = cot.sort_values('Report_Date_as_YYYY-MM-DD')
    return cot

cot = get_cot_data(lookback_years=7)

def render_asset_chart(asset_name, cot, asset_conf):
    market_name = asset_conf['market_and_exchange_names']
    label = asset_conf['label']
    df = cot[cot['Market_and_Exchange_Names'] == market_name].copy()
    if df.empty:
        st.error(f"No data found for {asset_name}.")
        return
    df = df.sort_values('Report_Date_as_YYYY-MM-DD')
    # Non-commercial = "Money Manager" (for financial futures)
    # For commodities: use "Managed Money"
    if 'Money_Manager_Long_Positions_All' in df.columns and 'Money_Manager_Short_Positions_All' in df.columns:
        df['net_noncom'] = df['Money_Manager_Long_Positions_All'] - df['Money_Manager_Short_Positions_All']
    elif 'Managed_Money_Long_All' in df.columns and 'Managed_Money_Short_All' in df.columns:
        df['net_noncom'] = df['Managed_Money_Long_All'] - df['Managed_Money_Short_All']
    else:
        st.error(f"Missing net non-commercial columns for {asset_name}.")
        return
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
