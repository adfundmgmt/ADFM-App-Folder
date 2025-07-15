import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from cot_reports import COT

st.set_page_config(page_title="CFTC Positioning Dashboard", layout="wide")
st.title("CFTC Net Non-Commercial Positioning Dashboard (cot_reports, Python-only)")

st.markdown("""
#### About This Tool

Live dashboard for **S&P 500, Nasdaq 100, Russell 2000, Gold, Oil, and Bitcoin** — pulls weekly net non-commercial futures positioning (% of open interest) directly from the official CFTC site (via [cot_reports](https://pypi.org/project/cot-reports/)), no uploads or API keys.
""")

# --- Asset Config (CFTC Codes and Market Names) ---
ASSET_CONFIG = {
    'SPY (S&P 500)':    {'cftc_code': '13874',  'market': 'E-MINI S&P 500'},
    'QQQ (Nasdaq 100)': {'cftc_code': '20974',  'market': 'E-MINI NASDAQ 100'},
    'IWM (Russell 2000)': {'cftc_code': '23974', 'market': 'E-MINI RUSSELL 2000'},
    'Gold (COMEX)':     {'cftc_code': '088691', 'market': 'GOLD'},
    'Oil (WTI Crude)':  {'cftc_code': '067651', 'market': 'WTI CRUDE OIL'},
    'Bitcoin (CME)':    {'cftc_code': '133741', 'market': 'BITCOIN'},
}

@st.cache_data(show_spinner=True)
def load_cot_data(cftc_code, market):
    cot = COT(report_type='futures_only', 
              report_format='legacy', 
              market_code=cftc_code, 
              market_name=market)
    df = cot.data
    df = df.rename(columns=str.title)
    df = df.sort_values('Date')
    return df

def render_asset_chart(asset_name, conf):
    df = load_cot_data(conf['cftc_code'], conf['market'])
    # Standard column names for legacy format
    if 'Noncomm Long' in df.columns and 'Noncomm Short' in df.columns and 'Open Interest (All)' in df.columns:
        net_noncom = df['Noncomm Long'] - df['Noncomm Short']
        pct_noncom = 100 * net_noncom / df['Open Interest (All)']
        x = df['Date']
        y = pct_noncom
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
    else:
        st.warning(f"Could not find standard columns for {asset_name}. Try updating cot_reports.")

# --- Render all six assets ---
for asset, conf in ASSET_CONFIG.items():
    st.subheader(asset)
    render_asset_chart(asset, conf)

st.info(
    "Green = net speculative long, Red = net speculative short. "
    "Extreme readings may mark crowding or reversal risk. "
    "All data updated weekly from CFTC via [cot_reports](https://pypi.org/project/cot-reports/)."
)
