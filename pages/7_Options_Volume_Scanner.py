import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

# Streamlit: Options Activity & Actionable Alerts Scanner

@st.cache_data
def fetch_volume_metrics(ticker: str, expiry: str) -> dict:
    chain = yf.Ticker(ticker).option_chain(expiry)
    calls = chain.calls.copy(); puts = chain.puts.copy()
    call_vol = calls['volume'].sum()
    put_vol = puts['volume'].sum()
    total_vol = call_vol + put_vol
    cp_ratio = call_vol / put_vol if put_vol>0 else np.nan
    return {'ticker':ticker,'expiry':expiry,'call_vol':call_vol,'put_vol':put_vol,'total_vol':total_vol,'cp_ratio':cp_ratio}

@st.cache_data
def get_expiries(ticker: str) -> list:
    return yf.Ticker(ticker).options or []

@st.cache_data
def build_volume_df(tickers: list, expiries: list) -> pd.DataFrame:
    rows=[]
    for t in tickers:
        for e in expiries:
            try: rows.append(fetch_volume_metrics(t,e))
            except: continue
    return pd.DataFrame(rows)

# --- UI Setup ---
st.title('Options Activity & Actionable Alerts')
# Inputs
raw = st.sidebar.text_input('Tickers (comma-separated)','AAPL,MSFT,GOOG')
tickers=[t.strip().upper() for t in raw.split(',') if t.strip()]
if not tickers:
    st.sidebar.error('Enter at least one ticker.')
    st.stop()
primary=tickers[0]
exps=get_expiries(primary)
expiries=st.sidebar.multiselect('Select Expiries',exps,default=exps[:3])
if not expiries:
    st.sidebar.warning('Select at least one expiry.')
    st.stop()
# Overview
metric = st.sidebar.selectbox('Overview Metric',['total_vol','cp_ratio'],format_func=lambda x:{'total_vol':'Total Volume','cp_ratio':'C/P Ratio'}[x])
# Build DataFrame
df=build_volume_df(tickers,expiries)
if df.empty:
    st.error('No data; verify inputs.')
    st.stop()
# Heatmap Overview
st.subheader('Volume Overview Heatmap')
hm = alt.Chart(df).mark_rect().encode(
    x=alt.X('expiry:N',title='Expiry'),
    y=alt.Y('ticker:N',title='Ticker'),
    color=alt.Color(f'{metric}:Q',title=metric),
    tooltip=['ticker','expiry','call_vol','put_vol','total_vol','cp_ratio']
).properties(width=700,height=300)
st.altair_chart(hm,use_container_width=True)

st.markdown('---')
# Strike-Level Actionable Alerts
sel = st.selectbox('Ticker & Expiry for Detailed Alerts',df.apply(lambda r:f"{r['ticker']} | {r['expiry']}",axis=1).tolist())
tk,ex=map(str.strip,sel.split('|'))
chain=yf.Ticker(tk).option_chain(ex)
combo=pd.concat([chain.calls.assign(type='call'),chain.puts.assign(type='put')],ignore_index=True)
combo['volume']=combo['volume'].fillna(0)
combo['openInterest']=combo['openInterest'].replace(0,np.nan)
combo['vol_oi']= (combo['volume']/combo['openInterest']).round(3)
# select top strikes
alerts=combo.sort_values('vol_oi',ascending=False).head(10)

# 2D Heatmap of Strike vs Vol/OI
st.subheader('Strike vs. Activity Heatmap')
m2 = alt.Chart(alerts).mark_rect().encode(
    x=alt.X('strike:O',title='Strike'),
    y=alt.Y('type:N',title='Option Type'),
    color=alt.Color('vol_oi:Q',title='Vol/OI'),
    tooltip=['type','strike','volume','openInterest','vol_oi']
).properties(width=700,height=250)
st.altair_chart(m2,use_container_width=True)

# Actionable Summary
st.subheader('Alerts Summary')
for _,r in alerts.iterrows():
    st.write(f"- **{r['type'].capitalize()}** @ strike {r['strike']}: vol/OI={r['vol_oi']} (vol={int(r['volume'])})")

# Footer
st.markdown("""
**Notes:**
- `C/P Ratio` = call_vol / put_vol
- `vol_oi` = volume / openInterest (higher = unusual activity)
""")
