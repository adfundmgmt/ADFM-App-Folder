import streamlit as st, pandas as pd, numpy as np, requests, re, time, yfinance as yf

st.set_page_config(page_title="Forward P/E vs 10Y", layout="wide")

UA = {"User-Agent":"Mozilla/5.0"}
FIELDS = ["Price","EPS_TTM","EPS_nextY","EPS_growth_5Y_pct","Fwd_PE_nextY"]

def safe_float(x):
    try: return float(x.replace("%","").replace(",",""))
    except: return np.nan

@st.cache_data(ttl=600)
def finviz_fetch(ticker):
    url=f"https://finviz.com/quote.ashx?t={ticker}"
    r=requests.get(url,headers=UA,timeout=10)
    if r.status_code!=200: return None
    html=r.text
    if "P/E" not in html: return None  # page likely blocked
    data={}
    for lbl, key in [
        ("Price","Price"),
        ("EPS (ttm)","EPS_TTM"),
        ("EPS next Y","EPS_nextY"),
        ("EPS next 5Y","EPS_growth_5Y_pct"),
        ("Forward P/E","Fwd_PE_nextY")
    ]:
        m=re.search(rf"{lbl}.*?</td>\s*<td[^>]*>(.*?)</td>",html,re.I|re.S)
        if m:
            val=re.sub(r"<.*?>","",m.group(1)).strip()
            data[key]=safe_float(val)
    return data if data else None

@st.cache_data(ttl=600)
def get_batch(tickers):
    rows=[]
    for t in tickers:
        info=finviz_fetch(t)
        if not info:
            try:
                y=yf.Ticker(t)
                info={"Price":y.fast_info["last_price"]}
            except Exception:
                info={"Price":np.nan}
        info["Ticker"]=t
        rows.append(info)
        time.sleep(0.25)
    df=pd.DataFrame(rows).set_index("Ticker")
    for c in FIELDS:
        if c not in df: df[c]=np.nan
        df[c]=pd.to_numeric(df[c],errors="coerce")
    return df

# Sidebar
with st.sidebar:
    st.subheader("Settings")
    ten_year=st.number_input("U.S. 10Y Treasury yield (%)",value=4.12,step=0.01)
    inv_pe=100/ten_year if ten_year>0 else np.nan
    st.metric("10Y Inverse P/E",f"{inv_pe:,.2f}x")
    tickers=st.text_input("Tickers","MSFT,NVDA,META,AAPL,GOOGL,AMZN,TSLA").upper().split(",")

df=get_batch([t.strip() for t in tickers if t.strip()])

# compute
for y in range(1,6):
    if y==1: df[f"EPS_Y{y}"]=df["EPS_nextY"]
    else: df[f"EPS_Y{y}"]=df[f"EPS_Y{y-1}"]*(1+df["EPS_growth_5Y_pct"].fillna(0)/100)
    df[f"FwdPE_Y{y}"]=df["Price"]/df[f"EPS_Y{y}"]
threshold=inv_pe
def tag(v): return "Cheap" if v<=threshold else "Expensive" if not pd.isna(v) else ""

# KPIs
cols=st.columns(len(df))
for i,t in enumerate(df.index):
    with cols[i]:
        price=df.at[t,"Price"]
        f1=df.at[t,"FwdPE_Y1"]
        label=tag(f1)
        color="green" if label=="Cheap" else "red"
        st.markdown(f"""
        <div style="border:1px solid #ddd;border-radius:10px;padding:14px;">
        <b>{t}</b><br>
        <span class='small'>Price</span><br>
        <b>${price:,.2f}</b><br>
        <span class='small'>Forward P/E Y1</span><br>
        <b>{f1 if not np.isnan(f1) else 0:.2f}x</b>
        <span style='color:{'green' if color=='green' else 'crimson'}'> {label}</span>
        </div>
        """,unsafe_allow_html=True)

# Tables
st.subheader("Inputs (Finviz)")
st.dataframe(df[FIELDS],use_container_width=True)
st.subheader("Forward P/E Path")
st.dataframe(df[[f"FwdPE_Y{y}" for y in range(1,6)]].style.format("{:.2f}x"),use_container_width=True)
