# finviz_forward_pe_app.py
# Requirements: streamlit, requests, pandas, numpy

import streamlit as st
import pandas as pd
import numpy as np
import re, requests

# ---------- setup ----------
st.set_page_config(page_title="Forward P/E vs 10Y", layout="wide")

st.markdown("""
<style>
.dataframe tbody td { padding:6px 8px!important; }
.dataframe thead th { padding:6px 8px!important; font-weight:600; }
.block-container { padding-top:1rem; padding-bottom:2rem; }
.kpi { border:1px solid #e6e6e6; border-radius:12px; padding:14px; background:#fff; }
.badge { padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; }
.badge.green { background:#d9f2e4; color:#146c2e; }
.badge.red { background:#f9d5d0; color:#7a1c15; }
.small { color:#666; font-size:12px; }
.section-title { font-size:18px; font-weight:700; margin:12px 0 6px 0; }
</style>
""", unsafe_allow_html=True)

st.title("Forward P/E vs U.S. 10Y Inverse")

UA = {"User-Agent":"Mozilla/5.0"}

# ---------- helpers ----------
def _num(x):
    try: return float(x)
    except: return np.nan

@st.cache_data(ttl=600)
def finviz_parse(ticker: str) -> dict:
    """Pulls key ratios from Finviz snapshot table."""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    r = requests.get(url, headers=UA, timeout=10)
    if r.status_code != 200: 
        return {"Ticker":ticker}
    html = r.text
    cells = re.findall(r'<td.*?>(.*?)</td>', html)
    out = {}
    for i in range(0, len(cells)-1, 2):
        label = re.sub(r'<.*?>', '', cells[i]).strip()
        val = re.sub(r'<.*?>', '', cells[i+1]).strip()
        out[label] = val
    def clean(v): 
        return _num(v.replace('%','').replace(',','')) if isinstance(v,str) else np.nan
    return {
        "Ticker": ticker,
        "Price": clean(out.get("Price")),
        "EPS_TTM": clean(out.get("EPS (ttm)")),
        "EPS_nextY": clean(out.get("EPS next Y")),
        "EPS_growth_5Y_pct": clean(out.get("EPS next 5Y")),
        "Fwd_PE_nextY": clean(out.get("Forward P/E"))
    }

@st.cache_data(ttl=600)
def fetch_finviz_batch(tickers):
    rows = []
    for t in tickers:
        try:
            rows.append(finviz_parse(t))
        except Exception:
            rows.append({"Ticker":t})
    df = pd.DataFrame(rows).set_index("Ticker")
    for c in ["Price","EPS_TTM","EPS_nextY","EPS_growth_5Y_pct","Fwd_PE_nextY"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------- sidebar ----------
with st.sidebar:
    st.subheader("Settings")
    ten_year = st.number_input("U.S. 10Y Treasury yield (%)", value=4.12, step=0.01, format="%.2f")
    inv_pe = 100.0 / ten_year if ten_year > 0 else np.nan
    st.metric("10Y Inverse P/E", f"{inv_pe:,.2f}x")

    st.subheader("Universe")
    tickers_input = st.text_input(
        "Tickers",
        value="MSFT, NVDA, META, AAPL, GOOGL, AMZN, TSLA",
        help="Comma separated symbols"
    )

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
df = fetch_finviz_batch(tickers)

# ---------- compute projections ----------
years = [1,2,3,4,5]
for y in years:
    if y == 1:
        df[f"EPS_Y{y}"] = df["EPS_nextY"]
    else:
        g = df["EPS_growth_5Y_pct"].fillna(0)/100.0
        df[f"EPS_Y{y}"] = df[f"EPS_Y{y-1}"]*(1+g)
for y in years:
    df[f"FwdPE_Y{y}"] = df["Price"]/df[f"EPS_Y{y}"]

threshold = inv_pe
def ce(v): 
    if pd.isna(v): return ""
    return "Cheap" if v<=threshold else "Expensive"
ce_tbl = pd.DataFrame(index=df.index)
for y in [1,2,3]:
    ce_tbl[f"Year {y}"] = df[f"FwdPE_Y{y}"].apply(ce)

# ---------- KPI cards ----------
cols = st.columns(len(df.index))
for i, t in enumerate(df.index):
    with cols[i]:
        price = df.at[t,"Price"]
        fwd1 = df.at[t,"FwdPE_Y1"]
        label = ce_tbl.at[t,"Year 1"] or "N/A"
        tag = "green" if label=="Cheap" else "red"
        st.markdown(f"""
        <div class="kpi">
          <div style="font-size:15px;font-weight:700;margin-bottom:4px">{t}</div>
          <div class="small">Price</div>
          <div style="font-size:20px;font-weight:700">${price:.2f}</div>
          <div class="small" style="margin-top:6px">Forward P/E Y1</div>
          <div style="display:flex;gap:8px;align-items:center">
            <div style="font-size:20px;font-weight:700">{fwd1:.2f}x</div>
            <span class="badge {tag}">{label}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

st.caption("Cheap or Expensive judged versus 10Y inverse P/E above.")

# ---------- main tables ----------
st.markdown('<div class="section-title">Inputs (Finviz)</div>', unsafe_allow_html=True)
st.dataframe(
    df[["Price","EPS_TTM","EPS_nextY","EPS_growth_5Y_pct","Fwd_PE_nextY"]]
    .rename(columns={
        "Price":"Current Price",
        "EPS_TTM":"EPS (TTM)",
        "EPS_nextY":"EPS next Y",
        "EPS_growth_5Y_pct":"EPS growth next 5Y (%)",
        "Fwd_PE_nextY":"Forward P/E (next Y)"
    })
    .style.format({
        "Current Price":"${:,.2f}",
        "EPS (TTM)":"{:.2f}",
        "EPS next Y":"{:.2f}",
        "EPS growth next 5Y (%)":"{:.2f}",
        "Forward P/E (next Y)":"{:.2f}"
    }),
    use_container_width=True
)

st.markdown('<div class="section-title">Forward P/E Path</div>', unsafe_allow_html=True)
pe_tbl = df[[f"FwdPE_Y{y}" for y in years]].copy()
pe_tbl.columns = [f"Year {y}" for y in years]
def color(v): 
    if pd.isna(v): return ""
    return "background-color:#d9f2e4" if v<=threshold else "background-color:#f9d5d0"
st.dataframe(pe_tbl.style.format("{:.2f}x").applymap(color), use_container_width=True)

st.markdown('<div class="section-title">Cheap / Expensive Summary</div>', unsafe_allow_html=True)
st.dataframe(
    ce_tbl.style.apply(
        lambda s: ["background-color:#d9f2e4" if v=="Cheap" else "background-color:#f9d5d0" for v in s],
        axis=1),
    use_container_width=True
)

# ---------- export ----------
out = pd.concat([df, ce_tbl.add_prefix("CE_")], axis=1)
st.download_button("Download CSV", out.to_csv().encode("utf-8"), file_name="forward_pe_finviz.csv")
