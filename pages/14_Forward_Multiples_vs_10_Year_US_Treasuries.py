# 14_Forward_Multiples_vs_10_Year_US_Treasuries.py
# Requirements: streamlit, yfinance, requests

import streamlit as st
import pandas as pd
import numpy as np
import re, requests
from typing import List, Optional

# -------------- Page setup and minimal styling --------------
st.set_page_config(page_title="Forward P/E vs 10Y", layout="wide")
st.markdown("""
<style>
.dataframe tbody td { padding: 6px 8px !important; }
.dataframe thead th { padding: 6px 8px !important; font-weight: 600; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
.kpi { border:1px solid #e6e6e6; border-radius:12px; padding:14px; background:#fff; }
.badge { padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; }
.badge.green { background:#d9f2e4; color:#146c2e; }
.badge.red { background:#f9d5d0; color:#7a1c15; }
.small { color:#666; font-size:12px; }
.section-title { font-size:18px; font-weight:700; margin: 10px 0 6px 0; }
</style>
""", unsafe_allow_html=True)

st.title("Forward P/E vs U.S. 10Y Inverse")

# -------------- Helpers --------------
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [".".join([str(x) for x in col]).strip() for col in df.columns.values]
    return df

def _to_num(x):
    try:
        return float(x)
    except Exception:
        return np.nan

@st.cache_data(ttl=300)
def finviz_eps_5y(ticker: str) -> Optional[float]:
    try:
        r = requests.get(f"https://finviz.com/quote.ashx?t={ticker}",
                         headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        if r.status_code != 200: 
            return None
        html = r.text
        m = re.search(r"EPS\s*next\s*5Y.*?</td>\s*<td[^>]*>(.*?)</td>",
                      html, re.IGNORECASE | re.DOTALL)
        if not m: 
            return None
        val = re.sub(r"<.*?>","",m.group(1)).strip().replace("%","").replace(",","")
        return float(val)
    except Exception:
        return None

@st.cache_data(ttl=300)
def fetch_from_yfinance(tickers: List[str]) -> pd.DataFrame:
    import yfinance as yf
    rows = []
    for t in tickers:
        tk = yf.Ticker(t)
        # price
        price = None
        try:
            price = float(getattr(tk, "fast_info", {}).get("last_price"))
        except Exception:
            pass
        if price is None:
            try:
                price = float((tk.history(period="1d")["Close"].tail(1)).iloc[0])
            except Exception:
                price = np.nan

        # info block
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        eps_ttm = _to_num(info.get("trailingEps"))

        # robust Y+1 EPS from earnings trend
        eps_next_y = np.nan
        try:
            trend = tk.get_earnings_trend()
            if isinstance(trend, pd.DataFrame) and not trend.empty:
                trend = _flatten_columns(trend.reset_index(drop=True))
                if "period" in trend.columns:
                    mask = trend["period"].astype(str).str.lower().isin(["y+1","nextyear"])
                    row = trend[mask]
                    if not row.empty:
                        candidates = [c for c in trend.columns if "eps" in c.lower() and ("avg" in c.lower() or "mean" in c.lower())]
                        # prefer epsTrend first, then earningsEstimate
                        ordered = sorted(candidates, key=lambda c: (0 if "epstrend" in c.lower() else 1, c))
                        for c in ordered:
                            v = _to_num(row.iloc[0][c])
                            if pd.notna(v) and v > 0:
                                eps_next_y = v
                                break
        except Exception:
            pass

        # fallback to forwardEps only if Y+1 EPS missing
        if pd.isna(eps_next_y):
            eps_next_y = _to_num(info.get("forwardEps"))

        # 5y growth: try trend longTerm first; else Finviz
        growth_5y = np.nan
        try:
            trend2 = tk.get_earnings_trend()
            if isinstance(trend2, pd.DataFrame) and "growth_longTerm" in trend2.columns:
                g = _to_num(trend2["growth_longTerm"].iloc[0])
                if pd.notna(g):
                    growth_5y = g * 100.0
        except Exception:
            pass
        if pd.isna(growth_5y):
            g = finviz_eps_5y(t)
            if g is not None:
                growth_5y = float(g)

        # forward PE next year: if Yahoo field missing, derive using Y+1 EPS
        fwd_pe_next_y = _to_num(info.get("forwardPE"))
        if (pd.isna(fwd_pe_next_y) or fwd_pe_next_y <= 0) and pd.notna(price) and pd.notna(eps_next_y) and eps_next_y > 0:
            fwd_pe_next_y = price / eps_next_y

        rows.append({
            "Ticker": t,
            "Current_Price": price,
            "EPS_TTM": eps_ttm,
            "EPS_nextY": eps_next_y,
            "EPS_growth_5Y_pct": growth_5y,
            "Fwd_PE_nextY": fwd_pe_next_y
        })
    df = pd.DataFrame(rows).set_index("Ticker")
    for c in ["Current_Price","EPS_TTM","EPS_nextY","EPS_growth_5Y_pct","Fwd_PE_nextY"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def fetch_live(tickers: List[str]) -> pd.DataFrame:
    df = fetch_from_yfinance(tickers)
    return df

# -------------- Sidebar --------------
with st.sidebar:
    st.subheader("Settings")
    ten_year = st.number_input("U.S. 10Y Treasury yield (%)", value=4.12, step=0.01, format="%.2f")
    inv_pe = 100.0 / ten_year if ten_year > 0 else np.nan
    st.metric("10Y Inverse P/E", f"{inv_pe:,.2f}x")

    st.subheader("Universe")
    tickers_input = st.text_input("Tickers", value="MSFT, NVDA, META, AAPL, GOOGL, AMZN, TSLA", help="Comma separated")

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
inputs = fetch_live(tickers)

# -------------- Calculations --------------
df = inputs.copy()
years = [1,2,3,4,5]

# EPS projection using 5Y CAGR
for y in years:
    if y == 1:
        df[f"EPS_Y{y}"] = df["EPS_nextY"]
    else:
        g = df["EPS_growth_5Y_pct"].fillna(0) / 100.0
        df[f"EPS_Y{y}"] = df[f"EPS_Y{y-1}"] * (1.0 + g)

# Forward P/E path at constant price
for y in years:
    df[f"FwdPE_Y{y}"] = df["Current_Price"] / df[f"EPS_Y{y}"]

# Cheap vs 10Y inverse
def ce_flag(v, thr):
    if pd.isna(v) or pd.isna(thr): return ""
    return "Cheap" if v <= thr else "Expensive"

threshold = inv_pe
ce = pd.DataFrame(index=df.index)
for y in [1,2,3]:
    ce[f"Year {y}"] = df[f"FwdPE_Y{y}"].apply(lambda v: ce_flag(v, threshold))

# -------------- KPI cards --------------
cols = st.columns(len(df.index))
for i, t in enumerate(df.index):
    with cols[i]:
        price = df.at[t, "Current_Price"]
        fwd1 = df.at[t, "FwdPE_Y1"]
        label = ce.at[t, "Year 1"] or "N/A"
        tag = "green" if label == "Cheap" else "red"
        st.markdown(f"""
        <div class="kpi">
          <div style="font-size:15px; font-weight:700; margin-bottom:4px">{t}</div>
          <div class="small">Price</div>
          <div style="font-size:20px; font-weight:700">${price:.2f}</div>
          <div class="small" style="margin-top:6px">Forward P/E Y1</div>
          <div style="display:flex; gap:8px; align-items:center">
            <div style="font-size:20px; font-weight:700">{fwd1:.2f}x</div>
            <span class="badge {tag}">{label}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

st.caption("Cheap or Expensive is judged versus the 10Y inverse P/E shown above.")

# -------------- Single-page tables --------------
st.markdown('<div class="section-title">Inputs</div>', unsafe_allow_html=True)
inputs_pretty = inputs.rename(columns={
    "EPS_TTM":"EPS (TTM)",
    "EPS_nextY":"EPS next Y (Y+1)",
    "EPS_growth_5Y_pct":"EPS growth next 5Y (%)",
    "Fwd_PE_nextY":"Forward P/E (next Y)",
    "Current_Price":"Current Price"
})
st.dataframe(
    inputs_pretty.style.format({
        "EPS (TTM)":"{:.2f}",
        "EPS next Y (Y+1)":"{:.2f}",
        "EPS growth next 5Y (%)":"{:.2f}",
        "Forward P/E (next Y)":"{:.2f}",
        "Current Price":"${:,.2f}",
    }),
    use_container_width=True
)

st.markdown('<div class="section-title">EPS Projection</div>', unsafe_allow_html=True)
eps_tbl = df[[f"EPS_Y{y}" for y in years]]
eps_tbl.columns = [f"Year {y}" for y in years]
st.dataframe(eps_tbl.style.format("{:.2f}").background_gradient(cmap="Greens", axis=None),
             use_container_width=True)

st.markdown('<div class="section-title">Forward P/E Path</div>', unsafe_allow_html=True)
fpe_tbl = df[[f"FwdPE_Y{y}" for y in years]]
fpe_tbl.columns = [f"Year {y}" for y in years]
def color_pe(val):
    if pd.isna(val) or pd.isna(threshold): return ""
    return "background-color: #d9f2e4" if val <= threshold else "background-color: #f9d5d0"
st.dataframe(fpe_tbl.style.format("{:.2f}x").applymap(color_pe),
             use_container_width=True)

st.markdown('<div class="section-title">Cheap or Expensive Summary</div>', unsafe_allow_html=True)
st.dataframe(
    ce.style.apply(lambda s: ["background-color: #d9f2e4" if v=="Cheap" else "background-color: #f9d5d0" for v in s], axis=1),
    use_container_width=True
)

# -------------- Export --------------
st.divider()
out = pd.concat([inputs, df[[f"EPS_Y{y}" for y in years]], df[[f"FwdPE_Y{y}" for y in years]], ce.add_prefix("CE_")], axis=1)
st.download_button("Download CSV", out.to_csv().encode("utf-8"),
                   file_name="forward_pe_vs_10y.csv", mime="text/csv")
