# streamlit_forward_pe_app.py
# Requirements:
#   pip install streamlit yahooquery yfinance requests bs4

import streamlit as st
import pandas as pd
import numpy as np
from typing import List

# -------------- Page setup --------------
st.set_page_config(page_title="Forward P/E vs 10Y", layout="wide")

# Minimal CSS to tighten density and add chip styling
st.markdown("""
<style>
/* compact tables */
.dataframe tbody td { padding: 6px 8px !important; }
.dataframe thead th { padding: 6px 8px !important; font-weight: 600; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
.kpi { border:1px solid #e6e6e6; border-radius:12px; padding:14px; background:#fff; }
.badge { padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; }
.badge.green { background:#d9f2e4; color:#146c2e; }
.badge.red { background:#f9d5d0; color:#7a1c15; }
.small { color:#666; font-size:12px; }
</style>
""", unsafe_allow_html=True)

st.title("Forward P/E Path vs U.S. 10Y Inverse")

# -------------- Data fetchers --------------
@st.cache_data(ttl=300)
def fetch_from_yahooquery(tickers: List[str]) -> pd.DataFrame:
    from yahooquery import Ticker
    yq = Ticker(tickers, asynchronous=True)

    price = yq.price or {}
    summary_detail = yq.summary_detail or {}
    key_stats = yq.key_stats or {}
    earnings_trend = yq.earnings_trend or {}
    analysis = yq.analysis or {}

    rows = []
    for t in tickers:
        p = (price.get(t) or {})
        sd = (summary_detail.get(t) or {})
        ks = (key_stats.get(t) or {})
        et = (earnings_trend.get(t) or {})
        an = (analysis.get(t) or {})

        last = p.get("regularMarketPrice") or p.get("postMarketPrice")
        eps_ttm = ks.get("trailingEps") or sd.get("trailingEps") or p.get("epsTrailingTwelveMonths")

        eps_next_y = None
        trend_list = et.get("trend") if isinstance(et.get("trend"), list) else []
        for node in trend_list:
            if str(node.get("period")).lower() in {"y+1", "nextyear"}:
                eps_next_y = (node.get("epsTrend") or {}).get("avg")

        growth_5y = None
        g = (et.get("growth") or {}).get("longTerm")
        if g is not None:
            growth_5y = float(g) * 100.0
        if growth_5y is None:
            ge = (an.get("growth_estimates") or {})
            for k in ["next_5_years","nextFiveYears","next5Years"]:
                node = ge.get(k)
                if isinstance(node, dict) and node.get("avg") is not None:
                    growth_5y = float(node["avg"]) * 100.0
                    break

        fwd_pe_next_y = sd.get("forwardPE")
        if (fwd_pe_next_y in (None, 0)) and last and eps_next_y:
            try:
                fwd_pe_next_y = float(last) / float(eps_next_y)
            except Exception:
                fwd_pe_next_y = None

        rows.append({
            "Ticker": t,
            "EPS_TTM": eps_ttm,
            "EPS_nextY": eps_next_y,
            "EPS_growth_5Y_pct": growth_5y,
            "Fwd_PE_nextY": fwd_pe_next_y,
            "Current_Price": last,
        })
    return pd.DataFrame(rows).set_index("Ticker")


@st.cache_data(ttl=3600)
def fetch_eps_5y_from_finviz(ticker: str) -> float | None:
    import requests
    from bs4 import BeautifulSoup
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
    if r.status_code != 200:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    cells = [c.get_text(strip=True) for c in soup.select("table.snapshot-table2 td")]
    for i in range(0, len(cells)-1, 2):
        if cells[i].lower() in {"eps next 5y","epsnext5y"}:
            val = cells[i+1].replace("%","").replace(",","")
            try:
                return float(val)
            except Exception:
                return None
    return None


def backfill_growth_with_finviz(df: pd.DataFrame) -> pd.DataFrame:
    for t in df.index:
        if pd.isna(df.at[t, "EPS_growth_5Y_pct"]):
            g = fetch_eps_5y_from_finviz(t)
            if g is not None:
                df.at[t, "EPS_growth_5Y_pct"] = g
    return df


def fetch_live(tickers: List[str]) -> pd.DataFrame:
    df = fetch_from_yahooquery(tickers)
    for c in ["EPS_TTM","EPS_nextY","EPS_growth_5Y_pct","Fwd_PE_nextY","Current_Price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = backfill_growth_with_finviz(df)
    return df

# -------------- Sidebar controls --------------
with st.sidebar:
    st.subheader("Settings")
    ten_year = st.number_input("U.S. 10Y Treasury yield (%)", value=4.12, step=0.01, format="%.2f")
    inv_pe = 100.0 / ten_year if ten_year > 0 else np.nan
    st.metric("10Y Inverse P/E", f"{inv_pe:,.2f}x")

    st.subheader("Universe")
    tickers_input = st.text_input(
        "Tickers",
        value="MSFT, NVDA, META, AAPL, GOOGL, AMZN, TSLA",
        help="Comma separated"
    )

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
df_inputs = fetch_live(tickers)

# -------------- Calculations --------------
df = df_inputs.copy()

years = [1,2,3,4,5]
for y in years:
    if y == 1:
        df[f"EPS_Y{y}"] = df["EPS_nextY"]
    else:
        g = df["EPS_growth_5Y_pct"].fillna(0) / 100.0
        df[f"EPS_Y{y}"] = df[f"EPS_Y{y-1}"] * (1.0 + g)

for y in years:
    df[f"FwdPE_Y{y}"] = df["Current_Price"] / df[f"EPS_Y{y}"]

threshold = inv_pe

def ce_flag(v):
    if pd.isna(v) or pd.isna(threshold):
        return ""
    return "Cheap" if v <= threshold else "Expensive"

ce = pd.DataFrame(index=df.index)
for y in [1,2,3]:
    ce[f"Year {y}"] = df[f"FwdPE_Y{y}"].apply(ce_flag)

# -------------- Header KPIs --------------
cols = st.columns(len(df.index))
for i, t in enumerate(df.index):
    with cols[i]:
        fwd1 = df.at[t, "FwdPE_Y1"]
        tag = "green" if ce.at[t,"Year 1"] == "Cheap" else "red"
        label = ce.at[t,"Year 1"] or "N/A"
        st.markdown(f"""
        <div class="kpi">
          <div style="font-size:15px; font-weight:700; margin-bottom:4px">{t}</div>
          <div class="small">Price</div>
          <div style="font-size:20px; font-weight:700">${df.at[t,'Current_Price']:.2f}</div>
          <div class="small" style="margin-top:6px">Forward P/E Y1</div>
          <div style="display:flex; gap:8px; align-items:center">
            <div style="font-size:20px; font-weight:700">{fwd1:.2f}x</div>
            <span class="badge {tag}">{label}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

st.caption("Cheap or Expensive is judged versus the 10Y inverse P/E shown above.")

# -------------- Tabs for detail --------------
tab1, tab2, tab3 = st.tabs(["Inputs", "Projections", "Forward P/E"])

with tab1:
    tbl = df_inputs.copy()
    tbl = tbl.rename(columns={
        "EPS_TTM":"EPS (TTM)",
        "EPS_nextY":"EPS next Y",
        "EPS_growth_5Y_pct":"EPS growth next 5Y (%)",
        "Fwd_PE_nextY":"Forward P/E (next Y)",
        "Current_Price":"Current Price"
    })
    st.dataframe(
        tbl.style.format({
            "EPS (TTM)":"{:.2f}",
            "EPS next Y":"{:.2f}",
            "EPS growth next 5Y (%)":"{:.2f}",
            "Forward P/E (next Y)":"{:.2f}",
            "Current Price":"${:,.2f}",
        }),
        use_container_width=True
    )

with tab2:
    eps = df[[f"EPS_Y{y}" for y in years]]
    eps.columns = [f"Year {y}" for y in years]
    st.dataframe(
        eps.style.format("{:.2f}").background_gradient(
            cmap="Greens", axis=None
        ),
        use_container_width=True
    )

with tab3:
    fpe = df[[f"FwdPE_Y{y}" for y in years]].copy()
    fpe.columns = [f"Year {y}" for y in years]

    def color_pe(val):
        if pd.isna(val) or pd.isna(threshold):
            return ""
        return "background-color: #d9f2e4" if val <= threshold else "background-color: #f9d5d0"

    styled = fpe.style.format("{:.2f}x").applymap(color_pe)
    st.dataframe(styled, use_container_width=True)

    st.markdown("**Cheap/Expensive summary**")
    st.dataframe(
        ce.style.apply(
            lambda s: ["background-color: #d9f2e4" if v=="Cheap" else "background-color: #f9d5d0" for v in s],
            axis=1
        ),
        use_container_width=True
    )

# -------------- Export --------------
st.divider()
out = pd.concat(
    [
        df_inputs.rename(columns={
            "EPS_TTM":"EPS_TTM",
            "EPS_nextY":"EPS_nextY",
            "EPS_growth_5Y_pct":"EPS_growth_5Y_pct",
            "Fwd_PE_nextY":"Fwd_PE_nextY",
            "Current_Price":"Current_Price"
        }),
        df[[f"EPS_Y{y}" for y in years]],
        df[[f"FwdPE_Y{y}" for y in years]],
        ce.add_prefix("CE_"),
    ],
    axis=1
)
csv = out.to_csv(index=True).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name="forward_pe_vs_10y.csv", mime="text/csv")
