# ai_compute_buildout_dashboard.py

"""
A single Streamlit app with 4 practical trackers for the AI + datacenter theme
across US, Europe ADRs, and Asia ADRs. No GA4. No Google Trends.

Tabs
1) Hyperscaler CapEx (SEC EDGAR XBRL facts)
2) Filings AI Mentions (10-Q/10-K keyword counts)
3) Power Demand & Prices (EIA API)
4) Vendor Revenue Trend (NVDA, AMD, SMCI, etc via SEC facts)

Notes
- Requires internet access when you run it locally or on Streamlit Cloud.
- SEC API requires a User-Agent header with your email or firm. Enter it in the sidebar.
- EIA API requires an API key (free). Enter it in the sidebar to enable the Power tab.
"""

from __future__ import annotations
import json
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="AI Compute Buildout Dashboard", page_icon="📊", layout="wide")

# =====================
# Sidebar: About
# =====================
with st.sidebar:
    st.title("AI Compute Buildout")
    st.caption("Decision trackers for the AI + datacenter theme. Data from SEC EDGAR and EIA.")
    with st.expander("About This Tool", expanded=True):
        st.write(
            """
            **What this does**
            - Pulls **CapEx** and **Revenue** time series from the SEC XBRL facts API for selected companies.
            - Scans the latest **10-Q/10-K** filings for **AI/data center** keyword counts.
            - Optional: fetches **US power demand and prices** from EIA for PJM, ERCOT, CAISO, MISO.

            **Why it helps**
            - CapEx lines at hyperscalers are a direct proxy for compute and data center build.
            - Filing mentions show narrative intensity that often leads real spend.
            - Power demand and price tighten when GPU buildouts stress grids.

            **Inputs needed**
            - SEC header: add your email or firm so EDGAR permits requests.
            - EIA key: paste your API key to enable power data.
            """
        )
    st.markdown("---")
    sec_header = st.text_input("SEC User-Agent (email or firm)", value="you@example.com")
    eia_key = st.text_input("EIA API key (optional for Power tab)", value="", type="password")

# =====================
# Static helpers
# =====================
SEC_BASE = "https://data.sec.gov"
HEADERS = lambda: {"User-Agent": sec_header or "you@example.com", "Accept-Encoding": "gzip", "Host": "data.sec.gov"}

# Common tickers and CIKs (leading zeros kept)
CIK_MAP: Dict[str, str] = {
    # Hyperscalers
    "AMZN": "0001018724",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",  # Alphabet
    "META": "0001326801",
    "ORCL": "0001341439",
    # Infra REITs
    "EQIX": "0001101239",
    "DLR":  "0001297996",
    # AI vendors / servers
    "NVDA": "0001045810",
    "AMD":  "0000002488",
    "SMCI": "0001629280",
    # Asia/EU ADRs with SEC reporting
    "TSM":  "0001046179",
    "ASML": "0000937480",
}

CAPEX_CANDIDATES = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PurchaseOfPropertyAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    "CapitalExpenditures",
]
REVENUE_CANDIDATES = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "Revenues",
]

KEYWORDS = ["data center", "datacenter", "GPU", "accelerator", "AI", "NVIDIA", "HBM", "inference", "training"]

EBA_SERIES = {
    # EIA EBA series ids: demand hourly. For CAISO use CISO, ERCOT ERCO
    "PJM": "EBA.PJM-ALL.D.H",
    "ERCOT": "EBA.ERCO-ALL.D.H",
    "CAISO": "EBA.CISO-ALL.D.H",
    "MISO": "EBA.MISO-ALL.D.H",
}
PRICE_SERIES = {
    # Retail industrial electricity price cents/kWh (monthly). State averages as simple proxies.
    # Users can adjust to precise hubs if they prefer.
    "US": "ELEC.PRICE.IND.US-ALL.M",
}

# =====================
# Caching wrappers
# =====================
@st.cache_data(show_spinner=False)
def sec_company_facts(cik: str) -> dict:
    url = f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS(), timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def sec_submissions(cik: str) -> dict:
    url = f"{SEC_BASE}/submissions/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS(), timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def fetch_eia_series(series_id: str, api_key: str, rows: int = 2000) -> pd.DataFrame:
    url = f"https://api.eia.gov/series/?api_key={api_key}&series_id={series_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    data = js["series"][0]["data"]
    df = pd.DataFrame(data, columns=["period", "value"])  # newest first
    # Handle hourly vs monthly formats
    if len(df.iloc[0, 0]) in (10, 13):  # hourly like YYYYMMDDHH or YYYYMMDDTHHZ
        df["Date"] = pd.to_datetime(df["period"].astype(str).str[:10], format="%Y%m%d%H", errors="coerce")
    else:
        # monthly YYYYMM
        df["Date"] = pd.to_datetime(df["period"].astype(str)+"01", format="%Y%m%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.sort_values("Date")

# =====================
# Parsing helpers
# =====================

def _best_concept(facts: dict, candidates: List[str]) -> Optional[Tuple[str, list]]:
    if not facts or "facts" not in facts:
        return None
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    best_name, best_list = None, None
    for name in candidates:
        obj = usgaap.get(name)
        if not obj:
            continue
        units = obj.get("units", {})
        # prefer USD then USDm
        for u in ("USD", "USDm", "USD Millions"):
            if u in units and units[u]:
                best_name, best_list = name, units[u]
                break
        if best_list:
            break
    if best_list is None:
        return None
    return best_name, best_list


def _to_quarterly_df(measures: list, filter_forms: Tuple[str, ...] = ("10-Q", "10-K", "20-F", "6-K")) -> pd.DataFrame:
    rows = []
    for m in measures:
        form = m.get("form", "")
        if filter_forms and form not in filter_forms:
            continue
        try:
            end = pd.to_datetime(m["end"])
        except Exception:
            continue
        val = pd.to_numeric(m.get("val"), errors="coerce")
        if pd.isna(val):
            continue
        rows.append({"Date": end, "value": float(val), "form": form, "fy": m.get("fy"), "fp": m.get("fp")})
    if not rows:
        return pd.DataFrame(columns=["Date", "value"])
    df = pd.DataFrame(rows).sort_values("Date").drop_duplicates("Date", keep="last")
    return df


def sec_timeseries_for_concept(ticker: str, candidates: List[str]) -> pd.DataFrame:
    cik = CIK_MAP[ticker]
    facts = sec_company_facts(cik)
    best = _best_concept(facts, candidates)
    if not best:
        return pd.DataFrame(columns=["Date", ticker])
    _, measures = best
    q = _to_quarterly_df(measures)
    if q.empty:
        return pd.DataFrame(columns=["Date", ticker])
    q = q[["Date", "value"]].rename(columns={"value": ticker})
    return q


def build_capex_panel(tickers: List[str]) -> pd.DataFrame:
    frames = []
    for t in tickers:
        try:
            df = sec_timeseries_for_concept(t, CAPEX_CANDIDATES)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="Date", how="outer")
    return out.sort_values("Date")


def build_revenue_panel(tickers: List[str]) -> pd.DataFrame:
    frames = []
    for t in tickers:
        try:
            df = sec_timeseries_for_concept(t, REVENUE_CANDIDATES)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="Date", how="outer")
    return out.sort_values("Date")

# =====================
# UI controls
# =====================
DEFAULT_HYPERS = ["AMZN", "MSFT", "GOOGL", "META", "ORCL"]
DEFAULT_REITS  = ["EQIX", "DLR"]
DEFAULT_VENDORS = ["NVDA", "AMD", "SMCI", "TSM", "ASML"]

sel_hypers = st.sidebar.multiselect("Hyperscalers", DEFAULT_HYPERS, default=DEFAULT_HYPERS)
sel_reits = st.sidebar.multiselect("Data center REITs", DEFAULT_REITS, default=DEFAULT_REITS)
sel_vendors = st.sidebar.multiselect("Vendors", DEFAULT_VENDORS, default=["NVDA", "AMD", "SMCI"])

st.sidebar.markdown("---")
fetch_btn = st.sidebar.button("Pull latest data", type="primary")

# =====================
# Tabs
# =====================

capex_tab, filings_tab, power_tab, revenue_tab = st.tabs([
    "Hyperscaler CapEx", "Filings AI Mentions", "Power Demand & Prices", "Vendor Revenue"
])

# ============= Tab 1: CapEx
with capex_tab:
    st.subheader("Hyperscaler CapEx (quarterly, USD)")
    if fetch_btn or "_capex" not in st.session_state:
        capex_df = build_capex_panel(sel_hypers)
        st.session_state["_capex"] = capex_df
    capex_df = st.session_state.get("_capex", pd.DataFrame())
    if capex_df.empty:
        st.info("No CapEx data returned. Check SEC header or try fewer tickers.")
    else:
        # LTM roll
        ltm = capex_df.set_index("Date").rolling(4).sum().reset_index()
        fig = go.Figure()
        for c in [col for col in ltm.columns if col != "Date"]:
            fig.add_trace(go.Scatter(x=ltm["Date"], y=ltm[c], mode="lines", name=f"{c} LTM"))
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(capex_df.tail(12), use_container_width=True)

# ============= Tab 2: Filings Mentions
with filings_tab:
    st.subheader("Keyword mentions in latest 10-Q/10-K")

    @st.cache_data(show_spinner=False)
    def scan_filings_counts(tickers: List[str], keywords: List[str]) -> pd.DataFrame:
        rows = []
        for t in tickers:
            try:
                cik = CIK_MAP[t]
                sub = sec_submissions(cik)
                # Gather last up to 6 relevant filings
                filings = pd.DataFrame(sub.get("filings", {}).get("recent", {}))
                if filings.empty:
                    continue
                mask = filings["form"].isin(["10-Q", "10-K"]).fillna(False)
                f = filings[mask].head(6)
                total_counts = {k: 0 for k in keywords}
                for _, r in f.iterrows():
                    acc = str(r["accessionNumber"]).replace("-", "")
                    doc = str(r["primaryDocument"])
                    url = f"https://www.sec.gov/Archives/edgar/data/{int(CIK_MAP[t])}/{acc}/{doc}"
                    try:
                        txt = requests.get(url, headers=HEADERS(), timeout=30).text.lower()
                        for kw in keywords:
                            total_counts[kw] += txt.count(kw.lower())
                    except Exception:
                        continue
                row = {"Ticker": t, **total_counts}
                rows.append(row)
            except Exception:
                continue
        return pd.DataFrame(rows)

    if fetch_btn or "_mentions" not in st.session_state:
        st.session_state["_mentions"] = scan_filings_counts(sel_hypers + sel_reits + sel_vendors, KEYWORDS)
    mentions = st.session_state.get("_mentions", pd.DataFrame())
    if mentions.empty:
        st.info("No filings found. Check SEC header or try again later.")
    else:
        st.dataframe(mentions.set_index("Ticker").sort_values("AI", ascending=False), use_container_width=True)
        st.caption("Counts are rough string matches across the latest few filings.")

# ============= Tab 3: Power
with power_tab:
    st.subheader("US Power Demand and Industrial Price")
    if not eia_key:
        st.info("Add an EIA API key in the sidebar to enable this tab.")
    else:
        bas = st.multiselect("Balancing Authorities", list(EBA_SERIES.keys()), default=["PJM", "ERCOT", "CAISO"]) 
        frames = []
        for ba in bas:
            try:
                s = fetch_eia_series(EBA_SERIES[ba], eia_key)
                s = s.rename(columns={"value": ba})
                frames.append(s[["Date", ba]])
            except Exception:
                continue
        if frames:
            out = frames[0]
            for f in frames[1:]:
                out = out.merge(f, on="Date", how="outer")
            fig = go.Figure()
            for c in [col for col in out.columns if col != "Date"]:
                fig.add_trace(go.Scatter(x=out["Date"], y=out[c], mode="lines", name=c))
            fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Demand (MW)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(out.tail(168), use_container_width=True)
        st.markdown("---")
        try:
            price = fetch_eia_series(PRICE_SERIES["US"], eia_key)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=price["Date"], y=price["value"], mode="lines", name="US Industrial c/kWh"))
            fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            st.caption("Industrial price series not available with current key.")

# ============= Tab 4: Vendor revenue
with revenue_tab:
    st.subheader("Vendor Revenue (quarterly, USD)")
    if fetch_btn or "_rev" not in st.session_state:
        rev_df = build_revenue_panel(sel_vendors)
        st.session_state["_rev"] = rev_df
    rev_df = st.session_state.get("_rev", pd.DataFrame())
    if rev_df.empty:
        st.info("No revenue data returned.")
    else:
        fig = go.Figure()
        for c in [col for col in rev_df.columns if col != "Date"]:
            fig.add_trace(go.Scatter(x=rev_df["Date"], y=rev_df[c], mode="lines", name=c))
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(rev_df.tail(12), use_container_width=True)

st.markdown("\n")
