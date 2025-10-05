# streamlit_forward_pe_app.py
# Requirements:
#   pip install streamlit yahooquery yfinance requests bs4

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any

st.set_page_config(page_title="Forward P/E vs 10Y", layout="wide")
st.title("Forward P/E Path vs U.S. 10Y Inverse")

# ----------------------------- Fetchers -----------------------------
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

        # Price
        last = p.get("regularMarketPrice") or p.get("postMarketPrice")

        # EPS TTM
        eps_ttm = ks.get("trailingEps") or sd.get("trailingEps") or p.get("epsTrailingTwelveMonths")

        # Forward EPS next year and long term growth
        eps_next_y = None
        growth_5y = None
        fwd_pe_next_y = sd.get("forwardPE")

        # Primary: earnings_trend
        trend_list = et.get("trend") if isinstance(et.get("trend"), list) else []
        for node in trend_list:
            if str(node.get("period")).lower() in {"y+1", "nextyear"}:
                eps_next_y = (node.get("epsTrend") or {}).get("avg")
        g = (et.get("growth") or {}).get("longTerm")
        if g is not None:
            growth_5y = float(g) * 100.0

        # Secondary: analysis growthEstimates
        if growth_5y is None:
            ge = (an.get("growth_estimates") or {})
            # Expected keys sometimes look like: {'next_5_years': {'avg': 0.15, ...}}
            for k in ["next_5_years", "nextFiveYears", "next5Years"]:
                node = ge.get(k)
                if isinstance(node, dict) and node.get("avg") is not None:
                    growth_5y = float(node["avg"]) * 100.0
                    break

        # Derive forward PE if missing
        if (fwd_pe_next_y in (None, 0)) and last and eps_next_y:
            try:
                fwd_pe_next_y = float(last) / float(eps_next_y)
            except Exception:
                fwd_pe_next_y = None

        rows.append(
            {
                "Ticker": t,
                "EPS_TTM": eps_ttm,
                "EPS_nextY": eps_next_y,
                "EPS_growth_5Y_pct": growth_5y,
                "Fwd_PE_nextY": fwd_pe_next_y,
                "Current_Price": last,
            }
        )
    return pd.DataFrame(rows).set_index("Ticker")


@st.cache_data(ttl=300)
def fetch_from_yfinance(tickers: List[str]) -> pd.DataFrame:
    import yfinance as yf

    rows = []
    for t in tickers:
        tk = yf.Ticker(t)
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        # Price
        last = None
        try:
            last = float(tk.fast_info.last_price)
        except Exception:
            last = info.get("regularMarketPrice")

        eps_ttm = info.get("trailingEps")
        eps_next_y = info.get("forwardEps")
        fwd_pe_next_y = info.get("forwardPE")
        growth_5y = None

        # Try new earnings trend API if available
        try:
            trend = tk.get_earnings_trend()
            if isinstance(trend, pd.DataFrame) and not trend.empty:
                nxt = trend[trend["period"].str.lower().isin(["y+1", "nextyear"])]
                if not nxt.empty:
                    if "epsTrend_avg" in nxt.columns:
                        eps_next_y = nxt.iloc[0]["epsTrend_avg"]
                if "growth_longTerm" in trend.columns and pd.notna(trend.iloc[0]["growth_longTerm"]):
                    growth_5y = float(trend.iloc[0]["growth_longTerm"]) * 100.0
        except Exception:
            pass

        if (not fwd_pe_next_y) and last and eps_next_y:
            try:
                fwd_pe_next_y = float(last) / float(eps_next_y)
            except Exception:
                fwd_pe_next_y = None

        rows.append(
            {
                "Ticker": t,
                "EPS_TTM": eps_ttm,
                "EPS_nextY": eps_next_y,
                "EPS_growth_5Y_pct": growth_5y,
                "Fwd_PE_nextY": fwd_pe_next_y,
                "Current_Price": last,
            }
        )
    return pd.DataFrame(rows).set_index("Ticker")


@st.cache_data(ttl=3600)
def fetch_eps_5y_from_finviz(ticker: str) -> float | None:
    # Finviz shows "EPS next 5Y" on the quote page. Use a light HTML parse.
    import requests
    from bs4 import BeautifulSoup

    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code != 200:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    # Look for the snapshot table cells
    cells = [c.get_text(strip=True) for c in soup.select("table.snapshot-table2 td")]
    # Cells alternate as Label, Value
    for i in range(0, len(cells) - 1, 2):
        if cells[i].lower() in {"eps next 5y", "epsnext5y"}:
            val = cells[i + 1].replace("%", "").replace(",", "")
            try:
                return float(val)
            except Exception:
                return None
    return None


def fill_missing_growth_with_finviz(df: pd.DataFrame) -> pd.DataFrame:
    for t in df.index:
        if pd.isna(df.at[t, "EPS_growth_5Y_pct"]):
            g = fetch_eps_5y_from_finviz(t)
            if g is not None:
                df.at[t, "EPS_growth_5Y_pct"] = g
    return df


def fetch_live(tickers: List[str]) -> pd.DataFrame:
    try:
        df = fetch_from_yahooquery(tickers)
    except Exception:
        df = fetch_from_yfinance(tickers)

    # Clean numeric
    for c in ["EPS_TTM", "EPS_nextY", "EPS_growth_5Y_pct", "Fwd_PE_nextY", "Current_Price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Backfill 5Y growth from Finviz if Yahoo missing
    df = fill_missing_growth_with_finviz(df)
    return df

# ----------------------------- UI -----------------------------
with st.sidebar:
    st.subheader("Global Settings")
    ten_year = st.number_input("U.S. 10Y Treasury yield (%)", value=4.12, step=0.01, format="%.2f")
    inv_multiple = 100.0 / ten_year if ten_year > 0 else np.nan
    st.metric("10Y Inverse P/E", f"{inv_multiple:,.2f}x")

    st.subheader("Universe")
    tickers_input = st.text_input(
        "Tickers (comma separated)",
        value="MSFT, NVDA, META, AAPL, GOOGL, AMZN, TSLA",
    )
    allow_edit_prices = st.checkbox("Allow manual price edits after fetch", value=False)

st.caption(
    "EPS next Y and 5Y growth are fetched from Yahoo first and backfilled from Finviz when missing. "
    "Forward P/E next Y is from Yahoo or derived as Price divided by EPS next Y."
)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
live_df = fetch_live(tickers)

st.subheader("Inputs")
df_edit = st.data_editor(
    live_df,
    use_container_width=True,
    num_rows="dynamic",
    disabled=None if allow_edit_prices else ["Current_Price"],
    column_config={
        "EPS_TTM": st.column_config.NumberColumn("EPS (TTM)", format="%.4f"),
        "EPS_nextY": st.column_config.NumberColumn("EPS next Y", format="%.4f"),
        "EPS_growth_5Y_pct": st.column_config.NumberColumn("EPS growth next 5Y (%)", format="%.2f"),
        "Fwd_PE_nextY": st.column_config.NumberColumn("Forward P/E (next Y)", format="%.2f"),
        "Current_Price": st.column_config.NumberColumn("Current Share Price", format="%.2f"),
    },
)

df = df_edit.copy()

# ----------------------------- Calculations -----------------------------
years = [1, 2, 3, 4, 5]
for y in years:
    if y == 1:
        df[f"EPS_Y{y}"] = df["EPS_nextY"]
    else:
        g = df["EPS_growth_5Y_pct"].fillna(0) / 100.0
        df[f"EPS_Y{y}"] = df[f"EPS_Y{y-1}"] * (1.0 + g)

for y in years:
    df[f"FwdPE_Y{y}"] = df["Current_Price"] / df[f"EPS_Y{y}"]

threshold = 100.0 / ten_year if ten_year > 0 else np.nan

def cheap_exp(val):
    if pd.isna(val) or pd.isna(threshold):
        return ""
    return "Cheap" if val <= threshold else "Expensive"

ce_table = pd.DataFrame(index=df.index)
for y in [1, 2, 3]:
    ce_table[f"Based on Forward P/E Year {y}"] = df[f"FwdPE_Y{y}"].apply(cheap_exp)

eps_proj_cols = [f"EPS_Y{y}" for y in years]
fpe_cols = [f"FwdPE_Y{y}" for y in years]
pretty_eps = df[eps_proj_cols].rename(columns=lambda c: c.replace("EPS_Y", "Year "))
pretty_fpe = df[fpe_cols].rename(columns=lambda c: c.replace("FwdPE_Y", "Year "))

# ----------------------------- Display -----------------------------
st.subheader("EPS Projection")
st.dataframe(pretty_eps.style.format("{:.2f}"), use_container_width=True)

st.subheader("Forward P/E Calculation")
st.dataframe(pretty_fpe.style.format("{:.2f}x"), use_container_width=True)

st.subheader("Versus U.S. 10Y Treasuries - Cheap or Expensive")
def style_ce(s):
    return ["background-color: #b7e1cd" if v == "Cheap" else "background-color: #f4c7c3" for v in s]
st.dataframe(ce_table.style.apply(style_ce, axis=1), use_container_width=True)

st.divider()
st.subheader("Download Results")
out = pd.concat(
    [
        df_edit,
        pretty_eps.add_prefix("EPS "),
        pretty_fpe.add_prefix("Fwd P/E "),
        ce_table,
    ],
    axis=1
)
csv = out.to_csv(index=True).encode("utf-8")
st.download_button("Download CSV", csv, file_name="forward_pe_vs_10y.csv", mime="text/csv")

st.caption("If a growth value still shows empty, Finviz may not have it or is rate limited. Edit the cell directly when needed.")
