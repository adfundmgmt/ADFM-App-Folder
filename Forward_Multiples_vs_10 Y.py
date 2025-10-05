# streamlit_forward_pe_app.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Forward P/E vs 10Y", layout="wide")
st.title("Forward P/E Path vs U.S. 10Y Inverse")

# ----------------------------- Helpers -----------------------------
@st.cache_data(ttl=300)
def fetch_from_yahooquery(tickers: list[str]) -> pd.DataFrame:
    try:
        from yahooquery import Ticker
    except Exception as e:
        raise RuntimeError("yahooquery is required. Try: pip install yahooquery") from e

    yq = Ticker(tickers, asynchronous=True)

    price = yq.price
    summary_detail = yq.summary_detail
    key_stats = yq.key_stats
    earnings_trend = yq.earnings_trend

    rows = []
    for t in tickers:
        p = price.get(t, {}) or {}
        sd = summary_detail.get(t, {}) or {}
        ks = key_stats.get(t, {}) or {}
        et = earnings_trend.get(t, {}) or {}

        # Current price
        last = p.get("regularMarketPrice") or p.get("postMarketPrice")

        # EPS TTM
        eps_ttm = (
            ks.get("trailingEps")
            or sd.get("trailingEps")
            or p.get("epsTrailingTwelveMonths")
        )

        # Forward EPS next year from earnings trend
        eps_next_y = None
        growth_5y = None
        fwd_pe_next_y = sd.get("forwardPE")

        trend_list = et.get("trend") if isinstance(et.get("trend"), list) else []
        for node in trend_list:
            if str(node.get("period")).lower() in {"y+1", "nextyear"}:
                eps_next_y = (node.get("epsTrend") or {}).get("avg")
            # growth longTerm is a float like 0.15 for 15 percent
            g = (et.get("growth") or {}).get("longTerm")
            if g is not None:
                growth_5y = g * 100.0

        # If forward PE missing, derive from price and forward EPS
        if fwd_pe_next_y in (None, 0) and eps_next_y:
            if last and eps_next_y:
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

    df = pd.DataFrame(rows).set_index("Ticker")
    return df


@st.cache_data(ttl=300)
def fetch_from_yfinance(tickers: list[str]) -> pd.DataFrame:
    # Secondary fallback for environments where yahooquery is blocked
    import yfinance as yf

    rows = []
    for t in tickers:
        tk = yf.Ticker(t)
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}
        last = None
        try:
            last = float(tk.fast_info.last_price)
        except Exception:
            try:
                last = float(info.get("regularMarketPrice"))
            except Exception:
                last = None

        eps_ttm = info.get("trailingEps")
        eps_next_y = info.get("forwardEps")  # note: this is next 12m EPS, often close to next fiscal Y
        growth_5y = None
        try:
            trend = tk.get_earnings_trend()  # returns DataFrame in newer yfinance
            if isinstance(trend, pd.DataFrame) and not trend.empty:
                node = trend[trend["period"].str.lower().isin(["y+1", "nextyear"])]
                if not node.empty and "epsTrend_avg" in node.columns:
                    eps_next_y = node.iloc[0]["epsTrend_avg"]
                if "growth_longTerm" in trend.columns:
                    g = trend.iloc[0]["growth_longTerm"]
                    growth_5y = float(g) * 100.0 if pd.notna(g) else None
        except Exception:
            pass

        fwd_pe_next_y = info.get("forwardPE")
        if not fwd_pe_next_y and last and eps_next_y:
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


def fetch_live(tickers: list[str]) -> pd.DataFrame:
    try:
        df = fetch_from_yahooquery(tickers)
    except Exception:
        df = fetch_from_yfinance(tickers)

    # Clean numeric types
    for c in ["EPS_TTM", "EPS_nextY", "EPS_growth_5Y_pct", "Fwd_PE_nextY", "Current_Price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

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
    keep_price_constant = st.checkbox("Freeze prices at fetch time", value=True)
    st.caption("If unchecked, you can paste your own Current Price after fetch.")

st.caption(
    "Inputs are fetched from Yahoo Finance. EPS next Y uses Yahoo earnings trend where available. "
    "5Y growth uses long term EPS growth from Yahoo. Forward P/E next Y comes from Yahoo or price divided by forward EPS."
)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

live_df = fetch_live(tickers)
if not keep_price_constant:
    # Allow user edits to price
    pass

st.subheader("Fetched Inputs")
editable = st.data_editor(
    live_df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "EPS_TTM": st.column_config.NumberColumn("EPS (TTM)", format="%.4f"),
        "EPS_nextY": st.column_config.NumberColumn("EPS next Y", format="%.4f"),
        "EPS_growth_5Y_pct": st.column_config.NumberColumn("EPS growth next 5Y (%)", format="%.2f"),
        "Fwd_PE_nextY": st.column_config.NumberColumn("Forward P/E (next Y)", format="%.2f"),
        "Current_Price": st.column_config.NumberColumn("Current Share Price", format="%.2f"),
    },
)

df = editable.copy()

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
        editable,                       # possibly edited inputs
        pretty_eps.add_prefix("EPS "),
        pretty_fpe.add_prefix("Fwd P/E "),
        ce_table,
    ],
    axis=1
)
csv = out.to_csv(index=True).encode("utf-8")
st.download_button("Download CSV", csv, file_name="forward_pe_vs_10y.csv", mime="text/csv")

st.caption("Notes: Data quality varies by ticker. If a field is missing, edit it directly in the grid.")
