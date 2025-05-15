from datetime import datetime
import pandas as pd
import yfinance as yf
import streamlit as st

# ── Fetch & filter expiries ────────────────────────────────────────────────
@st.cache(ttl=3600, show_spinner=False)
def get_valid_expiries(tkr):
    raw = yf.Ticker(tkr).options
    today = datetime.today().date()
    return [exp for exp in raw if pd.to_datetime(exp).date() >= today]

expiries = get_valid_expiries(ticker)
if not expiries:
    st.error(f"No upcoming expiries available for {ticker}.")
    st.stop()

expiry = st.sidebar.selectbox("Select Expiry", expiries)

# ── Try fetching the chain, validate non‑empty ─────────────────────────────
@st.cache(ttl=600, show_spinner=False)
def get_chain_safe(tkr, exp):
    return yf.Ticker(tkr).option_chain(exp)

try:
    chain = get_chain_safe(ticker, expiry)
    if chain.calls.empty and chain.puts.empty:
        raise ValueError
except Exception:
    st.warning(
        f"⚠️ Could not load an options chain for {ticker} @ {expiry}.  "
        "It may have just expired or not yet populated.  Please pick a different expiry."
    )
    st.stop()

# ── …then continue on with `chain.calls` and `chain.puts` as before. ───────
