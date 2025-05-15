from datetime import datetime

# ── Fetch & filter expiries ────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_valid_expiries(tkr):
    raw = yf.Ticker(tkr).options
    today = datetime.today().date()
    # keep only dates >= today
    return [exp for exp in raw if pd.to_datetime(exp).date() >= today]

expiries = get_valid_expiries(ticker)
if not expiries:
    st.error(f"No upcoming expiries available for {ticker}.")
    st.stop()

expiry = st.sidebar.selectbox("Select Expiry", expiries)

# ── Try fetching the chain, validate non‑empty ─────────────────────────────
try:
    chain = get_chain(ticker, expiry)
    # yfinance returns empty DataFrames if nothing’s available
    if chain.calls.empty and chain.puts.empty:
        raise ValueError("Empty chain")
except Exception:
    st.warning(f"⚠️ Could not load an options chain for {ticker} @ {expiry}. "
               "It may have just expired or not yet populated.  Please pick a different expiry.")
    st.stop()

# ── Proceed as before with `chain.calls` and `chain.puts` ─────────────────
