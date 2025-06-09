import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Simple Options Chain Viewer", layout="wide")
st.title('Simple Options Chain Viewer')

ticker = st.text_input('Ticker', 'AAPL').upper().strip()
if not ticker:
    st.warning("Enter a ticker symbol")
    st.stop()

# --- Try to load expiries from Yahoo Finance ---
try:
    tk = yf.Ticker(ticker)
    expiries = tk.options
except Exception as e:
    st.error(f"Could not connect to Yahoo: {e}")
    st.stop()

if not expiries:
    st.error(
        f"No options expiries returned for {ticker}. "
        "This is a Yahoo data issue or your IP is blocked. "
        "Try switching your network, using a VPN, or waiting a while and try again."
    )
    st.write("Debug info (expiries):", expiries)
    st.stop()

expiry = st.selectbox("Select Expiry", expiries)

# --- Try to load option chain for selected expiry ---
try:
    chain = tk.option_chain(expiry)
    st.subheader("Calls")
    st.dataframe(chain.calls)
    st.subheader("Puts")
    st.dataframe(chain.puts)
except Exception as e:
    st.error(f"Failed to load option chain for {ticker} @ {expiry}: {e}")

st.caption("© 2025 AD Fund Management LP")
