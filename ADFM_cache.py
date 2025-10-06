# adfm_cache.py
import os, datetime as dt, pandas as pd, yfinance as yf
import streamlit as st

DEFAULT_YEARS = 5

@st.cache_resource
def yf_client():
    return yf

@st.cache_data(ttl=3600, max_entries=50)
def load_ohlc(ticker: str, start=None, end=None):
    end = end or dt.date.today()
    start = start or (end - dt.timedelta(days=365*DEFAULT_YEARS))
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = df.reset_index().rename(columns={"Date": "date"})
    # downcast to save RAM
    for c in df.select_dtypes("float64").columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes("int64").columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df
