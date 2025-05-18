import streamlit as st
import requests
import pandas as pd
import re

st.set_page_config(page_title="SEC 10-Q/10-K Dashboard", layout="wide")
st.title("SEC Filing Metrics Dashboard")

@st.cache_data(ttl=24*3600, show_spinner=False)
def build_ticker_cik_map():
    url = 'https://www.sec.gov/include/ticker.txt'
    resp = requests.get(url)
    lookup = {}
    for line in resp.text.splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            tkr, cik = parts
            lookup[tkr.lower()] = cik.zfill(10)
    return lookup

def get_cik(ticker, lookup):
    # For debugging: print available keys
    if ticker.lower().strip() not in lookup:
        st.warning(f"First 10 tickers loaded: {list(lookup.keys())[:10]}")
    return lookup.get(ticker.lower().strip(), None)

def get_filing_metadata(cik, count=6):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        return None
    data = resp.json()
    filings = []
    for i, form in enumerate(data['filings']['recent']['form']):
        if form in ['10-Q', '10-K']:
            filings.append({
                "form": form,
                "date": data['filings']['recent']['filingDate'][i],
                "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                       f"{data['filings']['recent']['accessionNumber'][i].replace('-','')}/"
                       f"{data['filings']['recent']['primaryDocument'][i]}"
            })
            if len(filings) >= count:
                break
    return filings

def parse_metrics_from_table(df):
    metrics = {}
    possible_labels = {
        "revenue": ["total revenue", "net revenue", "revenues", "sales"],
        "net_income": ["net income", "net earnings", "net (loss)", "net loss"],
        "eps": ["earnings per share", "net income per share", "basic net income per share", "diluted net income per share"],
        "operating_cash_flow": ["net cash provided by operating activities", "net cash from operating activities", "operating cash flow"],
        "shares_outstanding": ["weighted average shares", "shares outstanding"]
    }
    df_flat = df.applymap(lambda x: str(x).lower() if pd.notnull(x) else "")
    for metric, labels in possible_labels.items():
        for label in labels:
            for idx, row in df_flat.iterrows():
                if any(label in cell for cell in row):
                    row_vals = [v for v in row if re.search(r"\d", v)]
                    if row_vals:
                        metrics[metric] = row_vals[-1]
                        break
            if metric in metrics:
                break
    return metrics

def extract_metrics_from_filing(url):
    try:
        tables = pd.read_html(url, flavor="bs4", header=0)
    except Exception:
        return None
    for df in tables[:3]:
        metrics = parse_metrics_from_table(df)
        if metrics:
            return metrics
    return {}

lookup = build_ticker_cik_map()

ticker = st.text_input("Enter Ticker Symbol (e.g., NVDA, AAPL, MSFT):", value="nvda")
cik = get_cik(ticker, lookup)

if not cik:
    st.error("Ticker not found. Please check spelling, use all lowercase (e.g., 'nvda'), and ensure it's a US-listed ticker.")
    st.info("Try typing the ticker in all lowercase. If still not found, see the first 10 tickers loaded above for reference.")
else:
    st.info(f"CIK for '{ticker.upper()}' is {cik}")
    filings = get_filing_metadata(cik, count=6)
    if not filings:
        st.warning("No 10-Q or 10-K filings found.")
    else:
        data = []
        st.subheader(f"Recent 10-Q and 10-K Filings for {ticker.upper()}")
        for f in filings:
            with st.spinner(f"Parsing {f['form']} from {f['date']}..."):
                metrics = extract_metrics_from_filing(f["url"])
            row = {
                "Form": f["form"],
                "Filing Date": f["date"],
                "Filing Link": f"[View Filing]({f['url']})",
                "Revenue": metrics.get("revenue", ""),
                "Net Income": metrics.get("net_income", ""),
                "EPS": metrics.get("eps", ""),
                "Operating Cash Flow": metrics.get("operating_cash_flow", ""),
                "Shares Outstanding": metrics.get("shares_outstanding", ""),
            }
            data.append(row)
        df = pd.DataFrame(data)
        st.write(df.to_markdown(index=False), unsafe_allow_html=True)
        st.download_button("Download CSV", df.to_csv(index=False), file_name=f"{ticker}_sec_filings.csv")
        st.caption("Metrics auto-extracted from first few tables in each SEC filing. If data missing, try viewing filing directly (link provided). For best results, use major US tickers.")
