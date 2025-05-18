import streamlit as st
import requests
from bs4 import BeautifulSoup
import difflib
import re

# Robust Ticker-to-CIK mapping with caching
@st.cache_data(ttl=24*3600, show_spinner=False)
def build_ticker_cik_map():
    url = 'https://www.sec.gov/include/ticker.txt'
    resp = requests.get(url)
    lookup = {}
    for line in resp.text.splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            tkr, cik = parts
            lookup[tkr.lower()] = cik.zfill(10)  # Always pad to 10 digits for EDGAR
    return lookup

def get_cik(ticker, lookup):
    return lookup.get(ticker.lower().strip(), None)

def get_latest_filing_urls(cik, filing_type, count=2):
    base = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {'User-Agent': 'Mozilla/5.0'}
    data = requests.get(base, headers=headers).json()
    filings = data['filings']['recent']
    urls = []
    for idx, form in enumerate(filings['form']):
        if form == filing_type and len(urls) < count:
            acc_no = filings['accessionNumber'][idx].replace('-', '')
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/{filings['primaryDocument'][idx]}"
            urls.append(url)
    return urls

def extract_section(html, section_name):
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(separator='\n')
    matches = [m.start() for m in re.finditer(section_name.lower(), text.lower())]
    if not matches:
        return "Section not found."
    idx = matches[0]
    return text[idx:idx+6000]

st.set_page_config(page_title="SEC Filing Change Detector (Ticker to CIK)", layout="wide")
st.title("SEC Filing Change Detector")

st.markdown("""
Enter a ticker (e.g., `NVDA`, `AAPL`) and this tool will auto-convert it to a CIK number, fetch the latest two filings (10-K/10-Q), and highlight what changed.
""")

ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL)", value="NVDA")
filing_type = st.selectbox("Select Filing Type", ["10-K", "10-Q"])
section = st.selectbox("Compare Which Section?", ["Risk Factors", "MD&A", "Financial Statements"])

lookup = build_ticker_cik_map()
cik = get_cik(ticker, lookup)

if cik:
    st.success(f"CIK for '{ticker.upper()}' is {cik}")
else:
    st.error("Ticker not found. Please check your input (use official US tickers, e.g. 'nvda').")

if cik and st.button("Compare Latest Filings"):
    with st.spinner("Fetching and parsing filings..."):
        urls = get_latest_filing_urls(cik, filing_type)
        if len(urls) < 2:
            st.error("Not enough filings found.")
        else:
            html1 = requests.get(urls[0], headers={'User-Agent':'Mozilla/5.0'}).text
            html2 = requests.get(urls[1], headers={'User-Agent':'Mozilla/5.0'}).text

            sec_map = {
                "Risk Factors": "risk factors",
                "MD&A": "management’s discussion",
                "Financial Statements": "financial statements"
            }
            sec_name = sec_map[section]

            part1 = extract_section(html1, sec_name)
            part2 = extract_section(html2, sec_name)

            st.subheader("Redline Diff – What Changed?")
            diff = difflib.HtmlDiff().make_table(
                part2.splitlines(),
                part1.splitlines(),
                "Previous", "Latest", context=True, numlines=6
            )
            st.markdown(diff, unsafe_allow_html=True)

            st.subheader("Actionable Summary (Experimental)")
            added = [line for line in difflib.unified_diff(
                part2.splitlines(), part1.splitlines(), n=0
            ) if line.startswith("+ ") and len(line.strip()) > 2]
            removed = [line for line in difflib.unified_diff(
                part2.splitlines(), part1.splitlines(), n=0
            ) if line.startswith("- ") and len(line.strip()) > 2]
            st.markdown("**Newly Added Text:**")
            st.write(added[:10] if added else "None")
            st.markdown("**Removed Text:**")
            st.write(removed[:10] if removed else "None")

st.caption("Tip: Use the official ticker for US-listed equities. If still not found, search [SEC’s company lookup](https://www.sec.gov/edgar/searchedgar/companysearch.html).")
