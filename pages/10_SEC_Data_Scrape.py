import streamlit as st
import requests
from bs4 import BeautifulSoup
import difflib
import re

# Helper: get company CIK
def get_cik(ticker):
    # Use SEC's full list
    url = 'https://www.sec.gov/include/ticker.txt'
    resp = requests.get(url)
    lookup = {}
    for line in resp.text.splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            tkr, cik = parts
            lookup[tkr.lower()] = cik.lstrip('0')  # Remove leading zeros for the main code
    return lookup.get(ticker.lower(), None)


# Helper: find latest two filings of specified type
def get_latest_filing_urls(cik, filing_type, count=2):
    base = f"https://data.sec.gov/submissions/CIK{int(cik):010}.json"
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

# Helper: extract section (very naive)
def extract_section(html, section_name):
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(separator='\n')
    matches = [m.start() for m in re.finditer(section_name.lower(), text.lower())]
    if not matches:
        return "Section not found."
    idx = matches[0]
    # crude window for diff
    return text[idx:idx+6000]

st.set_page_config(page_title="SEC Filing Change Detector", layout="wide")
st.title("SEC Filing Change Detector (No Dependencies)")

st.markdown("""
Compare the latest two 10-K/10-Q filings for any U.S. public company and instantly visualize **what changed** – new risk factors, edits to MD&A, pivots in language.
""")

ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL)", value="AAPL")
filing_type = st.selectbox("Select Filing Type", ["10-K", "10-Q"])
section = st.selectbox("Compare Which Section?", ["Risk Factors", "MD&A", "Financial Statements"])

if st.button("Compare Latest Filings"):
    with st.spinner("Fetching and parsing filings..."):
        cik = get_cik(ticker)
        if not cik:
            st.error("Ticker not found.")
        else:
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
