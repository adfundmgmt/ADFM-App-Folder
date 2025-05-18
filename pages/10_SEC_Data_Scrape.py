import streamlit as st
import os
from sec_edgar_downloader import Downloader
import difflib
from bs4 import BeautifulSoup
import glob

# Helper to extract a section (naive: just search by title)
def extract_section(text, section_name):
    lower = text.lower()
    idx = lower.find(section_name.lower())
    if idx == -1:
        return "Section not found."
    end_idx = lower.find("item", idx + 10)
    return text[idx:end_idx] if end_idx > idx else text[idx:idx + 8000]

# Sidebar and UI
st.set_page_config(page_title="SEC Filing Change Detector", layout="wide")
st.title("SEC Filing Change Detector")

st.markdown("""
Compare the latest two 10-K/10-Q filings for any U.S. public company and instantly visualize **what changed** – new risk factors, edits to MD&A, subtle pivots in language.
""")

ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL)", value="AAPL")
filing_type = st.selectbox("Select Filing Type", ["10-K", "10-Q"])
section = st.selectbox("Compare Which Section?", ["Risk Factors", "MD&A", "Financial Statements"])

if st.button("Compare Latest Filings"):
    with st.spinner("Fetching filings..."):
        dl = Downloader("filings_data")
        dl.get(filing_type, ticker, amount=2)

        filings = sorted(
            glob.glob(f"filings_data/sec-edgar-filings/{ticker}/{filing_type}/*/full-submission.txt"),
            reverse=True
        )

        if len(filings) < 2:
            st.error("Not enough filings found.")
        else:
            # Parse as plain text for now (HTML/XBRL parsing can be added)
            with open(filings[0], "r", encoding="utf-8", errors="ignore") as f1:
                text1 = f1.read()
            with open(filings[1], "r", encoding="utf-8", errors="ignore") as f2:
                text2 = f2.read()

            # Extract section
            section_map = {
                "Risk Factors": "risk factors",
                "MD&A": "management’s discussion",
                "Financial Statements": "financial statements"
            }
            sec_name = section_map[section]

            part1 = extract_section(text1, sec_name)
            part2 = extract_section(text2, sec_name)

            # Show diffs
            st.subheader("Redline Diff – What Changed?")
            diff = difflib.HtmlDiff().make_table(
                part2.splitlines(),  # older first
                part1.splitlines(),  # newer second
                "Previous", "Latest", context=True, numlines=6
            )
            st.markdown(diff, unsafe_allow_html=True)

            # Show actionable summary
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

            st.download_button("Download Raw Diff", diff, file_name=f"{ticker}_{filing_type}_diff.html")

st.markdown("---")
st.markdown("**Takeaway:** Use this tool after every earnings or annual report to spot material changes faster than the market—be it new legal risks, altered outlook, or quietly added cautionary language. For best results, layer with LLM-powered NLP on the extracted diffs.")

