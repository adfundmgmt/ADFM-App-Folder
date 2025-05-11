# Home.py
import streamlit as st

st.set_page_config(page_title="AD Fund Management Tools", layout="wide")

st.title("🔍 AD Fund Management LP — Analytics Suite")

st.markdown("""
Welcome to the internal analytics platform of **AD Fund Management LP**.

Use the sidebar to explore:

- 📈 **Monthly Seasonality Explorer**  
  Visualizes median monthly returns and hit rates for any stock, index, or commodity using Yahoo and FRED data.

- 📘 **YTD Analog Year Correlation Explorer**  
  Compares current year-to-date performance with past years to uncover historically similar market paths.

---

These tools are designed to enhance our tactical positioning and historical awareness, supporting faster, data-driven decision making.
""")

