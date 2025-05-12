# Home.py
import streamlit as st
st.set_page_config(page_title="AD Fund Management Tools", layout="wide")

st.title("AD Fund Management LP — Analytics Suite")

st.markdown("""
Welcome to the internal analytics and visualization platform of **AD Fund Management LP** — a tactical decision support system built to translate market noise into signal.

This suite of proprietary tools empowers the investment team to track historical analogs, spot inflection points in macro regimes, and uncover recurring patterns across asset classes. Designed for speed, clarity, and insight, these modules help reinforce high-conviction positioning in real time.

---

### Available Dashboards

- **Monthly Seasonality Explorer**  
  Identify consistent return patterns by visualizing **median monthly performance** and **hit rates** across equities, indices, and commodities. Pulls data from Yahoo Finance, with deep historical coverage via FRED for major benchmarks like the S&P 500, Nasdaq, and Dow. Helps isolate seasonal strength/weakness and guide timing decisions.

- **YTD Analog Year Correlation Explorer**  
  Compares the **year-to-date trajectory** of a chosen ticker to historical paths with the **highest correlation**, enabling rapid discovery of similar past environments. Useful for scenario planning, expectation anchoring, and understanding the behavioral structure of the current tape.

- **10-Year Yield Dashboard**  
  Plots **nominal** and **real 10-year Treasury yields** alongside the **inverted 63-day momentum of real yields** to flag potential regime shifts. Built to highlight the transition points between tightening and easing cycles—this dashboard serves as a leading macro overlay for both equity and rates positioning.

---

These tools are in continuous development and tailored to the evolving strategy of ADFM. They serve as a foundation for research, narrative building, and timely risk/reward calibration. Feedback, feature ideas, and bug reports are always welcome.
""")
