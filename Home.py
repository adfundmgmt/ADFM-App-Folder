# Home.py
import streamlit as st
st.set_page_config(page_title="AD Fund Management Tools", layout="wide")

st.title("AD Fund Management LP — Analytics Suite")

st.markdown("""
Welcome to the internal analytics dashboard for **AD Fund Management LP**.

These are proprietary tools ADFM has built in-house to support daily decision-making. The goal is simple: make it easier to spot patterns, track shifts, and move faster with more context.

---

### Tools Available (use the sidebar to launch):

- **Monthly Seasonality Explorer**  
  Looks at median monthly returns and hit rates for any stock, index, or commodity. Pulls from Yahoo Finance and FRED for deeper historical data. Helps us stay aware of recurring calendar trends and seasonality biases.

- **Market Memory Explorer**  
  Compares how this year is tracking versus past years with similar return paths. Useful for putting the current tape in historical context and building intuition around where we might go next.

- **Real Yield Dashboard**  
  Tracks nominal and real 10-year Treasury yields alongside the 63-day momentum of real yields (inverted). It’s a clean way to monitor macro regime changes—whether the environment is tilting toward tightening or easing.

- **Ticker Correlation Dashboard**  
  Offers an at-a-glance view of historical and rolling correlations to streamline relative-value analysis, diversification checks, and position sizing.

- **US Inflation Dashboard**  
  Dissects the latest CPI print with YoY and MoM breakdowns across headline, and core categories.

- **Technical Chart Explorer**  
  Visualize key technical indicators for any publicly traded ticker that is on Yahoo Finance (equities, commodities, credit).

- **Options Chain Viewer**  
  Fetch real-time call & put chains for any ticker and expiry via Yahoo Finance.

- **Breakout Scanner**  
  Screen for stocks breaking out to 20D, 50D, 100D, or 200D highs and view multi-timeframe RSI.

---

These tools aren’t overly complex, but they’ve become a key part of how we stay organized, especially in fast-moving markets. Always a work in progress—open to feedback or feature ideas.
""")
