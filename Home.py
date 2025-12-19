# Home.py
import streamlit as st

from ui_helpers import render_page_header

st.set_page_config(page_title="AD Fund Management Tools", layout="wide")

render_page_header(
    "AD Fund Management LP - Analytics Suite",
    subtitle="Internal dashboards for faster, more informed decisions.",
    description="Use the sidebar to launch any tool. Each page follows the same streamlined layout so you can focus on the outputs without re-learning controls.",
)

st.markdown(
    """
### Tools Available

- **Monthly Seasonality Explorer**  
  Looks at median monthly returns and hit rates for any stock, index, or commodity. Pulls from Yahoo Finance and FRED for deeper historical data. Helps us stay aware of recurring calendar trends and seasonality biases.

- **Market Memory Explorer**  
  Compares how this year is tracking versus past years with similar return paths. Useful for putting the current tape in historical context and building intuition around where we might go next.

- **Technical Chart Explorer**  
  Visualize key technical indicators for any publicly traded ticker that is on Yahoo Finance (equities, commodities, credit).

- **Sector Breadth and Rotation**  
  This dashboard monitors the relative performance and rotation of S&P 500 sectors to help identify leadership trends and market breadth dynamics.

- **Breakout Scanner**  
  Screen for stocks breaking out to 20D, 50D, 100D, or 200D highs and view multi-timeframe RSI.

- **Ratio Charts**  
  Tracks the relative performance of S&P cyclical and defensive sector ETFs (equal-weighted) to visualize risk-on/risk-off regime shifts in US equities.

- **Cross Asset Volatility Surface**  
  Track cross asset implied and realized volatility in one place and map it to a simple regime read that can steer gross and hedging.

- **ETF Flows Dashboard**  
  A dashboard of thematic, and global ETF flows - see where money is moving among major macro and innovation trades.
  
- **US Inflation Dashboard**  
  Dissects the latest CPI print with YoY and MoM breakdowns across headline, and core categories.

- **Real Yield Dashboard**  
  Tracks nominal and real 10-year Treasury yields alongside the 63-day momentum of real yields (inverted). It’s a clean way to monitor macro regime changes—whether the environment is tilting toward tightening or easing.

- **Liquidity Tracker**  
  Tracks Net Liquidity and relate it to policy and equities. Definition: Net Liquidity = WALCL - RRP - TGA. Units are billions. Sustained rises often coincide with risk-on conditions.

- **Market Stress Composite**  
  A single 0 to 100 stress score blending volatility, credit, curve, funding, and equity drawdown. Higher = more stress.

- **Unusual Options Flow Tracker**  
  Tracks unusual options flow tracker.

- **Factor Momentum Leadership**  
  Tracks style and macro factor leadership using ETF pairs. Each factor is a relative strength ratio scored on short and long momentum, trend structure, and inflection.
  
- **Weekly Cross-Asset Compass**  
  A weekly cross-asset readout that maps regime shifts, correlations, and volatility changes to clear actions, invalidations, and trade expressions in one place.

- **VIX Spike Deep Dive**  
  Evaluates short-horizon SPX behavior after a VIX spike. Support sizing and timing of tactical bounce or fade setups.

- **ADFM Public Equities Baskets**  
  A consolidated dashboard that tracks daily, equal-weight performance and technicals across all ADFM baskets, benchmarked to SPY, with aligned business-day panels and category-level drilldowns.

These tools aren’t overly complex, but they’ve become a key part of how we stay organized, especially in fast moving markets. Always a work in progress, open to feedback or feature ideas.
"""
)
