import sys
import os
import importlib
import textwrap
import streamlit as st

# ---------- Page setup ----------
# "centered" is friendlier on phones; avoids horizontal scroll traps.
st.set_page_config(page_title="AD Fund Management Tools", layout="centered", initial_sidebar_state="collapsed")

# Always show visible checkpoints first so the page never looks stuck
st.title("AD Fund Management LP - Analytics Suite")
st.write("Home page reached. If you see this line, Streamlit executed the entrypoint successfully.")

# ---------- Optional lightweight diagnostics ----------
# Toggle lets you verify runtime quickly on phone without opening logs
with st.expander("Diagnostics (optional)"):
    st.write("These checks help verify the app is healthy even if Streamlit Cloud logs are unavailable.")
    colA, colB = st.columns(2)
    with colA:
        st.write("Python version:", sys.version.split()[0])
        st.write("Working directory:", os.getcwd())
    with colB:
        # Quick import probe for common deps. Does not import heavy pages or run network I/O.
        required = [
            "pandas", "numpy", "yfinance", "matplotlib", "mplfinance", "plotly",
            "pandas_datareader", "requests", "packaging", "pytrends",
            "bs4", "lxml", "html5lib", "openpyxl", "sec_edgar_downloader", "cot_reports"
        ]
        ok, missing, failed = [], [], []
        for mod in required:
            try:
                importlib.import_module(mod)
                ok.append(mod)
            except ModuleNotFoundError:
                missing.append(mod)
            except Exception as e:
                failed.append(f"{mod}: {e}")
        st.write("Imports OK:", ", ".join(ok) if ok else "None")
        if missing:
            st.error("Missing modules: " + ", ".join(missing))
        if failed:
            st.error("Import errors: " + "; ".join(failed))
    st.caption("If a module is missing or erroring, update requirements.txt or pin Python with runtime.txt.")

# ---------- Mobile-friendly notes ----------
with st.expander("Mobile tips"):
    st.markdown(
        "On phones, the sidebar is hidden behind the top-left menu button. "
        "Use it to navigate between pages. If text looks cramped, rotate the phone to landscape."
    )

# ---------- Main content ----------
st.markdown("""
Welcome to the internal analytics dashboard for **AD Fund Management LP**.

These are proprietary tools ADFM has built in-house to support daily decision-making. The goal is simple: organize information, track shifts, and move faster with more context.

Use the sidebar to launch each tool. On a phone, open the top-left menu to access the sidebar. The home page is intentionally light so it renders quickly even on constrained networks or cold starts.

---

### Monthly Seasonality Explorer

Looks at median monthly returns and hit rates for any stock, index, or commodity. Pulls from Yahoo Finance and FRED where available to extend the history so seasonality is not just a short-sample illusion. The view helps maintain awareness of recurring calendar tendencies and when they tend to fail. It is not a signal in isolation. It informs risk concentration and entry timing around known seasonal windows.

---

### Market Memory Explorer

Compares how the current year is tracking versus historical years with similar cumulative return paths. The point is context, not prediction. It frames where markets have tended to accelerate or fade given similar path dependencies. You can tighten or loosen similarity thresholds and see how often analogs diverged after a match. Useful for calibrating conviction and sizing rather than calling the next week.

---

### Technical Chart Explorer

Visualizes key indicators for any ticker available via Yahoo Finance. Daily and weekly timeframes, moving averages, RSI, MACD, Bollinger bands, and volume overlays are included to keep the focus on structure and momentum. It is designed for quick checks across a list of names without having to leave the dashboard.

---

### Sector Breadth Dashboard

Monitors the relative performance and rotation of S&P 500 sectors with both cap-weighted and equal-weighted lenses so leadership is not mistaken for breadth. Breadth deterioration often precedes index-level weakness. Likewise, broadening participation after pullbacks tends to precede recoveries. This view tracks those transitions.

---

### Breakout Scanner

Surfaces names breaking to 20 day, 50 day, 100 day, or 200 day highs and overlays multi timeframe RSI to filter extended names. The point is to focus attention on strength that is expanding rather than one day spikes. Breakouts are ranked by persistence and follow-through statistics so the list is not just a daily noise screen.

---

### Ratio Charts

Tracks the relative performance of cyclical versus defensive sector ETFs and other risk appetite pairs. The objective is to keep a live handle on regime. When cyclical to defensive ratios trend higher alongside improving breadth and credit, risk tends to be rewarded. When those ratios roll over while credit widens, capital preservation becomes the priority.

---

### Options Chain Viewer

Fetches call and put chains for any ticker and expiry available from Yahoo Finance. Useful for quick reference on open interest distribution, approximate skew, and where dealers may be concentrated. It is not an execution tool. It is a context tool for positioning and gamma sensitivity checks before sizing.

---

### ETF Flows Dashboard

Tracks flows across major macro and thematic ETFs to see where capital is moving in or out. The goal is to catch early rotation, not to chase flows mechanically. Sustained inflows combined with improving relative strength tend to be more durable than one day surges around headlines.

---

### US Inflation Dashboard

Breaks down the latest CPI print with year over year and month over month views across headline and core categories. It emphasizes the components that drive market reaction and the pieces that carry forward momentum. The layout is designed to make it obvious where inflation pressures are broadening or narrowing.

---

### Real Yield Dashboard

Displays nominal and real 10 year Treasury yields along with the 63 day momentum of real yields, inverted for ease of comparison to risk assets. The aim is to keep a disciplined eye on the macro impulse that often leads equity factor performance. When real yield momentum eases, cyclicals and duration sensitive assets tend to get relief.

---

### Liquidity Tracker

Tracks Net Liquidity defined as WALCL minus RRP minus TGA, in billions. The idea is to keep one simple composite that links policy and liquidity with risk appetite. It is not a trading signal on its own, but sustained rises often coincide with environments where buying weakness is rewarded, and sustained declines often coincide with environments where rallies fade.

---

### Market Stress Composite

Blends volatility, credit, yield curve shape, funding indicators, and equity drawdown into a single 0 to 100 stress score. Higher means more stress. The point is to condense cross market risk so there is a shared reference for position sizing and exposure control across the book. It is designed to spike early enough to cut gross when needed and ease early enough to re risk responsibly.

---

These tools are iterative. They evolve with markets and with our process. If something feels slow on a phone, open the Diagnostics at the top, verify imports, and reload once. If a specific page fails, the home page will still render so you can navigate elsewhere.
""")

# ---------- Footer checkpoint ----------
st.info("Checkpoint: content rendered. If you reached this line on your phone, the app is functioning and the sidebar should be accessible from the top-left menu.")
