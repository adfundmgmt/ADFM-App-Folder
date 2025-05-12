"""
Sector & Single-Ticker Momentum — 3-D Surface Explorer
Author: ChatGPT / Arya Deniz

Streamlit app that renders a rotatable **3-D Plotly surface** to visualise
momentum / relative performance either for the 11 GICS sector ETFs **or** for a
user-supplied ticker at the 12-, 24-, and 36-month look-back horizons.

Axes mapping
────────────
• **x-axis**  : Category (Sector ticker *or* look-back window 12/24/36 months)  
• **y-axis**  : Observation date (month-end)                             
• **z-axis**  : Metric value (trailing total return or relative return)

Requirements (add to requirements.txt)
──────────────────────────────────────
  streamlit>=1.30
  yfinance>=0.2
  plotly>=5.20.0
  pandas>=2.2
"""

import datetime as dt
from functools import lru_cache
from typing import Literal, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ── Constants ────────────────────────────────────────────────────────────────
SECTORS = {
    "XLC": "Comm Svcs",
    "XLY": "Cons Disc",
    "XLP": "Cons Stap",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLK": "Tech",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
}
LOOKBACK_WINDOWS: List[int] = [12, 24, 36]

st.set_page_config(page_title="Momentum 3-D Surface", layout="wide")
st.title("📈 Sector & Ticker Momentum — 3-D Surface")

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    view_mode: Literal["Sector View", "Ticker View"] = st.radio(
        "View mode", ["Sector View", "Ticker View"], index=0
    )

    years_back = st.slider("Years of history", 3, 25, value=10)
    metric: Literal["Momentum", "Relative"] = st.radio(
        "Metric", ["Momentum", "Relative"], index=0, help="Relative = vs SPY or equal-weight basket"
    )

    if view_mode == "Sector View":
        momentum_window = st.slider("Momentum window (months)", 3, 24, value=12)
    else:  # Ticker View
        user_ticker = st.text_input("Ticker symbol", value="AAPL").upper().strip()

# ── Helpers ──────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_prices(tickers: tuple) -> pd.DataFrame:
    """Fetch adjusted close prices for given tickers (max history)."""
    df = yf.download(list(tickers), period="max", auto_adjust=True, progress=False)[
        "Close"
    ]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    df = df.dropna(how="all")
    return df

# ── Data prep based on mode ─────────────────────────────────────────────────
dt_end = pd.Timestamp.today().normalize()
dt_start = dt_end - pd.DateOffset(years=years_back)

if view_mode == "Sector View":
    sector_prices = load_prices(tuple(SECTORS.keys()))
    sector_prices = sector_prices.loc[dt_start:dt_end]

    # Month-end prices
    mprices = sector_prices.resample("M").last()

    # Momentum (rolling window selected by user)
    mom = mprices / mprices.shift(momentum_window) - 1
    mom = mom.dropna(how="all")

    # Equal-weight basket cumulative return
    basket_cum = (1 + mprices.pct_change()).cumprod()
    basket_eq = basket_cum.mean(axis=1)
    rel = (basket_cum.sub(basket_eq, axis=0) / basket_eq).dropna(how="all")

    z_df = mom if metric == "Momentum" else rel.loc[mom.index]

    # Axes arrays
    cats = list(SECTORS.keys())
    x_vals = np.arange(len(cats))
    z = z_df[cats].values

else:  # Ticker View
    tickers = (user_ticker, "SPY") if metric == "Relative" else (user_ticker,)
    prices = load_prices(tickers)
    prices = prices.loc[dt_start:dt_end]
    mprices = prices.resample("M").last()

    # Compute trailing window returns
    def trailing(df: pd.DataFrame, win: int) -> pd.Series:
        return df[user_ticker] / df[user_ticker].shift(win) - 1

    mom_cols = {
        f"{win}M": trailing(mprices, win)
        for win in LOOKBACK_WINDOWS
    }
    mom_df = pd.DataFrame(mom_cols).dropna(how="all")

    if metric == "Relative":
        rel_cols = {}
        spy_ret = {win: mprices["SPY"] / mprices["SPY"].shift(win) - 1 for win in LOOKBACK_WINDOWS}
        for win in LOOKBACK_WINDOWS:
            rel_cols[f"{win}M"] = mom_cols[f"{win}M"] - spy_ret[win]
        z_df = pd.DataFrame(rel_cols).dropna(how="all")
    else:
        z_df = mom_df

    cats = [f"{w}M" for w in LOOKBACK_WINDOWS]
    x_vals = np.arange(len(cats))
    z = z_df[cats].values

# ── Common arrays for y-axis (dates) ─────────────────────────────────────────
dates = z_df.index
x_mat = np.tile(x_vals, (len(dates), 1))
y_num = dates.astype(np.int64) // 10 ** 9

# ── Build 3-D surface ────────────────────────────────────────────────────────
fig = go.Figure(
    data=[
        go.Surface(
            z=z,
            x=x_mat,
            y=y_num[:, None] * np.ones(len(cats)),
            colorscale="Viridis",
            showscale=True,
        )
    ]
)

# Axis labels and ticks
if view_mode == "Sector View":
    x_ticktext = [SECTORS[t] for t in cats]
else:
    x_ticktext = cats

fig.update_layout(
    scene=dict(
        xaxis=dict(title="Category", tickmode="array", tickvals=x_vals, ticktext=x_ticktext),
        yaxis=dict(
            title="Date",
            tickmode="array",
            tickvals=y_num[:: max(1, len(y_num) // 10)],
            ticktext=[d.strftime("%Y-%m-%d") for d in dates][:: max(1, len(dates) // 10)],
        ),
        zaxis_title="Return (%)",
    ),
    margin=dict(l=0, r=0, b=0, t=40),
)

st.plotly_chart(fig, use_container_width=True)

# ── Footnote / caption ───────────────────────────────────────────────────────
if view_mode == "Sector View":
    st.caption(
        f"Sector View — Metric: {metric} (momentum window = {momentum_window} months). Rotate the surface to detect leadership shifts and mean-reversion set-ups."
    )
else:
    st.caption(
        f"Ticker View — {user_ticker} | Metric: {metric} (windows: 12/24/36 months). Use rotation to compare return horizons and spot breakout regimes."
    )
