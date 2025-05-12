"""
Sector & Single‑Ticker Momentum — 3‑D Surface Explorer (v3)
Author: ChatGPT / Arya Deniz

Clarified UI + richer annotations
────────────────────────────────
✓ Sidebar now includes a **"How to read this"** guide for quick context.  
✓ Hover shows Date | Category | Return% (no extra box).  
✓ Camera preset gives a cleaner oblique angle; layout widened.  
✓ Color‑bar labelled clearly as "Return (%)".

Unchanged core: Sector & Ticker modes with Momentum / Relative metrics.
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
    "XLC": "Comm Svcs",
    "XLY": "Cons Disc",
    "XLP": "Cons Stap",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLK": "Tech",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
}
LOOKBACK_WINDOWS: List[int] = [12, 24, 36]

st.set_page_config(page_title="Momentum 3‑D Surface", layout="wide")
st.title("📈 Sector & Ticker Momentum — 3‑D Surface (v3)")

# ═════════ Sidebar ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Controls")
    view_mode: Literal["Sector View", "Ticker View"] = st.radio(
        "View mode", ["Sector View", "Ticker View"], index=0
    )
    years_back = st.slider("Years of history", 3, 25, value=10)
    metric: Literal["Momentum", "Relative"] = st.radio(
        "Metric", ["Momentum", "Relative"], index=0,
        help="Relative = vs SPY (Ticker) or equal‑weight basket (Sector)"
    )
    if view_mode == "Sector View":
        momentum_window = st.slider("Momentum window (months)", 3, 24, value=12)
    else:
        user_ticker = st.text_input("Ticker symbol", value="AAPL").upper().strip()

    st.markdown("""
### How to read this surface
* **X‑axis** → Category (Sector symbol *or* 12│24│36‑month window).
* **Y‑axis** → Month‑end date.
* **Z‑axis** → Return (%).

**Momentum** = trailing total return for the selected window.  
**Relative** = excess return vs SPY (Ticker mode) or vs equal‑weight sector basket (Sector mode).

Hover the surface to see precise values; rotate to spot leadership shifts and mean‑reversion.
""")

# ── Helpers ──────────────────────────────────────────────────────────────────
@lru_cache(maxsize=8)
def load_prices(tickers: tuple) -> pd.DataFrame:
    df = yf.download(list(tickers), period="max", auto_adjust=True, progress=False)[
        "Close"
    ]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    return df.dropna(how="all")

# Date range
today = pd.Timestamp.today().normalize()
start = today - pd.DateOffset(years=years_back)

# ═════════ Data prep ═════════════════════════════════════════════════════════
if view_mode == "Sector View":
    raw_px = load_prices(tuple(SECTORS.keys())).loc[start:today]
    mpx = raw_px.resample("M").last()
    mom = mpx / mpx.shift(momentum_window) - 1
    mom = mom.dropna(how="all")

    basket = (1 + mpx.pct_change()).cumprod().mean(axis=1)
    rel = ((1 + mpx.pct_change()).cumprod().div(basket, axis=0) - 1).dropna(how="all")

    z_df = mom if metric == "Momentum" else rel.reindex_like(mom)
    z_df = z_df.dropna(how="all")

    cats = list(SECTORS.keys())
    x_vals = np.arange(len(cats))
    x_labels = [SECTORS[t] for t in cats]

else:  # Ticker View
    tickers = (user_ticker, "SPY") if metric == "Relative" else (user_ticker,)
    px = load_prices(tickers).loc[start:today]
    mpx = px.resample("M").last()

    def trailing(series: pd.Series, win: int) -> pd.Series:
        return series / series.shift(win) - 1

    mom_cols = {f"{w}M": trailing(mpx[user_ticker], w) for w in LOOKBACK_WINDOWS}
    mom_df = pd.DataFrame(mom_cols)

    if metric == "Momentum":
        z_df = mom_df.dropna(how="all")
    else:
        spy_cols = {w: trailing(mpx["SPY"], w) for w in LOOKBACK_WINDOWS}
        rel_df = pd.concat({f"{w}M": mom_cols[f"{w}M"] - spy_cols[w] for w in LOOKBACK_WINDOWS}, axis=1)
        z_df = rel_df.dropna(how="all")

    cats = [f"{w}M" for w in LOOKBACK_WINDOWS]
    x_vals = np.arange(len(cats))
    x_labels = cats

# ── Build matrices for Plotly ────────────────────────────────────────────────
if z_df.empty:
    st.error("Not enough data after filtering. Try shorter history or different ticker.")
    st.stop()

dates = z_df.index
z = z_df[cats].values  # rectangular
x_mat = np.tile(x_vals, (len(dates), 1))

y_num = (dates.astype("int64") // 10 ** 9).to_numpy()
y_mat = np.tile(y_num[:, None], (1, len(cats)))

# ═════════ Plotly Surface ════════════════════════════════════════════════════
fig = go.Figure(
    go.Surface(
        z=z,
        x=x_mat,
        y=y_mat,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="Return (%)"),
        hovertemplate="Date: %{customdata[0]}<br>Category: %{customdata[1]}<br>Return: %{z:.2%}<extra></extra>",
        customdata=np.dstack([
            np.tile(dates.strftime("%Y-%m-%d").to_numpy()[:, None], (1, len(cats))),
            np.tile(np.array(x_labels)[None, :], (len(dates), 1)),
        ]),
    )
)

fig.update_layout(
    scene=dict(
        camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
        xaxis=dict(title="Category", tickmode="array", tickvals=x_vals, ticktext=x_labels),
        yaxis=dict(
            title="Date",
            tickmode="array",
            tickvals=y_num[:: max(1, len(y_num) // 10)],
            ticktext=[d.strftime("%Y‑%m‑%d") for d in dates][:: max(1, len(dates) // 10)],
        ),
        zaxis=dict(title="Return (%)"),
        aspectratio=dict(x=1.2, y=2.0, z=0.7),
    ),
    margin=dict(l=0, r=0, b=0, t=40),
)

st.plotly_chart(fig, use_container_width=True)

# ── Caption ──────────────────────────────────────────────────────────────────
if view_mode == "Sector View":
    st.caption(f"Sector View │ {metric} │ {momentum_window}-month window")
else:
    st.caption(f"Ticker View │ {user_ticker} │ {metric} │ Horizons: 12/24/36‑month")
