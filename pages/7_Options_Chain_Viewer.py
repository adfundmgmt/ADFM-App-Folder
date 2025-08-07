import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import math
from datetime import datetime, date

import matplotlib.pyplot as plt
plt.style.use("default")

# ---------------- Page ----------------
st.set_page_config(page_title="Options Chain Viewer", layout="wide")
st.title("Options Chain Viewer")

with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
    - Interactive options chain with filters that matter
    - Volume by strike with ATM highlight
    - Summary metrics: put call ratio, expected move, average delta
    - Delta distribution to visualize skew
    - Optional tables and CSV export
    """)

# ---------------- Helpers ----------------
@st.cache_data(ttl=600)
def get_spot(tkr: str) -> float:
    h = yf.Ticker(tkr).history(period="1d")
    return float(h["Close"].iloc[-1]) if not h.empty else float("nan")

@st.cache_data(ttl=900)
def get_expiries(tkr: str):
    try:
        return yf.Ticker(tkr).options or []
    except Exception:
        return []

@st.cache_data(ttl=300)
def get_chain(tkr: str, exp: str) -> pd.DataFrame:
    """
    Returns a tidy chain with calls and puts combined, including a 'mid' column.
    """
    tk = yf.Ticker(tkr)
    ch = tk.option_chain(exp)
    calls = ch.calls.copy()
    puts  = ch.puts.copy()
    for df in (calls, puts):
        for col in ["bid","ask","lastPrice","volume","openInterest","impliedVolatility","strike"]:
            if col not in df.columns:
                df[col] = np.nan
        df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2.0
        df["mid"] = df["mid"].replace(0, np.nan)
    calls["type"] = "Call"
    puts["type"]  = "Put"
    return pd.concat([calls, puts], ignore_index=True)

def busdays_to_expiry(exp: str) -> int:
    today = date.today()
    ed = pd.to_datetime(exp).date()
    return int(np.busday_count(today, ed))

def bs_call_delta_from_d1(d1: np.ndarray) -> np.ndarray:
    # standard normal CDF via erf
    return 0.5 * (1.0 + np.erf(d1 / np.sqrt(2.0)))

# ---------------- Sidebar form with submit ----------------
with st.sidebar.form("controls", clear_on_submit=False):
    ticker = st.text_input("Ticker", "AAPL").strip().upper()

    # Load expiries live so the selectbox is always valid
    expiries = get_expiries(ticker) if ticker else []
    if not expiries:
        expiry = None
        st.caption("No expiries available")
    else:
        # default to the nearest expiry
        expiry = st.selectbox("Expiry", expiries, index=0)

    st.markdown("---")
    use_mid = st.selectbox("Price source", ["Mid (bid ask)", "Last"], index=0) == "Mid (bid ask)"
    hide_zero = st.checkbox("Hide zero OI Vol", value=True)
    min_oi = st.number_input("Min open interest", value=0, step=10)
    min_vol = st.number_input("Min volume", value=0, step=10)
    mny_win = st.slider("Moneyness window around spot percent", 5, 50, 25)

    st.markdown("---")
    r_pct = st.number_input("Risk free rate percent", value=4.5, step=0.25)
    q_pct = st.number_input("Dividend yield percent", value=0.0, step=0.25)

    submitted = st.form_submit_button("Run")

# Persist last valid params so charts remain after reruns
if submitted and ticker and expiry:
    st.session_state["_params"] = {
        "ticker": ticker,
        "expiry": expiry,
        "use_mid": use_mid,
        "hide_zero": hide_zero,
        "min_oi": int(min_oi),
        "min_vol": int(min_vol),
        "mny_win": int(mny_win),
        "r": float(r_pct) / 100.0,
        "q": float(q_pct) / 100.0,
    }

params = st.session_state.get("_params")
if not params:
    st.info("Set your inputs in the sidebar and click Run.")
    st.stop()

ticker = params["ticker"]
expiry = params["expiry"]
use_mid = params["use_mid"]
hide_zero = params["hide_zero"]
min_oi = params["min_oi"]
min_vol = params["min_vol"]
mny_win = params["mny_win"]
r = params["r"]
q = params["q"]

# ---------------- Data fetch and checks ----------------
spot = get_spot(ticker)
if not np.isfinite(spot):
    st.error("Could not retrieve spot. Try again later.")
    st.stop()

if not expiry:
    st.error("No valid expiry for this ticker.")
    st.stop()

try:
    chain = get_chain(ticker, expiry)
except Exception:
    st.error("Failed to load option chain. Try another expiry.")
    st.stop()

if chain.empty:
    st.error("Empty chain returned by data source.")
    st.stop()

# ---------------- Filters and enrich ----------------
# Hide zero OI Vol
if hide_zero:
    chain = chain[(chain["openInterest"] > 0) | (chain["volume"] > 0)]

# Min OI and Vol
if min_oi > 0:
    chain = chain[chain["openInterest"] >= min_oi]
if min_vol > 0:
    chain = chain[chain["volume"] >= min_vol]

# Moneyness window
low = spot * (1 - mny_win / 100.0)
high = spot * (1 + mny_win / 100.0)
chain = chain[(chain["strike"] >= low) & (chain["strike"] <= high)]

if chain.empty:
    st.warning("No contracts left after filtering. Adjust filters and click Run.")
    st.stop()

# Time to expiry in years
bdays = busdays_to_expiry(expiry)
T = max(bdays, 0) / 252.0

# Choose price source
px_src = chain["mid"] if use_mid else chain["lastPrice"]
chain["price_src"] = px_src
chain.loc[chain["price_src"].isna(), "price_src"] = chain["mid"]

# Compute deltas using Black Scholes with chain IV
sigma = chain["impliedVolatility"].astype(float).to_numpy(copy=False)
K = chain["strike"].astype(float).to_numpy(copy=False)
S = float(spot)
T_arr = np.full_like(K, T, dtype=float)

valid = (sigma > 0) & (T_arr > 0)
d1 = np.zeros_like(K, dtype=float)
# risk neutral with dividends: r - q
d1[valid] = (np.log(S / K[valid]) + (r - q + 0.5 * sigma[valid] ** 2) * T_arr[valid]) / (sigma[valid] * np.sqrt(T_arr[valid]))

call_delta = bs_call_delta_from_d1(d1)
# map to calls puts
is_call = (chain["type"] == "Call").to_numpy()
delta = np.where(is_call, call_delta, call_delta - 1.0)
delta[~valid] = np.nan
chain["delta"] = delta

# ATM identification
atm_idx = (chain["strike"] - spot).abs().idxmin()
atm_strike = float(chain.loc[atm_idx, "strike"])
atm_iv = float(chain.loc[atm_idx, "impliedVolatility"]) if pd.notna(chain.loc[atm_idx, "impliedVolatility"]) else float("nan")
expected_move = S * atm_iv * math.sqrt(T) if np.isfinite(atm_iv) and T > 0 else float("nan")

# Summary metrics
total_call_vol = int(chain.query("type == 'Call'")["volume"].sum())
total_put_vol  = int(chain.query("type == 'Put'")["volume"].sum())
put_call_ratio = (total_put_vol / total_call_vol) if total_call_vol > 0 else float("nan")
avg_delta      = float(chain["delta"].mean())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Spot", f"{spot:,.2f}")
c2.metric("Expiry business days", f"{bdays}")
c3.metric("ATM strike", f"{atm_strike:,.2f}")
c4.metric("ATM IV", f"{atm_iv:.2%}" if np.isfinite(atm_iv) else "n a")
c5.metric("Expected move 1 sigma", f"{expected_move:,.2f}" if np.isfinite(expected_move) else "n a")

st.markdown("---")

st.subheader("Summary")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Total call volume", f"{total_call_vol:,}")
s2.metric("Total put volume", f"{total_put_vol:,}")
s3.metric("Put call ratio", f"{put_call_ratio:.2f}" if np.isfinite(put_call_ratio) else "n a")
s4.metric("Average delta", f"{avg_delta:.2f}" if np.isfinite(avg_delta) else "n a")

st.caption(f"Spot {spot:,.2f}   ATM strike {atm_strike:,.2f}")

st.markdown("---")

# ---------------- Volume by strike with ATM highlight ----------------
st.subheader("Volume by strike")

chart_data = chain[["type","strike","volume","openInterest","delta"]].copy()
chart_data["highlight"] = np.where(chart_data["strike"] == atm_strike, "ATM", "Other")

vol_chart = (
    alt.Chart(chart_data)
      .mark_bar()
      .encode(
          x=alt.X("strike:Q", title="Strike"),
          y=alt.Y("volume:Q", title="Volume"),
          color=alt.condition(
              alt.datum.highlight == "ATM",
              alt.value("#FFD600"),
              alt.Color("type:N",
                        scale=alt.Scale(domain=["Call","Put"], range=["#1a9641","#d7191c"]),
                        legend=alt.Legend(title="Type"))
          ),
          tooltip=["type","strike","volume","openInterest","delta"]
      )
      .properties(height=400)
      .interactive()
)
st.altair_chart(vol_chart, use_container_width=True)
st.caption(f"Yellow bar highlights the ATM strike {atm_strike:,.2f}")

st.markdown("---")

# ---------------- Delta distribution by type ----------------
st.subheader("Delta distribution by option type")

dist = (
    alt.Chart(chain[["type","delta"]].dropna())
      .transform_density("delta", as_=["delta","density"], groupby=["type"])
      .mark_area(opacity=0.5)
      .encode(
          x=alt.X("delta:Q", title="Delta"),
          y=alt.Y("density:Q", title="Density"),
          color=alt.Color("type:N",
                          scale=alt.Scale(domain=["Call","Put"], range=["#1a9641","#d7191c"]),
                          legend=alt.Legend(title="Type"))
      )
      .properties(height=300)
)
st.altair_chart(dist, use_container_width=True)

# ---------------- Optional tables and download ----------------
with st.expander("Show calls and puts tables"):
    show_cols = ["type","strike","bid","ask","lastPrice","mid","openInterest","volume","impliedVolatility","delta"]
    st.dataframe(chain.query("type == 'Call'")[show_cols].sort_values("strike"), use_container_width=True)
    st.dataframe(chain.query("type == 'Put'")[show_cols].sort_values("strike"), use_container_width=True)

with st.expander("Download filtered chain as CSV"):
    st.download_button(
        "Download CSV",
        chain.to_csv(index=False),
        file_name=f"{ticker}_{expiry}_filtered_chain.csv",
        mime="text/csv"
    )

st.caption("© 2025 AD Fund Management LP")
