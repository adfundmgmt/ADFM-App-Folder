# options_chain_viewer.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import math
from datetime import datetime, date, time
import pytz
import matplotlib.pyplot as plt

plt.style.use("default")

# ---------------- Page ----------------
st.set_page_config(page_title="Options Chain Viewer", layout="wide")
st.title("Options Chain Viewer")

with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
    - Interactive options chain with filters that matter
    - Open interest by strike with ATM highlight
    - Summary metrics: put call ratio, expected move, average delta
    - Delta distribution to visualize skew
    - Optional tables and CSV export
    """)

# ---------------- Helpers ----------------
NY = pytz.timezone("America/New_York")

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
    Returns a tidy chain with calls and puts combined, includes clean 'mid' and IV fields.
    """
    tk = yf.Ticker(tkr)
    ch = tk.option_chain(exp)
    calls = ch.calls.copy()
    puts  = ch.puts.copy()

    needed = ["bid","ask","lastPrice","volume","openInterest","impliedVolatility","strike"]
    for df in (calls, puts):
        for col in needed:
            if col not in df.columns:
                df[col] = np.nan

        df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
        df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
        df["lastPrice"] = pd.to_numeric(df["lastPrice"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0).astype(int)
        df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")

        # Robust mid: only if both bid and ask are positive, else fall back to lastPrice
        has_ba = (df["bid"] > 0) & (df["ask"] > 0)
        df["mid"] = np.where(has_ba, (df["bid"] + df["ask"]) / 2.0, np.nan)
        df["mid"] = df["mid"].fillna(df["lastPrice"])

    # Keep side IV and later compute blended IV per strike
    calls.rename(columns={"impliedVolatility": "side_iv"}, inplace=True)
    puts.rename(columns={"impliedVolatility": "side_iv"}, inplace=True)

    calls["type"] = "Call"
    puts["type"]  = "Put"
    out = pd.concat([calls, puts], ignore_index=True)

    iv_blend = (
        out.groupby("strike", as_index=False)["side_iv"]
          .mean()
          .rename(columns={"side_iv": "blend_iv"})
    )
    out = out.merge(iv_blend, on="strike", how="left")
    return out

def busdays_to_expiry(exp: str) -> int:
    today = datetime.now(NY).date()
    ed = pd.to_datetime(exp).date()
    return int(np.busday_count(today, ed))

def act365_time_to_expiry(exp: str) -> float:
    """
    Calendar time to 4 pm New York on expiry, ACT/365.
    """
    now = datetime.now(NY)
    ed = pd.to_datetime(exp).date()
    expiry_dt = NY.localize(datetime.combine(ed, time(16, 0)))
    seconds = max((expiry_dt - now).total_seconds(), 0.0)
    return seconds / (365.0 * 24 * 60 * 60)

# Vectorized normal CDF (Hastings)
def norm_cdf(x):
    x = np.asarray(x, dtype=float)
    a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    L = np.abs(x)
    k = 1.0 / (1.0 + 0.2316419 * L)
    poly = (((a5*k + a4)*k + a3)*k + a2)*k + a1
    w = 1.0 - (1.0/np.sqrt(2*np.pi)) * np.exp(-0.5*L*L) * poly
    return np.where(x >= 0, w, 1.0 - w)

# ---------------- Sidebar controls (live, no form) ----------------
st.sidebar.subheader("Inputs")

# Persist last used ticker and expiry lock
if "locked_expiry_str" not in st.session_state:
    st.session_state.locked_expiry_str = None
if "lock_expiry" not in st.session_state:
    st.session_state.lock_expiry = False

ticker = st.sidebar.text_input("Ticker", st.session_state.get("last_ticker", "AAPL")).strip().upper()
st.session_state.last_ticker = ticker

expiries_list = get_expiries(ticker) if ticker else []

# Lock expiry across tickers
lock_expiry = st.sidebar.checkbox("Lock expiry across tickers", value=st.session_state.lock_expiry)
st.session_state.lock_expiry = lock_expiry

def nearest_expiry(target_iso: str, avail: list) -> str | None:
    if not avail:
        return None
    if target_iso in avail:
        return target_iso
    try:
        t = pd.to_datetime(target_iso)
        diffs = [abs(pd.to_datetime(e) - t) for e in avail]
        return avail[int(np.argmin(diffs))]
    except Exception:
        return avail[0]

# Pick expiry
if expiries_list:
    if lock_expiry and st.session_state.locked_expiry_str:
        expiry = nearest_expiry(st.session_state.locked_expiry_str, expiries_list)
        st.sidebar.selectbox("Expiry", expiries_list, index=expiries_list.index(expiry), key="expiry_dummy", disabled=True)
    else:
        expiry = st.sidebar.selectbox("Expiry", expiries_list, index=0, key="expiry_select")
        st.session_state.locked_expiry_str = expiry
else:
    expiry = None
    st.sidebar.caption("No expiries available")

st.sidebar.markdown("---")
use_mid = st.sidebar.selectbox("Price source", ["Mid (bid ask)", "Last"], index=0) == "Mid (bid ask)"
hide_zero = st.sidebar.checkbox("Hide zero OI Vol", value=True)
min_oi = st.sidebar.number_input("Min open interest", value=0, step=10)
min_vol = st.sidebar.number_input("Min volume", value=0, step=10)
mny_win = st.sidebar.slider("Moneyness window around spot percent", 5, 50, 25)

st.sidebar.markdown("---")
r_pct = st.sidebar.number_input("Risk free rate percent", value=4.5, step=0.25)
q_pct = st.sidebar.number_input("Dividend yield percent", value=0.0, step=0.25)

st.sidebar.markdown("---")
st.sidebar.subheader("Delta distribution controls")
ymax = st.sidebar.slider("Density y-max", 0.5, 5.0, 2.0, 0.1)
bandwidth = st.sidebar.slider("KDE bandwidth", 0.02, 0.20, 0.08, 0.01)
weight_mode = st.sidebar.selectbox("Weight density by", ["Contracts (unweighted)", "Open interest", "Volume"], index=0)

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
if hide_zero:
    chain = chain[(chain["openInterest"] > 0) | (chain["volume"] > 0)]
if min_oi > 0:
    chain = chain[chain["openInterest"] >= int(min_oi)]
if min_vol > 0:
    chain = chain[chain["volume"] >= int(min_vol)]

low = spot * (1 - mny_win / 100.0)
high = spot * (1 + mny_win / 100.0)
chain = chain[(chain["strike"] >= low) & (chain["strike"] <= high)]

if chain.empty:
    st.warning("No contracts left after filtering. Adjust filters.")
    st.stop()

# Price source
px_src = chain["mid"] if use_mid else chain["lastPrice"]
chain["price_src"] = px_src

# ---------------- Time to expiry ----------------
bdays = busdays_to_expiry(expiry)
T = act365_time_to_expiry(expiry)  # ACT/365 for pricing

# ---------------- ATM selection and expected move ----------------
atm_idx = (chain["strike"] - spot).abs().idxmin()
atm_strike = float(chain.loc[atm_idx, "strike"])
atm_iv_row = chain.loc[chain["strike"] == atm_strike, "blend_iv"]
atm_iv = float(atm_iv_row.mean()) if not atm_iv_row.empty else float("nan")

expected_move = spot * atm_iv * math.sqrt(T) if np.isfinite(atm_iv) and T > 0 else float("nan")

# ---------------- Deltas using Black Scholes with dividend yield ----------------
sigma = pd.to_numeric(chain["side_iv"], errors="coerce").astype(float).to_numpy(copy=False)
sigma = np.clip(sigma, 1e-6, 5.0)  # clamp to avoid artifacts

K = chain["strike"].astype(float).to_numpy(copy=False)
S = float(spot)
T_arr = np.full_like(K, T, dtype=float)

valid = (sigma > 0) & (T_arr > 0) & np.isfinite(S) & np.isfinite(K)

d1 = np.zeros_like(K, dtype=float)
if valid.any():
    d1[valid] = (np.log(S / K[valid]) + (r_pct/100.0 - q_pct/100.0 + 0.5 * sigma[valid] ** 2) * T_arr[valid]) / (sigma[valid] * np.sqrt(T_arr[valid]))

disc_q = np.exp(-(q_pct/100.0) * T_arr)
call_delta = np.zeros_like(K, dtype=float)
if valid.any():
    call_delta[valid] = disc_q[valid] * norm_cdf(d1[valid])

is_call = (chain["type"] == "Call").to_numpy()
delta = np.where(is_call, call_delta, call_delta - disc_q)
delta[~valid] = np.nan
chain["delta"] = delta

# ---------------- Summary metrics ----------------
total_call_vol = int(chain.query("type == 'Call'")["volume"].sum())
total_put_vol  = int(chain.query("type == 'Put'")["volume"].sum())
put_call_ratio = (total_put_vol / total_call_vol) if total_call_vol > 0 else float("nan")
avg_delta      = float(chain["delta"].mean())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Spot", f"{spot:,.2f}")
c2.metric("Business days to expiry", f"{bdays}")
c3.metric("ATM strike", f"{atm_strike:,.2f}")
c4.metric("ATM IV", f"{atm_iv:.2%}" if np.isfinite(atm_iv) else "n a")
c5.metric("Expected move 1 sigma", f"{expected_move:,.2f}" if np.isfinite(expected_move) else "n a")

st.caption("PCR and averages computed on the filtered window")
st.markdown("---")

st.subheader("Summary")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Total call volume", f"{total_call_vol:,}")
s2.metric("Total put volume", f"{total_put_vol:,}")
s3.metric("Put call ratio", f"{put_call_ratio:.2f}" if np.isfinite(put_call_ratio) else "n a")
s4.metric("Average delta", f"{avg_delta:.2f}" if np.isfinite(avg_delta) else "n a")

st.caption(f"Spot {spot:,.2f}   ATM strike {atm_strike:,.2f}")
st.markdown("---")

# ---------------- Open Interest by strike with ATM highlight ----------------
st.subheader("Open Interest by strike")

chart_data = chain[["type","strike","openInterest","volume","delta"]].copy()
chart_data["highlight"] = np.where(chart_data["strike"] == atm_strike, "ATM", "Other")

oi_chart = (
    alt.Chart(chart_data)
      .mark_bar()
      .encode(
          x=alt.X("strike:Q", title="Strike"),
          y=alt.Y("openInterest:Q", title="Open Interest"),
          color=alt.condition(
              alt.datum.highlight == "ATM",
              alt.value("#FFD600"),
              alt.Color("type:N",
                        scale=alt.Scale(domain=["Call","Put"], range=["#1a9641","#d7191c"]),
                        legend=alt.Legend(title="Type"))
          ),
          tooltip=["type","strike","openInterest","volume","delta"]
      )
      .properties(height=400)
      .interactive()
)
st.altair_chart(oi_chart, use_container_width=True)
st.caption(f"Yellow bar highlights the ATM strike {atm_strike:,.2f}")
st.markdown("---")

# ---------------- Delta distribution by option type ----------------
st.subheader("Delta distribution by option type")

df = chain[["type","delta","openInterest","volume"]].dropna().copy()

# If user picked weighted modes, fall back gracefully (Altair in this env lacks weights=)
if weight_mode != "Contracts (unweighted)":
    st.info("Density weighting by OI/Volume not available in this Altair build. Showing unweighted density.")
    # no 'w' used below; kept variable in case of future upgrade

dist = (
    alt.Chart(df)
      .transform_density(
          "delta",
          as_=["delta","density"],
          groupby=["type"],
          extent=[-1.0, 1.0],
          bandwidth=float(bandwidth)
      )
      .mark_area(opacity=0.5)
      .encode(
          x=alt.X("delta:Q", title="Delta", scale=alt.Scale(domain=[-1, 1])),
          y=alt.Y("density:Q", title="Density", scale=alt.Scale(domain=[0, float(ymax)])),
          color=alt.Color("type:N",
                          scale=alt.Scale(domain=["Call","Put"], range=["#1a9641","#d7191c"]),
                          legend=alt.Legend(title="Type"))
      )
      .properties(height=320)
)
st.altair_chart(dist, use_container_width=True)

# ---------------- Optional tables and download ----------------
with st.expander("Show calls and puts tables"):
    show_cols = ["type","strike","bid","ask","lastPrice","mid","openInterest","volume","side_iv","blend_iv","delta"]
    st.dataframe(chain.query("type == 'Call'")[show_cols].sort_values("strike"), use_container_width=True)
    st.dataframe(chain.query("type == 'Put'")[show_cols].sort_values("strike"), use_container_width=True)

with st.expander("Download filtered chain as CSV"):
    st.download_button(
        "Download CSV",
        chain.to_csv(index=False),
        file_name=f"{ticker}_{expiry}_filtered_chain.csv",
        mime="text/csv"
    )

# Persist locked expiry if the user wants it
if lock_expiry and expiry:
    st.session_state.locked_expiry_str = expiry

st.caption("© 2025 AD Fund Management LP")
