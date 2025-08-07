import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
from datetime import datetime, date

import altair as alt
import matplotlib.pyplot as plt
plt.style.use("default")

# -------------------------------- Page --------------------------------
st.set_page_config(page_title="Options Workstation", layout="wide")
st.title("Options Workstation")

st.sidebar.header("About")
st.sidebar.markdown(
    """
    Practical options dashboard:
    1) Build a multi-leg position and see live Greeks and payoff.
    2) Scan context: expected move, IV smile, OI and volume by strike.
    3) Useful filters: moneyness, min OI/volume, mid vs last.
    """
)

# ----------------------------- Inputs ---------------------------------
ticker = st.sidebar.text_input("Ticker", "AAPL").strip().upper()
if not ticker:
    st.stop()

r = st.sidebar.number_input("Risk-free rate (annual, %)", value=4.5, step=0.25) / 100.0
q = st.sidebar.number_input("Dividend yield (annual, %)", value=0.0, step=0.25) / 100.0
use_mid = st.sidebar.selectbox("Price source", ["Mid (bid/ask)", "Last"], index=0) == "Mid (bid/ask)"
hide_zero = st.sidebar.checkbox("Hide zero OI/Vol", value=True)
min_oi = st.sidebar.number_input("Min open interest", value=0, step=10)
min_vol = st.sidebar.number_input("Min volume", value=0, step=10)
moneyness_window = st.sidebar.slider("Moneyness window around spot (%)", 5, 50, 25)

# ------------------------ Data: spot and expiries ----------------------
@st.cache_data(ttl=600)
def get_spot(tkr: str):
    h = yf.Ticker(tkr).history(period="1d")
    return float(h["Close"].iloc[-1]) if not h.empty else np.nan

@st.cache_data(ttl=900)
def get_expiries(tkr: str):
    try:
        return yf.Ticker(tkr).options
    except Exception:
        return []

spot = get_spot(ticker)
if not np.isfinite(spot):
    st.error("Could not retrieve spot.")
    st.stop()

raw_exps = get_expiries(ticker)
if not raw_exps:
    st.error("No expiries available from Yahoo for this symbol.")
    st.stop()

expiry = st.sidebar.selectbox("Expiry", raw_exps)

# ------------------------ Chain fetch (no cache) -----------------------
def fetch_chain(tkr: str, exp: str):
    tk = yf.Ticker(tkr)
    ch = tk.option_chain(exp)
    calls = ch.calls.copy()
    puts = ch.puts.copy()
    for df in (calls, puts):
        for col in ["bid","ask","lastPrice","volume","openInterest","impliedVolatility","strike"]:
            if col not in df.columns:
                df[col] = np.nan
        df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2.0
        df["mid"] = df["mid"].replace(0, np.nan)
    calls["type"] = "Call"
    puts["type"] = "Put"
    return calls, puts

try:
    calls, puts = fetch_chain(ticker, expiry)
except Exception:
    st.error("Failed to load option chain. Try another expiry.")
    st.stop()

# Filter junk
chain = pd.concat([calls, puts], ignore_index=True)
if hide_zero:
    chain = chain[(chain["openInterest"] > 0) | (chain["volume"] > 0)]
if min_oi > 0:
    chain = chain[chain["openInterest"] >= min_oi]
if min_vol > 0:
    chain = chain[chain["volume"] >= min_vol]

# Moneyness filter
low = spot * (1 - moneyness_window / 100)
high = spot * (1 + moneyness_window / 100)
chain = chain[(chain["strike"] >= low) & (chain["strike"] <= high)]

if chain.empty:
    st.warning("No contracts left after filtering. Adjust filters.")
    st.stop()

# ------------------------ Time to expiry -------------------------------
def busdays_to_expiry(exp: str) -> int:
    today = date.today()
    ed = pd.to_datetime(exp).date()
    return int(np.busday_count(today, ed))

bdays = busdays_to_expiry(expiry)
T = max(bdays, 0) / 252.0

# --------------------------- Black-Scholes -----------------------------
def _N(x):  # cdf of standard normal
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_d1(S, K, r, q, sigma, T):
    return (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def bs_d2(d1, sigma, T):
    return d1 - sigma * math.sqrt(T)

def bs_price_greeks(S, K, r, q, sigma, T, is_call=True):
    """
    Returns price, delta, gamma, vega, theta (per 1 underlying share; vega per 1% vol = vega/100)
    """
    if sigma <= 0 or T <= 0:
        # approximate intrinsic greeks at expiry limit
        if is_call:
            price = max(0.0, S*math.exp(-q*T) - K*math.exp(-r*T))
            delta = 1.0 if S > K else 0.0
        else:
            price = max(0.0, K*math.exp(-r*T) - S*math.exp(-q*T))
            delta = -1.0 if S < K else 0.0
        return price, delta, 0.0, 0.0, 0.0

    d1 = bs_d1(S, K, r, q, sigma, T)
    d2 = bs_d2(d1, sigma, T)
    Nd1 = _N(d1)
    Nd2 = _N(d2)
    nd1 = math.exp(-0.5*d1*d1) / math.sqrt(2*math.pi)

    disc_q = math.exp(-q*T)
    disc_r = math.exp(-r*T)

    if is_call:
        price = S*disc_q*Nd1 - K*disc_r*Nd2
        delta = disc_q*Nd1
    else:
        price = K*disc_r*_N(-d2) - S*disc_q*_N(-d1)
        delta = disc_q*(Nd1 - 1)

    gamma = disc_q * nd1 / (S * sigma * math.sqrt(T))
    vega = S * disc_q * nd1 * math.sqrt(T)  # per 1 vol (ie 100% change)
    theta = (-S*disc_q*nd1*sigma/2/math.sqrt(T)
             - (r*K*disc_r*Nd2 if is_call else r*K*disc_r*_N(-d2))
             + (q*S*disc_q*Nd1 if is_call else q*S*disc_q*_N(-d1)))
    return price, delta, gamma, vega, theta

# ------------------------- Enrich chain --------------------------------
chain["moneyness"] = chain["strike"] / spot - 1.0
chain["price_src"] = chain["mid"] if use_mid else chain["lastPrice"]
chain.loc[chain["price_src"].isna(), "price_src"] = chain["mid"]
chain["price_src"] = chain["price_src"].astype(float)

# Greeks at current IV
def row_greeks(row):
    iv = float(row.get("impliedVolatility") or 0.0)
    if iv <= 0:
        return pd.Series({"delta": np.nan, "gamma": np.nan, "vega": np.nan, "theta": np.nan})
    is_call = row["type"] == "Call"
    _, d, g, v, th = bs_price_greeks(spot, row["strike"], r, q, iv, T, is_call=is_call)
    return pd.Series({"delta": d, "gamma": g, "vega": v/100.0, "theta": th})  # vega per 1 vol pct

greeks = chain.apply(row_greeks, axis=1)
chain = pd.concat([chain, greeks], axis=1)

# ---------------------- Expected move and ATM --------------------------
atm_idx = (chain["strike"] - spot).abs().idxmin()
atm_iv = float(chain.loc[atm_idx, "impliedVolatility"]) if pd.notna(chain.loc[atm_idx, "impliedVolatility"]) else np.nan
exp_move = spot * atm_iv * math.sqrt(T) if np.isfinite(atm_iv) and T > 0 else np.nan

c1, c2, c3, c4 = st.columns(4)
c1.metric("Spot", f"{spot:,.2f}")
c2.metric("Time to expiry (bdays)", f"{bdays}")
c3.metric("ATM IV", f"{atm_iv:.2%}" if np.isfinite(atm_iv) else "n/a")
c4.metric("Expected move (1-sigma)", f"{exp_move:,.2f}" if np.isfinite(exp_move) else "n/a")

st.markdown("---")

# -------------------------- Leg builder --------------------------------
st.subheader("Position builder")

max_legs = st.sidebar.slider("Number of legs", 1, 4, 2)
legs = []

strike_choices = sorted(chain["strike"].unique().tolist())

with st.form("legs_form", clear_on_submit=False):
    cols = st.columns([1,1,1,1,1,1])
    cols[0].markdown("**Side**")
    cols[1].markdown("**Type**")
    cols[2].markdown("**Strike**")
    cols[3].markdown("**Qty**")
    cols[4].markdown("**IV override (optional)**")
    cols[5].markdown("**Use custom price**")

    for i in range(max_legs):
        c = st.columns([1,1,1,1,1,1])
        side = c[0].selectbox(f"side_{i}", ["Buy","Sell"], key=f"side_{i}")
        otype = c[1].selectbox(f"type_{i}", ["Call","Put"], key=f"type_{i}")
        strike = c[2].selectbox(f"strike_{i}", strike_choices, index=min(len(strike_choices)-1, max(0, np.searchsorted(strike_choices, spot))), key=f"strike_{i}")
        qty = c[3].number_input(f"qty_{i}", min_value=-1000, max_value=1000, value=1, step=1, key=f"qty_{i}")
        iv_override = c[4].number_input(f"iv_{i}", min_value=0.0, max_value=5.0, value=0.0, step=0.01, key=f"iv_{i}")
        use_custom_px = c[5].checkbox(f"custom_px_{i}", value=False, key=f"custompx_{i}")

        # derive default price from chain
        sel = chain[(chain["type"]==otype) & (chain["strike"]==strike)]
        default_px = float(sel["price_src"].iloc[0]) if not sel.empty and pd.notna(sel["price_src"].iloc[0]) else 0.0
        price = c[5].number_input(f"price_{i}", min_value=0.0, value=float(default_px), step=0.05, key=f"price_{i}")
        if not use_custom_px:
            price = default_px

        legs.append({
            "side": side, "type": otype, "strike": float(strike),
            "qty": int(qty), "price": float(price),
            "iv": float(iv_override) if iv_override > 0 else float(sel["impliedVolatility"].iloc[0]) if not sel.empty and pd.notna(sel["impliedVolatility"].iloc[0]) else 0.0
        })

    submitted = st.form_submit_button("Update position")

# Compute position Greeks and payoff
def leg_sign(side): return 1 if side == "Buy" else -1

def payoff_at_expiry(S, leg):
    K = leg["strike"]
    ql = leg["qty"]
    sgn = leg_sign(leg["side"])
    if leg["type"] == "Call":
        intrinsic = max(S - K, 0.0)
    else:
        intrinsic = max(K - S, 0.0)
    # option contract is 100 shares
    return ql * (sgn * (intrinsic * 100.0) - sgn * leg["price"] * 100.0)

def greeks_now(leg):
    is_call = leg["type"] == "Call"
    sgn = leg_sign(leg["side"])
    price, d, g, v, th = bs_price_greeks(spot, leg["strike"], r, q, max(leg["iv"], 1e-8), T, is_call)
    # scale by 100 per contract and by qty, sign flips for sells
    scale = leg["qty"] * sgn * 100.0
    return {
        "price": price * scale,
        "delta": d * scale,
        "gamma": g * scale,
        "vega": v * scale,
        "theta": th * scale
    }

# aggregate Greeks
agg = {"price":0.0,"delta":0.0,"gamma":0.0,"vega":0.0,"theta":0.0,"credit":0.0}
for lg in legs:
    gk = greeks_now(lg)
    for k in ["price","delta","gamma","vega","theta"]:
        agg[k] += gk[k]
    # credit positive for sells, debit negative for buys
    agg["credit"] += leg_sign(lg["side"]) * (-lg["price"] * 100.0 * lg["qty"])

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Position delta", f"{agg['delta']:.1f}")
c2.metric("Gamma", f"{agg['gamma']:.4f}")
c3.metric("Vega (per 1 vol pct)", f"{agg['vega']:.1f}")
c4.metric("Theta (per day approx)", f"{agg['theta']/365.0:.1f}")
c5.metric("Theoretical value", f"{agg['price']:,.0f}")
c6.metric("Net credit (+) / debit (-)", f"{agg['credit']:,.0f}")

# Payoff chart
st.subheader("Payoff at expiry")
S_grid = np.linspace(spot * 0.5, spot * 1.5, 300)
payoff = np.zeros_like(S_grid)
for lg in legs:
    payoff += np.vectorize(lambda x: payoff_at_expiry(x, lg))(S_grid)

fig, ax = plt.subplots(figsize=(10,4))
ax.axhline(0, color="gray", lw=1)
ax.axvline(spot, color="gray", lw=1, ls=":")
ax.plot(S_grid, payoff, color="black", lw=2)
ax.set_xlabel("Underlying price at expiry")
ax.set_ylabel("P&L (USD)")
ax.grid(alpha=0.3, linestyle="--")
st.pyplot(fig, use_container_width=True)

st.markdown("---")

# ---------------------- Context: IV smile, OI/Vol ----------------------
st.subheader("IV smile and strike distribution")

smile_data = chain[["type","strike","impliedVolatility","moneyness","openInterest","volume"]].dropna()
smile = alt.Chart(smile_data).mark_circle(size=45, opacity=0.7).encode(
    x=alt.X("moneyness:Q", title="Moneyness (K/S - 1)"),
    y=alt.Y("impliedVolatility:Q", title="Implied vol"),
    color=alt.Color("type:N", scale=alt.Scale(domain=["Call","Put"], range=["#1a9641","#d7191c"])),
    size=alt.Size("openInterest:Q", title="Open interest"),
    tooltip=["type","strike","impliedVolatility","openInterest","volume"]
).properties(height=350)
st.altair_chart(smile, use_container_width=True)

st.caption("Point size scaled by open interest. Use filters on the left to focus the view.")

# OI and Volume by strike (stacked bars)
bars = chain.groupby(["strike","type"])[["openInterest","volume"]].sum().reset_index()
oi_chart = alt.Chart(bars).mark_bar().encode(
    x=alt.X("strike:Q", title="Strike"),
    y=alt.Y("openInterest:Q", title="Open interest"),
    color=alt.Color("type:N", scale=alt.Scale(domain=["Call","Put"], range=["#1a9641","#d7191c"])),
    tooltip=["type","strike","openInterest","volume"]
).properties(height=300)
st.altair_chart(oi_chart, use_container_width=True)

vol_chart = alt.Chart(bars).mark_bar().encode(
    x=alt.X("strike:Q", title="Strike"),
    y=alt.Y("volume:Q", title="Volume"),
    color=alt.Color("type:N", scale=alt.Scale(domain=["Call","Put"], range=["#1a9641","#d7191c"])),
    tooltip=["type","strike","openInterest","volume"]
).properties(height=300)
st.altair_chart(vol_chart, use_container_width=True)

st.markdown("---")

# -------------------------- Tables and export --------------------------
with st.expander("Show calls and puts tables"):
    show_cols = ["type","strike","bid","ask","lastPrice","mid","openInterest","volume","impliedVolatility","delta","gamma","vega","theta"]
    st.dataframe(chain.query("type=='Call'")[show_cols].sort_values("strike"), use_container_width=True)
    st.dataframe(chain.query("type=='Put'")[show_cols].sort_values("strike"), use_container_width=True)

with st.expander("Download filtered chain as CSV"):
    st.download_button(
        "Download CSV",
        chain.to_csv(index=False),
        file_name=f"{ticker}_{expiry}_filtered_chain.csv",
        mime="text/csv"
    )

st.caption("© 2025 AD Fund Management LP")
