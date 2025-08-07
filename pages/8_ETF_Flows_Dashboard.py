import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime, timedelta
import concurrent.futures

plt.style.use("default")
st.set_page_config(page_title="ETF Flows Dashboard", layout="wide")

# ───────────────── Sidebar ─────────────────
st.sidebar.title("ETF Flows")
st.sidebar.markdown(
    """
A dashboard of **thematic and global ETF flows** — see where money is moving.

**Method**: Dollar flow ≈ Δ(shares outstanding) × price (end of period).  
Flow % ≈ Dollar flow / AUM (end of period). Proxies when SO/AUM missing.
"""
)

# Lookback
today = pd.Timestamp.today().normalize()
spans = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "12 Months": 365,
    "YTD": (today - pd.Timestamp(year=today.year, month=1, day=1)).days,
}
period_label = st.sidebar.radio("Select Lookback Period", list(spans.keys()), index=1)
period_days = int(spans[period_label])
start_date = today - pd.Timedelta(days=period_days + 10)  # +buffer for alignment

# Universe
etf_info = {
    "MAGS": ("Mag 7", "Magnificent 7 stocks ETF"),
    "SMH": ("Semiconductors", "Semiconductor stocks (VanEck)"),
    "BOTZ": ("Robotics/AI", "Global robotics and AI leaders"),
    "ICLN": ("Clean Energy", "Global clean energy stocks"),
    "URNM": ("Uranium", "Uranium miners (Sprott)"),
    "ARKK": ("Innovation", "Disruptive growth stocks (ARK)"),
    "KWEB": ("China Internet", "China internet leaders (KraneShares)"),
    "FXI": ("China Large-Cap", "China mega-cap stocks"),
    "EWZ": ("Brazil", "Brazil large-cap equities"),
    "EEM": ("Emerging Markets", "EM equities (MSCI)"),
    "VWO": ("Emerging Markets", "EM equities (Vanguard)"),
    "VGK": ("Europe Large-Cap", "Developed Europe stocks (Vanguard)"),
    "FEZ": ("Eurozone", "Euro STOXX 50 ETF"),
    "ILF": ("Latin America", "Latin America 40 ETF"),
    "ARGT": ("Argentina", "Global X MSCI Argentina ETF"),
    "GLD": ("Gold", "SPDR Gold Trust ETF"),
    "SLV": ("Silver", "iShares Silver Trust ETF"),
    "DBC": ("Commodities", "Invesco DB Commodity Index ETF"),
    "HEDJ": ("Hedged Europe", "WisdomTree Europe Hedged Equity ETF"),
    "USMV": ("US Min Volatility", "iShares MSCI USA Min Volatility ETF"),
    "COWZ": ("US Free Cash Flow", "Pacer US Cash Cows 100 ETF"),
    "BITO": ("BTC Futures", "Bitcoin futures ETF"),
    "IBIT": ("Spot BTC", "BlackRock spot Bitcoin ETF"),
    "BIL": ("1–3mo T-Bills", "1–3 month U.S. Treasury bills"),
    "TLT": ("20+yr Treasuries", "20+ year U.S. Treasuries"),
    "SHV": ("0–1yr T-Bills", "Short-term Treasury bonds"),
}
TICKERS = list(etf_info.keys())

# ───────────────── Data helpers ─────────────────
@st.cache_data(ttl=900, show_spinner=True)
def fetch_prices(tickers, start, end):
    """Batch adjusted closes to align with SO dates."""
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    if isinstance(df.columns, pd.MultiIndex):
        closes = {t: df[(t, "Close")] for t in tickers if (t, "Close") in df.columns}
        if not closes:
            return pd.DataFrame()
        out = pd.DataFrame(closes).sort_index().ffill()
        return out
    else:
        # single ticker fallback
        return pd.DataFrame({tickers[0]: df["Close"]}).sort_index().ffill() if "Close" in df.columns else pd.DataFrame()

def _flow_from_so_and_price(so_series, price_series, end_price):
    """Compute dollar flow from ΔSO × end_price; also return AUM_end."""
    if so_series is None or so_series.dropna().empty:
        return None, None
    so = so_series.dropna().astype(float)
    # Align SO to trading days for the period
    aligned = so.reindex(price_series.index, method="ffill").dropna()
    if aligned.empty:
        return None, None
    so_start, so_end = float(aligned.iloc[0]), float(aligned.iloc[-1])
    flow_dollars = (so_end - so_start) * float(end_price)
    aum_end = so_end * float(end_price)
    return flow_dollars, aum_end

@st.cache_data(ttl=900, show_spinner=True)
def robust_flow_estimate(ticker, price_slice: pd.Series):
    """
    Best-effort flow and AUM.
    Prefers: Δ(Shares Outstanding) × Price_end. Falls back to info['totalAssets'] for AUM.
    Last-ditch proxy: price change × avg volume (clearly labeled as proxy).
    """
    try:
        tk = yf.Ticker(ticker)
        # Shares outstanding history (may be sparse by issuer)
        try:
            so_full = tk.get_shares_full()  # pandas Series
        except Exception:
            so_full = None

        price_slice = price_slice.dropna()
        if price_slice.empty:
            return {"flow": None, "flow_pct": None, "aum": None, "method": "no_price"}

        end_price = price_slice.iloc[-1]

        # Primary method: ΔSO × end_price
        flow, aum = _flow_from_so_and_price(so_full, price_slice, end_price)
        if flow is not None and aum not in (None, 0):
            return {"flow": float(flow), "flow_pct": float(flow / aum), "aum": float(aum), "method": "ΔSO×P_end"}

        # Secondary: issuer AUM if available
        info = {}
        try:
            info = tk.fast_info or {}
        except Exception:
            pass
        if not info:
            try:
                info = tk.info or {}
            except Exception:
                info = {}

        total_assets = info.get("totalAssets") or info.get("total_assets")
        if total_assets and price_slice.size >= 2:
            # When SO unavailable, estimate flow by Δ(estimated SO) = AUM/P
            so_est = (float(total_assets) / price_slice).dropna()
            flow2, aum2 = _flow_from_so_and_price(so_est, price_slice, end_price)
            if flow2 is not None and aum2 not in (None, 0):
                return {"flow": float(flow2), "flow_pct": float(flow2 / aum2), "aum": float(aum2), "method": "Δ(AUM/P)×P_end"}

        # Fallback proxy (worst): price Δ × avg volume
        if price_slice.size >= 2:
            proxy_flow = (price_slice.iloc[-1] - price_slice.iloc[0]) * float(price_slice.rolling(10, min_periods=1).mean().iloc[-1])
            # very rough AUM proxy: attempt fast_info market cap or default to $1B
            aum_proxy = float(info.get("marketCap") or 1_000_000_000.0)
            flow_pct = float(proxy_flow / aum_proxy) if aum_proxy != 0 else None
            return {"flow": float(proxy_flow), "flow_pct": flow_pct, "aum": aum_proxy, "method": "ΔP×Vol_proxy"}

        return {"flow": None, "flow_pct": None, "aum": None, "method": "no_data"}

    except Exception:
        return {"flow": None, "flow_pct": None, "aum": None, "method": "error"}

@st.cache_data(ttl=900, show_spinner=True)
def compute_flows(tickers, start_date, end_date):
    # Batch prices once
    prices = fetch_prices(tickers, start_date, end_date)
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Compute per-ticker in a thread pool, with a sane cap to avoid rate limits
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(tickers))) as ex:
        futures = {
            ex.submit(robust_flow_estimate, t, prices[t] if t in prices.columns else pd.Series(dtype=float)): t
            for t in tickers
        }
        for fut in concurrent.futures.as_completed(futures):
            t = futures[fut]
            results[t] = fut.result()

    rows = []
    for t in tickers:
        cat, desc = etf_info[t]
        res = results.get(t, {}) or {}
        rows.append({
            "Ticker": t,
            "Category": cat,
            "Description": desc,
            "Flow ($)": res.get("flow"),
            "Flow (%)": (res.get("flow_pct") * 100.0) if res.get("flow_pct") is not None else None,
            "AUM ($)": res.get("aum"),
            "Method": res.get("method"),
        })

    df = pd.DataFrame(rows)
    # Aggregate by category
    cat_df = df.groupby("Category", as_index=False).agg({"Flow ($)": "sum"})
    # For % at category level, show flow / sum AUM where available
    aum_cat = df.groupby("Category", as_index=False).agg({"AUM ($)": "sum"})
    cat_df = cat_df.merge(aum_cat, on="Category", how="left")
    cat_df["Flow (%)"] = np.where(cat_df["AUM ($)"].fillna(0) != 0, cat_df["Flow ($)"] / cat_df["AUM ($)"] * 100.0, np.nan)

    return df, cat_df

# ───────────────── Compute & prepare ─────────────────
tickers = TICKERS
df, cat_df = compute_flows(tickers, start_date, today)

# Label helpers
def money_label(x):
    if x is None or pd.isna(x):
        return ""
    s = "-" if x < 0 else "+"
    ax = abs(x)
    if ax >= 1e12:  return f"{s}${ax/1e12:,.0f}T"
    if ax >= 1e9:   return f"{s}${ax/1e9:,.0f}B"
    if ax >= 1e6:   return f"{s}${ax/1e6:,.0f}M"
    if ax >= 1e3:   return f"{s}${ax/1e3:,.0f}K"
    return f"{s}${ax:,.0f}"

# Sort by flow
df = df.sort_values("Flow ($)", ascending=False, na_position="last").reset_index(drop=True)
df["Label"] = [f"{etf_info[t][0]} ({t})" for t in df["Ticker"]]

# ───────────────── Main ─────────────────
st.title("ETF Flows Dashboard")
st.caption(f"Period: **{period_label}** (ending {today.date()})  •  Method priority: ΔSO×Price → Δ(AUM/P)×Price → proxy")

# — ETF bar chart —
chart_df = df.copy()
max_val = chart_df["Flow ($)"].dropna().abs().max() if chart_df["Flow ($)"].notna().any() else 0.0
buffer = max(1.0, max_val * 0.15)

fig, ax = plt.subplots(figsize=(15, max(6, len(chart_df) * 0.42)))
vals = chart_df["Flow ($)"].fillna(0.0).to_numpy()
colors = np.where(vals > 0, "#2ca02c", np.where(vals < 0, "#d62728", "#bdbdbd"))
bars = ax.barh(chart_df["Label"], vals, color=colors, alpha=0.85)

ax.set_xlabel("Estimated Flow ($)")
ax.set_title(f"ETF Proxy Flows — {period_label}")
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e9:,.0f}B' if abs(x)>=1e9 else f'${x/1e6:,.0f}M'))
x_lo = -buffer if np.nanmin(vals) < 0 else 0
x_hi = max_val + buffer
ax.set_xlim([x_lo, x_hi])

for bar, val in zip(bars, chart_df["Flow ($)"]):
    if pd.isna(val):
        continue
    label = money_label(val)
    x = bar.get_width()
    ha = "left" if val >= 0 else "right"
    offset = 0.01 * (x_hi - x_lo)
    ax.text(x + (offset if val >= 0 else -offset), bar.get_y() + bar.get_height()/2, label, va="center", ha=ha, fontsize=10)

plt.tight_layout()
st.pyplot(fig)
st.markdown("*Green = inflow, Red = outflow, Gray = missing/unknown.*")

# — Category rollup —
st.markdown("### Category Totals")
cat_v = cat_df.sort_values("Flow ($)", ascending=False, na_position="last")
fig2, ax2 = plt.subplots(figsize=(12, max(3, len(cat_v) * 0.45)))
vals2 = cat_v["Flow ($)"].fillna(0.0).to_numpy()
colors2 = np.where(vals2 > 0, "#2ca02c", np.where(vals2 < 0, "#d62728", "#bdbdbd"))
bars2 = ax2.barh(cat_v["Category"], vals2, color=colors2, alpha=0.9)
ax2.set_xlabel("Estimated Flow ($)")
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e9:,.0f}B' if abs(x)>=1e9 else f'${x/1e6:,.0f}M'))
x2_max = np.nanmax(np.abs(vals2)) if vals2.size else 0.0
ax2.set_xlim([-(x2_max*1.15), x2_max*1.15])
for bar, val in zip(bars2, cat_v["Flow ($)"]):
    if pd.isna(val): continue
    ax2.text(bar.get_width() + np.sign(val)*0.01*x2_max, bar.get_y()+bar.get_height()/2, money_label(val), va="center",
             ha=("left" if val>=0 else "right"), fontsize=10)
plt.tight_layout()
st.pyplot(fig2)

# — Top movers tables —
st.markdown("### Top Inflows & Outflows")
top_in  = df.head(5)[["Label","Flow ($)","Flow (%)","AUM ($)","Method"]].copy()
top_out = df.sort_values("Flow ($)").head(5)[["Label","Flow ($)","Flow (%)","AUM ($)","Method"]].copy()

def fmt_table(_df):
    out = _df.copy()
    out["Flow ($)"] = out["Flow ($)"].map(money_label)
    out["Flow (%)"] = out["Flow (%)"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
    out["AUM ($)"]  = out["AUM ($)"].map(lambda x: money_label(x).replace("+",""))
    return out.set_index("Label")

c1, c2 = st.columns(2)
with c1:
    st.write("**Top Inflows**")
    st.table(fmt_table(top_in))
with c2:
    st.write("**Top Outflows**")
    st.table(fmt_table(top_out))

# — Full table —
st.markdown("### Full Table")
full_tbl = df[["Ticker","Category","Flow ($)","Flow (%)","AUM ($)","Method","Description"]].copy()
full_tbl["Flow ($)"] = full_tbl["Flow ($)"].map(money_label)
full_tbl["Flow (%)"] = full_tbl["Flow (%)"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
full_tbl["AUM ($)"]  = full_tbl["AUM ($)"].map(lambda x: money_label(x).replace("+",""))
st.dataframe(full_tbl, use_container_width=True)

# — Status —
if df["Flow ($)"].isna().any():
    st.warning("Some ETFs missing SO/AUM; used fallback methods where needed (see Method column).")
else:
    st.success("All flows computed via ΔSO×Price.")

st.caption("© 2025 AD Fund Management LP")
