# ETF Flows — Core 20 with Upload Support (ΔSO × Close)
# ------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.style.use("default")
st.set_page_config(page_title="ETF Flows — Core 20", layout="wide")

# ========= Config =========
BASKET = {
    # Core beta
    "SPY": "S&P 500", "IVV": "S&P 500 (iShares)", "QQQ": "Nasdaq 100", "IWM": "Russell 2000",
    "EFA": "Developed ex-US", "EEM": "Emerging Markets",
    # Sectors (SPDR)
    "XLK": "Tech", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care", "XLI": "Industrials",
    # Credit & Rates
    "HYG": "High Yield", "LQD": "IG Corp", "TLT": "20+Y Treasuries", "IEF": "7–10Y Treas", "SHY": "1–3Y Treas",
    # Cash / metals / alt beta
    "BIL": "T-Bills 1–3M", "GLD": "Gold", "SLV": "Silver", "USMV": "US Min Vol",
}
TICKERS = list(BASKET.keys())

# ========= Sidebar =========
st.sidebar.title("ETF Flows — Core 20")
st.sidebar.markdown("""
Flows = **Δ Shares Outstanding × Close**.

- **Source:**
  - *Yahoo (best-effort)* — tries to fetch SO history. Many ETFs won’t have it.
  - *Upload (recommended)* — provide a CSV of daily SO for any subset.
""")

today = pd.Timestamp.today().normalize()
LOOKBACKS = {
    "1M": 30, "3M": 90, "6M": 180, "12M": 365,
    "YTD": (today - pd.Timestamp(year=today.year, month=1, day=1)).days,
}
lb_key = st.sidebar.selectbox("Lookback", list(LOOKBACKS.keys()), index=1)
days = int(LOOKBACKS[lb_key])
start = today - pd.Timedelta(days=days + 10)

agg = st.sidebar.selectbox("Aggregate", ["Daily", "Weekly"], index=1)
source = st.sidebar.selectbox("Source", ["Yahoo (best-effort)", "Upload (CSV)"], index=1)
show_missing = st.sidebar.checkbox("Show Missing audit", value=True)

uploaded = None
if source == "Upload (CSV)":
    st.sidebar.markdown("**CSV format** (wide or long):")
    st.sidebar.code("date,ticker,shares\n2024-01-02,SPY,123456789\n...", language="text")
    uploaded = st.sidebar.file_uploader("Upload Shares Outstanding CSV", type=["csv"])

# ========= Helpers =========
@st.cache_data(ttl=900, show_spinner=True)
def batch_prices(tickers, start_dt, end_dt):
    df = yf.download(tickers, start=start_dt, end=end_dt, auto_adjust=False,
                     progress=False, group_by="ticker", threads=True)
    if isinstance(df.columns, pd.MultiIndex):
        closes = {t: df[(t, "Close")] for t in tickers if (t, "Close") in df.columns}
        if not closes: return pd.DataFrame()
        out = pd.DataFrame(closes).sort_index().ffill()
    elif "Close" in df.columns:
        out = pd.DataFrame({tickers[0]: df["Close"]}).sort_index().ffill()
    else:
        return pd.DataFrame()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out

def fmt_money(x, signed=True, d=2):
    if x is None or pd.isna(x): return ""
    s = "+" if (signed and x > 0) else ("-" if (signed and x < 0) else "")
    ax = abs(float(x))
    if   ax >= 1e9: val, suf = ax/1e9, "B"
    elif ax >= 1e6: val, suf = ax/1e6, "M"
    elif ax >= 1e3: val, suf = ax/1e3, "k"
    else:           val, suf = ax, ""
    return f"{s}${val:,.{d}f}{suf}"

def tick_money(x, _):
    ax = abs(float(x)); s = "-" if x < 0 else ""
    if   ax >= 1e9: val, suf = ax/1e9, "B"
    elif ax >= 1e6: val, suf = ax/1e6, "M"
    elif ax >= 1e3: val, suf = ax/1e3, "k"
    else:           val, suf = ax, ""
    return f"{s}${val:,.2f}{suf}"

def normalize_so_units(so: pd.Series):
    so = (so or pd.Series(dtype=float)).dropna().astype(float)
    if so.empty: return so
    # If median looks like "millions", scale up to shares
    return so*1_000_000.0 if float(so.median()) < 1e4 else so

def parse_uploaded_so(file):
    df = pd.read_csv(file)
    cols = [c.lower() for c in df.columns]
    if set(["date","ticker","shares"]).issubset(cols):
        # long format
        df.columns = cols
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        pivot = df.pivot(index="date", columns="ticker", values="shares").sort_index()
        return pivot
    # wide format: first col is date, rest tickers
    df.columns = ["date"] + [c.strip().upper() for c in df.columns[1:]]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.set_index("date").sort_index()

def fetch_yahoo_so_series(ticker):
    try:
        so = yf.Ticker(ticker).get_shares_full()
    except Exception:
        return None
    if so is None:
        return None
    s = pd.Series(so).dropna()
    if s.empty:
        return None
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return normalize_so_units(s)

def compute_flows(px: pd.Series, so: pd.Series, freq: str):
    pxw = px.loc[start:today].dropna()
    if pxw.empty or so is None or so.dropna().empty:
        return None, None
    so_al = so.reindex(pxw.index, method="ffill").dropna()
    if so_al.empty: return None, None
    dso = so_al.diff().fillna(0.0)
    flow_daily = dso * pxw.loc[so_al.index]
    flow = flow_daily if freq == "D" else flow_daily.resample("W-FRI").sum()
    return flow.to_frame(name="Flow ($)"), float(flow.sum())

# ========= Data =========
closes = batch_prices(TICKERS, start, today)
if closes.empty:
    st.error("Couldn’t download prices. Try again later.")
    st.stop()

freq = "W" if agg == "Weekly" else "D"

# Shares Outstanding source
if source == "Upload (CSV)":
    if uploaded is None:
        st.info("Upload a CSV with daily Shares Outstanding to compute flows.")
        so_table = pd.DataFrame()
    else:
        so_table = parse_uploaded_so(uploaded)
else:
    # Yahoo best-effort: build a table of SO series we can actually get
    series = {}
    for t in TICKERS:
        s = fetch_yahoo_so_series(t)
        if s is not None:
            series[t] = s
    so_table = pd.concat(series, axis=1) if series else pd.DataFrame()

# ========= Compute flows per ticker =========
rows, per_ticker, missing = [], {}, []
for t in TICKERS:
    if t not in closes.columns:
        missing.append((t, BASKET[t], "no_price"))
        continue

    so_series = None
    if not so_table.empty:
        # support mismatched indices/columns
        if t in so_table.columns.get_level_values(-1):
            # MultiIndex (from Yahoo concat)
            if isinstance(so_table.columns, pd.MultiIndex):
                so_series = so_table.xs(t, axis=1, level=-1).squeeze()
            else:
                so_series = so_table[t]
        elif t in so_table.columns:
            so_series = so_table[t]

    df_flow, net = compute_flows(closes[t], so_series, freq)
    if df_flow is None:
        missing.append((t, BASKET[t], "no_SO"))
        continue

    per_ticker[t] = df_flow
    rows.append({"Ticker": t, "Label": f"{BASKET[t]} ({t})", "Net Flow ($)": net})

if not rows:
    st.error("No ETFs had usable Shares Outstanding for the chosen source.\n"
             "Tip: switch to **Upload (CSV)** and drop in SO for the ETFs you care about.")
    if show_missing and missing:
        st.markdown("### Missing (audit)")
        st.dataframe(pd.DataFrame(missing, columns=["Ticker","Label","Reason"]), use_container_width=True)
    st.stop()

df = pd.DataFrame(rows).sort_values("Net Flow ($)", ascending=False).reset_index(drop=True)

# ========= UI =========
st.title("ETF Flows — Core 20")
src_label = "Upload" if source.startswith("Upload") else "Yahoo (best-effort)"
st.caption(f"Source: **{src_label}** • Period: **{lb_key}** (ending {today.date()}) • Aggregation: **{agg}** • Flows = ΔSO × Close")

# Bar chart
vals = df["Net Flow ($)"].to_numpy()
fig, ax = plt.subplots(figsize=(15, max(6, len(df)*0.42)))
colors = np.where(vals >= 0, "#2ca02c", "#d62728")
bars = ax.barh(df["Label"], vals, color=colors, alpha=0.9)
ax.invert_yaxis()
ax.set_xlabel("Net Flow ($ with k/M/B)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(tick_money))

x_lo, x_hi = min(vals.min(), 0.0), max(vals.max(), 0.0)
pad = (x_hi - x_lo) * 0.15 if x_hi != x_lo else 1.0
ax.set_xlim([x_lo - pad, x_hi + pad])

for bar, v in zip(bars, vals):
    txt = fmt_money(v, signed=True, d=2)
    x = bar.get_width()
    ha = "left" if x >= 0 else "right"
    off = 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    ax.text(x + (off if x >= 0 else -off),
            bar.get_y() + bar.get_height()/2, txt, va="center", ha=ha, fontsize=10)

plt.tight_layout()
st.pyplot(fig)

# Summary
c1, c2, c3 = st.columns(3)
total_in  = df.loc[df["Net Flow ($)"] > 0, "Net Flow ($)"].sum()
total_out = df.loc[df["Net Flow ($)"] < 0, "Net Flow ($)"].sum()
coverage  = len(df) / len(TICKERS)
c1.metric("Total Inflows",  fmt_money(total_in, True))
c2.metric("Total Outflows", fmt_money(total_out, True))
c3.metric("Coverage", f"{coverage:.0%}")

# Time series (Total & Cumulative)
st.markdown("### Flow Time Series")
panel = pd.concat([per_ticker[t]["Flow ($)"].rename(t) for t in per_ticker], axis=1).fillna(0.0)
panel["Total Flow"] = panel.sum(axis=1)
panel["Cumulative"] = panel["Total Flow"].cumsum()

t1, t2 = st.columns(2)
with t1:
    st.line_chart(panel["Total Flow"], height=260, use_container_width=True)
    st.caption("Total flow per period")
with t2:
    st.line_chart(panel["Cumulative"], height=260, use_container_width=True)
    st.caption("Cumulative flow over the lookback")

# Top in/out tables
st.markdown("### Top Inflows & Outflows")
L, R = st.columns(2)
with L:
    top_in = df.head(5).copy()
    top_in["Net Flow ($)"] = top_in["Net Flow ($)"].map(lambda v: fmt_money(v, True))
    st.table(top_in.set_index("Label")[["Net Flow ($)"]])
with R:
    top_out = df.sort_values("Net Flow ($)").head(5).copy()
    top_out["Net Flow ($)"] = top_out["Net Flow ($)"].map(lambda v: fmt_money(v, True))
    st.table(top_out.set_index("Label")[["Net Flow ($)"]])

# Missing audit
if show_missing and missing:
    st.markdown("### Missing (audit)")
    st.dataframe(pd.DataFrame(missing, columns=["Ticker","Label","Reason"]), use_container_width=True)

st.caption("© 2025 AD Fund Management LP")
