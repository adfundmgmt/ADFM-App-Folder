import streamlit as st
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ============================== App Config ==============================
TITLE = "Liquidity & Fed Policy Tracker"
st.set_page_config(page_title=TITLE, layout="wide")

SERIES = {
    "WALCL": {"fred": "WALCL", "label": "Fed Balance Sheet (WALCL)", "unit": "Millions USD", "to_bil": True},
    "RRP":   {"fred": "RRPONTSYD", "label": "ON RRP (RRPONTSYD)", "unit": "Billions USD", "to_bil": False},
    "TGA":   {"fred": "WDTGAL", "label": "Treasury General Account (WDTGAL)", "unit": "Millions USD", "to_bil": True},
    "EFFR":  {"fred": "EFFR", "label": "Effective Fed Funds Rate (EFFR)", "unit": "%", "to_bil": False},
    "NFCI":  {"fred": "NFCI", "label": "Chicago Fed NFCI (NFCI)", "unit": "Index", "to_bil": False},
}

LOOKBACKS = {
    "1y": 1,
    "2y": 2,
    "3y": 3,
    "5y": 5,
    "10y": 10,
    "25y": 25,
    "Max": None,
}

DEFAULT_LOOKBACK = "25y"
DEFAULT_SMOOTH = 5

# Rebase robustness
REBASE_BASE_WINDOW = 10
RRP_BASE_FLOOR_B = 5.0

# ============================== Styling ==============================
CSS = """
<style>
/* tighten top padding */
.block-container { padding-top: 1.2rem; padding-bottom: 2.0rem; }

/* nicer sidebar typography */
section[data-testid="stSidebar"] * { font-size: 0.95rem; }

/* metric row spacing */
[data-testid="stMetric"] { padding: 0.65rem 0.75rem; border: 1px solid rgba(49,51,63,0.12); border-radius: 14px; }
[data-testid="stMetric"] label { font-size: 0.85rem; opacity: 0.85; }

/* subtle captions */
.smallcap { font-size: 0.85rem; opacity: 0.75; }
.pill {
  display: inline-block;
  padding: 0.20rem 0.55rem;
  border-radius: 999px;
  border: 1px solid rgba(49,51,63,0.15);
  font-size: 0.85rem;
  opacity: 0.90;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ============================== Helpers ==============================
def fmt_b(x):
    return "N/A" if x is None or pd.isna(x) else f"{x:,.0f} B"

def fmt_b_delta(x):
    return "" if x is None or pd.isna(x) else f"{x:+,.0f} B"

def fmt_pct(x):
    return "N/A" if x is None or pd.isna(x) else f"{x:.2f}%"

def fmt_pct_delta(x):
    return "" if x is None or pd.isna(x) else f"{x:+.2f}%"

def fmt_nfci(x):
    return "N/A" if x is None or pd.isna(x) else f"{x:.3f}"

def fmt_nfci_delta(x):
    return "" if x is None or pd.isna(x) else f"{x:+.3f}"

def safe_delta(s: pd.Series, periods: int):
    if s is None or s.empty or len(s) <= periods:
        return None
    a = s.iloc[-1]
    b = s.iloc[-(periods + 1)]
    if pd.isna(a) or pd.isna(b):
        return None
    return a - b

def rolling_mean(s: pd.Series, window: int):
    if window <= 1:
        return s
    return s.rolling(window, min_periods=1).mean()

def rebase(series: pd.Series, base_window: int = REBASE_BASE_WINDOW, min_base=None) -> pd.Series:
    s = series.copy()
    if s.isna().all():
        return pd.Series(index=s.index, data=100.0)
    head = s.dropna().iloc[: max(1, base_window)]
    base = head.median() if not head.empty else s.dropna().iloc[0]
    if min_base is not None:
        base = max(float(base), float(min_base))
    if base == 0 or pd.isna(base):
        return pd.Series(index=s.index, data=100.0)
    return (s / base) * 100.0

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_fred(series_map: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    tickers = [v["fred"] for v in series_map.values()]
    try:
        raw = pdr.DataReader(tickers, "fred", start, end)
    except Exception:
        return pd.DataFrame()

    inv = {v["fred"]: k for k, v in series_map.items()}
    raw = raw.rename(columns=inv)
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()
    return raw

def align_to_business_days(df: pd.DataFrame, ffill_limit_days: int = 10) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.date_range(df.index.min(), df.index.max(), freq="B")
    out = df.reindex(idx)
    # cap forward fill so stale weekly points do not drift too far
    out = out.ffill(limit=ffill_limit_days)
    return out

def regime_pill(netliq_4w: float, nfci_level: float, effr_4w: float):
    tags = []
    if netliq_4w is None or pd.isna(netliq_4w):
        tags.append("Liquidity: unknown")
    elif netliq_4w > 0:
        tags.append("Liquidity: easing")
    else:
        tags.append("Liquidity: draining")

    if nfci_level is None or pd.isna(nfci_level):
        tags.append("Conditions: unknown")
    elif nfci_level > 0:
        tags.append("Conditions: tight")
    else:
        tags.append("Conditions: loose")

    if effr_4w is None or pd.isna(effr_4w):
        tags.append("Policy: unknown")
    elif effr_4w > 0:
        tags.append("Policy: tightening")
    elif effr_4w < 0:
        tags.append("Policy: easing")
    else:
        tags.append("Policy: flat")

    return " | ".join(tags)

# ============================== Sidebar ==============================
with st.sidebar:
    st.markdown(f"### {TITLE}")
    st.markdown(
        """
        **Net Liquidity (B)** = WALCL(B) − RRP(B) − TGA(B)

        Use this to track liquidity impulse, policy stance, and financial conditions.
        """
    )
    st.markdown("---")

    lookback_label = st.selectbox("Lookback", list(LOOKBACKS.keys()), index=list(LOOKBACKS.keys()).index(DEFAULT_LOOKBACK))
    smooth = st.number_input("Smoothing (days)", min_value=1, max_value=30, value=DEFAULT_SMOOTH, step=1)

    align = st.selectbox("Frequency alignment", ["Business days (recommended)", "Raw FRED index"], index=0)

    with st.expander("Advanced"):
        ffill_limit = st.slider("Forward-fill cap (business days)", 1, 30, 10, 1)
        show_range_slider = st.checkbox("Show range slider", value=False)
        show_download = st.checkbox("Enable download", value=True)

    st.markdown('<div class="smallcap">Source: FRED via pandas-datareader</div>', unsafe_allow_html=True)

# ============================== Data Build ==============================
today = pd.Timestamp.today().normalize()

if LOOKBACKS[lookback_label] is None:
    start_fetch = pd.Timestamp("1980-01-01")
else:
    years = LOOKBACKS[lookback_label]
    # pull extra headroom for smooth and rebases
    start_fetch = (today - pd.DateOffset(years=years)) - pd.DateOffset(months=9)

raw = fetch_fred(SERIES, start_fetch, today)
if raw.empty:
    st.error("FRED fetch failed (empty response).")
    st.stop()

missing = [k for k in SERIES.keys() if k not in raw.columns]
if missing:
    st.warning(f"Missing series: {', '.join(missing)}")

df = raw.copy()

if align.startswith("Business days"):
    df = align_to_business_days(df, ffill_limit_days=ffill_limit)

# Trim to lookback window after alignment
if LOOKBACKS[lookback_label] is not None:
    start_lb = today - pd.DateOffset(years=LOOKBACKS[lookback_label])
    df = df[df.index >= start_lb]

# Require liquidity components
need = ["WALCL", "RRP", "TGA"]
df = df.dropna(subset=[c for c in need if c in df.columns])
if df.empty:
    st.error("No data available for selected settings.")
    st.stop()

# Units to billions for liquidity components
df["WALCL_b"] = df["WALCL"] / 1000.0
df["TGA_b"] = df["TGA"] / 1000.0
df["RRP_b"] = df["RRP"]  # already billions
df["NetLiq_b"] = df["WALCL_b"] - df["RRP_b"] - df["TGA_b"]

# Smoothed versions
df["WALCL_b_s"] = rolling_mean(df["WALCL_b"], smooth)
df["TGA_b_s"] = rolling_mean(df["TGA_b"], smooth)
df["RRP_b_s"] = rolling_mean(df["RRP_b"], smooth)
df["NetLiq_b_s"] = rolling_mean(df["NetLiq_b"], smooth)

if "EFFR" in df.columns:
    df["EFFR_s"] = rolling_mean(df["EFFR"], smooth)
if "NFCI" in df.columns:
    df["NFCI_s"] = rolling_mean(df["NFCI"], smooth)

# Rebased components
reb = pd.DataFrame(index=df.index)
reb["WALCL_idx"] = rebase(df["WALCL_b_s"])
reb["RRP_idx"] = rebase(df["RRP_b_s"], min_base=RRP_BASE_FLOOR_B)
reb["TGA_idx"] = rebase(df["TGA_b_s"])

# ============================== Header + KPIs ==============================
st.markdown("## Dashboard")

asof = df.index.max()
st.caption(f"As of {asof:%Y-%m-%d}")

netliq = df["NetLiq_b"].iloc[-1]
walcl = df["WALCL_b"].iloc[-1]
rrp = df["RRP_b"].iloc[-1]
tga = df["TGA_b"].iloc[-1]
effr = df["EFFR"].iloc[-1] if "EFFR" in df.columns else pd.NA
nfci = df["NFCI"].iloc[-1] if "NFCI" in df.columns else pd.NA

# Deltas on aligned series: 1w and 4w in business days
d_netliq_1w = safe_delta(df["NetLiq_b"], 5)
d_netliq_4w = safe_delta(df["NetLiq_b"], 21)
d_effr_4w = safe_delta(df["EFFR"], 21) if "EFFR" in df.columns else None
d_nfci_4w = safe_delta(df["NFCI"], 21) if "NFCI" in df.columns else None

pill_text = regime_pill(d_netliq_4w, nfci, d_effr_4w)
st.markdown(f'<span class="pill">{pill_text}</span>', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Net Liquidity", fmt_b(netliq), fmt_b_delta(d_netliq_1w))
c2.metric("WALCL", fmt_b(walcl))
c3.metric("RRP", fmt_b(rrp))
c4.metric("TGA", fmt_b(tga))
c5.metric("EFFR", fmt_pct(effr), fmt_pct_delta(d_effr_4w))
c6.metric("NFCI", fmt_nfci(nfci), fmt_nfci_delta(d_nfci_4w), help=">0 = tighter conditions")

# ============================== Main Views ==============================
tab_liq, tab_policy, tab_data, tab_method = st.tabs(["Liquidity", "Policy + Conditions", "Data", "Methodology"])

with tab_liq:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=("Net Liquidity (B)", "Components rebased to 100")
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["NetLiq_b_s"], name="Net Liquidity",
            hovertemplate="Date=%{x|%Y-%m-%d}<br>NetLiq=%{y:,.0f} B<extra></extra>",
            line=dict(width=2)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=reb.index, y=reb["WALCL_idx"], name="WALCL idx",
            hovertemplate="Date=%{x|%Y-%m-%d}<br>WALCL idx=%{y:.1f}<extra></extra>"
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=reb.index, y=reb["RRP_idx"], name="RRP idx",
            hovertemplate="Date=%{x|%Y-%m-%d}<br>RRP idx=%{y:.1f}<extra></extra>"
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=reb.index, y=reb["TGA_idx"], name="TGA idx",
            hovertemplate="Date=%{x|%Y-%m-%d}<br>TGA idx=%{y:.1f}<extra></extra>"
        ),
        row=2, col=1
    )

    fig.update_yaxes(title_text="Billions", row=1, col=1)
    fig.update_yaxes(title_text="Index", row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=820,
        legend=dict(orientation="h", x=0, y=1.12),
        margin=dict(l=55, r=30, t=60, b=55),
        hovermode="x unified",
    )
    fig.update_xaxes(rangeslider_visible=show_range_slider)

    st.plotly_chart(fig, use_container_width=True)

with tab_policy:
    fig2 = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Effective Fed Funds Rate (EFFR)", "Financial Conditions (NFCI)")
    )

    if "EFFR_s" in df.columns:
        fig2.add_trace(
            go.Scatter(
                x=df.index, y=df["EFFR_s"], name="EFFR",
                hovertemplate="Date=%{x|%Y-%m-%d}<br>EFFR=%{y:.2f}%<extra></extra>"
            ),
            row=1, col=1
        )

    if "NFCI_s" in df.columns:
        fig2.add_trace(
            go.Scatter(
                x=df.index, y=df["NFCI_s"], name="NFCI",
                hovertemplate="Date=%{x|%Y-%m-%d}<br>NFCI=%{y:.3f}<extra></extra>",
                line=dict(width=2)
            ),
            row=2, col=1
        )
        # Add a zero line for interpretation
        fig2.add_hline(y=0, line_width=1, line_dash="dot", row=2, col=1)

    fig2.update_yaxes(title_text="%", row=1, col=1)
    fig2.update_yaxes(title_text="Level", row=2, col=1)

    fig2.update_layout(
        template="plotly_white",
        height=700,
        legend=dict(orientation="h", x=0, y=1.12),
        margin=dict(l=55, r=30, t=60, b=55),
        hovermode="x unified",
    )
    fig2.update_xaxes(rangeslider_visible=show_range_slider)

    st.plotly_chart(fig2, use_container_width=True)

with tab_data:
    st.markdown("### Aligned dataset")
    out = pd.DataFrame(index=df.index)
    out.index.name = "Date"
    out["WALCL_B"] = df["WALCL_b"]
    out["RRP_B"] = df["RRP_b"]
    out["TGA_B"] = df["TGA_b"]
    out["NetLiq_B"] = df["NetLiq_b"]
    if "EFFR" in df.columns:
        out["EFFR_%"] = df["EFFR"]
    if "NFCI" in df.columns:
        out["NFCI"] = df["NFCI"]

    st.dataframe(out.tail(400), use_container_width=True, height=420)

    if 'show_download' in locals() and show_download:
        st.download_button(
            "Download CSV",
            out.to_csv(),
            file_name="liquidity_policy_tracker.csv",
            mime="text/csv"
        )

with tab_method:
    st.markdown(
        f"""
        **Definitions**

        Net Liquidity (B) = WALCL(B) − RRP(B) − TGA(B)

        **Unit handling**
        - WALCL and TGA are reported in millions, converted to billions.
        - RRPONTSYD is reported in billions.

        **Frequency alignment**
        - If you select business-day alignment, series are reindexed to business days and forward-filled with a cap of {ffill_limit} business days.
        - This prevents very stale weekly values from drifting too far when you are looking at short-term deltas.

        **Smoothing**
        - Rolling mean with a {smooth}-day window.

        **Rebasing**
        - Components are rebased to 100 using the median of the first {REBASE_BASE_WINDOW} observations.
        - RRP has a base floor of {RRP_BASE_FLOOR_B}B to avoid nonsense rebases when RRP is near zero.
        """
    )

st.caption("© 2025 AD Fund Management LP")
