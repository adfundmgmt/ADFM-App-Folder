############################################################
# Built by AD Fund Management LP
# 10-Year Nominal and Real Yield Dashboard
############################################################

from __future__ import annotations

from typing import List, Tuple, Optional
import io

import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from bs4 import BeautifulSoup

# ── Constants ─────────────────────────────────────────────
FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
REAL_SERIES = "DFII10"
NOM_SERIES = "DGS10"
REC_SERIES = "USREC"

DEFAULT_WIN = 63
DEFAULT_EASE_BP = 40
DEFAULT_TIGHT_BP = -40

REQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AD-Fund-Yield-Dashboard/2.0)"
}

# ── Streamlit Setup ──────────────────────────────────────
st.set_page_config(page_title="10-Year Yield Dashboard", layout="wide")
st.title("10-Year Nominal and Real Yield Dashboard")

# ── Helper Functions ─────────────────────────────────────
def fmt_pct(x: Optional[float], decimals: int = 2) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:.{decimals}f}%"

def fmt_bp(x: Optional[float], decimals: int = 0) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:+.{decimals}f} bp"

def last_delta(series: pd.Series) -> float:
    if len(series) < 2:
        return pd.NA
    return series.iloc[-1] - series.iloc[-2]

def contiguous(mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    segments: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    start_dt: Optional[pd.Timestamp] = None

    for ts, flag in mask.items():
        flag = bool(flag)
        if flag and start_dt is None:
            start_dt = ts
        elif (not flag) and start_dt is not None:
            segments.append((start_dt, ts))
            start_dt = None

    if start_dt is not None:
        segments.append((start_dt, mask.index[-1]))

    return segments

# ── Data Fetching ────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fred_series(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    params = {
        "id": series_id,
        "cosd": start.strftime("%Y-%m-%d"),
        "coed": end.strftime("%Y-%m-%d"),
    }

    r = requests.get(FRED_BASE, params=params, headers=REQ_HEADERS, timeout=20)
    r.raise_for_status()

    text = r.text.strip()
    if not text:
        raise ValueError(f"Empty FRED response for {series_id}")

    # Guard against HTML or non-CSV responses
    lower_text = text[:200].lower()
    if lower_text.startswith("<!doctype html") or lower_text.startswith("<html"):
        raise ValueError(f"FRED returned HTML instead of CSV for {series_id}")

    df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip() for c in df.columns]

    if "DATE" not in df.columns:
        raise ValueError(
            f"FRED response missing DATE column for {series_id}. Columns: {df.columns.tolist()}"
        )

    # Prefer exact series column, otherwise use the first non-DATE column
    value_col = series_id if series_id in df.columns else None
    if value_col is None:
        non_date_cols = [c for c in df.columns if c != "DATE"]
        if not non_date_cols:
            raise ValueError(
                f"FRED response missing value column for {series_id}. Columns: {df.columns.tolist()}"
            )
        value_col = non_date_cols[0]

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    df = df.dropna(subset=["DATE"]).set_index("DATE").sort_index()
    s = df[value_col].ffill().dropna()
    s.name = series_id

    if s.empty:
        raise ValueError(f"No usable observations returned for {series_id}")

    return s

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fomc_dates() -> List[pd.Timestamp]:
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    try:
        r = requests.get(url, headers=REQ_HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        dates: List[pd.Timestamp] = []
        for td in soup.select("td.eventdate"):
            txt = td.get_text(" ", strip=True)
            dt = pd.to_datetime(txt, errors="coerce")
            if pd.notnull(dt):
                dates.append(pd.Timestamp(dt).normalize())

        return sorted(set(dates))
    except Exception:
        return []

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_cpi_dates() -> List[pd.Timestamp]:
    url = "https://www.bls.gov/schedule/news_release/cpi.htm"
    try:
        r = requests.get(url, headers=REQ_HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        dates: List[pd.Timestamp] = []
        for td in soup.select("table.nrtable tbody tr td"):
            txt = td.get_text(" ", strip=True)
            dt = pd.to_datetime(txt, errors="coerce")
            if pd.notnull(dt):
                dates.append(pd.Timestamp(dt).normalize())

        return sorted(set(dates))
    except Exception:
        return []

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_recession_bands(start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    try:
        rec = fetch_fred_series(REC_SERIES, start, end)
        daily_index = pd.date_range(rec.index.min(), rec.index.max(), freq="D")
        rec = rec.reindex(daily_index).ffill()

        mask = rec > 0.5
        bands = contiguous(mask)
        return [(s.normalize(), e.normalize()) for s, e in bands]
    except Exception:
        return []

# ── Transform Layer ──────────────────────────────────────
def build_yield_dataframe(
    nominal: pd.Series,
    real: pd.Series,
    start: pd.Timestamp,
    win: int,
) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.concat(
        [
            nominal.rename("Nominal"),
            real.rename("Real"),
        ],
        axis=1,
    ).dropna()

    df = df[df.index >= start].copy()
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    # Inverted real-yield momentum in bp
    mom = -(df["Real"] - df["Real"].shift(win)) * 100.0
    mom = mom.dropna()

    if mom.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    df = df.loc[mom.index].copy()
    df["InflationCompProxy"] = df["Nominal"] - df["Real"]

    return df, mom

def classify_regime(mom_value: float, ease_bp: int, tight_bp: int) -> tuple[str, str]:
    if mom_value > ease_bp:
        return "Easing", "#16a34a"
    if mom_value < tight_bp:
        return "Tightening", "#dc2626"
    return "Neutral", "#6b7280"

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.header("About This Tool")
    st.markdown(
        """
        Track the 10-year nominal Treasury yield against the 10-year TIPS-implied real yield with a regime-focused lens.

        What it shows  
        • Nominal vs real 10-year yields  
        • Inverted rolling real-yield momentum in bp as a simple market-rate easing or tightening gauge  
        • Optional inflation-compensation proxy panel and recession shading  
        • Optional FOMC and CPI event markers

        Data  
        • FRED DGS10 for nominal 10Y  
        • FRED DFII10 for real 10Y  
        • FRED USREC for recession shading
        """
    )
    st.markdown("---")

    st.header("Settings")
    lookback = st.selectbox("Lookback Window", ["2y", "3y", "5y", "10y"], index=2)
    start_years = int(lookback[:-1])

    win = st.number_input(
        "Momentum Window (trading days)",
        min_value=21,
        max_value=252,
        value=DEFAULT_WIN,
        step=1,
    )
    ease_bp = st.number_input(
        "Easing Threshold (bp)",
        min_value=10,
        max_value=200,
        value=DEFAULT_EASE_BP,
        step=5,
    )
    tight_bp = st.number_input(
        "Tightening Threshold (bp)",
        min_value=-200,
        max_value=-10,
        value=DEFAULT_TIGHT_BP,
        step=5,
    )

    st.markdown("---")
    st.subheader("Overlays")
    show_regime = st.checkbox("Show Regime Shading", value=True)
    show_spread = st.checkbox("Show Inflation Compensation Proxy", value=False)
    show_fomc = st.checkbox("Show FOMC Markers", value=True)
    show_cpi = st.checkbox("Show CPI Markers", value=False)
    show_recession = st.checkbox("Show Recession Shading", value=True)

    st.markdown("---")
    st.caption("Source: FRED and official public event calendars")

# ── Load Data ────────────────────────────────────────────
today = pd.Timestamp.today().normalize()
start = today - pd.DateOffset(years=start_years)
extended_start = start - pd.Timedelta(days=max(180, win * 3))

try:
    nominal = fetch_fred_series(NOM_SERIES, extended_start, today)
except Exception as e:
    st.error(f"Failed loading {NOM_SERIES}: {e}")
    st.stop()

try:
    real = fetch_fred_series(REAL_SERIES, extended_start, today)
except Exception as e:
    st.error(f"Failed loading {REAL_SERIES}: {e}")
    st.stop()

df, mom = build_yield_dataframe(nominal=nominal, real=real, start=start, win=win)

if df.empty or mom.empty:
    st.warning("No overlapping yield data for the selected lookback.")
    st.stop()

latest = {
    "date": df.index[-1],
    "nominal": df["Nominal"].iloc[-1],
    "real": df["Real"].iloc[-1],
    "spread": df["InflationCompProxy"].iloc[-1],
    "mom": mom.iloc[-1],
    "mom_prev": mom.iloc[-2] if len(mom) > 1 else pd.NA,
    "d_nominal": last_delta(df["Nominal"]),
    "d_real": last_delta(df["Real"]),
    "d_spread": last_delta(df["InflationCompProxy"]),
}

regime_label, regime_color = classify_regime(latest["mom"], ease_bp, tight_bp)

# ── Metrics Row ──────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns([2, 2, 2.2, 2, 2.2])

c1.metric(
    "Nominal 10Y Yield",
    fmt_pct(latest["nominal"]),
    fmt_pct(latest["d_nominal"]),
)

c2.metric(
    "Real 10Y Yield",
    fmt_pct(latest["real"]),
    fmt_pct(latest["d_real"]),
)

c3.metric(
    f"Inverted {win}-Day Real Yield Momentum",
    fmt_bp(latest["mom"]),
    fmt_bp(latest["mom"] - latest["mom_prev"]) if pd.notna(latest["mom_prev"]) else "N/A",
    help=f"-(Current real yield - real yield {win} observations ago) × 100",
)

c4.metric(
    "Inflation Compensation Proxy",
    fmt_pct(latest["spread"]),
    fmt_pct(latest["d_spread"]),
    help="Nominal 10Y minus real 10Y",
)

c5.markdown(
    f"""
    <div style="padding-top:0.35rem;">
        <div style="
            display:inline-block;
            padding:0.35rem 0.7rem;
            border-radius:999px;
            background:{regime_color};
            color:white;
            font-weight:600;
            font-size:0.95rem;
        ">{regime_label}</div>
        <div style="margin-top:0.45rem; color:#6b7280; font-size:0.9rem;">
            as of {latest["date"].date()}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Figure Builder ───────────────────────────────────────
nrows = 3 if show_spread else 2
row_heights = [0.42, 0.33, 0.25] if show_spread else [0.58, 0.42]

subplot_titles = [
    "Nominal vs Real 10-Year Yield",
    f"{win}-Day Inverted Real Yield Momentum",
]
if show_spread:
    subplot_titles.append("10Y Inflation Compensation Proxy")

fig = make_subplots(
    rows=nrows,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=row_heights,
    subplot_titles=subplot_titles,
)

# Top panel
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Nominal"],
        name="Nominal 10Y",
        mode="lines",
        line=dict(color="#2563eb", width=2.2),
        hovertemplate="%{x|%Y-%m-%d}<br>Nominal 10Y: %{y:.2f}%<extra></extra>",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Real"],
        name="Real 10Y",
        mode="lines",
        line=dict(color="#f59e0b", width=2.2),
        hovertemplate="%{x|%Y-%m-%d}<br>Real 10Y: %{y:.2f}%<extra></extra>",
    ),
    row=1,
    col=1,
)

# Momentum panel
fig.add_trace(
    go.Scatter(
        x=mom.index,
        y=mom,
        name="Inverted Real Yield Momentum",
        mode="lines",
        line=dict(color="#dc2626", width=1.9, dash="dash"),
        hovertemplate="%{x|%Y-%m-%d}<br>Momentum: %{y:.0f} bp<extra></extra>",
    ),
    row=2,
    col=1,
)

for level, label in [
    (ease_bp, f"+{ease_bp} bp"),
    (0, "0 bp"),
    (tight_bp, f"{tight_bp} bp"),
]:
    fig.add_hline(
        y=level,
        line_width=1,
        line_dash="dot",
        line_color="rgba(100,100,100,0.7)",
        row=2,
        col=1,
    )
    fig.add_annotation(
        x=df.index.max(),
        y=level,
        xref="x2" if nrows >= 2 else "x",
        yref="y2",
        text=label,
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=10, color="rgba(80,80,80,0.9)"),
    )

# Regime shading
if show_regime:
    easing_mask = mom > ease_bp
    tightening_mask = mom < tight_bp

    for s, e in contiguous(easing_mask):
        fig.add_vrect(
            x0=s,
            x1=e,
            fillcolor="rgba(22,163,74,0.10)",
            line_width=0,
            row=2,
            col=1,
        )

    for s, e in contiguous(tightening_mask):
        fig.add_vrect(
            x0=s,
            x1=e,
            fillcolor="rgba(220,38,38,0.10)",
            line_width=0,
            row=2,
            col=1,
        )

# Spread panel
if show_spread:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["InflationCompProxy"],
            name="Inflation Compensation Proxy",
            mode="lines",
            line=dict(color="#16a34a", width=2.0),
            hovertemplate="%{x|%Y-%m-%d}<br>Proxy: %{y:.2f}%<extra></extra>",
        ),
        row=3,
        col=1,
    )

# Overlays
if show_recession:
    recession_bands = fetch_recession_bands(df.index.min(), df.index.max())
    for rs, re in recession_bands:
        if re >= df.index.min() and rs <= df.index.max():
            fig.add_vrect(
                x0=max(rs, df.index.min()),
                x1=min(re, df.index.max()),
                fillcolor="rgba(120,120,120,0.15)",
                line_width=0,
            )

if show_fomc:
    for dt in fetch_fomc_dates():
        if df.index.min() <= dt <= df.index.max():
            fig.add_vline(
                x=dt,
                line_width=1,
                line_dash="dashdot",
                line_color="rgba(37,99,235,0.25)",
            )

if show_cpi:
    for dt in fetch_cpi_dates():
        if df.index.min() <= dt <= df.index.max():
            fig.add_vline(
                x=dt,
                line_width=1,
                line_dash="dot",
                line_color="rgba(245,158,11,0.25)",
            )

# Axis and layout
fig.update_yaxes(title_text="Yield (%)", tickformat=".2f", row=1, col=1)
fig.update_yaxes(title_text="bp", tickformat=".0f", row=2, col=1)
if show_spread:
    fig.update_yaxes(title_text="Spread (%)", tickformat=".2f", row=3, col=1)

fig.update_xaxes(
    title_text="Date",
    tickformat="%b-%y",
    showgrid=False,
    row=nrows,
    col=1,
)

fig.update_layout(
    template="plotly_white",
    height=840 if show_spread else 720,
    margin=dict(l=60, r=30, t=70, b=50),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        x=0,
        y=1.08,
        xanchor="left",
        bgcolor="rgba(255,255,255,0.7)",
    ),
)

st.plotly_chart(fig, use_container_width=True)

# ── Download Section ─────────────────────────────────────
with st.expander("Download Data"):
    df_out = df.copy()
    df_out[f"Inverted_{win}d_Real_Yield_Momentum_bp"] = mom
    st.download_button(
        label="Download Yield Data (CSV)",
        data=df_out.to_csv(index=True, index_label="Date"),
        file_name="10yr_yield_dashboard.csv",
        mime="text/csv",
    )

# ── Methodology ──────────────────────────────────────────
with st.expander("Methodology and Interpretation", expanded=False):
    st.markdown(
        f"""
**Series**  
• Nominal 10Y: FRED `{NOM_SERIES}`  
• Real 10Y: FRED `{REAL_SERIES}`  
• Recessions: FRED `{REC_SERIES}`  

**Inverted real-yield momentum**  
• Formula: `-(current real yield - real yield {win} observations ago) × 100`  
• Units: basis points  
• Higher readings imply real yields have fallen over the lookback window  
• Lower readings imply real yields have risen over the lookback window  

**Regime thresholds**  
• Easing: momentum greater than {ease_bp} bp  
• Neutral: between {tight_bp} bp and {ease_bp} bp  
• Tightening: momentum below {tight_bp} bp  

**Inflation compensation proxy**  
• Nominal 10Y minus real 10Y  
• This is a rough market-implied inflation compensation measure, not a perfect breakeven series  

**Overlay notes**  
• FOMC and CPI markers come from public official calendars and may occasionally fail if source page structure changes  
• Recession shading is derived from FRED `USREC`
"""
    )

st.caption("© 2026 AD Fund Management LP")
