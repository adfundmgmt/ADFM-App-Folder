import time
from io import StringIO
from typing import Dict, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# ---------------- Config ----------------
TITLE = "Liquidity & Fed Policy Tracker"

SERIES = {
    "WALCL": "Fed Balance Sheet",
    "RRPONTSYD": "ON RRP",
    "WDTGAL": "Treasury General Account",
    "EFFR": "Effective Fed Funds Rate",
    "NFCI": "Chicago Fed NFCI",
}

REQUIRED_SERIES = ["WALCL", "RRPONTSYD", "WDTGAL"]
DEFAULT_SMOOTH = 5
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Fail fast. The old version could sit inside one Streamlit spinner for too long.
REQUEST_TIMEOUT = (4, 10)  # connect timeout, read timeout
MAX_RETRIES = 2
BACKOFF_SECONDS = 0.75

st.set_page_config(page_title=TITLE, layout="wide")

# ---------------- UI ----------------
CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1550px;
    }

    .adfm-title {
        font-size: 1.85rem;
        font-weight: 750;
        margin-bottom: 0.1rem;
        color: #111827;
    }

    .adfm-subtitle {
        font-size: 0.96rem;
        color: #6b7280;
        margin-bottom: 1.1rem;
    }

    .section-title {
        font-size: 1.02rem;
        font-weight: 750;
        color: #111827;
        margin-top: 0.35rem;
        margin-bottom: 0.45rem;
    }

    .section-subtitle {
        font-size: 0.88rem;
        color: #6b7280;
        margin-bottom: 0.85rem;
    }

    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 13px 15px 10px 15px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
        min-height: 91px;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.045em;
        margin-bottom: 0.45rem;
    }

    .metric-value {
        font-size: 1.42rem;
        font-weight: 750;
        color: #111827;
        line-height: 1.08;
    }

    .metric-footnote {
        font-size: 0.76rem;
        color: #9ca3af;
        margin-top: 0.42rem;
    }

    .info-box {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 0.8rem;
    }

    .warning-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 14px;
        padding: 13px 15px;
        margin-bottom: 0.9rem;
        color: #7c2d12;
        font-size: 0.9rem;
    }

    .stDownloadButton button {
        border-radius: 10px;
        font-weight: 650;
    }

    .stDataFrame {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        overflow: hidden;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown(f"<div class='adfm-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='adfm-subtitle'>Balance-sheet liquidity, funding drains, policy rate, and financial conditions.</div>",
    unsafe_allow_html=True,
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### About This Tool")
    st.markdown(
        """
Tracks the Fed balance sheet, ON RRP, Treasury cash balance, EFFR, and NFCI.

The core liquidity proxy is:

**Net Liquidity = WALCL - ON RRP - TGA**

WALCL and TGA are converted from millions to billions. ON RRP is already in billions.
        """
    )

    st.divider()
    st.markdown("### Controls")

    lookback = st.selectbox(
        "Lookback",
        ["1y", "2y", "3y", "5y", "10y"],
        index=2,
    )
    years = int(lookback.replace("y", ""))

    smooth_days = st.number_input(
        "Smoothing window",
        min_value=1,
        max_value=30,
        value=DEFAULT_SMOOTH,
        step=1,
    )

    smooth_policy = st.checkbox("Smooth EFFR and NFCI", value=False)
    show_raw_table = st.checkbox("Show raw data table", value=True)

    st.caption("Source: public FRED CSV endpoints. No API key required.")

# ---------------- Helpers ----------------
def _clean_fred_csv(text: str, series_id: str) -> pd.Series:
    df = pd.read_csv(StringIO(text))

    if df.empty or len(df.columns) < 2:
        raise ValueError(f"No usable CSV returned for {series_id}")

    date_col = df.columns[0]
    value_col = df.columns[1]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col].replace(".", pd.NA), errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    if df.empty:
        raise ValueError(f"No valid dates returned for {series_id}")

    s = pd.Series(df[value_col].values, index=df[date_col], name=series_id)
    s = s[~s.index.duplicated(keep="last")]
    s = s.dropna()

    if s.empty:
        raise ValueError(f"No numeric observations returned for {series_id}")

    return s


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_fred_series(series_id: str, start_iso: str, end_iso: str) -> pd.Series:
    """Fetch one FRED series with short timeouts and explicit retries."""
    params = {
        "id": series_id,
        "cosd": start_iso,
        "coed": end_iso,
    }
    headers = {
        "User-Agent": "ADFM-Liquidity-Tracker/1.0",
        "Accept": "text/csv,*/*;q=0.8",
    }

    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(
                FRED_URL,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()

            if "observation_date" not in r.text[:250] and series_id not in r.text[:250]:
                raise ValueError(f"Unexpected FRED response for {series_id}")

            return _clean_fred_csv(r.text, series_id)

        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_SECONDS * attempt)

    raise RuntimeError(f"{series_id} failed after {MAX_RETRIES} attempts: {last_error}")


def load_all_series(start: pd.Timestamp, end: pd.Timestamp) -> Tuple[Dict[str, pd.Series], Dict[str, str]]:
    raw: Dict[str, pd.Series] = {}
    errors: Dict[str, str] = {}

    start_iso = start.strftime("%Y-%m-%d")
    end_iso = end.strftime("%Y-%m-%d")

    progress = st.progress(0)
    status_line = st.empty()

    items = list(SERIES.items())
    for i, (series_id, label) in enumerate(items, start=1):
        status_line.caption(f"Loading {label} ({series_id})")
        try:
            raw[series_id] = fetch_fred_series(series_id, start_iso, end_iso)
        except Exception as exc:
            errors[series_id] = str(exc)
            raw[series_id] = pd.Series(dtype="float64", name=series_id)
        progress.progress(i / len(items))

    progress.empty()
    status_line.empty()

    return raw, errors


def smooth(s: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if window <= 1:
        return s
    return s.rolling(window, min_periods=1).mean()


def rebase(s: pd.Series, floor: Optional[float] = None) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    valid = s.dropna()

    if valid.empty:
        return pd.Series(index=s.index, dtype="float64", name=s.name)

    base = valid.iloc[:10].median()
    if floor is not None:
        base = max(base, floor)

    if pd.isna(base) or base == 0:
        return pd.Series(index=s.index, dtype="float64", name=s.name)

    return (s / base) * 100.0


def chg(series: pd.Series, periods: int) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if len(valid) <= periods:
        return float("nan")
    return valid.iloc[-1] - valid.iloc[-1 - periods]


def fmt_b(x: float) -> str:
    return "N/A" if pd.isna(x) else f"{x:,.0f}B"


def fmt_delta_b(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:,.0f}B"


def fmt_pct(x: float) -> str:
    return "N/A" if pd.isna(x) else f"{x:.2f}%"


def fmt_num(x: float) -> str:
    return "N/A" if pd.isna(x) else f"{x:.3f}"


def metric_card(label: str, value: str, footnote: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-footnote">{footnote}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- Data ----------------
today = pd.Timestamp.today().normalize()
start = today - pd.DateOffset(years=years)

raw, errors = load_all_series(start, today)

missing_required = [series_id for series_id in REQUIRED_SERIES if raw.get(series_id, pd.Series(dtype="float64")).empty]

if missing_required:
    st.error(f"Required FRED series failed to load: {', '.join(missing_required)}")
    st.markdown(
        """
        <div class="warning-box">
        The app is no longer stuck. FRED did not return the required liquidity series before the timeout.
        This usually means the runtime cannot reach fred.stlouisfed.org, the network is blocking outbound requests,
        or FRED is temporarily slow. The failed series and exact errors are below.
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Error details", expanded=True):
        for series_id, message in errors.items():
            st.write(f"**{series_id}:** {message}")
    st.stop()

# Keep optional series optional. If EFFR or NFCI fails, the dashboard still loads.
def optional_series(series_id: str) -> pd.Series:
    s = raw.get(series_id, pd.Series(dtype="float64", name=series_id))
    if s.empty:
        return pd.Series(dtype="float64", name=series_id)
    return s

frames = [
    raw["WALCL"].rename("WALCL"),
    raw["RRPONTSYD"].rename("RRP"),
    raw["WDTGAL"].rename("TGA"),
    optional_series("EFFR").rename("EFFR"),
    optional_series("NFCI").rename("NFCI"),
]

df = pd.concat(frames, axis=1).sort_index()

# Convert the irregular FRED calendar into a clean daily calendar.
# Weekly balance-sheet/TGA/NFCI observations carry forward until the next release.
daily_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
df = df.reindex(daily_index).ffill()
df = df[df.index >= start].copy()
df = df.dropna(subset=["WALCL", "RRP", "TGA"])

if df.empty:
    st.error("The working dataframe is empty after cleaning and alignment.")
    with st.expander("Error details"):
        st.write(errors)
    st.stop()

# ---------------- Units ----------------
# FRED units:
# WALCL: millions of dollars. WDTGAL: millions of dollars. RRPONTSYD: billions of dollars.
df["WALCL_b"] = pd.to_numeric(df["WALCL"], errors="coerce") / 1000.0
df["RRP_b"] = pd.to_numeric(df["RRP"], errors="coerce")
df["TGA_b"] = pd.to_numeric(df["TGA"], errors="coerce") / 1000.0
df["EFFR"] = pd.to_numeric(df["EFFR"], errors="coerce")
df["NFCI"] = pd.to_numeric(df["NFCI"], errors="coerce")

df["NetLiq"] = df["WALCL_b"] - df["RRP_b"] - df["TGA_b"]

# ---------------- Smoothed ----------------
df["NetLiq_s"] = smooth(df["NetLiq"], smooth_days)
df["WALCL_s"] = smooth(df["WALCL_b"], smooth_days)
df["RRP_s"] = smooth(df["RRP_b"], smooth_days)
df["TGA_s"] = smooth(df["TGA_b"], smooth_days)

df["EFFR_s"] = smooth(df["EFFR"], smooth_days) if smooth_policy else df["EFFR"]
df["NFCI_s"] = smooth(df["NFCI"], smooth_days) if smooth_policy else df["NFCI"]

reb = pd.DataFrame(index=df.index)
reb["WALCL_idx"] = rebase(df["WALCL_b"])
reb["RRP_idx"] = rebase(df["RRP_b"], floor=5.0)
reb["TGA_idx"] = rebase(df["TGA_b"])

latest = df.iloc[-1]
latest_date = df.index[-1].strftime("%b %d, %Y")

# ---------------- Snapshot ----------------
st.markdown("<div class='section-title'>Snapshot</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='section-subtitle'>Latest available reading: {latest_date}. Values are shown in billions unless noted.</div>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    metric_card("Net Liquidity", fmt_b(latest["NetLiq"]), "WALCL - RRP - TGA")
with c2:
    metric_card("1W Change", fmt_delta_b(chg(df["NetLiq"], 7)), "Net liquidity")
with c3:
    metric_card("1M Change", fmt_delta_b(chg(df["NetLiq"], 30)), "Net liquidity")
with c4:
    metric_card("WALCL", fmt_b(latest["WALCL_b"]), "Fed assets")
with c5:
    metric_card("TGA", fmt_b(latest["TGA_b"]), "Treasury cash")
with c6:
    metric_card("RRP", fmt_b(latest["RRP_b"]), "ON reverse repo")

c7, c8, c9, c10 = st.columns(4)
with c7:
    metric_card("EFFR", fmt_pct(latest["EFFR"]), "Policy rate")
with c8:
    metric_card("NFCI", fmt_num(latest["NFCI"]), "Above 0 = tighter")
with c9:
    metric_card("TGA 1W", fmt_delta_b(chg(df["TGA_b"], 7)), "Cash drain/source")
with c10:
    metric_card("RRP 1W", fmt_delta_b(chg(df["RRP_b"], 7)), "Funding drain/source")

# ---------------- Quick Read ----------------
st.markdown("<div class='section-title'>Quick Read</div>", unsafe_allow_html=True)

net_1w = chg(df["NetLiq"], 7)
net_1m = chg(df["NetLiq"], 30)
policy_state = "tighter than average" if pd.notna(latest["NFCI"]) and latest["NFCI"] > 0 else "easier than average"

quick_left, quick_right = st.columns([1.25, 1])
with quick_left:
    st.markdown(
        f"""
        <div class="info-box">
        <b>Liquidity impulse</b><br><br>
        Net liquidity is <b>{fmt_b(latest["NetLiq"])}</b>, with a <b>{fmt_delta_b(net_1w)}</b> one-week move and a <b>{fmt_delta_b(net_1m)}</b> one-month move.
        The relevant pressure point is whether Treasury cash rebuilding and residual RRP usage are absorbing reserves faster than the Fed balance sheet path offsets them.
        </div>
        """,
        unsafe_allow_html=True,
    )

with quick_right:
    st.markdown(
        f"""
        <div class="info-box">
        <b>Policy and conditions</b><br><br>
        EFFR is <b>{fmt_pct(latest["EFFR"])}</b>. NFCI is <b>{fmt_num(latest["NFCI"])}</b>, implying financial conditions are currently <b>{policy_state}</b> on the Chicago Fed measure.
        <br><br>
        Liquidity smoothing: <b>{smooth_days} day(s)</b>. Policy smoothing: <b>{"on" if smooth_policy else "off"}</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

if errors:
    optional_errors = {k: v for k, v in errors.items() if k not in REQUIRED_SERIES}
    if optional_errors:
        st.warning("The dashboard loaded, but one or more optional policy/conditions series failed.")
        with st.expander("Optional series error details"):
            for series_id, message in optional_errors.items():
                st.write(f"**{series_id}:** {message}")

# ---------------- Chartbook ----------------
st.markdown("<div class='section-title'>Chartbook</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>Liquidity path, components, policy rate, and financial conditions.</div>",
    unsafe_allow_html=True,
)

fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.055,
    subplot_titles=(
        "Net Liquidity",
        "Core Components Rebased to 100",
        "Effective Fed Funds Rate",
        "Financial Conditions",
    ),
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["NetLiq_s"],
        name="Net Liquidity",
        mode="lines",
        line=dict(width=2.8, color="#111827"),
        hovertemplate="%{x|%Y-%m-%d}<br>Net Liquidity: %{y:,.0f}B<extra></extra>",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=reb.index,
        y=reb["WALCL_idx"],
        name="WALCL",
        mode="lines",
        line=dict(width=2.2, color="#2563eb"),
        hovertemplate="%{x|%Y-%m-%d}<br>WALCL Index: %{y:.1f}<extra></extra>",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=reb.index,
        y=reb["RRP_idx"],
        name="RRP",
        mode="lines",
        line=dict(width=2.2, color="#dc2626"),
        hovertemplate="%{x|%Y-%m-%d}<br>RRP Index: %{y:.1f}<extra></extra>",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=reb.index,
        y=reb["TGA_idx"],
        name="TGA",
        mode="lines",
        line=dict(width=2.2, color="#059669"),
        hovertemplate="%{x|%Y-%m-%d}<br>TGA Index: %{y:.1f}<extra></extra>",
    ),
    row=2,
    col=1,
)

if df["EFFR_s"].notna().any():
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["EFFR_s"],
            name="EFFR",
            mode="lines",
            line=dict(width=2.2, color="#f59e0b"),
            hovertemplate="%{x|%Y-%m-%d}<br>EFFR: %{y:.2f}%<extra></extra>",
        ),
        row=3,
        col=1,
    )

if df["NFCI_s"].notna().any():
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#9ca3af", row=4, col=1)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["NFCI_s"],
            name="NFCI",
            mode="lines",
            line=dict(width=2.2, color="#7c3aed"),
            hovertemplate="%{x|%Y-%m-%d}<br>NFCI: %{y:.3f}<extra></extra>",
        ),
        row=4,
        col=1,
    )

fig.update_layout(
    template="plotly_white",
    height=1080,
    margin=dict(l=55, r=25, t=55, b=35),
    legend=dict(
        orientation="h",
        x=0,
        y=1.035,
        xanchor="left",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.78)",
    ),
    hovermode="x unified",
)

fig.update_annotations(font=dict(size=13, color="#111827"))
fig.update_yaxes(title_text="Billions", showgrid=True, gridcolor="#eceff3", zeroline=False, row=1, col=1)
fig.update_yaxes(title_text="Index", showgrid=True, gridcolor="#eceff3", zeroline=False, row=2, col=1)
fig.update_yaxes(title_text="%", showgrid=True, gridcolor="#eceff3", zeroline=False, row=3, col=1)
fig.update_yaxes(title_text="Index", showgrid=True, gridcolor="#eceff3", zeroline=False, row=4, col=1)
fig.update_xaxes(tickformat="%b-%y", showgrid=False, title_text="Date", row=4, col=1)

st.plotly_chart(fig, use_container_width=True)

# ---------------- Lower Panels ----------------
left, right = st.columns([1.2, 0.8])

with left:
    if show_raw_table:
        st.markdown("<div class='section-title'>Raw Data</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>Latest 50 observations from the cleaned daily working dataset.</div>",
            unsafe_allow_html=True,
        )
        show = df[["WALCL_b", "RRP_b", "TGA_b", "EFFR", "NFCI", "NetLiq"]].copy()
        show.columns = ["WALCL (B)", "RRP (B)", "TGA (B)", "EFFR (%)", "NFCI", "Net Liquidity (B)"]
        st.dataframe(show.tail(50), use_container_width=True, height=420)

with right:
    st.markdown("<div class='section-title'>Download</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Export the cleaned working dataset used in the dashboard.</div>",
        unsafe_allow_html=True,
    )
    csv = df[["WALCL_b", "RRP_b", "TGA_b", "EFFR", "NFCI", "NetLiq"]].to_csv(index_label="Date").encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name="liquidity_tracker.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("<div class='section-title' style='margin-top:1.2rem;'>Methodology</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="info-box">
        <b>Formula</b><br>
        Net Liquidity = WALCL - ON RRP - TGA
        <br><br>
        <b>Unit treatment</b><br>
        WALCL and WDTGAL are converted from millions to billions. RRPONTSYD is already in billions.
        <br><br>
        <b>Calendar treatment</b><br>
        FRED series are aligned to a daily calendar and forward-filled. Weekly series carry forward until the next release.
        <br><br>
        <b>Smoothing</b><br>
        Liquidity series use a <b>{smooth_days}-day</b> rolling average. Policy smoothing is <b>{"enabled" if smooth_policy else "disabled"}</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption("© 2026 AD Fund Management LP")
