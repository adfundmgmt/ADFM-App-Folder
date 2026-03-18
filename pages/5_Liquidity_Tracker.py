import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------- Config ----------------
st.set_page_config(page_title="Liquidity & Fed Policy Tracker", layout="wide")
st.title("Liquidity & Fed Policy Tracker")

SERIES = {
    "WALCL": "Fed Balance Sheet",
    "RRPONTSYD": "ON RRP",
    "WDTGAL": "Treasury General Account",
    "EFFR": "Effective Fed Funds Rate",
    "NFCI": "Chicago Fed NFCI",
}

DEFAULT_LOOKBACK_YEARS = 3
DEFAULT_SMOOTH = 5


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    lookback = st.selectbox("Lookback", ["1y", "2y", "3y", "5y", "10y"], index=2)
    years = int(lookback[:-1])

    smooth_days = st.number_input(
        "Smoothing window",
        min_value=1,
        max_value=30,
        value=DEFAULT_SMOOTH,
        step=1,
    )

    smooth_policy = st.checkbox("Smooth EFFR and NFCI", value=False)

    st.markdown("---")
    st.caption("Source: public FRED CSV endpoints")
    st.caption("No API key required")


# ---------------- Loaders ----------------
def fetch_fred_csv(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}"
        f"&cosd={start.strftime('%Y-%m-%d')}"
        f"&coed={end.strftime('%Y-%m-%d')}"
    )

    r = requests.get(url, timeout=30)
    r.raise_for_status()

    df = pd.read_csv(pd.io.common.StringIO(r.text))
    if df.empty or len(df.columns) < 2:
        raise ValueError(f"No usable data returned for {series_id}")

    date_col = df.columns[0]
    value_col = df.columns[1]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    s = pd.Series(df[value_col].values, index=df[date_col], name=series_id)
    return s


@st.cache_data(ttl=60 * 60)
def load_all_series(start: pd.Timestamp, end: pd.Timestamp):
    out = {}
    errors = {}

    for series_id in SERIES:
        try:
            out[series_id] = fetch_fred_csv(series_id, start, end)
        except Exception as e:
            errors[series_id] = str(e)
            out[series_id] = pd.Series(dtype="float64", name=series_id)

    return out, errors


def smooth(s: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return s
    return s.rolling(window, min_periods=1).mean()


def rebase(s: pd.Series, floor: float | None = None) -> pd.Series:
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


def fmt_b(x):
    return "N/A" if pd.isna(x) else f"{x:,.0f} B"


def fmt_pct(x):
    return "N/A" if pd.isna(x) else f"{x:.2f}%"


def fmt_num(x):
    return "N/A" if pd.isna(x) else f"{x:.3f}"


# ---------------- Data ----------------
today = pd.Timestamp.today().normalize()
start = today - pd.DateOffset(years=years)

with st.spinner("Loading data..."):
    raw, errors = load_all_series(start, today)

required = ["WALCL", "RRPONTSYD", "WDTGAL"]
missing = [x for x in required if raw[x].empty]

if missing:
    st.error(f"Required series failed to load: {', '.join(missing)}")
    if errors:
        with st.expander("Error details"):
            for k, v in errors.items():
                st.write(f"{k}: {v}")
    st.stop()

df = pd.concat(
    [
        raw["WALCL"].rename("WALCL"),
        raw["RRPONTSYD"].rename("RRP"),
        raw["WDTGAL"].rename("TGA"),
        raw["EFFR"].rename("EFFR"),
        raw["NFCI"].rename("NFCI"),
    ],
    axis=1,
).sort_index().ffill()

df = df[df.index >= start].copy()
df = df.dropna(subset=["WALCL", "RRP", "TGA"])

if df.empty:
    st.error("Dataframe is empty after cleaning.")
    st.stop()

# ---------------- Units ----------------
# WALCL and WDTGAL are in millions. RRPONTSYD is already billions.
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

if smooth_policy:
    df["EFFR_s"] = smooth(df["EFFR"], smooth_days)
    df["NFCI_s"] = smooth(df["NFCI"], smooth_days)
else:
    df["EFFR_s"] = df["EFFR"]
    df["NFCI_s"] = df["NFCI"]

reb = pd.DataFrame(index=df.index)
reb["WALCL_idx"] = rebase(df["WALCL_b"])
reb["RRP_idx"] = rebase(df["RRP_b"], floor=5.0)
reb["TGA_idx"] = rebase(df["TGA_b"])

# ---------------- Metrics ----------------
latest = df.iloc[-1]

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Net Liquidity", fmt_b(latest["NetLiq"]))
c2.metric("WALCL", fmt_b(latest["WALCL_b"]))
c3.metric("RRP", fmt_b(latest["RRP_b"]))
c4.metric("TGA", fmt_b(latest["TGA_b"]))
c5.metric("EFFR", fmt_pct(latest["EFFR"]))
c6.metric("NFCI", fmt_num(latest["NFCI"]))

if errors:
    noncritical = {k: v for k, v in errors.items() if k not in required}
    if noncritical:
        st.warning("Some non-critical series failed to load.")
        with st.expander("Error details"):
            for k, v in noncritical.items():
                st.write(f"{k}: {v}")

# ---------------- Chart ----------------
fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=(
        "Net Liquidity (Billions)",
        "Components Rebased to 100",
        "Effective Fed Funds Rate",
        "Financial Conditions (NFCI)",
    ),
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["NetLiq_s"],
        name="Net Liquidity",
        line=dict(width=2.5),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(x=reb.index, y=reb["WALCL_idx"], name="WALCL idx", line=dict(width=2)),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(x=reb.index, y=reb["RRP_idx"], name="RRP idx", line=dict(width=2)),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(x=reb.index, y=reb["TGA_idx"], name="TGA idx", line=dict(width=2)),
    row=2,
    col=1,
)

if df["EFFR_s"].notna().any():
    fig.add_trace(
        go.Scatter(x=df.index, y=df["EFFR_s"], name="EFFR", line=dict(width=2)),
        row=3,
        col=1,
    )

if df["NFCI_s"].notna().any():
    fig.add_hline(y=0, line_width=1, line_dash="dot", row=4, col=1)
    fig.add_trace(
        go.Scatter(x=df.index, y=df["NFCI_s"], name="NFCI", line=dict(width=2)),
        row=4,
        col=1,
    )

fig.update_layout(
    template="plotly_white",
    height=1050,
    margin=dict(l=50, r=20, t=50, b=40),
    legend=dict(orientation="h", x=0, y=1.03),
)

fig.update_yaxes(title_text="Billions", row=1, col=1)
fig.update_yaxes(title_text="Index", row=2, col=1)
fig.update_yaxes(title_text="%", row=3, col=1)
fig.update_yaxes(title_text="Index", row=4, col=1)
fig.update_xaxes(tickformat="%b-%y", row=4, col=1)

st.plotly_chart(fig, use_container_width=True)

# ---------------- Data Table ----------------
with st.expander("Show raw data"):
    show = df[["WALCL_b", "RRP_b", "TGA_b", "EFFR", "NFCI", "NetLiq"]].copy()
    st.dataframe(show.tail(50), use_container_width=True)

# ---------------- Download ----------------
csv = df[["WALCL_b", "RRP_b", "TGA_b", "EFFR", "NFCI", "NetLiq"]].to_csv().encode("utf-8")
st.download_button(
    "Download CSV",
    data=csv,
    file_name="liquidity_tracker.csv",
    mime="text/csv",
)
