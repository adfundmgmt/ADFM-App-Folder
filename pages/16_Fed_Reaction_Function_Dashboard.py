import re
from io import StringIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from bs4 import BeautifulSoup

# ---------------- Config ----------------
TITLE = "Fed Reaction Function Dashboard"

SERIES: Dict[str, str] = {
    "CUSR0000SASL2RS": "Core Services Less Rent of Shelter",
    "FRBATLWGT3M": "Atlanta Fed Wage Growth Tracker",
    "UNRATE": "Unemployment Rate",
    "ANFCI": "Adjusted National Financial Conditions Index",
    "BAMLH0A0HYM2": "US High Yield OAS",
    "DFF": "Effective Fed Funds Rate",
    "DFEDTARU": "Fed Funds Target Range Upper Limit",
    "FEDTARMD": "SEP Median Federal Funds Rate",
    "UNRATEMD": "SEP Median Unemployment Rate",
    "PCEINFMDTOY": "SEP Median PCE Inflation",
    "CPCEINFMDTOY": "SEP Median Core PCE Inflation",
}

DEFAULT_LOOKBACK_YEARS = 5
DEFAULT_STATEMENT_URL = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260318a.htm"
DEFAULT_MINUTES_URL = "https://www.federalreserve.gov/monetarypolicy/fomcminutes20260318.htm"

HAWKISH_PATTERNS = {
    r"\bhigher for longer\b": 2.0,
    r"\bongoing increases\b": 2.0,
    r"\bsome additional policy firming\b": 2.0,
    r"\bupside risks to inflation\b": 1.5,
    r"\binflation remains elevated\b": 1.0,
    r"\binflation remains somewhat elevated\b": 1.0,
    r"\brestrictive\b": 0.5,
    r"\bstrong labor market\b": 0.5,
    r"\bsolid pace\b": 0.5,
    r"\bprepared to adjust policy as appropriate\b": 0.5,
    r"\bnot appropriate to reduce\b": 1.5,
}

DOVISH_PATTERNS = {
    r"\bbegin to reduce\b": 2.0,
    r"\breduce the target range\b": 2.0,
    r"\bgreater confidence\b": 1.5,
    r"\binflation has eased\b": 1.0,
    r"\blabor market conditions have eased\b": 1.0,
    r"\bunemployment has moved up\b": 1.0,
    r"\beconomic activity has slowed\b": 1.0,
    r"\brisks to employment\b": 1.25,
    r"\brisk(s)? to the outlook\b": 0.5,
    r"\bpolicy is well positioned\b": 0.5,
    r"\bcarefully assess incoming data\b": 0.25,
}

st.set_page_config(page_title=TITLE, layout="wide")

# ---------------- Style ----------------
CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 2.0rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .adfm-title {
        font-size: 2.0rem;
        font-weight: 700;
        margin-bottom: -0.15rem;
        color: #111827;
    }

    .adfm-subtitle {
        font-size: 0.98rem;
        color: #6b7280;
        margin-bottom: 1.2rem;
    }

    .section-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #111827;
        margin-top: 0.4rem;
        margin-bottom: 0.5rem;
    }

    .section-subtitle {
        font-size: 0.9rem;
        color: #6b7280;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 16px 10px 16px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
        min-height: 92px;
    }

    .metric-label {
        font-size: 0.78rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.45rem;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
        line-height: 1.1;
    }

    .metric-footnote {
        font-size: 0.78rem;
        color: #9ca3af;
        margin-top: 0.4rem;
    }

    .info-box {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 0.8rem;
    }

    .sidebar-box {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 14px 10px 14px;
        margin-bottom: 1rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown(f"<div class='adfm-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='adfm-subtitle'>Map core services ex housing, wage growth, unemployment, financial conditions, market pricing, Fed language, and SEP medians into a single policy-bias readout.</div>",
    unsafe_allow_html=True,
)

# ---------------- Helpers ----------------
def fetch_fred_csv(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}"
        f"&cosd={start.strftime('%Y-%m-%d')}"
        f"&coed={end.strftime('%Y-%m-%d')}"
    )

    r = requests.get(url, timeout=30)
    r.raise_for_status()

    text = r.text.strip()
    if not text:
        raise ValueError(f"Empty response for {series_id}")

    df = pd.read_csv(StringIO(text))
    if df.empty or len(df.columns) < 2:
        raise ValueError(f"No usable data returned for {series_id}")

    date_col = df.columns[0]
    value_col = df.columns[1]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    return pd.Series(df[value_col].values, index=df[date_col], name=series_id)

@st.cache_data(ttl=60 * 60, show_spinner=False)
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

@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_fed_text(url: str) -> str:
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 300:
        raise ValueError(f"Could not extract enough text from {url}")

    return text

def pct_change_yoy(s: pd.Series) -> pd.Series:
    if s.empty:
        return pd.Series(dtype="float64")
    return s.pct_change(12) * 100.0

def pct_change_3m_ann(s: pd.Series) -> pd.Series:
    if s.empty:
        return pd.Series(dtype="float64")
    return ((s / s.shift(3)) ** 4 - 1.0) * 100.0

def latest_valid(s: pd.Series) -> Tuple[Optional[pd.Timestamp], float]:
    s = s.dropna()
    if s.empty:
        return None, float("nan")
    return s.index[-1], float(s.iloc[-1])

def get_sep_for_year(s: pd.Series, year: int) -> Optional[float]:
    s = s.dropna()
    if s.empty:
        return None
    same_year = s[s.index.year == year]
    if same_year.empty:
        return None
    return float(same_year.iloc[-1])

def fmt_pct(x) -> str:
    return "N/A" if pd.isna(x) else f"{x:.2f}%"

def fmt_num(x) -> str:
    return "N/A" if pd.isna(x) else f"{x:.2f}"

def metric_card(label: str, value: str, footnote: str = ""):
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

def score_text(text: str) -> Tuple[float, List[str], List[str]]:
    if not text:
        return 0.0, [], []

    text = text.lower()
    hawk_hits = []
    dove_hits = []
    score = 0.0

    for pattern, weight in HAWKISH_PATTERNS.items():
        if re.search(pattern, text):
            score += weight
            hawk_hits.append(pattern.replace(r"\b", "").replace("\\", ""))

    for pattern, weight in DOVISH_PATTERNS.items():
        if re.search(pattern, text):
            score -= weight
            dove_hits.append(pattern.replace(r"\b", "").replace("\\", ""))

    return score, hawk_hits, dove_hits

def reaction_summary(total_score: float) -> str:
    if total_score <= -2:
        return "easing bias rising"
    if total_score >= 2:
        return "tightening risk re-emerging"
    return "hold bias intact"

def make_line_chart(
    s: pd.Series,
    title: str,
    y_title: str,
    months: int,
    ref_value: Optional[float] = None,
    ref_name: str = "Reference",
) -> go.Figure:
    fig = go.Figure()

    s = s.dropna().copy()
    if not s.empty:
        cutoff = s.index.max() - pd.DateOffset(months=months)
        s = s[s.index >= cutoff]
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=title,
                line=dict(width=2.5),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
            )
        )

    if ref_value is not None and not pd.isna(ref_value):
        fig.add_hline(
            y=ref_value,
            line_dash="dash",
            annotation_text=ref_name,
            annotation_position="top left",
        )

    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=55, b=20),
        title=title,
        yaxis_title=y_title,
        legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom"),
        hovermode="x unified",
    )
    return fig

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("### About This Tool")
    st.markdown(
        """
This dashboard is meant to capture the policy reaction function instead of staring at one inflation print.

**What it shows**
- Core services ex housing momentum
- Wage growth and unemployment versus SEP
- Financial conditions and credit spread context
- Market-implied cuts versus SEP median
- Fed language from the latest statement and minutes

**Final output**
- easing bias rising
- hold bias intact
- tightening risk re-emerging
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("### Settings")
    lookback = st.selectbox("Lookback", ["2y", "3y", "5y", "10y"], index=2)
    years = int(lookback[:-1])
    lookback_months = years * 12

    statement_url = st.text_input("Latest FOMC statement URL", value=DEFAULT_STATEMENT_URL)
    minutes_url = st.text_input("Latest FOMC minutes URL", value=DEFAULT_MINUTES_URL)

    market_mode = st.radio(
        "Market pricing input",
        ["Manual cuts", "Manual implied end-year rate"],
        index=0,
    )

    market_implied_cuts_bps = 50
    market_implied_end_rate_manual = 3.10

    if market_mode == "Manual cuts":
        market_implied_cuts_bps = st.number_input(
            "Cumulative cuts over next 12 months (bps)",
            min_value=-100,
            max_value=400,
            value=50,
            step=25,
        )
    else:
        market_implied_end_rate_manual = st.number_input(
            "Market-implied end-year fed funds rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=3.10,
            step=0.05,
            format="%.2f",
        )

    st.caption("Source: public FRED CSV endpoints and official Federal Reserve webpages")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Data ----------------
today = pd.Timestamp.today().normalize()
start = today - pd.DateOffset(years=years)

with st.spinner("Loading macro data and Fed language..."):
    raw, errors = load_all_series(start, today)

required = ["CUSR0000SASL2RS", "FRBATLWGT3M", "UNRATE", "ANFCI", "DFEDTARU", "FEDTARMD"]
missing = [x for x in required if raw[x].empty]

if missing:
    st.error(f"Required series failed to load: {', '.join(missing)}")
    if errors:
        with st.expander("Error details"):
            for k, v in errors.items():
                st.write(f"{k}: {v}")
    st.stop()

try:
    statement_text = fetch_fed_text(statement_url)
    statement_error = None
except Exception as e:
    statement_text = ""
    statement_error = str(e)

try:
    minutes_text = fetch_fed_text(minutes_url)
    minutes_error = None
except Exception as e:
    minutes_text = ""
    minutes_error = str(e)

df = pd.concat(
    [
        raw["CUSR0000SASL2RS"].rename("supercore"),
        raw["FRBATLWGT3M"].rename("wages"),
        raw["UNRATE"].rename("unrate"),
        raw["ANFCI"].rename("anfci"),
        raw["BAMLH0A0HYM2"].rename("hy_oas"),
        raw["DFF"].rename("dff"),
        raw["DFEDTARU"].rename("fed_target_upper"),
        raw["FEDTARMD"].rename("sep_fed"),
        raw["UNRATEMD"].rename("sep_unrate"),
        raw["PCEINFMDTOY"].rename("sep_pce"),
        raw["CPCEINFMDTOY"].rename("sep_core_pce"),
    ],
    axis=1,
).sort_index().ffill()

df = df[df.index >= start].copy()

supercore_yoy = pct_change_yoy(df["supercore"])
supercore_3m = pct_change_3m_ann(df["supercore"])

current_year = today.year

supercore_yoy_dt, supercore_yoy_last = latest_valid(supercore_yoy)
supercore_3m_dt, supercore_3m_last = latest_valid(supercore_3m)
wage_dt, wage_last = latest_valid(df["wages"])
un_dt, un_last = latest_valid(df["unrate"])
anfci_dt, anfci_last = latest_valid(df["anfci"])
hy_dt, hy_last = latest_valid(df["hy_oas"])
dff_dt, dff_last = latest_valid(df["dff"])
upper_dt, upper_last = latest_valid(df["fed_target_upper"])

sep_fed_curr = get_sep_for_year(df["sep_fed"], current_year)
sep_un_curr = get_sep_for_year(df["sep_unrate"], current_year)
sep_pce_curr = get_sep_for_year(df["sep_pce"], current_year)
sep_core_curr = get_sep_for_year(df["sep_core_pce"], current_year)

if pd.isna(upper_last):
    upper_last = 4.50

if market_mode == "Manual cuts":
    market_implied_end_rate = float(upper_last) - float(market_implied_cuts_bps) / 100.0
else:
    market_implied_end_rate = float(market_implied_end_rate_manual)
    market_implied_cuts_bps = int(round((float(upper_last) - float(market_implied_end_rate)) * 100.0))

# ---------------- Scoring ----------------
metric_rows = []

# Inflation
if pd.isna(supercore_3m_last):
    inflation_pts = 0
    inflation_state = "n/a"
elif supercore_3m_last >= 4.0:
    inflation_pts = 2
    inflation_state = "re-accelerating"
elif supercore_3m_last >= 3.0:
    inflation_pts = 1
    inflation_state = "sticky"
elif supercore_3m_last <= 2.25:
    inflation_pts = -2
    inflation_state = "cooling fast"
elif supercore_3m_last <= 2.75:
    inflation_pts = -1
    inflation_state = "cooling"
else:
    inflation_pts = 0
    inflation_state = "mixed"

metric_rows.append(
    {
        "Factor": "Core services ex housing",
        "Current": f"{fmt_num(supercore_3m_last)}% (3m ann)",
        "Reference": f"{fmt_num(supercore_yoy_last)}% YoY | SEP core PCE {fmt_num(sep_core_curr)}%",
        "Read": inflation_state,
        "Score": inflation_pts,
    }
)

# Wages
if pd.isna(wage_last):
    wage_pts = 0
    wage_state = "n/a"
elif wage_last >= 4.5:
    wage_pts = 2
    wage_state = "too hot"
elif wage_last >= 4.0:
    wage_pts = 1
    wage_state = "still firm"
elif wage_last <= 3.0:
    wage_pts = -2
    wage_state = "cool enough"
elif wage_last <= 3.5:
    wage_pts = -1
    wage_state = "cooling"
else:
    wage_pts = 0
    wage_state = "balanced"

metric_rows.append(
    {
        "Factor": "Wage growth",
        "Current": fmt_pct(wage_last),
        "Reference": "Atlanta Fed Wage Growth Tracker",
        "Read": wage_state,
        "Score": wage_pts,
    }
)

# Unemployment vs SEP
if pd.isna(un_last) or sep_un_curr is None or pd.isna(sep_un_curr):
    un_pts = 0
    un_state = "n/a"
    un_gap_txt = "N/A"
else:
    un_gap = un_last - sep_un_curr
    un_gap_txt = f"{un_gap:+.2f}"
    if un_gap >= 0.30:
        un_pts = -2
        un_state = "labor cooling faster than SEP"
    elif un_gap >= 0.10:
        un_pts = -1
        un_state = "modestly softer than SEP"
    elif un_gap <= -0.30:
        un_pts = 2
        un_state = "labor still tighter than SEP"
    elif un_gap <= -0.10:
        un_pts = 1
        un_state = "slightly tighter than SEP"
    else:
        un_pts = 0
        un_state = "near SEP"

metric_rows.append(
    {
        "Factor": "Unemployment",
        "Current": fmt_pct(un_last),
        "Reference": f"SEP median {fmt_num(sep_un_curr)}% | gap {un_gap_txt}",
        "Read": un_state,
        "Score": un_pts,
    }
)

# Financial conditions
if pd.isna(anfci_last):
    fci_pts = 0
    fci_state = "n/a"
elif anfci_last >= 0.50:
    fci_pts = -2
    fci_state = "materially tighter"
elif anfci_last >= 0.15:
    fci_pts = -1
    fci_state = "tighter"
elif anfci_last <= -0.50:
    fci_pts = 2
    fci_state = "very easy"
elif anfci_last <= -0.15:
    fci_pts = 1
    fci_state = "easy"
else:
    fci_pts = 0
    fci_state = "near neutral"

metric_rows.append(
    {
        "Factor": "Financial conditions",
        "Current": f"ANFCI {fmt_num(anfci_last)} | HY OAS {fmt_num(hy_last)}%",
        "Reference": "Positive ANFCI = tighter than average",
        "Read": fci_state,
        "Score": fci_pts,
    }
)

# Market implied cuts vs SEP
if sep_fed_curr is None or pd.isna(sep_fed_curr):
    mkt_pts = 0
    mkt_state = "n/a"
else:
    fed_gap = market_implied_end_rate - sep_fed_curr
    if fed_gap <= -0.50:
        mkt_pts = -2
        mkt_state = "market pricing much more easing than SEP"
    elif fed_gap <= -0.25:
        mkt_pts = -1
        mkt_state = "market leaning easier than SEP"
    elif fed_gap >= 0.50:
        mkt_pts = 2
        mkt_state = "market pricing tighter path than SEP"
    elif fed_gap >= 0.25:
        mkt_pts = 1
        mkt_state = "market fading cuts vs SEP"
    else:
        mkt_pts = 0
        mkt_state = "near SEP path"

metric_rows.append(
    {
        "Factor": "Market-implied cuts",
        "Current": f"{market_implied_cuts_bps} bps | implied end rate {fmt_num(market_implied_end_rate)}%",
        "Reference": f"SEP median end-{current_year} rate {fmt_num(sep_fed_curr)}%",
        "Read": mkt_state,
        "Score": mkt_pts,
    }
)

# Fed language
statement_score, statement_hawk, statement_dove = score_text(statement_text)
minutes_score, minutes_hawk, minutes_dove = score_text(minutes_text)
language_score = round(0.65 * statement_score + 0.35 * minutes_score, 2)

if language_score <= -2.0:
    lang_pts = -2
    lang_state = "clear easing lean"
elif language_score <= -0.50:
    lang_pts = -1
    lang_state = "softening tone"
elif language_score >= 2.0:
    lang_pts = 2
    lang_state = "hawkish hold"
elif language_score >= 0.50:
    lang_pts = 1
    lang_state = "guarded / restrictive"
else:
    lang_pts = 0
    lang_state = "balanced hold"

metric_rows.append(
    {
        "Factor": "Fed language",
        "Current": f"score {fmt_num(language_score)}",
        "Reference": "Statement and minutes keyword score",
        "Read": lang_state,
        "Score": lang_pts,
    }
)

score_df = pd.DataFrame(metric_rows)
total_score = int(score_df["Score"].sum())
policy_call = reaction_summary(total_score)

# ---------------- Snapshot ----------------
st.markdown("<div class='section-title'>Snapshot</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>Latest available readings across the reaction function.</div>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Dashboard Output", policy_call.title(), "Composite policy-bias readout")
with c2:
    metric_card("Reaction Score", f"{total_score:+d}", "Sum of factor scores")
with c3:
    metric_card("Fed Funds Upper", fmt_pct(upper_last), "Current target range upper bound")
with c4:
    metric_card("SEP End-Year Median", fmt_pct(sep_fed_curr), "Median fed funds projection")

# ---------------- Quick Read ----------------
st.markdown("<div class='section-title'>Quick Read</div>", unsafe_allow_html=True)

dovish_drivers = score_df[score_df["Score"] < 0]["Factor"].tolist()
hawkish_drivers = score_df[score_df["Score"] > 0]["Factor"].tolist()

driver_text = "the factors are clustered around neutral"
if dovish_drivers and hawkish_drivers:
    driver_text = (
        f"easing pressure is coming from {', '.join(dovish_drivers[:3])}, "
        f"offset by {', '.join(hawkish_drivers[:3])}"
    )
elif dovish_drivers:
    driver_text = f"easing pressure is coming from {', '.join(dovish_drivers[:3])}"
elif hawkish_drivers:
    driver_text = f"hawkish pressure is coming from {', '.join(hawkish_drivers[:3])}"

quick_left, quick_right = st.columns([1.35, 1])

with quick_left:
    st.markdown(
        f"""
        <div class="info-box">
        <b>Tape read</b><br><br>
        The model reads <b>{policy_call}</b>. Right now {driver_text}. This is built to approximate the way the market maps incoming macro data against the Fed's own language and SEP path rather than reducing the tape to one CPI print.
        </div>
        """,
        unsafe_allow_html=True,
    )

with quick_right:
    st.markdown(
        f"""
        <div class="info-box">
        <b>Current state</b><br><br>
        Core services ex housing is running at <b>{fmt_num(supercore_3m_last)}%</b> on a 3-month annualized basis, wage growth is <b>{fmt_num(wage_last)}%</b>, unemployment is <b>{fmt_num(un_last)}%</b>, and ANFCI is <b>{fmt_num(anfci_last)}</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- Warnings ----------------
if errors:
    noncritical = {k: v for k, v in errors.items() if k not in required}
    if noncritical:
        st.warning("Some non-critical FRED series failed to load.")
        with st.expander("Series error details"):
            for k, v in noncritical.items():
                st.write(f"{k}: {v}")

if statement_error or minutes_error:
    st.warning("Some Fed language pages failed to load.")
    with st.expander("Fed page error details"):
        if statement_error:
            st.write(f"Statement: {statement_error}")
        if minutes_error:
            st.write(f"Minutes: {minutes_error}")

# ---------------- Scorecard ----------------
st.markdown("<div class='section-title'>Reaction Function Scorecard</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>Each input is mapped into a simple hawkish or dovish score.</div>",
    unsafe_allow_html=True,
)

st.dataframe(score_df, use_container_width=True, hide_index=True)

# ---------------- Chartbook ----------------
st.markdown("<div class='section-title'>Chartbook</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>Macro state, policy context, and where market pricing sits relative to the SEP.</div>",
    unsafe_allow_html=True,
)

left, right = st.columns((1.1, 1.0))

with left:
    st.plotly_chart(
        make_line_chart(
            supercore_3m,
            "Core Services ex Housing, 3m Annualized",
            "%",
            months=lookback_months,
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        make_line_chart(
            df["wages"],
            "Atlanta Fed Wage Growth Tracker",
            "%",
            months=lookback_months,
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        make_line_chart(
            df["unrate"],
            "Unemployment Rate vs SEP Median",
            "%",
            months=lookback_months,
            ref_value=sep_un_curr,
            ref_name="SEP median",
        ),
        use_container_width=True,
    )

with right:
    st.plotly_chart(
        make_line_chart(
            df["anfci"],
            "Financial Conditions (ANFCI)",
            "Index",
            months=lookback_months,
            ref_value=0.0,
            ref_name="Neutral",
        ),
        use_container_width=True,
    )

    fed_compare = pd.DataFrame(
        {
            "Series": [
                "Current fed funds upper",
                "Market-implied end-year rate",
                "SEP median end-year rate",
            ],
            "Value": [upper_last, market_implied_end_rate, sep_fed_curr],
        }
    ).dropna()

    fig_bars = go.Figure()
    fig_bars.add_trace(
        go.Bar(
            x=fed_compare["Series"],
            y=fed_compare["Value"],
            hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
        )
    )
    fig_bars.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=55, b=20),
        title="Where policy is vs where market and SEP think it goes",
        yaxis_title="%",
    )
    st.plotly_chart(fig_bars, use_container_width=True)

    sep_tbl = pd.DataFrame(
        {
            "SEP Median": ["Fed funds", "Unemployment", "PCE", "Core PCE"],
            f"{current_year}": [sep_fed_curr, sep_un_curr, sep_pce_curr, sep_core_curr],
        }
    )
    st.markdown("**SEP Medians**")
    st.dataframe(sep_tbl, use_container_width=True, hide_index=True)

# ---------------- Fed Language ----------------
st.markdown("<div class='section-title'>Fed Language</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-subtitle'>Simple statement and minutes scoring to approximate whether the language is leaning easier or more restrictive.</div>",
    unsafe_allow_html=True,
)

lang1, lang2 = st.columns(2)

with lang1:
    st.markdown(f"**Statement score:** {fmt_num(statement_score)}")
    st.write("Hawkish hits: " + (", ".join(statement_hawk[:6]) if statement_hawk else "none"))
    st.write("Dovish hits: " + (", ".join(statement_dove[:6]) if statement_dove else "none"))
    st.text_area(
        "Latest FOMC statement excerpt",
        value=statement_text[:2500] if statement_text else "Could not load statement text.",
        height=220,
    )

with lang2:
    st.markdown(f"**Minutes score:** {fmt_num(minutes_score)}")
    st.write("Hawkish hits: " + (", ".join(minutes_hawk[:6]) if minutes_hawk else "none"))
    st.write("Dovish hits: " + (", ".join(minutes_dove[:6]) if minutes_dove else "none"))
    st.text_area(
        "Latest FOMC minutes excerpt",
        value=minutes_text[:2500] if minutes_text else "Could not load minutes text.",
        height=220,
    )

# ---------------- Diagnostics ----------------
with st.expander("Diagnostics and data freshness"):
    def dt_or_na(x):
        return "N/A" if x is None else str(x.date())

    freshness = pd.DataFrame(
        [
            ["Core services ex housing", dt_or_na(supercore_3m_dt), "CUSR0000SASL2RS"],
            ["Wage growth", dt_or_na(wage_dt), "FRBATLWGT3M"],
            ["Unemployment", dt_or_na(un_dt), "UNRATE"],
            ["ANFCI", dt_or_na(anfci_dt), "ANFCI"],
            ["HY OAS", dt_or_na(hy_dt), "BAMLH0A0HYM2"],
            ["Fed funds upper", dt_or_na(upper_dt), "DFEDTARU"],
            ["SEP fed funds", str(df["sep_fed"].dropna().index.max().date()) if df["sep_fed"].dropna().any() else "N/A", "FEDTARMD"],
            ["SEP unemployment", str(df["sep_unrate"].dropna().index.max().date()) if df["sep_unrate"].dropna().any() else "N/A", "UNRATEMD"],
            ["SEP PCE", str(df["sep_pce"].dropna().index.max().date()) if df["sep_pce"].dropna().any() else "N/A", "PCEINFMDTOY"],
            ["SEP core PCE", str(df["sep_core_pce"].dropna().index.max().date()) if df["sep_core_pce"].dropna().any() else "N/A", "CPCEINFMDTOY"],
        ],
        columns=["Series", "Latest observation", "FRED ID"],
    )
    st.dataframe(freshness, use_container_width=True, hide_index=True)
    st.markdown("**Fed URLs used**")
    st.code(f"Statement: {statement_url}\nMinutes: {minutes_url}")

st.caption("Design choice: this uses CPI services less rent of shelter as the public proxy for core services ex housing. SEP inflation medians are reported in PCE and core PCE.")
