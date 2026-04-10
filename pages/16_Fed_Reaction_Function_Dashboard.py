import math
import re
import time
from datetime import datetime
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title="Fed Reaction Function Dashboard", layout="wide")

# =========================
# Constants
# =========================
SERIES: Dict[str, Dict[str, str]] = {
    # Closest stable public proxy for "core services ex housing":
    # CPI Services Less Rent of Shelter
    "supercore_cpi": {
        "id": "CUSR0000SASL2RS",
        "label": "CPI Services Less Rent of Shelter",
    },
    # Atlanta Fed Wage Growth Tracker
    "wage_growth": {
        "id": "FRBATLWGT3M",
        "label": "Atlanta Fed Wage Growth Tracker",
    },
    "unemployment": {
        "id": "UNRATE",
        "label": "Unemployment Rate",
    },
    # Chicago Fed Adjusted National Financial Conditions Index
    "anfci": {
        "id": "ANFCI",
        "label": "Adjusted National Financial Conditions Index",
    },
    # ICE BofA US High Yield OAS
    "hy_oas": {
        "id": "BAMLH0A0HYM2",
        "label": "US High Yield OAS",
    },
    "fed_funds_effective": {
        "id": "DFF",
        "label": "Effective Fed Funds Rate",
    },
    # Upper bound of target range
    "fed_target_upper": {
        "id": "DFEDTARU",
        "label": "Fed Funds Target Range Upper Limit",
    },
    # SEP medians on FRED
    "sep_fed_funds": {
        "id": "FEDTARMD",
        "label": "SEP Median Federal Funds Rate",
    },
    "sep_unemployment": {
        "id": "UNRATEMD",
        "label": "SEP Median Unemployment Rate",
    },
    "sep_pce": {
        "id": "PCEINFMDTOY",
        "label": "SEP Median PCE Inflation",
    },
    "sep_core_pce": {
        "id": "CPCEINFMDTOY",
        "label": "SEP Median Core PCE Inflation",
    },
}

DEFAULT_STATEMENT_URL = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260318a.htm"
DEFAULT_MINUTES_URL = "https://www.federalreserve.gov/monetarypolicy/fomcminutes20260318.htm"

# Lightweight keyword scoring for Fed language
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
    r"\bgreat(er)? confidence\b": -0.75,  # slightly offsets if in context of easing confidence
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

# =========================
# Helpers
# =========================
def init_session_state() -> None:
    if "last_good_data" not in st.session_state:
        st.session_state["last_good_data"] = {}
    if "last_good_text" not in st.session_state:
        st.session_state["last_good_text"] = {}

def requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }
    )
    return s

@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_fred_series(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    s = requests_session()
    last_err = None
    for timeout in (15, 25, 40):
        try:
            r = s.get(url, timeout=timeout)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
            if df.empty or len(df.columns) < 2:
                raise ValueError(f"Unexpected FRED CSV format for {series_id}")
            df.columns = ["DATE", "VALUE"]
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
            df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
            df = df.dropna(subset=["DATE"]).set_index("DATE").sort_index()
            ser = df["VALUE"].dropna()
            if ser.empty:
                raise ValueError(f"No valid data returned for {series_id}")
            return ser
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise RuntimeError(f"Failed to fetch FRED series {series_id}: {last_err}")

@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_html_text(url: str) -> str:
    s = requests_session()
    last_err = None
    for timeout in (15, 25, 40):
        try:
            r = s.get(url, timeout=timeout)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            text = soup.get_text(" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) < 300:
                raise ValueError("Extracted text is too short")
            return text
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise RuntimeError(f"Failed to fetch page text {url}: {last_err}")

def safe_series(key: str) -> pd.Series:
    try:
        ser = fetch_fred_series(SERIES[key]["id"])
        st.session_state["last_good_data"][key] = ser
        return ser
    except Exception:
        cached = st.session_state["last_good_data"].get(key)
        if cached is not None:
            return cached
        raise

def safe_text(cache_key: str, url: str) -> str:
    try:
        text = fetch_html_text(url)
        st.session_state["last_good_text"][cache_key] = text
        return text
    except Exception:
        cached = st.session_state["last_good_text"].get(cache_key)
        if cached is not None:
            return cached
        raise

def latest_valid(ser: pd.Series) -> Tuple[pd.Timestamp, float]:
    ser = ser.dropna()
    return ser.index[-1], float(ser.iloc[-1])

def pct_change_yoy(ser: pd.Series) -> pd.Series:
    return ser.pct_change(12) * 100.0

def pct_change_3m_ann(ser: pd.Series) -> pd.Series:
    return ((ser / ser.shift(3)) ** 4 - 1.0) * 100.0

def z_last(ser: pd.Series, window: int = 156) -> float:
    s = ser.dropna().tail(window)
    if len(s) < max(20, window // 4):
        return float("nan")
    std = float(s.std(ddof=0))
    if std == 0:
        return 0.0
    return float((s.iloc[-1] - s.mean()) / std)

def get_sep_for_year(ser: pd.Series, year: int) -> Optional[float]:
    if ser.dropna().empty:
        return None
    tmp = ser.dropna().copy()
    exact = tmp[tmp.index.year == year]
    if not exact.empty:
        return float(exact.iloc[-1])
    return None

def score_text(text: str) -> Tuple[float, List[str], List[str]]:
    t = text.lower()
    hawk_hits = []
    dove_hits = []
    score = 0.0
    for pattern, w in HAWKISH_PATTERNS.items():
        if re.search(pattern, t):
            score += w
            hawk_hits.append(pattern.replace(r"\b", "").replace("\\", ""))
    for pattern, w in DOVISH_PATTERNS.items():
        if re.search(pattern, t):
            score -= w
            dove_hits.append(pattern.replace(r"\b", "").replace("\\", ""))
    return score, hawk_hits, dove_hits

def classify_metric_direction(value: float, bands: List[Tuple[float, str, int]], higher_is_hawkish: bool = True) -> Tuple[str, int]:
    """
    bands example for higher_is_hawkish:
    [(2.75, "cooling", -1), (4.0, "sticky", 1), (9e9, "hot", 2)]
    """
    if math.isnan(value):
        return "n/a", 0
    if higher_is_hawkish:
        for upper, label, pts in bands:
            if value <= upper:
                return label, pts
    else:
        for upper, label, pts in bands:
            if value <= upper:
                return label, pts
    return "n/a", 0

def format_delta(curr: float, ref: Optional[float]) -> str:
    if ref is None or pd.isna(ref):
        return "n/a"
    d = curr - ref
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.2f}"

def make_line_chart(
    ser: pd.Series,
    title: str,
    y_title: str,
    ref_value: Optional[float] = None,
    ref_name: str = "Reference",
    months: int = 60,
) -> go.Figure:
    s = ser.dropna().copy()
    cutoff = s.index.max() - pd.DateOffset(months=months)
    s = s[s.index >= cutoff]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s.values,
            mode="lines",
            name=title,
            line=dict(width=2),
        )
    )
    if ref_value is not None and not pd.isna(ref_value):
        fig.add_hline(y=ref_value, line_dash="dash", annotation_text=ref_name, annotation_position="top left")
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=55, b=20),
        title=title,
        xaxis_title="",
        yaxis_title=y_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
    )
    return fig

def reaction_summary(total_score: float) -> str:
    if total_score <= -2.0:
        return "easing bias rising"
    if total_score >= 2.0:
        return "tightening risk re-emerging"
    return "hold bias intact"

def score_badge(score: int) -> str:
    if score >= 2:
        return "hawkish"
    if score <= -2:
        return "dovish"
    if score == 1:
        return "slightly hawkish"
    if score == -1:
        return "slightly dovish"
    return "neutral"

def markdown_small(text: str) -> None:
    st.markdown(f"<div style='font-size:0.92rem; color:#666;'>{text}</div>", unsafe_allow_html=True)

# =========================
# App
# =========================
init_session_state()

st.title("Fed Reaction Function Dashboard")
st.caption("Map inflation persistence, labor cooling, financial conditions, market pricing, Fed language, and SEP medians into a single policy-bias readout.")

with st.sidebar:
    st.header("Inputs")
    st.markdown("Use official sources by default. Update the links after each FOMC meeting.")
    statement_url = st.text_input("Latest FOMC statement URL", value=DEFAULT_STATEMENT_URL)
    minutes_url = st.text_input("Latest FOMC minutes URL", value=DEFAULT_MINUTES_URL)

    st.markdown("---")
    st.subheader("Market pricing")
    market_mode = st.radio(
        "How to feed market-implied cuts",
        options=["Manual input", "Infer from end-2026 rate"],
        index=0,
    )

    current_target_upper = safe_series("fed_target_upper").dropna().iloc[-1]

    if market_mode == "Manual input":
        market_implied_cuts_bps = st.number_input(
            "Cumulative cuts priced over next 12 months (bps)",
            min_value=-100,
            max_value=400,
            value=50,
            step=25,
            help="Pull this from CME FedWatch or your preferred rates screen.",
        )
        market_implied_end_rate = float(current_target_upper) - float(market_implied_cuts_bps) / 100.0
    else:
        market_implied_end_rate = st.number_input(
            "Market-implied end-2026 fed funds rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=3.10,
            step=0.05,
            format="%.2f",
            help="Use the implied year-end rate from fed funds futures / OIS.",
        )
        market_implied_cuts_bps = int(round((float(current_target_upper) - float(market_implied_end_rate)) * 100.0))

    st.markdown("---")
    lookback_months = st.slider("Chart lookback (months)", min_value=24, max_value=120, value=60, step=12)

    st.markdown("---")
    st.subheader("About this tool")
    markdown_small(
        "This dashboard is built for the actual reaction function, not a single inflation print. "
        "It uses public FRED data for the macro state, official Fed webpages for language, and FOMC SEP median series distributed through FRED."
    )

# Load data
with st.spinner("Loading macro data and Fed language..."):
    supercore = safe_series("supercore_cpi")
    supercore_yoy = pct_change_yoy(supercore)
    supercore_3m = pct_change_3m_ann(supercore)

    wages = safe_series("wage_growth")
    unrate = safe_series("unemployment")
    anfci = safe_series("anfci")
    hy_oas = safe_series("hy_oas")
    dff = safe_series("fed_funds_effective")
    fed_target_upper = safe_series("fed_target_upper")

    sep_fed = safe_series("sep_fed_funds")
    sep_un = safe_series("sep_unemployment")
    sep_pce = safe_series("sep_pce")
    sep_core = safe_series("sep_core_pce")

    statement_text = safe_text("statement", statement_url)
    minutes_text = safe_text("minutes", minutes_url)

# Latest data
today = datetime.now()
current_year = today.year

supercore_yoy_dt, supercore_yoy_last = latest_valid(supercore_yoy)
supercore_3m_dt, supercore_3m_last = latest_valid(supercore_3m)
wage_dt, wage_last = latest_valid(wages)
un_dt, un_last = latest_valid(unrate)
anfci_dt, anfci_last = latest_valid(anfci)
hy_dt, hy_last = latest_valid(hy_oas)
dff_dt, dff_last = latest_valid(dff)
upper_dt, upper_last = latest_valid(fed_target_upper)

sep_fed_curr = get_sep_for_year(sep_fed, current_year)
sep_un_curr = get_sep_for_year(sep_un, current_year)
sep_pce_curr = get_sep_for_year(sep_pce, current_year)
sep_core_curr = get_sep_for_year(sep_core, current_year)

# Scoring
metric_rows = []

# Inflation proxy
if supercore_3m_last >= 4.0:
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

metric_rows.append({
    "Factor": "Core services ex housing",
    "Current": f"{supercore_3m_last:.2f}% (3m ann)",
    "Reference": f"{supercore_yoy_last:.2f}% YoY | SEP core PCE {sep_core_curr:.2f}%" if sep_core_curr is not None else f"{supercore_yoy_last:.2f}% YoY",
    "Read": inflation_state,
    "Score": inflation_pts,
})

# Wage growth
if wage_last >= 4.5:
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

metric_rows.append({
    "Factor": "Wage growth",
    "Current": f"{wage_last:.2f}%",
    "Reference": "Atlanta Fed Wage Growth Tracker",
    "Read": wage_state,
    "Score": wage_pts,
})

# Unemployment vs SEP
un_gap = None if sep_un_curr is None else un_last - sep_un_curr
if un_gap is None:
    un_pts = 0
    un_state = "n/a"
elif un_gap >= 0.30:
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

metric_rows.append({
    "Factor": "Unemployment",
    "Current": f"{un_last:.2f}%",
    "Reference": f"SEP median {sep_un_curr:.2f}% | gap {format_delta(un_last, sep_un_curr)}" if sep_un_curr is not None else "SEP n/a",
    "Read": un_state,
    "Score": un_pts,
})

# Financial conditions
if anfci_last >= 0.50:
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

metric_rows.append({
    "Factor": "Financial conditions",
    "Current": f"ANFCI {anfci_last:.2f} | HY OAS {hy_last:.2f}%",
    "Reference": "Positive ANFCI = tighter than average",
    "Read": fci_state,
    "Score": fci_pts,
})

# Market implied cuts vs SEP
fed_gap = None if sep_fed_curr is None else market_implied_end_rate - sep_fed_curr
if fed_gap is None:
    mkt_pts = 0
    mkt_state = "n/a"
elif fed_gap <= -0.50:
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

metric_rows.append({
    "Factor": "Market-implied cuts",
    "Current": f"{market_implied_cuts_bps} bps | implied end rate {market_implied_end_rate:.2f}%",
    "Reference": f"SEP median end-{current_year} rate {sep_fed_curr:.2f}%" if sep_fed_curr is not None else "SEP n/a",
    "Read": mkt_state,
    "Score": mkt_pts,
})

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

hits = []
if statement_hawk:
    hits.append("statement hawkish: " + ", ".join(statement_hawk[:3]))
if statement_dove:
    hits.append("statement dovish: " + ", ".join(statement_dove[:3]))
if minutes_hawk:
    hits.append("minutes hawkish: " + ", ".join(minutes_hawk[:3]))
if minutes_dove:
    hits.append("minutes dovish: " + ", ".join(minutes_dove[:3]))

metric_rows.append({
    "Factor": "Fed language",
    "Current": f"score {language_score:.2f}",
    "Reference": " | ".join(hits[:2]) if hits else "No keyword hits",
    "Read": lang_state,
    "Score": lang_pts,
})

score_df = pd.DataFrame(metric_rows)
total_score = int(score_df["Score"].sum())
policy_call = reaction_summary(total_score)

# Header cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dashboard output", policy_call.title())
c2.metric("Reaction score", f"{total_score:+d}")
c3.metric("Current fed funds upper", f"{upper_last:.2f}%")
c4.metric("SEP median end-year rate", f"{sep_fed_curr:.2f}%" if sep_fed_curr is not None else "n/a")

# Tape summary
drivers = score_df.sort_values("Score")
dovish_drivers = drivers[drivers["Score"] < 0]["Factor"].tolist()
hawkish_drivers = drivers[drivers["Score"] > 0]["Factor"].tolist()

summary_parts = []
if dovish_drivers:
    summary_parts.append("easing pressure is coming from " + ", ".join(dovish_drivers[:3]))
if hawkish_drivers:
    summary_parts.append("offset by " + ", ".join(hawkish_drivers[:3]))
if not summary_parts:
    summary_parts.append("the macro inputs are clustered around a holding pattern")

st.markdown(
    f"""
    <div style="padding:16px 18px; border:1px solid #ddd; border-radius:10px; background:#fafafa;">
    <b>Tape read:</b> The model reads <b>{policy_call}</b>. Right now {summary_parts[0]}. 
    This is a decision aid, not a mechanical forecast. The inflation proxy is a services-ex-housing CPI measure, while the SEP inflation median is core PCE, so that comparison is directional rather than apples-to-apples.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("")
st.subheader("Reaction function scorecard")
st.dataframe(
    score_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Score": st.column_config.NumberColumn(format="%d"),
    },
)

left, right = st.columns((1.1, 1.0))
with left:
    st.subheader("Macro state")
    st.plotly_chart(
        make_line_chart(supercore_3m, "Core Services ex Housing, 3m Annualized", "%", months=lookback_months),
        use_container_width=True,
    )
    st.plotly_chart(
        make_line_chart(wages, "Atlanta Fed Wage Growth Tracker", "%", months=lookback_months),
        use_container_width=True,
    )
    st.plotly_chart(
        make_line_chart(unrate, "Unemployment Rate vs SEP Median", "%", ref_value=sep_un_curr, ref_name="SEP median", months=lookback_months),
        use_container_width=True,
    )

with right:
    st.subheader("Policy context")
    st.plotly_chart(
        make_line_chart(anfci, "Financial Conditions (ANFCI)", "Index", ref_value=0.0, ref_name="Neutral", months=lookback_months),
        use_container_width=True,
    )

    fed_compare = pd.DataFrame(
        {
            "Series": ["Current fed funds upper", "Market-implied end-year rate", "SEP median end-year rate"],
            "Value": [
                upper_last,
                market_implied_end_rate,
                np.nan if sep_fed_curr is None else sep_fed_curr,
            ],
        }
    ).dropna()

    fig_bars = go.Figure()
    fig_bars.add_trace(go.Bar(x=fed_compare["Series"], y=fed_compare["Value"]))
    fig_bars.update_layout(
        title="Where policy is vs where market and SEP think it goes",
        height=320,
        margin=dict(l=20, r=20, t=55, b=20),
        yaxis_title="%",
        template="plotly_white",
    )
    st.plotly_chart(fig_bars, use_container_width=True)

    sep_tbl = pd.DataFrame(
        {
            "SEP median": ["Fed funds", "Unemployment", "PCE", "Core PCE"],
            f"{current_year}": [
                sep_fed_curr,
                sep_un_curr,
                sep_pce_curr,
                sep_core_curr,
            ],
        }
    )
    st.markdown("**SEP medians**")
    st.dataframe(sep_tbl, use_container_width=True, hide_index=True)

st.subheader("Fed language")
lang1, lang2 = st.columns(2)
with lang1:
    st.markdown(f"**Statement score:** {statement_score:.2f}")
    st.write(
        "Hawkish hits: " + (", ".join(statement_hawk[:6]) if statement_hawk else "none")
    )
    st.write(
        "Dovish hits: " + (", ".join(statement_dove[:6]) if statement_dove else "none")
    )
    st.text_area(
        "Latest FOMC statement excerpt",
        value=statement_text[:2500],
        height=220,
    )

with lang2:
    st.markdown(f"**Minutes score:** {minutes_score:.2f}")
    st.write(
        "Hawkish hits: " + (", ".join(minutes_hawk[:6]) if minutes_hawk else "none")
    )
    st.write(
        "Dovish hits: " + (", ".join(minutes_dove[:6]) if minutes_dove else "none")
    )
    st.text_area(
        "Latest FOMC minutes excerpt",
        value=minutes_text[:2500],
        height=220,
    )

with st.expander("Diagnostics and data freshness"):
    freshness = pd.DataFrame(
        [
            ["Core services ex housing", str(supercore_3m_dt.date()), SERIES["supercore_cpi"]["id"]],
            ["Wage growth", str(wage_dt.date()), SERIES["wage_growth"]["id"]],
            ["Unemployment", str(un_dt.date()), SERIES["unemployment"]["id"]],
            ["ANFCI", str(anfci_dt.date()), SERIES["anfci"]["id"]],
            ["HY OAS", str(hy_dt.date()), SERIES["hy_oas"]["id"]],
            ["Fed funds upper", str(upper_dt.date()), SERIES["fed_target_upper"]["id"]],
            ["SEP fed funds", str(sep_fed.dropna().index.max().date()) if not sep_fed.dropna().empty else "n/a", SERIES["sep_fed_funds"]["id"]],
            ["SEP unemployment", str(sep_un.dropna().index.max().date()) if not sep_un.dropna().empty else "n/a", SERIES["sep_unemployment"]["id"]],
            ["SEP PCE", str(sep_pce.dropna().index.max().date()) if not sep_pce.dropna().empty else "n/a", SERIES["sep_pce"]["id"]],
            ["SEP core PCE", str(sep_core.dropna().index.max().date()) if not sep_core.dropna().empty else "n/a", SERIES["sep_core_pce"]["id"]],
        ],
        columns=["Series", "Latest observation", "FRED ID"],
    )
    st.dataframe(freshness, use_container_width=True, hide_index=True)
    st.markdown("**Fed URLs used**")
    st.code(f"Statement: {statement_url}\nMinutes: {minutes_url}")

st.markdown("---")
markdown_small(
    "Design choice: this tool uses CPI services less rent of shelter as the closest public, stable proxy for core services ex housing; the Fed's preferred inflation target remains PCE, and the SEP inflation medians are reported in PCE and core PCE."
)
