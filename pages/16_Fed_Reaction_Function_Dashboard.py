import re
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title="Fed Reaction Function Dashboard", layout="wide")

TITLE = "Fed Reaction Function Dashboard"
DEFAULT_STATEMENT_URL = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260318a.htm"
DEFAULT_MINUTES_URL = "https://www.federalreserve.gov/monetarypolicy/fomcminutes20260318.htm"

SERIES = {
    "supercore": "CUSR0000SASL2RS",          # CPI services less rent of shelter
    "wages": "FRBATLWGT3MMAWMHWGO",          # Atlanta Fed wage tracker
    "unrate": "UNRATE",
    "anfci": "ANFCI",
    "hy_oas": "BAMLH0A0HYM2",
    "fed_upper": "DFEDTARU",
    "sep_fed": "FEDTARMD",
    "sep_unrate": "UNRATEMD",
    "sep_pce": "PCECTPIMD",
    "sep_core_pce": "JCXFEMD",
}

HAWK = {
    r"\bhigher for longer\b": 2.0,
    r"\bongoing increases\b": 2.0,
    r"\bsome additional policy firming\b": 2.0,
    r"\bupside risks to inflation\b": 1.5,
    r"\binflation remains elevated\b": 1.0,
    r"\brestrictive\b": 0.5,
    r"\bstrong labor market\b": 0.5,
    r"\bnot appropriate to reduce\b": 1.5,
}

DOVE = {
    r"\bbegin to reduce\b": 2.0,
    r"\breduce the target range\b": 2.0,
    r"\bgreater confidence\b": 1.5,
    r"\binflation has eased\b": 1.0,
    r"\blabor market conditions have eased\b": 1.0,
    r"\bunemployment has moved up\b": 1.0,
    r"\beconomic activity has slowed\b": 1.0,
    r"\brisks to employment\b": 1.25,
}

st.markdown(f"# {TITLE}")
st.caption("Map core services ex housing, wage growth, unemployment, financial conditions, market pricing, Fed language, and SEP medians into a single policy-bias readout.")

def fred_csv(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={start:%Y-%m-%d}&coed={end:%Y-%m-%d}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    if df.empty or len(df.columns) < 2:
        raise ValueError(f"No usable data for {series_id}")
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    df = df.dropna(subset=[df.columns[0]]).sort_values(df.columns[0])
    return pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0], name=series_id)

@st.cache_data(ttl=3600, show_spinner=False)
def load_series(start: pd.Timestamp, end: pd.Timestamp):
    out, errs = {}, {}
    for k, sid in SERIES.items():
        try:
            out[k] = fred_csv(sid, start, end)
        except Exception as e:
            out[k] = pd.Series(dtype="float64", name=k)
            errs[k] = f"{sid}: {e}"
    return out, errs

@st.cache_data(ttl=1800, show_spinner=False)
def page_text(url: str) -> str:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
    return text

def yoy(s: pd.Series) -> pd.Series:
    return s.pct_change(12) * 100

def ann3m(s: pd.Series) -> pd.Series:
    return ((s / s.shift(3)) ** 4 - 1) * 100

def last(s: pd.Series):
    s = s.dropna()
    return float("nan") if s.empty else float(s.iloc[-1])

def sep_val(s: pd.Series, year: int):
    s = s.dropna()
    s = s[s.index.year == year]
    return None if s.empty else float(s.iloc[-1])

def txt_score(text: str):
    score, hawk_hits, dove_hits = 0.0, [], []
    t = text.lower() if text else ""
    for p, w in HAWK.items():
        if re.search(p, t):
            score += w
            hawk_hits.append(p.replace(r"\b", "").replace("\\", ""))
    for p, w in DOVE.items():
        if re.search(p, t):
            score -= w
            dove_hits.append(p.replace(r"\b", "").replace("\\", ""))
    return score, hawk_hits, dove_hits

def classify(total):
    if total <= -2:
        return "easing bias rising"
    if total >= 2:
        return "tightening risk re-emerging"
    return "hold bias intact"

def fig_line(s: pd.Series, title: str, months: int, ytitle: str = "%", ref=None, ref_name="Reference"):
    s = s.dropna()
    if not s.empty:
        s = s[s.index >= s.index.max() - pd.DateOffset(months=months)]
    fig = go.Figure()
    if not s.empty:
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=title, line=dict(width=2.5)))
    if ref is not None and pd.notna(ref):
        fig.add_hline(y=ref, line_dash="dash", annotation_text=ref_name, annotation_position="top left")
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=20, r=20, t=50, b=20), title=title, yaxis_title=ytitle)
    return fig

with st.sidebar:
    years = st.selectbox("Lookback", [2, 3, 5, 10], index=2)
    market_mode = st.radio("Market pricing input", ["Cuts in bps", "Implied end-year rate"], index=0)
    cuts_bps = st.number_input("Cuts over next 12 months (bps)", -100, 400, 50, 25) if market_mode == "Cuts in bps" else None
    implied_rate = st.number_input("Implied end-year fed funds rate (%)", 0.0, 10.0, 3.10, 0.05, format="%.2f") if market_mode == "Implied end-year rate" else None
    use_language = st.checkbox("Include Fed language", value=False)
    statement_url = st.text_input("Statement URL", DEFAULT_STATEMENT_URL) if use_language else DEFAULT_STATEMENT_URL
    minutes_url = st.text_input("Minutes URL", DEFAULT_MINUTES_URL) if use_language else DEFAULT_MINUTES_URL

today = pd.Timestamp.today().normalize()
start = today - pd.DateOffset(years=years)

with st.spinner("Loading data..."):
    raw, errors = load_series(start, today)

required = ["supercore", "wages", "unrate", "anfci", "fed_upper", "sep_fed", "sep_unrate", "sep_pce", "sep_core_pce"]
missing = [k for k in required if raw[k].empty]
if missing:
    st.error("Required series failed to load: " + ", ".join(SERIES[k] for k in missing))
    with st.expander("Error details"):
        for v in errors.values():
            st.write(v)
    st.stop()

df = pd.concat(raw.values(), axis=1, keys=raw.keys()).ffill()
supercore_3m = ann3m(df["supercore"])
supercore_yoy = yoy(df["supercore"])

year = today.year
upper = last(df["fed_upper"])
sep_fed = sep_val(df["sep_fed"], year)
sep_un = sep_val(df["sep_unrate"], year)
sep_pce = sep_val(df["sep_pce"], year)
sep_core = sep_val(df["sep_core_pce"], year)

if market_mode == "Cuts in bps":
    implied_end = upper - cuts_bps / 100.0
else:
    implied_end = implied_rate
    cuts_bps = int(round((upper - implied_end) * 100))

rows = []

x = last(supercore_3m)
if x >= 4.0:
    pts, read = 2, "re-accelerating"
elif x >= 3.0:
    pts, read = 1, "sticky"
elif x <= 2.25:
    pts, read = -2, "cooling fast"
elif x <= 2.75:
    pts, read = -1, "cooling"
else:
    pts, read = 0, "mixed"
rows.append(["Core services ex housing", f"{x:.2f}% (3m ann)", f"{last(supercore_yoy):.2f}% YoY | SEP core PCE {sep_core:.2f}%", read, pts])

x = last(df["wages"])
if x >= 4.5:
    pts, read = 2, "too hot"
elif x >= 4.0:
    pts, read = 1, "still firm"
elif x <= 3.0:
    pts, read = -2, "cool enough"
elif x <= 3.5:
    pts, read = -1, "cooling"
else:
    pts, read = 0, "balanced"
rows.append(["Wage growth", f"{x:.2f}%", "Atlanta Fed tracker", read, pts])

x = last(df["unrate"])
gap = x - sep_un
if gap >= 0.30:
    pts, read = -2, "labor cooling faster than SEP"
elif gap >= 0.10:
    pts, read = -1, "modestly softer than SEP"
elif gap <= -0.30:
    pts, read = 2, "labor still tighter than SEP"
elif gap <= -0.10:
    pts, read = 1, "slightly tighter than SEP"
else:
    pts, read = 0, "near SEP"
rows.append(["Unemployment", f"{x:.2f}%", f"SEP median {sep_un:.2f}% | gap {gap:+.2f}", read, pts])

x = last(df["anfci"])
hy = last(df["hy_oas"])
if x >= 0.50:
    pts, read = -2, "materially tighter"
elif x >= 0.15:
    pts, read = -1, "tighter"
elif x <= -0.50:
    pts, read = 2, "very easy"
elif x <= -0.15:
    pts, read = 1, "easy"
else:
    pts, read = 0, "near neutral"
rows.append(["Financial conditions", f"ANFCI {x:.2f} | HY OAS {hy:.2f}%", "Positive ANFCI = tighter", read, pts])

gap = implied_end - sep_fed
if gap <= -0.50:
    pts, read = -2, "market pricing much more easing than SEP"
elif gap <= -0.25:
    pts, read = -1, "market leaning easier than SEP"
elif gap >= 0.50:
    pts, read = 2, "market pricing tighter path than SEP"
elif gap >= 0.25:
    pts, read = 1, "market fading cuts vs SEP"
else:
    pts, read = 0, "near SEP path"
rows.append(["Market-implied cuts", f"{cuts_bps} bps | implied end rate {implied_end:.2f}%", f"SEP median end-{year} rate {sep_fed:.2f}%", read, pts])

stmt_text, mins_text = "", ""
stmt_score = mins_score = 0.0
stmt_h = stmt_d = mins_h = mins_d = []
if use_language:
    try:
        stmt_text = page_text(statement_url)
        mins_text = page_text(minutes_url)
        stmt_score, stmt_h, stmt_d = txt_score(stmt_text)
        mins_score, mins_h, mins_d = txt_score(mins_text)
    except Exception as e:
        st.warning(f"Fed language load failed: {e}")

lang_score = round(0.65 * stmt_score + 0.35 * mins_score, 2)
if lang_score <= -2:
    pts, read = -2, "clear easing lean"
elif lang_score <= -0.5:
    pts, read = -1, "softening tone"
elif lang_score >= 2:
    pts, read = 2, "hawkish hold"
elif lang_score >= 0.5:
    pts, read = 1, "guarded / restrictive"
else:
    pts, read = 0, "balanced hold"
rows.append(["Fed language", f"score {lang_score:.2f}", "Statement and minutes keyword score", read, pts])

score_df = pd.DataFrame(rows, columns=["Factor", "Current", "Reference", "Read", "Score"])
total = int(score_df["Score"].sum())
call = classify(total)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dashboard output", call.title())
c2.metric("Reaction score", f"{total:+d}")
c3.metric("Fed funds upper", f"{upper:.2f}%")
c4.metric("SEP end-year median", f"{sep_fed:.2f}%")

st.dataframe(score_df, use_container_width=True, hide_index=True)

l, r = st.columns(2)
with l:
    st.plotly_chart(fig_line(supercore_3m, "Core Services ex Housing, 3m Annualized", years * 12), use_container_width=True)
    st.plotly_chart(fig_line(df["wages"], "Atlanta Fed Wage Growth Tracker", years * 12), use_container_width=True)
    st.plotly_chart(fig_line(df["unrate"], "Unemployment Rate vs SEP Median", years * 12, ref=sep_un, ref_name="SEP median"), use_container_width=True)
with r:
    st.plotly_chart(fig_line(df["anfci"], "Financial Conditions (ANFCI)", years * 12, ytitle="Index", ref=0, ref_name="Neutral"), use_container_width=True)
    bar = pd.DataFrame({"Series": ["Current fed funds upper", "Market-implied end-year rate", "SEP median end-year rate"], "Value": [upper, implied_end, sep_fed]})
    fig = go.Figure(go.Bar(x=bar["Series"], y=bar["Value"]))
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=20, r=20, t=50, b=20), title="Where policy is vs where market and SEP think it goes", yaxis_title="%")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        pd.DataFrame({"SEP median": ["Fed funds", "Unemployment", "PCE", "Core PCE"], str(year): [sep_fed, sep_un, sep_pce, sep_core]}),
        use_container_width=True,
        hide_index=True,
    )

if use_language:
    a, b = st.columns(2)
    with a:
        st.markdown(f"**Statement score:** {stmt_score:.2f}")
        st.write("Hawkish hits:", ", ".join(stmt_h[:6]) if stmt_h else "none")
        st.write("Dovish hits:", ", ".join(stmt_d[:6]) if stmt_d else "none")
        st.text_area("Statement excerpt", stmt_text[:2500] if stmt_text else "No text loaded.", height=220)
    with b:
        st.markdown(f"**Minutes score:** {mins_score:.2f}")
        st.write("Hawkish hits:", ", ".join(mins_h[:6]) if mins_h else "none")
        st.write("Dovish hits:", ", ".join(mins_d[:6]) if mins_d else "none")
        st.text_area("Minutes excerpt", mins_text[:2500] if mins_text else "No text loaded.", height=220)

with st.expander("Diagnostics"):
    if errors:
        for v in errors.values():
            st.write(v)
