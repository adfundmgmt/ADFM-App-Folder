# app.py
# ADFM | FOMC Scenario Engine
#
# What this app does
# - Scrapes Federal Reserve FOMC calendar pages and individual meeting pages
# - Pulls meeting statements, minutes, press conference transcripts, and SEP materials where available
# - Extracts text from HTML and PDF sources
# - Builds a historical analog engine using document similarity
# - Scores current FOMC tone across scenarios:
#     1) Higher for longer
#     2) Restrictive hold
#     3) Dovish hold / pivot setup
#     4) Cutting cycle
#     5) Stagflation risk
# - Pulls live macro series from FRED CSV endpoints with retries and caching
# - Produces a clean Streamlit output tailored for discretionary macro work
#
# Run:
#   pip install streamlit pandas numpy requests beautifulsoup4 plotly pypdf PyPDF2 python-dateutil
#   streamlit run app.py

from __future__ import annotations

import io
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dateutil import parser as dtparser

# Robust PDF reader import for Streamlit Cloud and local environments
PDF_ENGINE = None
try:
    from pypdf import PdfReader  # preferred
    PDF_ENGINE = "pypdf"
except Exception:
    try:
        from PyPDF2 import PdfReader  # fallback
        PDF_ENGINE = "PyPDF2"
    except Exception:
        PdfReader = None
        PDF_ENGINE = None

# =========================
# App config
# =========================

st.set_page_config(
    page_title="ADFM | FOMC Scenario Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
)

FED_BASE = "https://www.federalreserve.gov"
FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"

REQUEST_TIMEOUT = 25
MAX_RETRIES = 3
RETRY_SLEEP = 1.5

# =========================
# Utility helpers
# =========================

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def pct_fmt(x, decimals=1):
    if pd.isna(x):
        return "N/A"
    return f"{x:.{decimals}f}%"


def num_fmt(x, decimals=2):
    if pd.isna(x):
        return "N/A"
    return f"{x:.{decimals}f}"


def zscore(series: pd.Series, window: int = 252) -> pd.Series:
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std()
    return (series - roll_mean) / roll_std


def annualized_3m_from_index(series: pd.Series) -> pd.Series:
    return ((series / series.shift(3)) ** 4 - 1) * 100


def yoy(series: pd.Series) -> pd.Series:
    return (series / series.shift(12) - 1) * 100


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def fetch_url(url: str, session: Optional[requests.Session] = None, timeout: int = REQUEST_TIMEOUT) -> requests.Response:
    sess = session or make_session()
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = sess.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP * (attempt + 1))
    raise last_err


def absolute_url(href: str) -> str:
    if not href:
        return ""
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        return FED_BASE + href
    return FED_BASE + "/" + href


def try_parse_date(text: str) -> Optional[pd.Timestamp]:
    try:
        return pd.Timestamp(dtparser.parse(text, fuzzy=True))
    except Exception:
        return None


# =========================
# FRED data loader
# =========================

@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_fred_series_csv(series_id: str, start: str = "1990-01-01") -> pd.DataFrame:
    url = FRED_CSV_URL.format(series_id=series_id, start=start)
    resp = fetch_url(url)
    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty or "DATE" not in df.columns or series_id not in df.columns:
        raise ValueError(f"Malformed FRED response for {series_id}")
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    df = df.dropna(subset=["DATE"]).rename(columns={"DATE": "date", series_id: "value"})
    return df[["date", "value"]].copy()


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_macro_panel() -> pd.DataFrame:
    # FEDTARMD = Effective Federal Funds Target midpoint
    # UNRATE = unemployment rate
    # PCEPILFE = Core PCE index
    # CPIAUCSL = CPI index
    # GS10 = 10Y Treasury
    # DGS2 = 2Y Treasury
    # NFCI = Chicago Fed National Financial Conditions Index
    # PAYEMS = nonfarm payrolls
    series = {
        "FEDTARMD": "1990-01-01",
        "UNRATE": "1990-01-01",
        "PCEPILFE": "1990-01-01",
        "CPIAUCSL": "1990-01-01",
        "GS10": "1990-01-01",
        "DGS2": "1990-01-01",
        "NFCI": "1990-01-01",
        "PAYEMS": "1990-01-01",
    }

    panel = None
    failures = []

    for sid, start in series.items():
        try:
            s = fetch_fred_series_csv(sid, start=start).rename(columns={"value": sid})
            panel = s if panel is None else panel.merge(s, on="date", how="outer")
        except Exception as e:
            failures.append(f"{sid}: {e}")

    if panel is None:
        raise RuntimeError("Failed to load all required macro series.")

    panel = panel.sort_values("date").reset_index(drop=True)

    panel["core_pce_yoy"] = yoy(panel["PCEPILFE"])
    panel["cpi_yoy"] = yoy(panel["CPIAUCSL"])
    panel["payems_3m_ann"] = annualized_3m_from_index(panel["PAYEMS"])
    panel["2s10s"] = panel["GS10"] - panel["DGS2"]
    panel["real_policy_proxy"] = panel["FEDTARMD"] - panel["core_pce_yoy"]

    panel.attrs["failures"] = failures
    return panel


# =========================
# Fed scraping
# =========================

@dataclass
class MeetingDocLinks:
    statement_html: Optional[str] = None
    statement_pdf: Optional[str] = None
    minutes_html: Optional[str] = None
    minutes_pdf: Optional[str] = None
    presser_pdf: Optional[str] = None
    projections_pdf: Optional[str] = None
    projections_html: Optional[str] = None
    implementation_note: Optional[str] = None


@dataclass
class MeetingRecord:
    meeting_date: pd.Timestamp
    title: str
    page_url: str
    year: int
    doc_links: MeetingDocLinks = field(default_factory=MeetingDocLinks)
    statement_text: str = ""
    minutes_text: str = ""
    presser_text: str = ""
    projections_text: str = ""

    @property
    def combined_text(self) -> str:
        return clean_text(
            " ".join(
                [
                    self.statement_text[:30000],
                    self.minutes_text[:50000],
                    self.presser_text[:40000],
                    self.projections_text[:20000],
                ]
            )
        )


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def fetch_html(url: str) -> str:
    return fetch_url(url).text


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def fetch_pdf_text(url: str) -> str:
    if PdfReader is None:
        raise ImportError(
            "No PDF reader library is installed. Add 'pypdf>=4.0.0' to requirements.txt "
            "or install PyPDF2 as a fallback."
        )

    resp = fetch_url(url)
    bio = io.BytesIO(resp.content)

    try:
        reader = PdfReader(bio)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF from {url}: {e}")

    pages = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
            pages.append(txt)
        except Exception:
            pages.append("")

    return clean_text("\n".join(pages))


def extract_main_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    candidates = []
    for sel in ["#article", "main", ".col-xs-12", ".col-md-8", ".panel", "body"]:
        found = soup.select(sel)
        if found:
            candidates.extend(found)

    if not candidates:
        candidates = [soup]

    texts = []
    for c in candidates[:5]:
        txt = c.get_text(" ", strip=True)
        if len(txt) > 1000:
            texts.append(txt)

    if not texts:
        texts = [soup.get_text(" ", strip=True)]

    text = max(texts, key=len)
    return clean_text(text)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def fetch_html_text(url: str) -> str:
    html = fetch_html(url)
    return extract_main_html_text(html)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def get_fomc_calendar_meetings() -> List[Dict]:
    html = fetch_html(FOMC_CALENDAR_URL)
    soup = BeautifulSoup(html, "html.parser")

    meetings = []
    links = soup.find_all("a", href=True)

    seen = set()
    for a in links:
        href = a["href"]
        text = normalize_whitespace(a.get_text(" ", strip=True))
        full_url = absolute_url(href)

        if re.search(r"/monetarypolicy/fomc.*\d{8}\.htm", href, flags=re.I):
            if full_url in seen:
                continue
            seen.add(full_url)

            date_match = re.search(r"(\d{8})", href)
            if not date_match:
                continue

            dt = pd.to_datetime(date_match.group(1), format="%Y%m%d", errors="coerce")
            if pd.isna(dt):
                continue

            title = text or f"FOMC Meeting {dt.date()}"
            meetings.append(
                {
                    "meeting_date": dt,
                    "title": title,
                    "page_url": full_url,
                    "year": int(dt.year),
                }
            )

    meetings = sorted(meetings, key=lambda x: x["meeting_date"], reverse=True)

    dedup = {}
    for m in meetings:
        dedup[m["meeting_date"].date()] = m
    meetings = list(dedup.values())
    meetings = sorted(meetings, key=lambda x: x["meeting_date"], reverse=True)
    return meetings


def parse_meeting_page_docs(page_url: str) -> MeetingDocLinks:
    html = fetch_html(page_url)
    soup = BeautifulSoup(html, "html.parser")
    docs = MeetingDocLinks()

    anchors = soup.find_all("a", href=True)

    for a in anchors:
        label = normalize_whitespace(a.get_text(" ", strip=True)).lower()
        href = absolute_url(a["href"])

        if "statement" in label and href.endswith(".htm"):
            docs.statement_html = docs.statement_html or href
        elif "statement" in label and href.endswith(".pdf"):
            docs.statement_pdf = docs.statement_pdf or href

        elif "minutes" in label and href.endswith(".htm"):
            docs.minutes_html = docs.minutes_html or href
        elif "minutes" in label and href.endswith(".pdf"):
            docs.minutes_pdf = docs.minutes_pdf or href

        elif "press conference transcript" in label and href.endswith(".pdf"):
            docs.presser_pdf = docs.presser_pdf or href

        elif "projections materials" in label and href.endswith(".pdf"):
            docs.projections_pdf = docs.projections_pdf or href
        elif "projections materials" in label and href.endswith(".htm"):
            docs.projections_html = docs.projections_html or href

        elif "implementation note" in label:
            docs.implementation_note = docs.implementation_note or href

    if not docs.presser_pdf:
        for a in anchors:
            href = absolute_url(a["href"])
            if re.search(r"presconf.*\.pdf$", href, flags=re.I):
                docs.presser_pdf = href
                break

    if not docs.minutes_pdf:
        for a in anchors:
            href = absolute_url(a["href"])
            if re.search(r"fomcminutes\d{8}\.pdf$", href, flags=re.I):
                docs.minutes_pdf = href
                break

    if not docs.minutes_html:
        for a in anchors:
            href = absolute_url(a["href"])
            if re.search(r"fomcminutes\d{8}\.htm$", href, flags=re.I):
                docs.minutes_html = href
                break

    return docs


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_meeting_record(page_url: str, meeting_date: pd.Timestamp, title: str, year: int) -> MeetingRecord:
    docs = parse_meeting_page_docs(page_url)

    rec = MeetingRecord(
        meeting_date=meeting_date,
        title=title,
        page_url=page_url,
        year=year,
        doc_links=docs,
    )

    try:
        if docs.statement_html:
            rec.statement_text = fetch_html_text(docs.statement_html)
        elif docs.statement_pdf:
            rec.statement_text = fetch_pdf_text(docs.statement_pdf)
    except Exception:
        rec.statement_text = ""

    try:
        if docs.minutes_html:
            rec.minutes_text = fetch_html_text(docs.minutes_html)
        elif docs.minutes_pdf:
            rec.minutes_text = fetch_pdf_text(docs.minutes_pdf)
    except Exception:
        rec.minutes_text = ""

    try:
        if docs.presser_pdf:
            rec.presser_text = fetch_pdf_text(docs.presser_pdf)
    except Exception:
        rec.presser_text = ""

    try:
        if docs.projections_html:
            rec.projections_text = fetch_html_text(docs.projections_html)
        elif docs.projections_pdf:
            rec.projections_text = fetch_pdf_text(docs.projections_pdf)
    except Exception:
        rec.projections_text = ""

    return rec


# =========================
# Text analytics
# =========================

STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "have", "were", "will", "their",
    "they", "been", "into", "which", "would", "there", "about", "could", "should", "while",
    "than", "also", "because", "these", "those", "such", "each", "other", "more", "less",
    "very", "over", "under", "some", "most", "much", "many", "only", "both", "within",
    "committee", "federal", "reserve", "board", "meeting", "monetary", "policy", "percent",
    "participants", "economic", "economy", "market", "markets", "inflation", "labor",
    "employment", "growth", "price", "prices", "rates", "rate", "financial", "conditions",
}


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    toks = re.findall(r"\b[a-z][a-z0-9\-]{2,}\b", text)
    toks = [t for t in toks if t not in STOPWORDS]
    return toks


def tf_vector(text: str) -> Counter:
    return Counter(tokenize(text))


def cosine_counter(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def count_keywords(text: str, terms: List[str]) -> int:
    text_l = (text or "").lower()
    return sum(text_l.count(t.lower()) for t in terms)


SCENARIOS = {
    "Higher for longer": {
        "positive_terms": [
            "higher for longer", "ongoing inflation", "upside risks to inflation", "restrictive",
            "further policy firming", "not sufficiently restrictive", "elevated inflation",
            "resilient demand", "strong labor market", "tight labor market"
        ],
        "negative_terms": [
            "disinflation", "cooling labor market", "softer growth", "rate cuts", "lower rates"
        ],
    },
    "Restrictive hold": {
        "positive_terms": [
            "well positioned", "wait for more evidence", "data dependent", "balance of risks",
            "maintain current target range", "monitor incoming data", "uncertain outlook",
            "restrictive stance", "hold rates steady"
        ],
        "negative_terms": [
            "imminent cuts", "need to cut", "rapid deterioration", "severe contraction"
        ],
    },
    "Dovish hold / pivot setup": {
        "positive_terms": [
            "disinflation", "progress on inflation", "downside risks to employment",
            "policy is restrictive", "cooling labor market", "slower demand", "moderating wages",
            "balanced risks", "scope to adjust", "additional adjustments"
        ],
        "negative_terms": [
            "upside inflation risks", "further firming", "reacceleration"
        ],
    },
    "Cutting cycle": {
        "positive_terms": [
            "cut the target range", "lower the target range", "downside risks",
            "deterioration in labor market", "weaker activity", "recession", "support the economy",
            "ease policy", "policy accommodation"
        ],
        "negative_terms": [
            "elevated inflation", "strong labor market", "further firming"
        ],
    },
    "Stagflation risk": {
        "positive_terms": [
            "higher energy prices", "elevated inflation", "slower growth", "weaker activity",
            "supply shock", "inflation remains above target", "downside risks to growth",
            "adverse supply developments", "tariffs", "geopolitical risks"
        ],
        "negative_terms": [
            "broad disinflation", "strong productivity", "cooling prices without growth damage"
        ],
    },
}

HAWK_TERMS = [
    "further policy firming", "upside risks to inflation", "elevated inflation",
    "restrictive", "strong labor market", "tight labor market", "not sufficiently restrictive",
]
DOVE_TERMS = [
    "disinflation", "downside risks to employment", "cooling labor market",
    "softer growth", "policy is restrictive", "balanced risks", "lower rates", "rate cuts"
]


def score_scenarios(doc_text: str, macro_snapshot: Dict[str, float]) -> pd.DataFrame:
    rows = []

    for scenario, cfg in SCENARIOS.items():
        pos = count_keywords(doc_text, cfg["positive_terms"])
        neg = count_keywords(doc_text, cfg["negative_terms"])
        raw = pos - 0.7 * neg

        if scenario == "Higher for longer":
            if macro_snapshot.get("core_pce_yoy", np.nan) > 2.7:
                raw += 2.0
            if macro_snapshot.get("unrate", np.nan) < 4.3:
                raw += 1.0
            if macro_snapshot.get("real_policy_proxy", np.nan) < 1.0:
                raw += 0.5

        elif scenario == "Restrictive hold":
            if 2.2 <= macro_snapshot.get("core_pce_yoy", np.nan) <= 3.0:
                raw += 1.0
            if 4.0 <= macro_snapshot.get("unrate", np.nan) <= 4.8:
                raw += 1.0

        elif scenario == "Dovish hold / pivot setup":
            if macro_snapshot.get("core_pce_yoy", np.nan) < 2.8:
                raw += 1.0
            if macro_snapshot.get("unrate_3m_change", np.nan) > 0.2:
                raw += 1.2
            if macro_snapshot.get("payems_3m_ann", np.nan) < 1.5:
                raw += 0.8

        elif scenario == "Cutting cycle":
            if macro_snapshot.get("unrate_3m_change", np.nan) > 0.4:
                raw += 2.0
            if macro_snapshot.get("payems_3m_ann", np.nan) < 0:
                raw += 1.5
            if macro_snapshot.get("nfci", np.nan) > 0.4:
                raw += 1.0

        elif scenario == "Stagflation risk":
            if macro_snapshot.get("core_pce_yoy", np.nan) > 2.8 and macro_snapshot.get("payems_3m_ann", np.nan) < 1.0:
                raw += 2.0
            if macro_snapshot.get("unrate_3m_change", np.nan) > 0.2 and macro_snapshot.get("core_pce_yoy", np.nan) > 2.8:
                raw += 1.5

        rows.append({"scenario": scenario, "raw_score": raw})

    df = pd.DataFrame(rows).sort_values("raw_score", ascending=False).reset_index(drop=True)

    x = df["raw_score"].astype(float).values
    x = x - np.nanmax(x)
    probs = np.exp(x)
    probs = probs / probs.sum() if probs.sum() > 0 else np.repeat(1 / len(df), len(df))
    df["probability"] = probs * 100
    return df


def hawk_dove_balance(doc_text: str) -> Dict[str, float]:
    hawk = count_keywords(doc_text, HAWK_TERMS)
    dove = count_keywords(doc_text, DOVE_TERMS)
    total = hawk + dove
    balance = 0.0 if total == 0 else (hawk - dove) / total
    return {
        "hawk_hits": hawk,
        "dove_hits": dove,
        "balance": balance,
    }


def top_terms(text: str, n: int = 30) -> pd.DataFrame:
    c = tf_vector(text)
    common = c.most_common(n)
    return pd.DataFrame(common, columns=["term", "count"])


# =========================
# Analog engine
# =========================

def build_similarity_frame(current: MeetingRecord, history: List[MeetingRecord]) -> pd.DataFrame:
    current_vec = tf_vector(current.combined_text)
    rows = []

    for rec in history:
        if rec.meeting_date >= current.meeting_date:
            continue
        sim = cosine_counter(current_vec, tf_vector(rec.combined_text))
        rows.append({
            "meeting_date": rec.meeting_date,
            "title": rec.title,
            "year": rec.year,
            "page_url": rec.page_url,
            "similarity": sim,
        })

    if not rows:
        return pd.DataFrame(columns=["meeting_date", "title", "year", "page_url", "similarity"])

    df = pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)
    return df


# =========================
# Macro snapshot
# =========================

def latest_macro_snapshot(panel: pd.DataFrame) -> Dict[str, float]:
    df = panel.sort_values("date").copy()
    latest = df.iloc[-1]
    prev_3m = df.iloc[-4] if len(df) >= 4 else latest

    snap = {
        "date": latest["date"],
        "fed_target_mid": safe_float(latest.get("FEDTARMD")),
        "unrate": safe_float(latest.get("UNRATE")),
        "core_pce_yoy": safe_float(latest.get("core_pce_yoy")),
        "cpi_yoy": safe_float(latest.get("cpi_yoy")),
        "payems_3m_ann": safe_float(latest.get("payems_3m_ann")),
        "gs10": safe_float(latest.get("GS10")),
        "dgs2": safe_float(latest.get("DGS2")),
        "curve_2s10s": safe_float(latest.get("2s10s")),
        "nfci": safe_float(latest.get("NFCI")),
        "real_policy_proxy": safe_float(latest.get("real_policy_proxy")),
        "unrate_3m_change": safe_float(latest.get("UNRATE")) - safe_float(prev_3m.get("UNRATE")),
    }
    return snap


def macro_regime_label(snap: Dict[str, float]) -> str:
    inf = snap.get("core_pce_yoy", np.nan)
    ur = snap.get("unrate", np.nan)
    ur_chg = snap.get("unrate_3m_change", np.nan)
    real = snap.get("real_policy_proxy", np.nan)

    if inf > 2.8 and ur < 4.3 and real < 2.0:
        return "Inflation persistence / policy still restrictive but not breaking labor"
    if inf > 2.8 and ur_chg > 0.2:
        return "Late-cycle stagflation risk"
    if inf < 2.6 and ur_chg > 0.2:
        return "Pivot setup / labor softening"
    if inf < 2.4 and ur_chg > 0.4:
        return "Cutting cycle setup"
    return "Restrictive hold / data-dependent plateau"


# =========================
# Streamlit UI
# =========================

st.title("ADFM | FOMC Scenario Engine")
st.caption("Historical analogs, scenario probabilities, Fed language shifts, and current macro overlay")

with st.sidebar:
    st.subheader("Settings")
    max_meetings = st.slider("Meetings to load", min_value=8, max_value=40, value=18, step=2)
    include_minutes = st.checkbox("Use minutes", value=True)
    include_pressers = st.checkbox("Use press conference transcripts", value=True)
    include_sep = st.checkbox("Use SEP materials", value=False)
    show_raw_text = st.checkbox("Show source text excerpts", value=False)

    st.markdown("---")
    st.caption(f"PDF engine: {PDF_ENGINE or 'none'}")

    st.markdown(
        """
        **About This Tool**

        This app scrapes the Federal Reserve's FOMC meeting pages, then builds a scenario engine
        around the text of statements, minutes, and press conferences. It overlays current macro
        conditions from FRED and finds the closest historical analog meetings by document similarity.

        It is designed for discretionary macro use, not as a forecasting black box.
        """
    )

with st.spinner("Loading macro panel from FRED..."):
    macro_panel = load_macro_panel()
macro_snap = latest_macro_snapshot(macro_panel)

with st.spinner("Loading FOMC meeting list from the Federal Reserve..."):
    meetings_meta = get_fomc_calendar_meetings()

if not meetings_meta:
    st.error("No FOMC meetings found from the Fed calendar page.")
    st.stop()

meetings_meta = meetings_meta[:max_meetings]

col_a, col_b, col_c, col_d, col_e = st.columns(5)
col_a.metric("Fed Target Mid", pct_fmt(macro_snap["fed_target_mid"], 2))
col_b.metric("Core PCE YoY", pct_fmt(macro_snap["core_pce_yoy"], 2))
col_c.metric("Unemployment", pct_fmt(macro_snap["unrate"], 1))
col_d.metric("3M Δ Unemployment", pct_fmt(macro_snap["unrate_3m_change"], 1))
col_e.metric("2s10s", pct_fmt(macro_snap["curve_2s10s"], 2))

st.markdown(f"**Current macro regime:** {macro_regime_label(macro_snap)}")

with st.expander("Macro snapshot details", expanded=False):
    st.write(
        {
            "as_of": str(pd.Timestamp(macro_snap["date"]).date()),
            "fed_target_mid": macro_snap["fed_target_mid"],
            "core_pce_yoy": macro_snap["core_pce_yoy"],
            "cpi_yoy": macro_snap["cpi_yoy"],
            "unrate": macro_snap["unrate"],
            "unrate_3m_change": macro_snap["unrate_3m_change"],
            "payems_3m_ann": macro_snap["payems_3m_ann"],
            "10Y": macro_snap["gs10"],
            "2Y": macro_snap["dgs2"],
            "2s10s": macro_snap["curve_2s10s"],
            "NFCI": macro_snap["nfci"],
            "real_policy_proxy": macro_snap["real_policy_proxy"],
        }
    )
    failures = macro_panel.attrs.get("failures", [])
    if failures:
        st.warning("Some non-critical macro series failed to load:")
        st.code("\n".join(failures))

st.markdown("---")

loaded_records: List[MeetingRecord] = []
load_errors = []

progress = st.progress(0)
status = st.empty()

for i, m in enumerate(meetings_meta):
    try:
        status.text(f"Loading meeting {i+1}/{len(meetings_meta)}: {m['meeting_date'].date()}")
        rec = load_meeting_record(
            page_url=m["page_url"],
            meeting_date=m["meeting_date"],
            title=m["title"],
            year=m["year"],
        )

        rec.statement_text = rec.statement_text or ""
        rec.minutes_text = rec.minutes_text if include_minutes else ""
        rec.presser_text = rec.presser_text if include_pressers else ""
        rec.projections_text = rec.projections_text if include_sep else ""

        loaded_records.append(rec)

    except Exception as e:
        load_errors.append(f"{m['meeting_date'].date()}: {e}")

    progress.progress((i + 1) / len(meetings_meta))

status.empty()
progress.empty()

if not loaded_records:
    st.error("No meeting records loaded.")
    if load_errors:
        st.code("\n".join(load_errors))
    st.stop()

loaded_records = sorted(loaded_records, key=lambda x: x.meeting_date, reverse=True)

current_options = {
    f"{r.meeting_date.date()} | {r.title[:90]}": idx for idx, r in enumerate(loaded_records)
}
selected_label = st.selectbox("Meeting to analyze", list(current_options.keys()), index=0)
current_rec = loaded_records[current_options[selected_label]]

doc_text = current_rec.combined_text
scenario_df = score_scenarios(doc_text, macro_snap)
hd = hawk_dove_balance(doc_text)

left, right = st.columns([1.2, 1.4])

with left:
    st.subheader("Scenario probabilities")
    fig = go.Figure()
    fig.add_bar(
        x=scenario_df["probability"],
        y=scenario_df["scenario"],
        orientation="h",
    )
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Probability (%)",
        yaxis_title="",
        yaxis=dict(categoryorder="total ascending"),
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Fed tone balance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Hawk hits", f"{hd['hawk_hits']}")
    c2.metric("Dove hits", f"{hd['dove_hits']}")
    c3.metric("Balance", f"{hd['balance']:.2f}")

    top_scenario = scenario_df.iloc[0]
    st.markdown(f"**Base case:** {top_scenario['scenario']} ({top_scenario['probability']:.1f}%)")
    st.markdown(
        """
        The model blends textual cues from the selected FOMC document set with the current macro backdrop.
        Treat it as a scenario map and analog finder, not a deterministic forecast.
        """
    )

st.markdown("---")

st.subheader("Closest historical analog meetings")
sim_df = build_similarity_frame(current_rec, loaded_records)
top_n = st.slider("Number of analogs", 3, 10, 5, 1)

if sim_df.empty:
    st.info("No historical analogs available yet.")
else:
    show_df = sim_df.head(top_n).copy()
    show_df["meeting_date"] = show_df["meeting_date"].dt.date
    show_df["similarity"] = (show_df["similarity"] * 100).round(1)
    st.dataframe(show_df[["meeting_date", "title", "similarity", "page_url"]], use_container_width=True)

    analog_rows = []
    rec_map = {r.page_url: r for r in loaded_records}
    for _, row in show_df.iterrows():
        rec = rec_map.get(row["page_url"])
        if rec:
            analog_scen = score_scenarios(rec.combined_text, macro_snap)
            analog_rows.append({
                "meeting_date": rec.meeting_date.date(),
                "title": rec.title,
                "top_scenario": analog_scen.iloc[0]["scenario"],
                "probability": round(float(analog_scen.iloc[0]["probability"]), 1),
                "similarity": row["similarity"],
            })

    if analog_rows:
        st.markdown("**Analog read-through**")
        st.dataframe(pd.DataFrame(analog_rows), use_container_width=True)

st.markdown("---")

doc_col1, doc_col2 = st.columns(2)

with doc_col1:
    st.subheader("Document availability")
    docs_data = {
        "Statement HTML": current_rec.doc_links.statement_html,
        "Statement PDF": current_rec.doc_links.statement_pdf,
        "Minutes HTML": current_rec.doc_links.minutes_html,
        "Minutes PDF": current_rec.doc_links.minutes_pdf,
        "Press Conference PDF": current_rec.doc_links.presser_pdf,
        "Projections PDF": current_rec.doc_links.projections_pdf,
        "Projections HTML": current_rec.doc_links.projections_html,
        "Implementation Note": current_rec.doc_links.implementation_note,
        "Meeting Page": current_rec.page_url,
    }
    docs_df = pd.DataFrame(
        [{"document": k, "url": v if v else ""} for k, v in docs_data.items()]
    )
    st.dataframe(docs_df, use_container_width=True)

with doc_col2:
    st.subheader("Top terms in selected document bundle")
    terms_df = top_terms(doc_text, n=25)
    st.dataframe(terms_df, use_container_width=True)

if show_raw_text:
    st.markdown("---")
    st.subheader("Source text excerpts")

    tab1, tab2, tab3, tab4 = st.tabs(["Statement", "Minutes", "Press Conference", "Projections"])

    with tab1:
        st.text_area("Statement text", current_rec.statement_text[:12000], height=300)

    with tab2:
        st.text_area("Minutes text", current_rec.minutes_text[:12000], height=300)

    with tab3:
        st.text_area("Press conference text", current_rec.presser_text[:12000], height=300)

    with tab4:
        st.text_area("Projections text", current_rec.projections_text[:12000], height=300)

st.markdown("---")

st.subheader("Macro context")
macro_plot = macro_panel.dropna(subset=["date"]).copy()
chart_choice = st.selectbox(
    "Macro chart",
    [
        "Core PCE YoY vs Fed Target Mid",
        "Unemployment Rate",
        "2s10s Curve",
        "NFCI",
        "Payrolls 3M Annualized",
    ],
)

fig2 = go.Figure()

if chart_choice == "Core PCE YoY vs Fed Target Mid":
    fig2.add_scatter(x=macro_plot["date"], y=macro_plot["core_pce_yoy"], name="Core PCE YoY")
    fig2.add_scatter(x=macro_plot["date"], y=macro_plot["FEDTARMD"], name="Fed Target Mid")
elif chart_choice == "Unemployment Rate":
    fig2.add_scatter(x=macro_plot["date"], y=macro_plot["UNRATE"], name="Unemployment Rate")
elif chart_choice == "2s10s Curve":
    fig2.add_scatter(x=macro_plot["date"], y=macro_plot["2s10s"], name="2s10s")
elif chart_choice == "NFCI":
    fig2.add_scatter(x=macro_plot["date"], y=macro_plot["NFCI"], name="NFCI")
elif chart_choice == "Payrolls 3M Annualized":
    fig2.add_scatter(x=macro_plot["date"], y=macro_plot["payems_3m_ann"], name="Payrolls 3M Annualized")

fig2.update_layout(
    height=420,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis_title="Date",
    yaxis_title="Value",
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.subheader("ADFM readout")

base = scenario_df.iloc[0]["scenario"]
second = scenario_df.iloc[1]["scenario"] if len(scenario_df) > 1 else None


def build_readout(base_case: str, second_case: Optional[str], snap: Dict[str, float], hd_map: Dict[str, float]) -> str:
    tone = "hawkish" if hd_map["balance"] > 0.15 else "dovish" if hd_map["balance"] < -0.15 else "balanced"
    out = []

    out.append(
        f"The selected FOMC bundle maps most closely to a **{base_case}** regime, with Fed language reading as **{tone}** on the hawk-dove balance."
    )

    out.append(
        f"Macro overlay matters here. Core PCE is running at **{pct_fmt(snap['core_pce_yoy'], 2)}**, unemployment is **{pct_fmt(snap['unrate'], 1)}**, and the 3-month change in unemployment is **{pct_fmt(snap['unrate_3m_change'], 1)}**."
    )

    if base_case == "Higher for longer":
        out.append(
            "That combination argues the Committee still sees policy restraint as necessary because inflation is not convincingly back inside the target corridor and labor has not broken enough to force a fast pivot."
        )
    elif base_case == "Restrictive hold":
        out.append(
            "That usually means the Fed thinks policy is already restrictive, but it does not yet have enough confidence to move. The path is sideways on rates until either inflation falls further or labor clearly weakens."
        )
    elif base_case == "Dovish hold / pivot setup":
        out.append(
            "That is usually the pre-pivot zone. The Committee is still holding, but the language starts to shift toward downside labor risk, cumulative restraint, and optionality around future adjustments."
        )
    elif base_case == "Cutting cycle":
        out.append(
            "That setup points to a materially weaker activity backdrop where labor and growth deterioration are beginning to dominate the inflation problem."
        )
    elif base_case == "Stagflation risk":
        out.append(
            "That is the most difficult policy mix because inflation pressure and growth weakness coexist, which raises the odds of policy hesitation, cross-asset volatility, and sharper disagreement between the Fed and market pricing."
        )

    if second_case:
        out.append(
            f"The nearest alternative read is **{second_case}**, which is useful because the market usually trades the transition between those two states before the Fed fully admits it."
        )

    return "\n\n".join(out)


st.markdown(build_readout(base, second, macro_snap, hd))

if load_errors:
    with st.expander("Load errors", expanded=False):
        st.code("\n".join(load_errors))
