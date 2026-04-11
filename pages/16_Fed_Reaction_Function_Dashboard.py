from __future__ import annotations

import io
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup

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

CONNECT_TIMEOUT = 3.05
READ_TIMEOUT = 8
MAX_RETRIES = 1
RETRY_SLEEP = 0.5

# =========================
# Utility helpers
# =========================


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


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


def annualized_3m_from_index(series: pd.Series) -> pd.Series:
    return ((series / series.shift(3)) ** 4 - 1) * 100


def yoy(series: pd.Series) -> pd.Series:
    return (series / series.shift(12) - 1) * 100


def fetch_url(url: str) -> requests.Response:
    headers = {"User-Agent": USER_AGENT}
    last_err = None

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                url,
                headers=headers,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_err = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP * (attempt + 1))

    raise last_err


def absolute_url(href: str) -> str:
    return urljoin(FED_BASE, href or "")


# =========================
# FRED data loader
# =========================

FRED_SERIES_STARTS = {
    "DFEDTARL": "2015-01-01",
    "DFEDTARU": "2015-01-01",
    "UNRATE": "2015-01-01",
    "PCEPILFE": "2015-01-01",
    "CPIAUCSL": "2015-01-01",
    "GS10": "2015-01-01",
    "DGS2": "2015-01-01",
    "NFCI": "2015-01-01",
    "PAYEMS": "2015-01-01",
}


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_fred_series_csv(series_id: str, start: str = "2015-01-01") -> pd.DataFrame:
    url = FRED_CSV_URL.format(series_id=series_id, start=start)
    resp = fetch_url(url)
    df = pd.read_csv(io.StringIO(resp.text))

    if df.empty or "DATE" not in df.columns or series_id not in df.columns:
        raise ValueError(f"Malformed FRED response for {series_id}")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    df = df.dropna(subset=["DATE"]).rename(columns={"DATE": "date", series_id: "value"})
    return df[["date", "value"]].copy()


def build_fallback_macro_panel(error_msg: str = "") -> pd.DataFrame:
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    panel = pd.DataFrame(
        [
            {
                "date": today,
                "DFEDTARL": np.nan,
                "DFEDTARU": np.nan,
                "UNRATE": np.nan,
                "PCEPILFE": np.nan,
                "CPIAUCSL": np.nan,
                "GS10": np.nan,
                "DGS2": np.nan,
                "NFCI": np.nan,
                "PAYEMS": np.nan,
                "FEDTARMD": np.nan,
                "core_pce_yoy": np.nan,
                "cpi_yoy": np.nan,
                "payems_3m_ann": np.nan,
                "2s10s": np.nan,
                "real_policy_proxy": np.nan,
            }
        ]
    )
    panel.attrs["failures"] = [error_msg] if error_msg else ["Live FRED load failed."]
    return panel


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_macro_panel() -> pd.DataFrame:
    panel = None
    failures = []

    for sid, start in FRED_SERIES_STARTS.items():
        try:
            s = fetch_fred_series_csv(sid, start=start).rename(columns={"value": sid})
            panel = s if panel is None else panel.merge(s, on="date", how="outer")
        except Exception as exc:
            failures.append(f"{sid}: {exc}")

    if panel is None:
        return build_fallback_macro_panel("All live FRED series failed to load.")

    panel = panel.sort_values("date").reset_index(drop=True)

    for sid in FRED_SERIES_STARTS:
        if sid not in panel.columns:
            panel[sid] = np.nan

    base_cols = list(FRED_SERIES_STARTS.keys())
    panel[base_cols] = panel[base_cols].ffill()

    panel["FEDTARMD"] = (panel["DFEDTARL"] + panel["DFEDTARU"]) / 2
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


def doc_links_to_dict(docs: MeetingDocLinks) -> Dict[str, Optional[str]]:
    return {
        "statement_html": docs.statement_html,
        "statement_pdf": docs.statement_pdf,
        "minutes_html": docs.minutes_html,
        "minutes_pdf": docs.minutes_pdf,
        "presser_pdf": docs.presser_pdf,
        "projections_pdf": docs.projections_pdf,
        "projections_html": docs.projections_html,
        "implementation_note": docs.implementation_note,
    }


def merge_doc_links(seed: MeetingDocLinks, extra: MeetingDocLinks) -> MeetingDocLinks:
    data = doc_links_to_dict(seed)
    for key, value in doc_links_to_dict(extra).items():
        if value and not data.get(key):
            data[key] = value
    return MeetingDocLinks(**data)


def has_any_doc_link(docs: MeetingDocLinks) -> bool:
    return any(doc_links_to_dict(docs).values())


def assign_doc_link(docs: MeetingDocLinks, href: str, label: str) -> None:
    href_l = href.lower()
    label_l = normalize_whitespace(label).lower()

    if re.search(r"/newsevents/pressreleases/monetary\d{8}a\.htm$", href_l):
        docs.statement_html = docs.statement_html or href
        return

    if re.search(r"/newsevents/pressreleases/monetary\d{8}a1\.pdf$", href_l):
        docs.statement_pdf = docs.statement_pdf or href
        return

    if re.search(r"/newsevents/pressreleases/monetary\d{8}a2\.pdf$", href_l) or "implementation note" in label_l:
        docs.implementation_note = docs.implementation_note or href
        return

    if re.search(r"/monetarypolicy/fomcminutes\d{8}\.htm$", href_l):
        docs.minutes_html = docs.minutes_html or href
        return

    if re.search(r"/monetarypolicy/fomcminutes\d{8}\.pdf$", href_l):
        docs.minutes_pdf = docs.minutes_pdf or href
        return

    if re.search(r"/monetarypolicy/fomcpresconf\d{8}\.pdf$", href_l) or "press conference transcript" in label_l:
        docs.presser_pdf = docs.presser_pdf or href
        return

    if re.search(r"/monetarypolicy/fomcprojtabl\d{8}\.htm$", href_l):
        docs.projections_html = docs.projections_html or href
        return

    if re.search(r"/monetarypolicy/fomcprojtabl\d{8}\.pdf$", href_l):
        docs.projections_pdf = docs.projections_pdf or href
        return


def extract_meeting_date_from_url(href: str) -> Optional[pd.Timestamp]:
    match = re.search(r"(\d{8})", href or "")
    if not match:
        return None
    dt = pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")
    return None if pd.isna(dt) else dt


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
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF from {url}: {exc}")

    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
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

    return clean_text(max(texts, key=len))


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def fetch_html_text(url: str) -> str:
    return extract_main_html_text(fetch_html(url))


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def get_fomc_calendar_meetings() -> List[Dict]:
    html = fetch_html(FOMC_CALENDAR_URL)
    soup = BeautifulSoup(html, "html.parser")
    grouped: Dict[pd.Timestamp, Dict] = {}

    for a in soup.find_all("a", href=True):
        href = absolute_url(a["href"])
        label = normalize_whitespace(a.get_text(" ", strip=True))
        meeting_date = extract_meeting_date_from_url(href)

        if meeting_date is None:
            continue

        if not (
            re.search(r"/monetarypolicy/fomcpresconf\d{8}\.htm$", href, flags=re.I)
            or re.search(r"/monetarypolicy/fomcminutes\d{8}\.(pdf|htm)$", href, flags=re.I)
            or re.search(r"/monetarypolicy/fomcprojtabl\d{8}\.(pdf|htm)$", href, flags=re.I)
            or re.search(r"/newsevents/pressreleases/monetary\d{8}a\d?\.((pdf)|(htm))$", href, flags=re.I)
        ):
            continue

        meta = grouped.setdefault(
            meeting_date,
            {
                "meeting_date": meeting_date,
                "title": f"FOMC Meeting {meeting_date.date()}",
                "page_url": FOMC_CALENDAR_URL,
                "year": int(meeting_date.year),
                "doc_links_obj": MeetingDocLinks(),
            },
        )

        assign_doc_link(meta["doc_links_obj"], href, label)

        if re.search(r"/monetarypolicy/fomcpresconf\d{8}\.htm$", href, flags=re.I):
            meta["page_url"] = href

    meetings = []
    for _, meta in sorted(grouped.items(), key=lambda x: x[0], reverse=True):
        if has_any_doc_link(meta["doc_links_obj"]) or meta["page_url"] != FOMC_CALENDAR_URL:
            meetings.append(
                {
                    "meeting_date": meta["meeting_date"],
                    "title": meta["title"],
                    "page_url": meta["page_url"],
                    "year": meta["year"],
                    "doc_links": doc_links_to_dict(meta["doc_links_obj"]),
                }
            )

    return meetings


def parse_meeting_page_docs(page_url: str) -> MeetingDocLinks:
    html = fetch_html(page_url)
    soup = BeautifulSoup(html, "html.parser")
    docs = MeetingDocLinks()

    anchors = soup.find_all("a", href=True)
    for a in anchors:
        href = absolute_url(a["href"])
        label = normalize_whitespace(a.get_text(" ", strip=True))
        assign_doc_link(docs, href, label)

    if not docs.presser_pdf:
        for a in anchors:
            href = absolute_url(a["href"])
            if re.search(r"fomcpresconf\d{8}\.pdf$", href, flags=re.I):
                docs.presser_pdf = href
                break

    return docs


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_meeting_record(
    page_url: str,
    meeting_date: pd.Timestamp,
    title: str,
    year: int,
    seed_doc_links: Optional[Dict[str, Optional[str]]] = None,
) -> MeetingRecord:
    docs = MeetingDocLinks(**(seed_doc_links or {}))

    if page_url and page_url != FOMC_CALENDAR_URL:
        try:
            docs = merge_doc_links(docs, parse_meeting_page_docs(page_url))
        except Exception:
            pass

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
    return [t for t in toks if t not in STOPWORDS]


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
            "resilient demand", "strong labor market", "tight labor market",
        ],
        "negative_terms": [
            "disinflation", "cooling labor market", "softer growth", "rate cuts", "lower rates",
        ],
    },
    "Restrictive hold": {
        "positive_terms": [
            "well positioned", "wait for more evidence", "data dependent", "balance of risks",
            "maintain current target range", "monitor incoming data", "uncertain outlook",
            "restrictive stance", "hold rates steady",
        ],
        "negative_terms": [
            "imminent cuts", "need to cut", "rapid deterioration", "severe contraction",
        ],
    },
    "Dovish hold / pivot setup": {
        "positive_terms": [
            "disinflation", "progress on inflation", "downside risks to employment",
            "policy is restrictive", "cooling labor market", "slower demand", "moderating wages",
            "balanced risks", "scope to adjust", "additional adjustments",
        ],
        "negative_terms": [
            "upside inflation risks", "further firming", "reacceleration",
        ],
    },
    "Cutting cycle": {
        "positive_terms": [
            "cut the target range", "lower the target range", "downside risks",
            "deterioration in labor market", "weaker activity", "recession", "support the economy",
            "ease policy", "policy accommodation",
        ],
        "negative_terms": [
            "elevated inflation", "strong labor market", "further firming",
        ],
    },
    "Stagflation risk": {
        "positive_terms": [
            "higher energy prices", "elevated inflation", "slower growth", "weaker activity",
            "supply shock", "inflation remains above target", "downside risks to growth",
            "adverse supply developments", "tariffs", "geopolitical risks",
        ],
        "negative_terms": [
            "broad disinflation", "strong productivity", "cooling prices without growth damage",
        ],
    },
}

HAWK_TERMS = [
    "further policy firming", "upside risks to inflation", "elevated inflation",
    "restrictive", "strong labor market", "tight labor market", "not sufficiently restrictive",
]

DOVE_TERMS = [
    "disinflation", "downside risks to employment", "cooling labor market",
    "softer growth", "policy is restrictive", "balanced risks", "lower rates", "rate cuts",
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
    return {"hawk_hits": hawk, "dove_hits": dove, "balance": balance}


def top_terms(text: str, n: int = 30) -> pd.DataFrame:
    return pd.DataFrame(tf_vector(text).most_common(n), columns=["term", "count"])


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
        rows.append(
            {
                "meeting_date": rec.meeting_date,
                "title": rec.title,
                "year": rec.year,
                "similarity": sim,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["meeting_date", "title", "year", "similarity"])

    return pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)


# =========================
# Macro snapshot
# =========================

def latest_macro_snapshot(panel: pd.DataFrame) -> Dict[str, float]:
    df = panel.sort_values("date").copy()
    latest = df.iloc[-1]

    if len(df) > 1:
        anchor_date = latest["date"] - pd.DateOffset(months=3)
        hist = df[df["date"] <= anchor_date]
        prev_3m = hist.iloc[-1] if not hist.empty else df.iloc[0]
    else:
        prev_3m = latest

    return {
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
    max_meetings = st.slider("Meetings to load", min_value=6, max_value=24, value=10, step=2)
    include_minutes = st.checkbox("Use minutes", value=True)
    include_pressers = st.checkbox("Use press conference transcripts", value=True)
    include_sep = st.checkbox("Use SEP materials", value=False)
    show_raw_text = st.checkbox("Show source text excerpts", value=False)

    st.markdown("---")
    st.caption(f"PDF engine: {PDF_ENGINE or 'none'}")

with st.spinner("Loading macro panel from FRED..."):
    macro_panel = load_macro_panel()

macro_snap = latest_macro_snapshot(macro_panel)
macro_failures = macro_panel.attrs.get("failures", [])

if macro_failures:
    st.warning(
        "Live macro data was partially unavailable, so the app is using fallback values where needed. "
        "Meeting analysis should still render."
    )

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
col_d.metric("3M Delta Unemployment", pct_fmt(macro_snap["unrate_3m_change"], 1))
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
    if macro_failures:
        st.warning("Some macro series failed to load:")
        st.code("\n".join(macro_failures))

st.markdown("---")

loaded_records: List[MeetingRecord] = []
load_errors = []

progress = st.progress(0)
status = st.empty()

for i, m in enumerate(meetings_meta):
    try:
        status.text(f"Loading meeting {i + 1}/{len(meetings_meta)}: {m['meeting_date'].date()}")

        rec = load_meeting_record(
            page_url=m["page_url"],
            meeting_date=m["meeting_date"],
            title=m["title"],
            year=m["year"],
            seed_doc_links=m.get("doc_links"),
        )

        rec.statement_text = rec.statement_text or ""
        rec.minutes_text = rec.minutes_text if include_minutes else ""
        rec.presser_text = rec.presser_text if include_pressers else ""
        rec.projections_text = rec.projections_text if include_sep else ""

        loaded_records.append(rec)

    except Exception as exc:
        load_errors.append(f"{m['meeting_date'].date()}: {exc}")

    progress.progress((i + 1) / len(meetings_meta))

status.empty()
progress.empty()

if not loaded_records:
    st.error("No meeting records loaded.")
    if load_errors:
        st.code("\n".join(load_errors))
    st.stop()

loaded_records = sorted(loaded_records, key=lambda x: x.meeting_date, reverse=True)

default_index = 0
for idx, record in enumerate(loaded_records):
    if record.combined_text:
        default_index = idx
        break

current_options = {
    f"{r.meeting_date.date()} | {r.title[:90]}": idx for idx, r in enumerate(loaded_records)
}
selected_label = st.selectbox("Meeting to analyze", list(current_options.keys()), index=default_index)
current_rec = loaded_records[current_options[selected_label]]

doc_text = current_rec.combined_text
scenario_df = score_scenarios(doc_text, macro_snap)
hd = hawk_dove_balance(doc_text)

if not doc_text:
    st.warning(
        "No source text was extracted for the selected meeting. "
        "The scenario output is relying mostly on the macro overlay."
    )

left, right = st.columns([1.2, 1.4])

with left:
    st.subheader("Scenario probabilities")
    fig = go.Figure()
    fig.add_bar(x=scenario_df["probability"], y=scenario_df["scenario"], orientation="h")
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
    st.dataframe(show_df[["meeting_date", "title", "similarity"]], use_container_width=True)

    analog_rows = []
    rec_map = {r.meeting_date.date(): r for r in loaded_records}

    for _, row in show_df.iterrows():
        rec = rec_map.get(row["meeting_date"])
        if rec:
            analog_scen = score_scenarios(rec.combined_text, macro_snap)
            analog_rows.append(
                {
                    "meeting_date": rec.meeting_date.date(),
                    "title": rec.title,
                    "top_scenario": analog_scen.iloc[0]["scenario"],
                    "probability": round(float(analog_scen.iloc[0]["probability"]), 1),
                    "similarity": row["similarity"],
                }
            )

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
    docs_df = pd.DataFrame([{"document": k, "url": v if v else ""} for k, v in docs_data.items()])
    st.dataframe(docs_df, use_container_width=True)

with doc_col2:
    st.subheader("Top terms in selected document bundle")
    st.dataframe(top_terms(doc_text, n=25), use_container_width=True)

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
