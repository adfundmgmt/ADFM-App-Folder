import re
import html
import time
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

APP_TITLE = "Fed Speeches Tone Underwriter"
BASE_URL = "https://www.federalreserve.gov"
INDEX_URL = f"{BASE_URL}/newsevents/speeches-testimony.htm"
DATA_DIR = Path("fed_tone_data")
DB_PATH = DATA_DIR / "fed_tone.sqlite"
REQUEST_TIMEOUT = 30
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
)

DATA_DIR.mkdir(parents=True, exist_ok=True)

SPEAKER_WEIGHTS = {
    "Jerome H. Powell": 3.0,
    "Philip N. Jefferson": 1.5,
    "John C. Williams": 1.5,
    "Christopher J. Waller": 1.5,
    "Michelle W. Bowman": 1.2,
    "Michael S. Barr": 1.1,
    "Lisa D. Cook": 1.0,
    "Adriana D. Kugler": 1.0,
    "Austan D. Goolsbee": 1.0,
    "Mary C. Daly": 1.0,
    "Raphael W. Bostic": 1.0,
    "Thomas I. Barkin": 1.0,
    "Lorie K. Logan": 1.2,
    "Neel Kashkari": 1.1,
    "Susan M. Collins": 1.0,
    "Patrick T. Harker": 1.0,
    "Alberto G. Musalem": 1.0,
    "Beth M. Hammack": 1.0,
    "Stephen I. Miran": 1.0,
    "Lael Brainard": 1.0,
}

LEXICON = {
    "hawkish": [
        "higher for longer", "restrictive", "sufficiently restrictive", "upside risk",
        "inflation remains too high", "inflation is too high", "persistent inflation",
        "reaccelerat", "not yet done", "additional firming", "further tightening",
        "vigilant", "upward pressure on prices", "elevated inflation", "inflation pressure",
        "price stability", "tight labor market", "strong labor market", "overheating",
        "policy restraint", "maintain restraint", "firming", "tightening",
        "hold rates higher", "risk of inflation", "unanchored inflation expectations",
        "still above target", "premature to ease", "not appropriate to cut",
        "more work to do", "upside risks to inflation", "inflation persistence",
        "restrictive stance", "we are prepared to", "need to remain", "longer than expected",
    ],
    "dovish": [
        "disinflation", "cooling labor market", "softening labor market", "downside risk",
        "growth is slowing", "economic slowdown", "below-trend growth", "normalizing inflation",
        "further progress on inflation", "policy can respond", "room to ease", "easing",
        "rate cuts", "cut rates", "lower rates", "less restrictive", "downward path",
        "labor market is moderating", "balanced risks", "two-sided risks", "weak demand",
        "headwinds", "slack", "unemployment is rising", "financial conditions tightened",
        "act as appropriate", "support the labor market", "downside risks to employment",
        "can be patient", "can adjust policy", "if needed we can", "could move lower",
    ],
    "inflation_concern": [
        "inflation", "prices", "price pressures", "services inflation", "core inflation",
        "shelter inflation", "goods inflation", "inflation expectations", "price stability",
    ],
    "labor_concern": [
        "labor market", "employment", "unemployment", "job growth", "payroll",
        "wages", "hiring", "layoffs", "slack", "participation",
    ],
    "growth_concern": [
        "growth", "activity", "demand", "consumer spending", "investment", "slowdown",
        "recession", "weakness", "output", "expansion",
    ],
    "financial_stability": [
        "financial stability", "banking", "banks", "stress", "liquidity", "funding",
        "market functioning", "credit conditions", "treasury market", "vulnerabilities",
    ],
    "balance_sheet": [
        "balance sheet", "quantitative tightening", "qt", "runoff", "reserves",
        "securities holdings", "mbs", "treasury holdings",
    ],
    "uncertainty_risk": [
        "uncertain", "uncertainty", "risk management", "careful", "monitor", "watching",
        "data dependent", "incoming data", "proceed carefully", "humble", "attentive",
    ],
}

SNIPPET_PATTERNS = {
    "hawkish": LEXICON["hawkish"],
    "dovish": LEXICON["dovish"],
    "inflation": LEXICON["inflation_concern"],
    "labor": LEXICON["labor_concern"],
    "growth": LEXICON["growth_concern"],
}

STOP_PHRASES = [
    "watch live",
    "for media inquiries",
    "last update",
    "return to text",
]

POLICY_RELEVANCE_TERMS = [
    "policy", "monetary policy", "committee", "fomc", "federal funds rate", "rate",
    "rates", "outlook", "forecast", "incoming data", "data dependent", "financial conditions",
    "current", "currently", "today", "going forward", "at this time", "we expect",
    "we will", "we may", "we remain", "our stance", "appropriate", "balance of risks",
    "inflation remains", "labor market remains", "economic outlook", "target range",
    "price stability", "maximum employment", "restrictive", "easing", "tightening",
]

HISTORICAL_TERMS = [
    "in the early", "in the late", "in 19", "in 20", "during the", "history", "historical",
    "legacy", "served", "was appointed", "reappointed", "under president", "at that time",
    "in the 1980s", "in the 1970s", "years ago", "decades ago", "career", "throughout his career",
    "he led", "he served", "he held firm", "volcker", "grew up", "biography",
]

CEREMONIAL_TERMS = [
    "award", "honor", "ceremony", "commencement", "tribute", "memorial", "anniversary",
    "public service", "integrity", "legacy", "congratulations", "pleasure to be here",
    "humbling honor", "thank you for inviting me", "it is a pleasure", "introduction",
]

QUOTE_ATTRIBUTION_TERMS = [
    "he said", "she said", "they said", "volcker said", "as he noted", "as she noted",
    "critics argued", "in a speech", "he acknowledged", "she acknowledged", "according to",
]

FORWARD_LOOKING_TERMS = [
    "will", "may", "could", "expect", "outlook", "going forward", "at this time",
    "over coming", "over the coming", "in coming months", "we remain", "we are prepared",
    "if needed", "from here", "currently", "today", "incoming data",
]

FOOTER_JUNK_PATTERNS = [
    "about the fed news & events",
    "monetary policy supervision & regulation",
    "financial stability payment systems",
    "economic research data",
    "consumers & communities",
    "connect with the board",
    "tools and information",
    "contact publications",
    "freedom of information",
    "office of inspector general",
    "budget & performance",
    "website policies",
    "privacy program",
    "federal reserve facebook",
    "federal reserve instagram",
    "federal reserve youtube",
    "federal reserve flickr",
    "federal reserve linkedin",
    "federal reserve threads",
    "bluesky page",
    "20th street and constitution avenue",
]

ACCESSIBILITY_JUNK_PATTERNS = [
    "accessible keys for video",
    "toggles play/pause",
    "seeks the video forwards and back",
    "toggles mute on/off",
    "toggles fullscreen on/off",
    "caption on/off",
]


@dataclass
class ToneResult:
    net_score: float
    hawkish_score: float
    dovish_score: float
    inflation_concern: float
    labor_concern: float
    growth_concern: float
    financial_stability: float
    balance_sheet: float
    uncertainty_risk: float
    word_count: int
    policy_relevance: float
    live_signal_share: float


def make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            url TEXT PRIMARY KEY,
            event_type TEXT,
            year INTEGER,
            date TEXT,
            title TEXT,
            speaker TEXT,
            role TEXT,
            venue TEXT,
            pdf_url TEXT,
            body_text TEXT,
            word_count INTEGER,
            scraped_at TEXT,
            hawkish_score REAL,
            dovish_score REAL,
            net_score REAL,
            inflation_concern REAL,
            labor_concern REAL,
            growth_concern REAL,
            financial_stability REAL,
            balance_sheet REAL,
            uncertainty_risk REAL,
            policy_relevance REAL,
            live_signal_share REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scrape_log (
            ts TEXT,
            level TEXT,
            message TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def db_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def log_event(level: str, message: str) -> None:
    conn = db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO scrape_log (ts, level, message) VALUES (?, ?, ?)",
        (pd.Timestamp.utcnow().isoformat(), level, message[:4000]),
    )
    conn.commit()
    conn.close()


def clear_log() -> None:
    conn = db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM scrape_log")
    conn.commit()
    conn.close()


def load_log(limit: int = 250) -> pd.DataFrame:
    init_db()
    conn = db_connection()
    df = pd.read_sql_query(
        f"SELECT * FROM scrape_log ORDER BY ts DESC LIMIT {int(limit)}",
        conn,
    )
    conn.close()
    return df


def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace("\xa0", " ")
    text = text.replace("â", "'").replace("â", '"').replace("â", '"').replace("â", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_phrase(phrase: str) -> str:
    return re.escape(phrase.lower())


def count_phrase_hits(text: str, phrases: List[str]) -> int:
    low = text.lower()
    hits = 0
    for phrase in phrases:
        pattern = normalize_phrase(phrase)
        hits += len(re.findall(pattern, low))
    return hits


def safe_mean_std(series: pd.Series) -> Tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0, 1.0
    mean = float(s.mean())
    std = float(s.std(ddof=0))
    if std <= 1e-12:
        std = 1.0
    return mean, std


def zscore_value(value: float, mean: float, std: float) -> float:
    if pd.isna(value):
        return np.nan
    return float((value - mean) / std)


def bounded_z(z: float, limit: float = 3.0) -> float:
    if pd.isna(z):
        return np.nan
    return float(np.clip(z, -limit, limit))


def percentile_rank(series: pd.Series, value: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or pd.isna(value):
        return np.nan
    return float((s <= value).mean() * 100.0)


def tone_bucket(z: float) -> str:
    if pd.isna(z):
        return "Unknown"
    if z >= 1.25:
        return "Very Hawkish"
    if z >= 0.35:
        return "Hawkish"
    if z <= -1.25:
        return "Very Dovish"
    if z <= -0.35:
        return "Dovish"
    return "Neutral"


def emphasis_bucket(z: float) -> str:
    if pd.isna(z):
        return "Unknown"
    if z >= 1.25:
        return "Very High"
    if z >= 0.35:
        return "High"
    if z <= -1.25:
        return "Very Low"
    if z <= -0.35:
        return "Low"
    return "Neutral"


def policy_bucket(x: float) -> str:
    if pd.isna(x):
        return "Unknown"
    if x >= 0.70:
        return "High"
    if x >= 0.40:
        return "Medium"
    return "Low"


def squash_score(x: float, scale: float = 2.5, max_abs: float = 2.2) -> float:
    if pd.isna(x):
        return np.nan
    return float(np.tanh(x / scale) * max_abs)


def split_paragraphs(text: str) -> List[str]:
    text = clean_text(text)
    raw = re.split(r"\n\s*\n|(?<=\.)\s{2,}", text)
    out = []
    for p in raw:
        p = clean_text(p)
        if len(p.split()) >= 12:
            out.append(p)
    return out


def contains_any(text: str, phrases: List[str]) -> int:
    low = text.lower()
    return sum(1 for p in phrases if p.lower() in low)


def paragraph_policy_relevance(paragraph: str) -> float:
    low = paragraph.lower()
    words = max(len(re.findall(r"\b[a-z][a-z\-']+\b", low)), 1)

    policy_hits = contains_any(low, POLICY_RELEVANCE_TERMS)
    forward_hits = contains_any(low, FORWARD_LOOKING_TERMS)
    historical_hits = contains_any(low, HISTORICAL_TERMS)
    ceremonial_hits = contains_any(low, CEREMONIAL_TERMS)
    quote_hits = contains_any(low, QUOTE_ATTRIBUTION_TERMS)

    topical_hits = (
        count_phrase_hits(low, LEXICON["inflation_concern"])
        + count_phrase_hits(low, LEXICON["labor_concern"])
        + count_phrase_hits(low, LEXICON["growth_concern"])
        + count_phrase_hits(low, LEXICON["financial_stability"])
        + count_phrase_hits(low, LEXICON["balance_sheet"])
    )

    score = 0.0
    score += 0.32 * min(policy_hits, 4)
    score += 0.25 * min(forward_hits, 4)
    score += 0.10 * min(topical_hits / max(words, 1) * 50.0, 2.5)
    score -= 0.22 * min(historical_hits, 4)
    score -= 0.18 * min(ceremonial_hits, 4)
    score -= 0.15 * min(quote_hits, 3)

    if "fomc" in low or "committee" in low or "federal funds rate" in low:
        score += 0.35
    if "today" in low and ("inflation" in low or "labor market" in low or "policy" in low):
        score += 0.20
    if re.search(r"\bin\s+(19|20)\d{2}\b", low):
        score -= 0.15

    return float(np.clip(score, 0.0, 1.0))


def paragraph_directional_tone(paragraph: str) -> Dict[str, float]:
    low = paragraph.lower()
    words = max(len(re.findall(r"\b[a-z][a-z\-']+\b", low)), 1)

    hawk = count_phrase_hits(low, LEXICON["hawkish"]) / words * 1000.0
    dove = count_phrase_hits(low, LEXICON["dovish"]) / words * 1000.0
    infl = count_phrase_hits(low, LEXICON["inflation_concern"]) / words * 1000.0
    labor = count_phrase_hits(low, LEXICON["labor_concern"]) / words * 1000.0
    growth = count_phrase_hits(low, LEXICON["growth_concern"]) / words * 1000.0
    fs = count_phrase_hits(low, LEXICON["financial_stability"]) / words * 1000.0
    bs = count_phrase_hits(low, LEXICON["balance_sheet"]) / words * 1000.0
    unc = count_phrase_hits(low, LEXICON["uncertainty_risk"]) / words * 1000.0

    # directional score emphasizes actual stance language, not topic mentions
    net = (
        1.15 * hawk
        - 1.15 * dove
        + 0.05 * infl
        - 0.03 * labor
        - 0.03 * growth
        - 0.02 * fs
    )

    return {
        "hawk": hawk,
        "dove": dove,
        "infl": infl,
        "labor": labor,
        "growth": growth,
        "fs": fs,
        "bs": bs,
        "unc": unc,
        "net": net,
    }


def score_text(text: str) -> ToneResult:
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        paragraphs = [clean_text(text)]

    weighted = []
    relevance_vals = []

    for p in paragraphs:
        rel = paragraph_policy_relevance(p)
        direction = paragraph_directional_tone(p)

        # floor keeps some small signal from disappearing entirely
        w = max(0.08, rel)
        relevance_vals.append(rel)
        weighted.append((w, direction, p))

    total_w = sum(w for w, _, _ in weighted) or 1.0

    hawk = sum(w * d["hawk"] for w, d, _ in weighted) / total_w
    dove = sum(w * d["dove"] for w, d, _ in weighted) / total_w
    infl = sum(w * d["infl"] for w, d, _ in weighted) / total_w
    labor = sum(w * d["labor"] for w, d, _ in weighted) / total_w
    growth = sum(w * d["growth"] for w, d, _ in weighted) / total_w
    fs = sum(w * d["fs"] for w, d, _ in weighted) / total_w
    bs = sum(w * d["bs"] for w, d, _ in weighted) / total_w
    unc = sum(w * d["unc"] for w, d, _ in weighted) / total_w
    net = sum(w * d["net"] for w, d, _ in weighted) / total_w

    live_signal_share = float(np.mean([1.0 if r >= 0.45 else 0.0 for r in relevance_vals])) if relevance_vals else 0.0
    policy_relevance = float(np.mean(relevance_vals)) if relevance_vals else 0.0

    # final dampener for low-relevance speeches
    damp = 0.35 + 0.65 * policy_relevance

    net = squash_score(net * damp, scale=2.2, max_abs=2.1)
    hawk = squash_score(hawk * damp, scale=2.0, max_abs=2.3)
    dove = squash_score(dove * damp, scale=2.0, max_abs=2.3)

    # topic salience remains visible, but low-policy speeches cannot dominate
    topic_damp = 0.55 + 0.45 * policy_relevance
    infl = squash_score(infl * topic_damp, scale=2.8, max_abs=2.4)
    labor = squash_score(labor * topic_damp, scale=2.8, max_abs=2.4)
    growth = squash_score(growth * topic_damp, scale=2.8, max_abs=2.4)
    fs = squash_score(fs * topic_damp, scale=2.6, max_abs=2.2)
    bs = squash_score(bs * topic_damp, scale=2.6, max_abs=2.2)
    unc = squash_score(unc * topic_damp, scale=2.6, max_abs=2.2)

    word_count = max(len(re.findall(r"\b[a-z][a-z\-']+\b", clean_text(text).lower())), 1)

    return ToneResult(
        net_score=net,
        hawkish_score=hawk,
        dovish_score=dove,
        inflation_concern=infl,
        labor_concern=labor,
        growth_concern=growth,
        financial_stability=fs,
        balance_sheet=bs,
        uncertainty_risk=unc,
        word_count=word_count,
        policy_relevance=policy_relevance,
        live_signal_share=live_signal_share,
    )


def fetch_html(session: requests.Session, url: str) -> BeautifulSoup:
    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def parse_year_links(index_soup: BeautifulSoup) -> List[Tuple[str, str, int]]:
    items: List[Tuple[str, str, int]] = []

    for a in index_soup.find_all("a", href=True):
        href = urljoin(BASE_URL, a["href"])
        text = clean_text(a.get_text(" ", strip=True))
        if not text.isdigit():
            continue
        year = int(text)
        href_low = href.lower()

        if re.search(r"/newsevents/\d{4}-speeches\.htm$", href_low):
            items.append(("speech", href, year))
        elif re.search(r"/newsevents/\d{4}-testimony\.htm$", href_low):
            items.append(("testimony", href, year))

    dedup = {}
    for event_type, href, year in items:
        dedup[(event_type, year)] = (event_type, href, year)

    return sorted(dedup.values(), key=lambda x: (x[2], x[0]), reverse=True)


def parse_date_text(raw: str) -> Optional[str]:
    raw = clean_text(raw)
    if not raw:
        return None

    for fmt in ("%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return pd.to_datetime(raw, format=fmt).strftime("%Y-%m-%d")
        except Exception:
            pass

    try:
        dt = pd.to_datetime(raw, errors="raise")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def parse_index_items(year_soup: BeautifulSoup, event_type: str, year: int) -> List[Dict[str, str]]:
    text_lines = []
    for line in year_soup.get_text("\n", strip=True).split("\n"):
        line = clean_text(line)
        if line:
            text_lines.append(line)

    article_links: List[Tuple[str, str]] = []
    for a in year_soup.find_all("a", href=True):
        href = urljoin(BASE_URL, a["href"])
        href_low = href.lower()
        title = clean_text(a.get_text(" ", strip=True))
        if not title:
            continue
        if re.search(r"/newsevents/(speech|testimony)/.+\.htm$", href_low):
            article_links.append((title, href))

    dedup_links = []
    seen = set()
    for title, href in article_links:
        if href in seen:
            continue
        seen.add(href)
        dedup_links.append((title, href))

    items: List[Dict[str, str]] = []

    for title, full_url in dedup_links:
        title_idx = None
        for idx, line in enumerate(text_lines):
            if line == title:
                title_idx = idx
                break

        date_hint = None
        speaker_hint = None
        venue_hint = None
        role_hint = None

        if title_idx is not None:
            search_window_before = text_lines[max(0, title_idx - 3):title_idx]
            search_window_after = text_lines[title_idx + 1:title_idx + 6]

            for cand in reversed(search_window_before):
                maybe_date = parse_date_text(cand)
                if maybe_date:
                    date_hint = maybe_date
                    break

            for cand in search_window_after:
                low = cand.lower()
                if "watch live" in low:
                    continue
                if speaker_hint is None and any(
                    low.startswith(prefix) for prefix in [
                        "chair ",
                        "vice chair ",
                        "vice chair for supervision ",
                        "governor ",
                        "president ",
                    ]
                ):
                    speaker_hint = cand
                    role_hint = cand
                    continue

                if venue_hint is None and (
                    cand.startswith("At ")
                    or cand.startswith("Before ")
                    or cand.startswith("At the ")
                    or cand.startswith("Before the ")
                ):
                    venue_hint = cand

        items.append(
            {
                "event_type": event_type,
                "year": int(year),
                "title": title,
                "url": full_url,
                "date_hint": date_hint,
                "speaker_hint": speaker_hint,
                "role_hint": role_hint,
                "venue_hint": venue_hint,
            }
        )

    return items


def extract_pdf_url(soup: BeautifulSoup, page_url: str) -> Optional[str]:
    for a in soup.find_all("a", href=True):
        href = clean_text(a.get("href", ""))
        label = clean_text(a.get_text(" ", strip=True)).lower()
        if href.lower().endswith(".pdf"):
            return urljoin(page_url, href)
        if label == "pdf":
            return urljoin(page_url, href)
    return None


def extract_role_and_speaker(raw_speaker_line: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not raw_speaker_line:
        return None, None

    s = clean_text(raw_speaker_line)
    if not s:
        return None, None

    prefixes = [
        "Vice Chair for Supervision ",
        "Vice Chair ",
        "Chair ",
        "Governor ",
        "President ",
    ]

    for prefix in prefixes:
        if s.startswith(prefix):
            return s, s

    return None, s


def extract_meta(soup: BeautifulSoup, page_url: str, hints: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    result = {
        "title": None,
        "date": None,
        "speaker": None,
        "role": None,
        "venue": None,
        "pdf_url": None,
    }

    title_node = soup.find(["h1", "h2", "h3"])
    if title_node:
        title_txt = clean_text(title_node.get_text(" ", strip=True))
        if title_txt.lower() not in {"speech", "testimony"}:
            result["title"] = title_txt

    lines = [clean_text(x) for x in soup.get_text("\n", strip=True).split("\n") if clean_text(x)]

    for line in lines[:120]:
        maybe_date = parse_date_text(line)
        if maybe_date and result["date"] is None:
            result["date"] = maybe_date
            continue

        low = line.lower()

        if result["speaker"] is None and (
            low.startswith("chair ")
            or low.startswith("vice chair ")
            or low.startswith("vice chair for supervision ")
            or low.startswith("governor ")
            or low.startswith("president ")
        ):
            role, speaker = extract_role_and_speaker(line)
            result["role"] = role
            result["speaker"] = speaker
            continue

        if result["venue"] is None and (
            line.startswith("At ")
            or line.startswith("Before ")
            or line.startswith("At the ")
            or line.startswith("Before the ")
        ):
            result["venue"] = line

    result["pdf_url"] = extract_pdf_url(soup, page_url)

    if result["title"] is None:
        result["title"] = hints.get("title")
    if result["date"] is None:
        result["date"] = hints.get("date_hint")
    if result["speaker"] is None:
        hinted_role, hinted_speaker = extract_role_and_speaker(hints.get("speaker_hint"))
        result["speaker"] = hinted_speaker
        if result["role"] is None:
            result["role"] = hinted_role
    if result["role"] is None:
        result["role"] = hints.get("role_hint")
    if result["venue"] is None:
        result["venue"] = hints.get("venue_hint")

    return result


def strip_known_junk(text: str) -> str:
    t = clean_text(text)
    low = t.lower()

    for pat in ACCESSIBILITY_JUNK_PATTERNS:
        if pat in low:
            idx = low.find(pat)
            next_thank = re.search(r"(thank you|good morning|good afternoon|it is a pleasure|let me)", low[idx:])
            if next_thank:
                cut = idx + next_thank.start()
                t = t[cut:]
                low = t.lower()
                break

    for pat in FOOTER_JUNK_PATTERNS:
        if pat in low:
            idx = low.find(pat)
            t = t[:idx]
            low = t.lower()
            break

    last_update_match = re.search(r"last update:", low)
    if last_update_match:
        t = t[: last_update_match.start()]

    return clean_text(t)


def extract_body_text(soup: BeautifulSoup) -> str:
    lines = [clean_text(x) for x in soup.get_text("\n", strip=True).split("\n") if clean_text(x)]

    skip_patterns = [
        "share",
        "pdf",
        "watch live",
        "return to text",
        "back to top",
        "subscribe to rss",
        "subscribe to email",
        "board of governors of the federal reserve system",
        "for media inquiries",
    ]

    body_lines: List[str] = []
    started = False

    for line in lines:
        low = line.lower()

        if not started:
            if any(
                low.startswith(prefix) for prefix in [
                    "at ", "before ", "at the ", "before the ",
                    "thank you", "good morning", "good afternoon", "good evening",
                    "mr. chairman", "chairman", "madam chair", "let me",
                    "today i", "this evening", "i would like", "it is a pleasure",
                ]
            ):
                started = True

        if started:
            if any(bad in low for bad in skip_patterns):
                continue
            if re.match(r"^\d+\.$", line):
                continue
            if low in {"speech", "testimony"}:
                continue
            body_lines.append(line)

    body = "\n\n".join(body_lines).strip()

    if len(body.split()) < 120:
        paragraphs = []
        for p in soup.find_all("p"):
            txt = clean_text(p.get_text(" ", strip=True))
            if not txt:
                continue
            low = txt.lower()
            if any(bad in low for bad in skip_patterns):
                continue
            if len(txt.split()) >= 5:
                paragraphs.append(txt)
        body = "\n\n".join(paragraphs).strip()

    body = strip_known_junk(body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body


def merge_preferred(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if v and clean_text(str(v)):
            return clean_text(str(v))
    return None


def parse_document(session: requests.Session, item: Dict[str, str]) -> Dict[str, Optional[str]]:
    soup = fetch_html(session, item["url"])
    meta = extract_meta(soup, item["url"], item)
    body_text = extract_body_text(soup)
    score = score_text(body_text)

    return {
        "url": item["url"],
        "event_type": item["event_type"],
        "year": item["year"],
        "date": merge_preferred(meta["date"], item.get("date_hint")),
        "title": merge_preferred(meta["title"], item.get("title")),
        "speaker": merge_preferred(meta["speaker"], item.get("speaker_hint")),
        "role": merge_preferred(meta["role"], item.get("role_hint")),
        "venue": merge_preferred(meta["venue"], item.get("venue_hint")),
        "pdf_url": meta.get("pdf_url"),
        "body_text": body_text,
        "word_count": score.word_count,
        "scraped_at": pd.Timestamp.utcnow().isoformat(),
        "hawkish_score": score.hawkish_score,
        "dovish_score": score.dovish_score,
        "net_score": score.net_score,
        "inflation_concern": score.inflation_concern,
        "labor_concern": score.labor_concern,
        "growth_concern": score.growth_concern,
        "financial_stability": score.financial_stability,
        "balance_sheet": score.balance_sheet,
        "uncertainty_risk": score.uncertainty_risk,
        "policy_relevance": score.policy_relevance,
        "live_signal_share": score.live_signal_share,
    }


def upsert_document(conn: sqlite3.Connection, doc: Dict[str, Optional[str]]) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO documents (
            url, event_type, year, date, title, speaker, role, venue, pdf_url, body_text,
            word_count, scraped_at, hawkish_score, dovish_score, net_score, inflation_concern,
            labor_concern, growth_concern, financial_stability, balance_sheet, uncertainty_risk,
            policy_relevance, live_signal_share
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            event_type=excluded.event_type,
            year=excluded.year,
            date=excluded.date,
            title=excluded.title,
            speaker=excluded.speaker,
            role=excluded.role,
            venue=excluded.venue,
            pdf_url=excluded.pdf_url,
            body_text=excluded.body_text,
            word_count=excluded.word_count,
            scraped_at=excluded.scraped_at,
            hawkish_score=excluded.hawkish_score,
            dovish_score=excluded.dovish_score,
            net_score=excluded.net_score,
            inflation_concern=excluded.inflation_concern,
            labor_concern=excluded.labor_concern,
            growth_concern=excluded.growth_concern,
            financial_stability=excluded.financial_stability,
            balance_sheet=excluded.balance_sheet,
            uncertainty_risk=excluded.uncertainty_risk,
            policy_relevance=excluded.policy_relevance,
            live_signal_share=excluded.live_signal_share
        """,
        (
            doc["url"], doc["event_type"], doc["year"], doc["date"], doc["title"], doc["speaker"],
            doc["role"], doc["venue"], doc["pdf_url"], doc["body_text"], doc["word_count"],
            doc["scraped_at"], doc["hawkish_score"], doc["dovish_score"], doc["net_score"],
            doc["inflation_concern"], doc["labor_concern"], doc["growth_concern"],
            doc["financial_stability"], doc["balance_sheet"], doc["uncertainty_risk"],
            doc["policy_relevance"], doc["live_signal_share"]
        ),
    )
    conn.commit()


def get_existing_urls(conn: sqlite3.Connection) -> set:
    cur = conn.cursor()
    cur.execute("SELECT url FROM documents")
    return {row[0] for row in cur.fetchall()}


@st.cache_data(show_spinner=False, ttl=300)
def load_documents() -> pd.DataFrame:
    init_db()
    conn = db_connection()
    df = pd.read_sql_query("SELECT * FROM documents", conn)
    conn.close()

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    num_cols = [
        "word_count", "hawkish_score", "dovish_score", "net_score",
        "inflation_concern", "labor_concern", "growth_concern",
        "financial_stability", "balance_sheet", "uncertainty_risk",
        "policy_relevance", "live_signal_share",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "policy_relevance" not in df.columns:
        df["policy_relevance"] = np.nan
    if "live_signal_share" not in df.columns:
        df["live_signal_share"] = np.nan

    df = df.sort_values(["date", "title"], ascending=[True, True]).reset_index(drop=True)
    return df


def refresh_corpus(max_years: int = 6, force_refresh: bool = False, progress_bar=None, status_box=None) -> Dict[str, int]:
    init_db()
    clear_log()

    stats = {
        "year_pages_found": 0,
        "index_items_found": 0,
        "attempted_docs": 0,
        "inserted_docs": 0,
        "skipped_existing": 0,
        "failed_docs": 0,
    }

    session = make_session()
    conn = db_connection()

    try:
        if status_box is not None:
            status_box.info("Loading speeches/testimony index...")
        index_soup = fetch_html(session, INDEX_URL)
        year_links = parse_year_links(index_soup)

        if not year_links:
            raise RuntimeError("No yearly speech/testimony links were parsed from the Fed index page.")

        by_type: Dict[str, List[Tuple[str, str, int]]] = {"speech": [], "testimony": []}
        for event_type, url, year in year_links:
            by_type[event_type].append((event_type, url, year))

        selected_year_links: List[Tuple[str, str, int]] = []
        for event_type in ["speech", "testimony"]:
            selected_year_links.extend(by_type[event_type][:max_years])

        selected_year_links = sorted(selected_year_links, key=lambda x: (x[2], x[0]), reverse=True)
        stats["year_pages_found"] = len(selected_year_links)

        existing = get_existing_urls(conn)
        all_items: List[Dict[str, str]] = []

        for idx, (event_type, year_url, year) in enumerate(selected_year_links, start=1):
            try:
                if status_box is not None:
                    status_box.info(f"Loading {event_type} index for {year} ({idx}/{len(selected_year_links)})")
                year_soup = fetch_html(session, year_url)
                items = parse_index_items(year_soup, event_type, year)
                all_items.extend(items)
                log_event("INFO", f"Parsed {len(items)} index items from {year_url}")
                time.sleep(0.1)
            except Exception as exc:
                log_event("ERROR", f"Failed year index {year_url}: {repr(exc)}")

        dedup = {}
        for item in all_items:
            dedup[item["url"]] = item
        all_items = sorted(dedup.values(), key=lambda x: (x["year"], x["url"]), reverse=True)
        stats["index_items_found"] = len(all_items)

        if not all_items:
            raise RuntimeError("Parsed zero speech/testimony documents from the selected yearly index pages.")

        for i, item in enumerate(all_items, start=1):
            stats["attempted_docs"] += 1
            if progress_bar is not None:
                progress_bar.progress(i / max(len(all_items), 1))
            if status_box is not None:
                status_box.info(f"Processing {i:,}/{len(all_items):,}: {item['title'][:120]}")

            if item["url"] in existing and not force_refresh:
                stats["skipped_existing"] += 1
                continue

            try:
                doc = parse_document(session, item)
                if not doc["body_text"] or int(doc["word_count"] or 0) < 80:
                    stats["failed_docs"] += 1
                    log_event("WARN", f"Skipped short/empty body for {item['url']} | words={doc.get('word_count')}")
                    continue

                upsert_document(conn, doc)
                stats["inserted_docs"] += 1
                log_event(
                    "INFO",
                    f"Stored {doc.get('date')} | {doc.get('speaker')} | {doc.get('title')} | "
                    f"words={doc.get('word_count')} | policy_relevance={doc.get('policy_relevance')}"
                )
                time.sleep(0.05)
            except Exception as exc:
                stats["failed_docs"] += 1
                log_event("ERROR", f"Failed document {item['url']}: {repr(exc)}")

    finally:
        conn.close()
        load_documents.clear()

    return stats


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    corpus_mean, corpus_std = safe_mean_std(out["net_score"])
    infl_mean, infl_std = safe_mean_std(out["inflation_concern"])
    labor_mean, labor_std = safe_mean_std(out["labor_concern"])
    growth_mean, growth_std = safe_mean_std(out["growth_concern"])
    fs_mean, fs_std = safe_mean_std(out["financial_stability"])
    unc_mean, unc_std = safe_mean_std(out["uncertainty_risk"])

    out["tone_z_fed"] = out["net_score"].apply(lambda x: bounded_z(zscore_value(x, corpus_mean, corpus_std)))
    out["inflation_z_fed"] = out["inflation_concern"].apply(lambda x: bounded_z(zscore_value(x, infl_mean, infl_std)))
    out["labor_z_fed"] = out["labor_concern"].apply(lambda x: bounded_z(zscore_value(x, labor_mean, labor_std)))
    out["growth_z_fed"] = out["growth_concern"].apply(lambda x: bounded_z(zscore_value(x, growth_mean, growth_std)))
    out["fs_z_fed"] = out["financial_stability"].apply(lambda x: bounded_z(zscore_value(x, fs_mean, fs_std)))
    out["uncertainty_z_fed"] = out["uncertainty_risk"].apply(lambda x: bounded_z(zscore_value(x, unc_mean, unc_std)))

    out["speaker_weight"] = out["speaker"].map(SPEAKER_WEIGHTS).fillna(1.0)

    out["tone_z_speaker"] = np.nan
    out["inflation_z_speaker"] = np.nan
    out["labor_z_speaker"] = np.nan
    out["growth_z_speaker"] = np.nan

    for speaker, idx in out.groupby("speaker", dropna=False).groups.items():
        subset = out.loc[idx]
        m, s = safe_mean_std(subset["net_score"])
        out.loc[idx, "tone_z_speaker"] = subset["net_score"].apply(lambda x: bounded_z(zscore_value(x, m, s)))
        m, s = safe_mean_std(subset["inflation_concern"])
        out.loc[idx, "inflation_z_speaker"] = subset["inflation_concern"].apply(lambda x: bounded_z(zscore_value(x, m, s)))
        m, s = safe_mean_std(subset["labor_concern"])
        out.loc[idx, "labor_z_speaker"] = subset["labor_concern"].apply(lambda x: bounded_z(zscore_value(x, m, s)))
        m, s = safe_mean_std(subset["growth_concern"])
        out.loc[idx, "growth_z_speaker"] = subset["growth_concern"].apply(lambda x: bounded_z(zscore_value(x, m, s)))

    out["tone_percentile_fed"] = out["net_score"].apply(lambda x: percentile_rank(out["net_score"], x))
    out["inflation_percentile_fed"] = out["inflation_concern"].apply(lambda x: percentile_rank(out["inflation_concern"], x))
    out["labor_percentile_fed"] = out["labor_concern"].apply(lambda x: percentile_rank(out["labor_concern"], x))
    out["growth_percentile_fed"] = out["growth_concern"].apply(lambda x: percentile_rank(out["growth_concern"], x))

    out["stance"] = out["tone_z_fed"].apply(tone_bucket)
    out["speaker_delta"] = out["tone_z_speaker"]
    out["policy_bucket"] = out["policy_relevance"].apply(policy_bucket)

    return out


def aggregate_series(df: pd.DataFrame, freq: str = "30D") -> pd.DataFrame:
    if df.empty:
        return df

    frame = df.copy().dropna(subset=["date"])
    if frame.empty:
        return frame

    if freq == "30D":
        min_date = frame["date"].min().normalize()
        frame["bucket"] = min_date + (((frame["date"] - min_date).dt.days // 30) * pd.Timedelta(days=30))
    else:
        frame["bucket"] = frame["date"].dt.to_period(freq).dt.start_time

    grouped = (
        frame.groupby("bucket")
        .apply(
            lambda x: pd.Series(
                {
                    "tone_z_fed": np.average(x["tone_z_fed"], weights=x["speaker_weight"]),
                    "inflation_z_fed": np.average(x["inflation_z_fed"], weights=x["speaker_weight"]),
                    "labor_z_fed": np.average(x["labor_z_fed"], weights=x["speaker_weight"]),
                    "growth_z_fed": np.average(x["growth_z_fed"], weights=x["speaker_weight"]),
                    "documents": len(x),
                    "policy_relevance": np.average(x["policy_relevance"].fillna(0.0), weights=x["speaker_weight"]),
                }
            )
        )
        .reset_index()
        .rename(columns={"bucket": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    grouped["tone_z_fed_smooth"] = grouped["tone_z_fed"].ewm(span=4, adjust=False).mean()
    grouped["tone_3m_ma"] = grouped["tone_z_fed_smooth"].rolling(3, min_periods=1).mean()
    grouped["tone_6m_ma"] = grouped["tone_z_fed_smooth"].rolling(6, min_periods=1).mean()

    return grouped


def split_sentences(text: str) -> List[str]:
    text = clean_text(text)
    parts = re.split(r"(?<=[\.\?\!])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def find_snippets(text: str, phrases: List[str], max_snippets: int = 4) -> List[str]:
    if not text:
        return []

    sentences = split_sentences(text)
    found = []

    for sent in sentences:
        low = sent.lower()
        if any(phrase.lower() in low for phrase in phrases):
            found.append(sent)
        if len(found) >= max_snippets:
            break

    return found


def highlight_terms(text: str, phrases: List[str]) -> str:
    highlighted = html.escape(text)
    for phrase in sorted(set(phrases), key=len, reverse=True):
        if len(phrase) < 3:
            continue
        pattern = re.compile(re.escape(html.escape(phrase)), re.IGNORECASE)
        highlighted = pattern.sub(
            lambda m: f"<mark style='background-color:#ffe58f'>{m.group(0)}</mark>",
            highlighted,
        )
    return highlighted


def speaker_matrix(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    if df.empty:
        return df

    agg = (
        df.groupby("speaker", dropna=False)
        .agg(
            docs=("url", "count"),
            tone_z_fed=("tone_z_fed", "mean"),
            tone_z_speaker=("tone_z_speaker", "mean"),
            policy_relevance=("policy_relevance", "mean"),
            last_date=("date", "max"),
        )
        .reset_index()
    )

    latest = (
        df.sort_values("date")
        .groupby("speaker", dropna=False)
        .tail(1)[["speaker", "tone_z_speaker", "tone_z_fed", "stance"]]
        .rename(columns={"tone_z_speaker": "latest_vs_own", "tone_z_fed": "latest_vs_fed"})
    )

    agg = agg.merge(latest, on="speaker", how="left")
    agg["weight"] = agg["speaker"].map(SPEAKER_WEIGHTS).fillna(1.0)
    agg["avg_stance"] = agg["tone_z_fed"].apply(tone_bucket)
    agg = agg.sort_values(["weight", "docs", "tone_z_fed"], ascending=[False, False, False]).head(top_n)
    return agg


def render_about() -> None:
    st.sidebar.markdown("### About This Tool")
    st.sidebar.markdown(
        """
        <div style="padding:0.85rem 1rem;border:1px solid #d9d9d9;border-radius:0.7rem;background:#fafafa;line-height:1.65">
        This dashboard scrapes Federal Reserve speeches and testimony directly from the Board website, scores each document on hawkish and dovish language, and shows how speaker tone is shifting relative to both that speaker’s own history and the broader Fed baseline.
        <br><br>
        The scoring engine now separates topic mentions from live policy signal, discounts ceremonial and historical passages, and weights paragraphs by policy relevance before assigning tone.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_diagnostics(log_df: pd.DataFrame, stats: Optional[Dict[str, int]] = None) -> None:
    st.subheader("Diagnostics")

    if stats:
        a, b, c, d, e, f = st.columns(6)
        a.metric("Year pages", stats.get("year_pages_found", 0))
        b.metric("Index items", stats.get("index_items_found", 0))
        c.metric("Attempted docs", stats.get("attempted_docs", 0))
        d.metric("Inserted docs", stats.get("inserted_docs", 0))
        e.metric("Skipped existing", stats.get("skipped_existing", 0))
        f.metric("Failed docs", stats.get("failed_docs", 0))

    if log_df.empty:
        st.info("No scrape log yet.")
    else:
        st.dataframe(log_df, use_container_width=True, hide_index=True, height=260)


def format_z(x: float) -> str:
    if pd.isna(x):
        return "n.a."
    return f"{x:+.2f}σ"


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "n.a."
    return f"{x:.0f}th %ile"


def stance_color(z: float) -> Tuple[str, str]:
    if pd.isna(z):
        return "#efefef", "#555555"
    if z <= -0.35:
        return "#dff3e4", "#1e6b35"
    if z >= 0.35:
        return "#fbe3e4", "#8a1f2b"
    return "#efefef", "#555555"


def make_badge(label: str, z: float) -> str:
    bg, fg = stance_color(z)
    return (
        f"<span style='display:inline-block;padding:0.28rem 0.6rem;border-radius:999px;"
        f"background:{bg};color:{fg};font-weight:600;font-size:0.88rem'>{html.escape(label)}</span>"
    )


def z_dashboard_figure(series: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    y_series = pd.to_numeric(series["tone_z_fed_smooth"], errors="coerce").dropna()
    ma3_series = pd.to_numeric(series["tone_3m_ma"], errors="coerce").dropna()
    ma6_series = pd.to_numeric(series["tone_6m_ma"], errors="coerce").dropna()
    combined_y = pd.concat([y_series, ma3_series, ma6_series], axis=0).dropna()

    if combined_y.empty:
        y_low, y_high = -1.5, 1.5
    else:
        y_min = float(combined_y.min())
        y_max = float(combined_y.max())
        pad = max(0.18, (y_max - y_min) * 0.18)
        y_low = np.floor((y_min - pad) / 0.25) * 0.25
        y_high = np.ceil((y_max + pad) / 0.25) * 0.25
        if (y_high - y_low) < 1.5:
            mid = (y_high + y_low) / 2
            y_low = mid - 0.75
            y_high = mid + 0.75
        y_low = max(y_low, -3.0)
        y_high = min(y_high, 3.0)

    def add_band(y0, y1, color, opacity):
        band_low = max(y0, y_low)
        band_high = min(y1, y_high)
        if band_high > band_low:
            fig.add_hrect(y0=band_low, y1=band_high, fillcolor=color, opacity=opacity, line_width=0)

    add_band(-3.0, -1.25, "rgb(46, 125, 50)", 0.12)
    add_band(-1.25, -0.35, "rgb(46, 125, 50)", 0.06)
    add_band(-0.35, 0.35, "rgb(160, 160, 160)", 0.05)
    add_band(0.35, 1.25, "rgb(183, 28, 28)", 0.06)
    add_band(1.25, 3.0, "rgb(183, 28, 28)", 0.12)

    fig.add_trace(
        go.Scatter(
            x=series["date"],
            y=series["tone_z_fed_smooth"],
            mode="lines+markers",
            name="Institutional tone z-score",
            line=dict(width=2.4),
            marker=dict(size=6),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}σ<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=series["date"],
            y=series["tone_3m_ma"],
            mode="lines",
            name="3-bucket mean",
            line=dict(dash="dot", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}σ<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=series["date"],
            y=series["tone_6m_ma"],
            mode="lines",
            name="6-bucket mean",
            line=dict(dash="dash", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}σ<extra></extra>",
        )
    )

    x_min = series["date"].min()
    x_max = series["date"].max()

    y_span = y_high - y_low
    if y_span <= 1.5:
        dtick = 0.25
    elif y_span <= 3.0:
        dtick = 0.5
    else:
        dtick = 1.0

    fig.update_layout(
        height=470,
        margin=dict(l=20, r=20, t=84, b=28),
        title=dict(text="Fed tone over time", x=0.01, xanchor="left", y=0.98, yanchor="top"),
        yaxis_title="Z-score vs Fed baseline",
        xaxis_title="",
        xaxis=dict(
            range=[x_min, x_max],
            showgrid=True,
            tickformat="%Y",
            dtick="M12",
            automargin=True,
        ),
        yaxis=dict(
            range=[y_low, y_high],
            tickmode="linear",
            dtick=dtick,
            zeroline=True,
            zerolinewidth=1,
            automargin=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.7)",
        ),
    )

    return fig


def speaker_matrix_figure(matrix: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        matrix,
        x="latest_vs_fed",
        y="speaker",
        size="docs",
        color="latest_vs_own",
        hover_data=["avg_stance", "last_date", "docs", "latest_vs_own", "latest_vs_fed", "policy_relevance"],
        title="Speaker matrix",
        height=470,
        color_continuous_scale="RdYlGn_r",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=68, b=25),
        xaxis_title="Latest speech vs Fed baseline (z-score)",
        yaxis_title="",
        coloraxis_colorbar_title="Vs own history",
    )
    return fig


def clean_latest_table(view: pd.DataFrame) -> pd.DataFrame:
    latest = view.sort_values("date", ascending=False).copy()
    latest["tone"] = latest["tone_z_fed"].apply(format_z)
    latest["vs own"] = latest["tone_z_speaker"].apply(format_z)
    latest["fed %ile"] = latest["tone_percentile_fed"].apply(format_pct)
    latest["inflation"] = latest["inflation_z_fed"].apply(format_z)
    latest["labor"] = latest["labor_z_fed"].apply(format_z)
    latest["growth"] = latest["growth_z_fed"].apply(format_z)
    latest["date"] = latest["date"].dt.strftime("%Y-%m-%d")
    latest["source"] = latest["url"]

    latest = latest[
        ["date", "speaker", "event_type", "stance", "tone", "vs own", "fed %ile", "inflation", "labor", "growth", "source"]
    ].rename(columns={"event_type": "type"})

    return latest


def scorecard_dataframe(doc: pd.Series) -> pd.DataFrame:
    rows = [
        ("Overall tone", doc["tone_z_fed"], doc["tone_percentile_fed"], tone_bucket(doc["tone_z_fed"])),
        ("Vs own history", doc["tone_z_speaker"], np.nan, tone_bucket(doc["tone_z_speaker"])),
        ("Policy relevance", doc["policy_relevance"] * 2.0 - 1.0 if pd.notna(doc["policy_relevance"]) else np.nan, np.nan, policy_bucket(doc["policy_relevance"])),
        ("Inflation salience", doc["inflation_z_fed"], doc["inflation_percentile_fed"], emphasis_bucket(doc["inflation_z_fed"])),
        ("Labor salience", doc["labor_z_fed"], doc["labor_percentile_fed"], emphasis_bucket(doc["labor_z_fed"])),
        ("Growth salience", doc["growth_z_fed"], doc["growth_percentile_fed"], emphasis_bucket(doc["growth_z_fed"])),
        ("Financial stability", doc["fs_z_fed"], np.nan, emphasis_bucket(doc["fs_z_fed"])),
        ("Uncertainty", doc["uncertainty_z_fed"], np.nan, emphasis_bucket(doc["uncertainty_z_fed"])),
    ]
    return pd.DataFrame(rows, columns=["dimension", "z", "percentile", "bucket"])


def scorecard_figure(scorecard: pd.DataFrame) -> go.Figure:
    plot_df = scorecard.copy().sort_values("z", ascending=True)

    colors = []
    for dim, z in zip(plot_df["dimension"], plot_df["z"]):
        if pd.isna(z):
            colors.append("#bdbdbd")
        elif dim == "Policy relevance":
            colors.append("#7e57c2")
        elif z <= -0.35:
            colors.append("#66bb6a")
        elif z >= 0.35:
            colors.append("#ef5350")
        else:
            colors.append("#bdbdbd")

    fig = go.Figure(
        go.Bar(
            x=plot_df["z"],
            y=plot_df["dimension"],
            orientation="h",
            text=[f"{z:+.2f}σ" if pd.notna(z) else "n.a." for z in plot_df["z"]],
            textposition="outside",
            marker_color=colors,
            hovertemplate="%{y}<br>%{x:.2f}<extra></extra>",
        )
    )
    fig.add_vline(x=-0.35, line_dash="dot", line_width=1)
    fig.add_vline(x=0.35, line_dash="dot", line_width=1)
    fig.add_vline(x=-1.25, line_dash="dash", line_width=1)
    fig.add_vline(x=1.25, line_dash="dash", line_width=1)
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        title="Document scorecard",
        xaxis_title="Relative score",
        yaxis_title="",
        xaxis=dict(range=[-3, 3]),
    )
    return fig


def doc_summary_html(doc: pd.Series) -> str:
    tone = make_badge(tone_bucket(doc["tone_z_fed"]), doc["tone_z_fed"])
    own = make_badge(f"Vs own: {tone_bucket(doc['tone_z_speaker'])}", doc["tone_z_speaker"])
    infl = make_badge(f"Inflation: {emphasis_bucket(doc['inflation_z_fed'])}", doc["inflation_z_fed"])
    labor = make_badge(f"Labor: {emphasis_bucket(doc['labor_z_fed'])}", doc["labor_z_fed"])
    growth = make_badge(f"Growth: {emphasis_bucket(doc['growth_z_fed'])}", doc["growth_z_fed"])

    policy_bucket_value = doc.get("policy_bucket", "Unknown")
    policy_relevance_value = doc.get("policy_relevance", np.nan)
    live_signal_value = doc.get("live_signal_share", np.nan)

    if pd.isna(live_signal_value):
        live_signal_text = "n.a."
    else:
        live_signal_text = f"{int(round(float(live_signal_value) * 100))}%"

    if pd.isna(policy_relevance_value):
        policy_relevance_text = "n.a."
    else:
        policy_relevance_text = f"{float(policy_relevance_value):.2f}"

    policy_style = "#efe7ff" if policy_bucket_value == "High" else "#f4f4f4"
    policy_text = "#4a2ea8" if policy_bucket_value == "High" else "#555555"
    policy_badge = (
        f"<span style='display:inline-block;padding:0.28rem 0.6rem;border-radius:999px;"
        f"background:{policy_style};color:{policy_text};font-weight:600;font-size:0.88rem'>"
        f"Policy relevance: {html.escape(str(policy_bucket_value))}</span>"
    )

    return f"""
    <div style="padding:0.9rem 1rem;border:1px solid #e6e6e6;border-radius:0.8rem;background:#fafafa">
        <div style="font-size:1.05rem;font-weight:700;margin-bottom:0.45rem">{html.escape(doc['title'] or 'Untitled')}</div>
        <div style="margin-bottom:0.55rem;color:#555">
            {html.escape(str(doc['date'].date()) if pd.notna(doc['date']) else 'Unknown date')} |
            {html.escape(doc['speaker'] or 'Unknown speaker')} |
            {html.escape((doc['event_type'] or '').title())}
        </div>
        <div style="display:flex;gap:0.4rem;flex-wrap:wrap;margin-bottom:0.6rem">
            {policy_badge}
            {tone}
            {own}
            {infl}
            {labor}
            {growth}
        </div>
        <div style="font-size:0.92rem;color:#555">
            Fed baseline: {format_z(doc['tone_z_fed'])} |
            vs own history: {format_z(doc['tone_z_speaker'])} |
            Fed percentile: {format_pct(doc['tone_percentile_fed'])} |
            Live-signal share: {live_signal_text} |
            Policy relevance score: {policy_relevance_text}
        </div>
    </div>
    """


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Primary-source underwriting of Fed communication tone from speeches and testimony.")
    init_db()
    render_about()

    with st.sidebar:
        st.subheader("Controls")
        years_to_pull = st.slider("Years to scan from latest backward", min_value=1, max_value=20, value=6)
        force_refresh = st.checkbox("Force refresh existing documents", value=False)
        refresh_clicked = st.button("Refresh corpus", use_container_width=True)

    refresh_stats = None

    if refresh_clicked:
        progress_bar = st.sidebar.progress(0)
        status_box = st.sidebar.empty()
        with st.spinner("Refreshing Fed corpus..."):
            try:
                refresh_stats = refresh_corpus(
                    max_years=years_to_pull,
                    force_refresh=force_refresh,
                    progress_bar=progress_bar,
                    status_box=status_box,
                )
                progress_bar.empty()
                status_box.empty()
            except Exception as exc:
                progress_bar.empty()
                status_box.empty()
                st.error(f"Refresh failed: {exc}")

    df = load_documents()
    log_df = load_log()

    if refresh_stats is not None:
        st.caption(
            f"Refresh complete. Inserted {refresh_stats['inserted_docs']} documents out of "
            f"{refresh_stats['attempted_docs']} attempted."
        )

    if df.empty:
        st.warning("No local corpus found yet.")
        render_diagnostics(log_df, refresh_stats)
        st.stop()

    df = compute_features(df)
    st.caption(f"Loaded {len(df):,} documents.")

    min_date_ts = df["date"].dropna().min()
    max_date_ts = df["date"].dropna().max()

    if pd.isna(min_date_ts) or pd.isna(max_date_ts):
        st.error("Corpus loaded, but parsed dates are missing across the dataset.")
        render_diagnostics(log_df, refresh_stats)
        st.stop()

    min_date = min_date_ts.date()
    max_date = max_date_ts.date()

    with st.sidebar:
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        speakers = sorted([x for x in df["speaker"].dropna().unique().tolist() if x])
        selected_speakers = st.multiselect("Speakers", options=speakers, default=[])
        event_type_options = sorted(df["event_type"].dropna().unique().tolist())
        event_types = st.multiselect("Event types", options=event_type_options, default=event_type_options)
        search = st.text_input("Search title or transcript")
        min_policy = st.slider("Minimum policy relevance", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    elif isinstance(date_range, list) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start_date, end_date = df["date"].min(), df["date"].max()

    view = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()

    if selected_speakers:
        view = view[view["speaker"].isin(selected_speakers)]

    if event_types:
        view = view[view["event_type"].isin(event_types)]

    if min_policy > 0:
        view = view[view["policy_relevance"].fillna(0.0) >= min_policy]

    if search.strip():
        q = search.strip().lower()
        view = view[
            view["title"].fillna("").str.lower().str.contains(q, na=False)
            | view["body_text"].fillna("").str.lower().str.contains(q, na=False)
        ]

    if view.empty:
        st.warning("No documents match the current filters.")
        render_diagnostics(log_df, refresh_stats)
        st.stop()

    series = aggregate_series(view, freq="30D")
    matrix = speaker_matrix(view, top_n=12)

    top_left, top_right = st.columns([1.8, 1.2])

    with top_left:
        st.plotly_chart(
            z_dashboard_figure(series),
            use_container_width=True,
            config={"displaylogo": False},
        )

    with top_right:
        if not matrix.empty:
            st.plotly_chart(
                speaker_matrix_figure(matrix),
                use_container_width=True,
                config={"displaylogo": False},
            )
        else:
            st.info("No speaker matrix available.")

    st.subheader("Latest communication")
    latest_table = clean_latest_table(view)

    st.dataframe(
        latest_table,
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config={
            "speaker": st.column_config.TextColumn(width="large"),
            "type": st.column_config.TextColumn(width="small"),
            "stance": st.column_config.TextColumn(width="medium"),
            "tone": st.column_config.TextColumn(width="small"),
            "vs own": st.column_config.TextColumn(width="small"),
            "fed %ile": st.column_config.TextColumn(width="small"),
            "inflation": st.column_config.TextColumn(width="small"),
            "labor": st.column_config.TextColumn(width="small"),
            "growth": st.column_config.TextColumn(width="small"),
            "source": st.column_config.LinkColumn("open"),
        },
    )

    st.subheader("Transcript underwrite")
    options = view.sort_values("date", ascending=False).copy()
    options["label"] = options.apply(
        lambda r: f"{r['date'].date() if pd.notna(r['date']) else 'n.a.'} | {r['speaker'] or 'Unknown'} | {r['title']}",
        axis=1,
    )
    selected_label = st.selectbox("Choose a document", options["label"].tolist(), index=0)
    doc = options.loc[options["label"] == selected_label].iloc[0]

    st.markdown(doc_summary_html(doc), unsafe_allow_html=True)

    upper_left, upper_right = st.columns([1.55, 1.0])

    with upper_left:
        st.markdown("**Highlighted transcript excerpt**")
        excerpt = clean_text(doc["body_text"])[:6000]
        patterns = list({p for vals in SNIPPET_PATTERNS.values() for p in vals})
        st.markdown(
            f"<div style='line-height:1.65;padding:0.95rem 1rem;border:1px solid #e6e6e6;border-radius:0.8rem;background:#fff'>{highlight_terms(excerpt, patterns)}</div>",
            unsafe_allow_html=True,
        )

        hawk_snips = find_snippets(doc["body_text"], SNIPPET_PATTERNS["hawkish"], max_snippets=3)
        dove_snips = find_snippets(doc["body_text"], SNIPPET_PATTERNS["dovish"], max_snippets=3)
        infl_snips = find_snippets(doc["body_text"], SNIPPET_PATTERNS["inflation"], max_snippets=2)
        labor_snips = find_snippets(doc["body_text"], SNIPPET_PATTERNS["labor"], max_snippets=2)

        st.markdown("**Passages driving the score**")
        if hawk_snips:
            st.markdown("Hawkish passages")
            for s in hawk_snips:
                st.markdown(f"> {s}")
        if dove_snips:
            st.markdown("Dovish passages")
            for s in dove_snips:
                st.markdown(f"> {s}")
        if infl_snips:
            st.markdown("Inflation passages")
            for s in infl_snips:
                st.markdown(f"> {s}")
        if labor_snips:
            st.markdown("Labor passages")
            for s in labor_snips:
                st.markdown(f"> {s}")

    with upper_right:
        st.markdown("**Document interpretation**")
        st.markdown(
            f"""
            <div style="padding:0.95rem 1rem;border:1px solid #e6e6e6;border-radius:0.8rem;background:#fafafa;line-height:1.7">
                <div><strong>Title</strong>: {html.escape(doc['title'] or 'Unknown')}</div>
                <div><strong>Speaker</strong>: {html.escape(doc['speaker'] or 'Unknown')}</div>
                <div><strong>Role</strong>: {html.escape(doc['role'] or 'Unknown')}</div>
                <div><strong>Date</strong>: {html.escape(str(doc['date'].date()) if pd.notna(doc['date']) else 'Unknown')}</div>
                <div><strong>Type</strong>: {html.escape((doc['event_type'] or '').title())}</div>
                <div><strong>Policy relevance</strong>: {html.escape(doc['policy_bucket'])} ({doc['policy_relevance']:.2f})</div>
                <div><strong>Fed baseline</strong>: {format_z(doc['tone_z_fed'])} ({tone_bucket(doc['tone_z_fed'])})</div>
                <div><strong>Vs own history</strong>: {format_z(doc['tone_z_speaker'])} ({tone_bucket(doc['tone_z_speaker'])})</div>
                <div><strong>Fed percentile</strong>: {format_pct(doc['tone_percentile_fed'])}</div>
                <div><strong>Live-signal share</strong>: {int(round((doc['live_signal_share'] or 0) * 100))}%</div>
                <div><strong>Source</strong>: <a href="{html.escape(doc['url'])}" target="_blank">open original</a></div>
                {"<div><strong>PDF</strong>: <a href='" + html.escape(doc['pdf_url']) + "' target='_blank'>open pdf</a></div>" if doc.get("pdf_url") else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

        scorecard = scorecard_dataframe(doc)
        st.plotly_chart(
            scorecard_figure(scorecard),
            use_container_width=True,
            config={"displaylogo": False},
        )

        scorecard_display = scorecard.copy()
        scorecard_display["z"] = scorecard_display["z"].apply(format_z)
        scorecard_display["percentile"] = scorecard_display["percentile"].apply(format_pct)
        st.dataframe(scorecard_display, use_container_width=True, hide_index=True, height=285)

    st.subheader("Export")
    export_cols = [
        "date", "speaker", "role", "event_type", "title", "policy_relevance", "live_signal_share", "stance",
        "tone_z_fed", "tone_z_speaker", "tone_percentile_fed",
        "inflation_z_fed", "labor_z_fed", "growth_z_fed",
        "financial_stability", "balance_sheet", "uncertainty_risk", "url"
    ]
    csv_bytes = (
        view.sort_values("date", ascending=False)[export_cols]
        .to_csv(index=False)
        .encode("utf-8")
    )
    st.download_button(
        "Download filtered dataset as CSV",
        data=csv_bytes,
        file_name="fed_tone_filtered.csv",
        mime="text/csv",
    )

    render_diagnostics(log_df, refresh_stats)


if __name__ == "__main__":
    main()
