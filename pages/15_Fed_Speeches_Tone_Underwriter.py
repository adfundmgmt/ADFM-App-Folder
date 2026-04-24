
import re
import html
import json
import time
import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import yfinance as yf
except Exception:
    yf = None


APP_TITLE = "Fed Communication Underwriter"
BASE_URL = "https://www.federalreserve.gov"
INDEX_URL = f"{BASE_URL}/newsevents/speeches-testimony.htm"

DATA_DIR = Path("fed_tone_data")
DB_PATH = DATA_DIR / "fed_tone.sqlite"
REQUEST_TIMEOUT = 30

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
)

DATA_DIR.mkdir(parents=True, exist_ok=True)

PASTEL_GREEN = "#52b788"
PASTEL_RED = "#e85d5d"
PASTEL_GREY = "#8b949e"
SOFT_BLUE = "#4f8cc9"
SOFT_PURPLE = "#8e6cc8"
TEXT_GREY = "#5f6b7a"

ROLE_PREFIXES = [
    "Governor and Vice Chair for Supervision",
    "Vice Chair for Supervision",
    "Vice Chair",
    "Chair",
    "Governor",
    "President",
]

SPEAKER_WEIGHTS = {
    "Jerome H. Powell": 3.0,
    "Philip N. Jefferson": 1.6,
    "John C. Williams": 1.6,
    "Christopher J. Waller": 1.5,
    "Michelle W. Bowman": 1.25,
    "Michael S. Barr": 1.2,
    "Lisa D. Cook": 1.1,
    "Adriana D. Kugler": 1.1,
    "Austan D. Goolsbee": 1.05,
    "Mary C. Daly": 1.05,
    "Raphael W. Bostic": 1.05,
    "Thomas I. Barkin": 1.05,
    "Lorie K. Logan": 1.15,
    "Neel Kashkari": 1.1,
    "Susan M. Collins": 1.05,
    "Patrick T. Harker": 1.0,
    "Alberto G. Musalem": 1.0,
    "Beth M. Hammack": 1.0,
    "Stephen I. Miran": 1.0,
    "Lael Brainard": 1.0,
}

ROLE_WEIGHTS = {
    "Chair": 3.0,
    "Vice Chair": 1.6,
    "Vice Chair for Supervision": 1.4,
    "Governor and Vice Chair for Supervision": 1.4,
    "Governor": 1.15,
    "President": 1.0,
}

HAWKISH_PHRASES = {
    "higher for longer": 2.6,
    "restrictive": 1.7,
    "sufficiently restrictive": 2.3,
    "restrictive stance": 2.3,
    "policy restraint": 1.9,
    "maintain restraint": 2.1,
    "inflation remains too high": 2.6,
    "inflation is too high": 2.4,
    "inflation remains elevated": 2.0,
    "elevated inflation": 1.7,
    "persistent inflation": 2.0,
    "inflation persistence": 2.0,
    "upside risks to inflation": 2.2,
    "upside risk": 1.5,
    "risk of inflation": 1.5,
    "price stability": 1.0,
    "still above target": 1.7,
    "above our target": 1.6,
    "strong labor market": 1.2,
    "tight labor market": 1.3,
    "tight labor markets": 1.3,
    "wage pressures": 1.4,
    "overheating": 1.8,
    "further tightening": 2.6,
    "additional firming": 2.6,
    "firming": 1.5,
    "tightening": 1.8,
    "raise rates": 2.4,
    "raise the target range": 2.5,
    "prepared to raise": 2.6,
    "prepared to adjust": 1.2,
    "not yet done": 2.1,
    "premature to ease": 2.6,
    "premature to cut": 2.6,
    "not appropriate to cut": 2.8,
    "more work to do": 1.9,
    "longer than expected": 1.8,
    "vigilant": 1.4,
    "unanchored inflation expectations": 2.5,
    "inflation expectations": 1.0,
    "reaccelerat": 2.2,
}

DOVISH_PHRASES = {
    "disinflation": 1.6,
    "further progress on inflation": 1.9,
    "inflation has eased": 1.8,
    "inflation is easing": 1.8,
    "inflation is moving down": 1.8,
    "downward path": 1.5,
    "normalizing inflation": 1.5,
    "balanced risks": 1.7,
    "two-sided risks": 1.8,
    "downside risk": 1.5,
    "downside risks to employment": 2.1,
    "support the labor market": 2.2,
    "cooling labor market": 1.9,
    "softening labor market": 2.0,
    "labor market is moderating": 1.8,
    "labor market has cooled": 1.9,
    "unemployment is rising": 2.1,
    "slack": 1.3,
    "job growth has slowed": 1.8,
    "growth is slowing": 1.5,
    "economic slowdown": 1.7,
    "below-trend growth": 1.5,
    "weak demand": 1.5,
    "headwinds": 1.2,
    "financial conditions tightened": 1.5,
    "less restrictive": 2.4,
    "room to ease": 2.6,
    "rate cuts": 2.5,
    "cut rates": 2.6,
    "lower rates": 2.5,
    "easing": 2.0,
    "policy can respond": 1.7,
    "act as appropriate": 1.2,
    "can adjust policy": 1.4,
    "can be patient": 1.1,
    "if needed we can": 1.4,
    "could move lower": 1.8,
}

TOPIC_PHRASES = {
    "inflation": {
        "inflation": 1.0,
        "prices": 0.8,
        "price pressures": 1.2,
        "services inflation": 1.4,
        "core inflation": 1.3,
        "pce inflation": 1.4,
        "shelter inflation": 1.2,
        "goods inflation": 1.0,
        "inflation expectations": 1.3,
        "price stability": 1.2,
    },
    "labor": {
        "labor market": 1.4,
        "employment": 1.0,
        "maximum employment": 1.5,
        "unemployment": 1.3,
        "job growth": 1.2,
        "payroll": 1.1,
        "wages": 1.0,
        "hiring": 1.0,
        "layoffs": 1.0,
        "slack": 1.0,
        "participation": 0.8,
    },
    "growth": {
        "growth": 1.0,
        "activity": 0.8,
        "demand": 0.9,
        "consumer spending": 1.1,
        "investment": 0.9,
        "slowdown": 1.1,
        "recession": 1.5,
        "weakness": 1.1,
        "output": 0.8,
        "expansion": 0.8,
    },
    "financial_stability": {
        "financial stability": 1.5,
        "banking": 1.0,
        "banks": 0.9,
        "stress": 1.2,
        "liquidity": 1.1,
        "funding": 1.0,
        "market functioning": 1.3,
        "credit conditions": 1.2,
        "treasury market": 1.1,
        "vulnerabilities": 1.1,
    },
    "balance_sheet": {
        "balance sheet": 1.5,
        "quantitative tightening": 1.5,
        "qt": 1.2,
        "runoff": 1.2,
        "reserves": 1.0,
        "securities holdings": 1.0,
        "mbs": 0.9,
        "treasury holdings": 0.9,
    },
    "uncertainty": {
        "uncertain": 1.0,
        "uncertainty": 1.0,
        "risk management": 1.2,
        "careful": 0.8,
        "monitor": 0.7,
        "watching": 0.7,
        "data dependent": 1.2,
        "incoming data": 1.1,
        "proceed carefully": 1.2,
        "humble": 0.7,
        "attentive": 0.7,
    },
}

POLICY_RELEVANCE_TERMS = [
    "policy", "monetary policy", "committee", "fomc", "federal funds rate",
    "target range", "rate", "rates", "outlook", "forecast", "incoming data",
    "data dependent", "financial conditions", "current", "currently", "today",
    "going forward", "at this time", "we expect", "we will", "we may",
    "we remain", "our stance", "appropriate", "balance of risks", "inflation remains",
    "labor market remains", "economic outlook", "maximum employment", "price stability",
    "restrictive", "easing", "tightening", "fed funds", "real rates", "nominal rates",
]

FORWARD_LOOKING_TERMS = [
    "will", "may", "could", "expect", "outlook", "going forward", "at this time",
    "over coming", "over the coming", "in coming months", "we remain", "we are prepared",
    "if needed", "from here", "currently", "today", "incoming data", "next few meetings",
]

HISTORICAL_TERMS = [
    "in the early", "in the late", "in 19", "in 20", "during the", "history",
    "historical", "legacy", "served", "was appointed", "reappointed",
    "under president", "at that time", "in the 1980s", "in the 1970s",
    "years ago", "decades ago", "career", "throughout his career", "volcker",
    "grew up", "biography",
]

CEREMONIAL_TERMS = [
    "award", "honor", "ceremony", "commencement", "tribute", "memorial",
    "anniversary", "public service", "integrity", "legacy", "congratulations",
    "pleasure to be here", "humbling honor", "thank you for inviting me",
    "it is a pleasure", "introduction",
]

QUOTE_ATTRIBUTION_TERMS = [
    "he said", "she said", "they said", "as he noted", "as she noted",
    "critics argued", "in a speech", "according to", "he acknowledged",
    "she acknowledged",
]

NEGATION_TERMS = {"not", "no", "never", "neither", "nor", "without", "hardly", "scarcely"}
HEDGE_TERMS = [
    "somewhat", "modestly", "gradually", "to some extent", "may", "might",
    "could", "uncertain", "uncertainty", "risk management", "proceed carefully",
]
INTENSIFIER_TERMS = [
    "clearly", "significantly", "materially", "firmly", "strongly",
    "substantially", "persistently", "meaningfully",
]
CONTRAST_CUES = ["but", "however", "although", "though", "yet", "nevertheless", "nonetheless"]

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

SKIP_LINE_PATTERNS = [
    "share",
    "pdf",
    "watch live",
    "return to text",
    "back to top",
    "subscribe to rss",
    "subscribe to email",
    "board of governors of the federal reserve system",
    "for media inquiries",
    "last update",
]

MARKET_ASSETS = {
    "SPY": {"label": "SPY", "type": "return", "expected_if_hawkish": -1.0},
    "QQQ": {"label": "QQQ", "type": "return", "expected_if_hawkish": -1.0},
    "TLT": {"label": "TLT", "type": "return", "expected_if_hawkish": -1.0},
    "IEF": {"label": "IEF", "type": "return", "expected_if_hawkish": -1.0},
    "HYG": {"label": "HYG", "type": "return", "expected_if_hawkish": -1.0},
    "UUP": {"label": "UUP", "type": "return", "expected_if_hawkish": 1.0},
    "GLD": {"label": "GLD", "type": "return", "expected_if_hawkish": -0.5},
}

DOCUMENT_COLUMNS = {
    "url": "TEXT PRIMARY KEY",
    "event_type": "TEXT",
    "year": "INTEGER",
    "date": "TEXT",
    "title": "TEXT",
    "speaker": "TEXT",
    "role": "TEXT",
    "venue": "TEXT",
    "pdf_url": "TEXT",
    "body_text": "TEXT",
    "body_hash": "TEXT",
    "word_count": "INTEGER",
    "source_quality": "REAL",
    "scraped_at": "TEXT",
    "first_seen_at": "TEXT",
    "last_checked_at": "TEXT",
    "hawkish_score": "REAL",
    "dovish_score": "REAL",
    "net_score": "REAL",
    "inflation_concern": "REAL",
    "labor_concern": "REAL",
    "growth_concern": "REAL",
    "financial_stability": "REAL",
    "balance_sheet": "REAL",
    "uncertainty_risk": "REAL",
    "policy_relevance": "REAL",
    "live_signal_share": "REAL",
    "driving_snippets_json": "TEXT",
}


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
    source_quality: float
    driving_snippets: List[Dict[str, Any]]


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
    adapter = HTTPAdapter(max_retries=retries, pool_connections=16, pool_maxsize=16)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def db_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = db_connection()
    cur = conn.cursor()

    column_sql = ", ".join([f"{name} {decl}" for name, decl in DOCUMENT_COLUMNS.items()])
    cur.execute(f"CREATE TABLE IF NOT EXISTS documents ({column_sql})")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scrape_log (
            ts TEXT,
            level TEXT,
            message TEXT
        )
        """
    )

    cur.execute("PRAGMA table_info(documents)")
    existing_cols = {row[1] for row in cur.fetchall()}
    for name, decl in DOCUMENT_COLUMNS.items():
        if name not in existing_cols:
            base_decl = decl.replace(" PRIMARY KEY", "")
            cur.execute(f"ALTER TABLE documents ADD COLUMN {name} {base_decl}")

    conn.commit()
    conn.close()


def log_event(level: str, message: str) -> None:
    init_db()
    conn = db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO scrape_log (ts, level, message) VALUES (?, ?, ?)",
        (pd.Timestamp.utcnow().isoformat(), level, str(message)[:4000]),
    )
    conn.commit()
    conn.close()


def clear_log() -> None:
    init_db()
    conn = db_connection()
    conn.execute("DELETE FROM scrape_log")
    conn.commit()
    conn.close()


def clear_documents() -> None:
    init_db()
    conn = db_connection()
    conn.execute("DELETE FROM documents")
    conn.commit()
    conn.close()
    try:
        load_documents.clear()
    except Exception:
        pass


def load_log(limit: int = 300) -> pd.DataFrame:
    init_db()
    conn = db_connection()
    df = pd.read_sql_query(
        "SELECT * FROM scrape_log ORDER BY ts DESC LIMIT ?",
        conn,
        params=(int(limit),),
    )
    conn.close()
    return df


def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    replacements = {
        "\xa0": " ",
        "â": "'",
        "â": '"',
        "â": '"',
        "â": "-",
        "â": "-",
        "’": "'",
        "“": '"',
        "”": '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_date_text(raw: str) -> Optional[str]:
    raw = clean_text(raw)
    if not raw:
        return None

    raw = re.sub(r"^(Date:|Released on:)\s*", "", raw, flags=re.IGNORECASE)
    for fmt in ("%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return pd.to_datetime(raw, format=fmt).strftime("%Y-%m-%d")
        except Exception:
            pass

    try:
        dt = pd.to_datetime(raw, errors="raise")
        if pd.isna(dt):
            return None
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def canonicalize_speaker_name(raw: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if raw is None:
        return None, None

    s = clean_text(raw)
    if not s:
        return None, None

    s = re.sub(r"^\s*by\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\[[^\]]+\]", "", s)
    s = re.sub(r"\([^)]*\)", "", s)
    s = clean_text(s)

    for prefix in sorted(ROLE_PREFIXES, key=len, reverse=True):
        if s.lower().startswith(prefix.lower() + " "):
            name = clean_text(s[len(prefix):])
            name = re.sub(r",.*$", "", name).strip()
            return prefix, name or None

    match = re.match(r"^(.+?),\s*(Chair|Vice Chair|Governor|President|Vice Chair for Supervision)$", s)
    if match:
        name = clean_text(match.group(1))
        role = clean_text(match.group(2))
        return role, name

    return None, s


def speaker_weight(speaker: Optional[str], role: Optional[str]) -> float:
    if speaker in SPEAKER_WEIGHTS:
        return float(SPEAKER_WEIGHTS[speaker])
    if role in ROLE_WEIGHTS:
        return float(ROLE_WEIGHTS[role])
    return 1.0


def body_hash(text: str) -> str:
    return hashlib.sha256(clean_text(text).encode("utf-8", errors="ignore")).hexdigest()


def word_count(text: str) -> int:
    return len(re.findall(r"\b[a-zA-Z][a-zA-Z\-']+\b", clean_text(text)))


def split_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?;])\s+", text)
    return [clean_text(p) for p in pieces if clean_text(p)]


def split_paragraphs(text: str) -> List[str]:
    raw = re.split(r"\n\s*\n|(?<=\.)\s{2,}", text or "")
    out = []
    for p in raw:
        p = clean_text(p)
        if len(p.split()) >= 10:
            out.append(p)
    if not out and clean_text(text):
        out = [clean_text(text)]
    return out


def phrase_pattern(phrase: str) -> str:
    phrase = phrase.lower().strip()
    if " " not in phrase and phrase.endswith(("at", "er", "ing")):
        return r"\b" + re.escape(phrase) + r"\w*\b"
    return r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"


def weighted_hits(sentence: str, phrase_weights: Dict[str, float]) -> float:
    low = clean_text(sentence).lower()
    if not low:
        return 0.0

    total = 0.0
    has_hedge = any(term in low for term in HEDGE_TERMS)
    has_intensifier = any(term in low for term in INTENSIFIER_TERMS)

    for phrase, weight in phrase_weights.items():
        pattern = phrase_pattern(phrase)
        for match in re.finditer(pattern, low):
            prefix_words = low[: match.start()].split()
            lookback = prefix_words[-6:] if prefix_words else []
            hit = float(weight)

            if any(w in NEGATION_TERMS for w in lookback):
                hit *= 0.25
            if has_hedge:
                hit *= 0.85
            if has_intensifier:
                hit *= 1.10
            if "?" in low:
                hit *= 0.75

            total += hit

    return float(total)


def weighted_topic_hits(sentence: str, phrase_weights: Dict[str, float]) -> float:
    low = clean_text(sentence).lower()
    total = 0.0
    for phrase, weight in phrase_weights.items():
        total += len(re.findall(phrase_pattern(phrase), low)) * float(weight)
    return float(total)


def count_contains(text: str, phrases: List[str]) -> int:
    low = clean_text(text).lower()
    return sum(1 for phrase in phrases if phrase in low)


def sentence_policy_relevance(sentence: str) -> float:
    low = clean_text(sentence).lower()
    if not low:
        return 0.0

    words = max(word_count(low), 1)

    policy_hits = count_contains(low, POLICY_RELEVANCE_TERMS)
    forward_hits = count_contains(low, FORWARD_LOOKING_TERMS)
    historical_hits = count_contains(low, HISTORICAL_TERMS)
    ceremonial_hits = count_contains(low, CEREMONIAL_TERMS)
    quote_hits = count_contains(low, QUOTE_ATTRIBUTION_TERMS)

    topical_hits = sum(weighted_topic_hits(low, terms) for terms in TOPIC_PHRASES.values())

    score = 0.0
    score += 0.23 * min(policy_hits, 5)
    score += 0.20 * min(forward_hits, 4)
    score += 0.09 * min((topical_hits / words) * 35.0, 4.0)

    if "fomc" in low or "committee" in low or "federal funds rate" in low or "target range" in low:
        score += 0.28
    if "today" in low and ("policy" in low or "inflation" in low or "labor market" in low):
        score += 0.20
    if "we" in low and ("will" in low or "expect" in low or "remain" in low or "prepared" in low):
        score += 0.16

    score -= 0.20 * min(historical_hits, 4)
    score -= 0.18 * min(ceremonial_hits, 4)
    score -= 0.14 * min(quote_hits, 3)

    if re.search(r"\bin\s+(19|20)\d{2}\b", low):
        score -= 0.12

    return float(np.clip(score, 0.0, 1.0))


def sentence_direction(sentence: str) -> Dict[str, float]:
    low = clean_text(sentence).lower()
    if not low:
        return {
            "hawk": 0.0,
            "dove": 0.0,
            "inflation": 0.0,
            "labor": 0.0,
            "growth": 0.0,
            "financial_stability": 0.0,
            "balance_sheet": 0.0,
            "uncertainty": 0.0,
            "net": 0.0,
        }

    hawk = weighted_hits(low, HAWKISH_PHRASES)
    dove = weighted_hits(low, DOVISH_PHRASES)

    for cue in CONTRAST_CUES:
        token = f" {cue} "
        if token in low:
            pre, post = low.split(token, 1)
            pre_h = weighted_hits(pre, HAWKISH_PHRASES)
            pre_d = weighted_hits(pre, DOVISH_PHRASES)
            post_h = weighted_hits(post, HAWKISH_PHRASES)
            post_d = weighted_hits(post, DOVISH_PHRASES)
            hawk = 0.70 * pre_h + 1.35 * post_h
            dove = 0.70 * pre_d + 1.35 * post_d
            break

    topics = {
        key: weighted_topic_hits(low, terms)
        for key, terms in TOPIC_PHRASES.items()
    }

    net = (
        1.15 * hawk
        - 1.15 * dove
        + 0.08 * topics["inflation"]
        - 0.05 * topics["labor"]
        - 0.04 * topics["growth"]
        - 0.02 * topics["financial_stability"]
    )

    return {
        "hawk": float(hawk),
        "dove": float(dove),
        "inflation": float(topics["inflation"]),
        "labor": float(topics["labor"]),
        "growth": float(topics["growth"]),
        "financial_stability": float(topics["financial_stability"]),
        "balance_sheet": float(topics["balance_sheet"]),
        "uncertainty": float(topics["uncertainty"]),
        "net": float(net),
    }


def squash_score(x: float, scale: float = 2.2, max_abs: float = 2.3) -> float:
    if pd.isna(x):
        return np.nan
    return float(np.tanh(float(x) / scale) * max_abs)


def source_quality_score(text: str) -> float:
    txt = clean_text(text)
    wc = word_count(txt)
    if wc <= 0:
        return 0.0

    low = txt.lower()
    junk_hits = sum(1 for pat in FOOTER_JUNK_PATTERNS + ACCESSIBILITY_JUNK_PATTERNS if pat in low)
    length_score = min(wc / 700.0, 1.0)
    structure_score = 1.0 if wc >= 250 else 0.65
    junk_penalty = min(junk_hits * 0.18, 0.55)

    if wc < 120:
        length_score *= 0.45
        structure_score *= 0.55

    return float(np.clip(0.60 * length_score + 0.40 * structure_score - junk_penalty, 0.0, 1.0))


def score_text(text: str) -> ToneResult:
    clean = clean_text(text)
    sentences = split_sentences(clean)
    if not sentences:
        sentences = [clean] if clean else []

    rows = []
    for sent in sentences:
        wc = max(word_count(sent), 1)
        rel = sentence_policy_relevance(sent)
        direction = sentence_direction(sent)
        signal_strength = abs(direction["net"]) + direction["hawk"] + direction["dove"]
        rows.append(
            {
                "sentence": sent,
                "words": wc,
                "relevance": rel,
                "signal_strength": signal_strength,
                **direction,
            }
        )

    total_words = max(sum(r["words"] for r in rows), 1)
    weighted_words = max(sum(max(0.06, r["relevance"]) * r["words"] for r in rows), 1e-9)

    def weighted_density(key: str) -> float:
        raw = sum(max(0.06, r["relevance"]) * r[key] for r in rows)
        return float(raw / weighted_words * 1000.0)

    hawk = weighted_density("hawk")
    dove = weighted_density("dove")
    infl = weighted_density("inflation")
    labor = weighted_density("labor")
    growth = weighted_density("growth")
    fs = weighted_density("financial_stability")
    bs = weighted_density("balance_sheet")
    unc = weighted_density("uncertainty")
    net = weighted_density("net")

    policy_relevance = float(np.average([r["relevance"] for r in rows], weights=[r["words"] for r in rows])) if rows else 0.0
    live_signal_share = float(np.mean([1.0 if r["relevance"] >= 0.45 else 0.0 for r in rows])) if rows else 0.0

    damp = 0.35 + 0.65 * policy_relevance
    topic_damp = 0.52 + 0.48 * policy_relevance

    driving = sorted(
        [
            {
                "sentence": r["sentence"],
                "relevance": round(float(r["relevance"]), 3),
                "net": round(float(r["net"]), 3),
                "hawk": round(float(r["hawk"]), 3),
                "dove": round(float(r["dove"]), 3),
            }
            for r in rows
            if r["signal_strength"] > 0
        ],
        key=lambda x: abs(x["net"]) + x["hawk"] + x["dove"] + x["relevance"],
        reverse=True,
    )[:8]

    return ToneResult(
        net_score=squash_score(net * damp, scale=2.0, max_abs=2.3),
        hawkish_score=squash_score(hawk * damp, scale=2.0, max_abs=2.5),
        dovish_score=squash_score(dove * damp, scale=2.0, max_abs=2.5),
        inflation_concern=squash_score(infl * topic_damp, scale=2.7, max_abs=2.5),
        labor_concern=squash_score(labor * topic_damp, scale=2.7, max_abs=2.5),
        growth_concern=squash_score(growth * topic_damp, scale=2.7, max_abs=2.5),
        financial_stability=squash_score(fs * topic_damp, scale=2.7, max_abs=2.3),
        balance_sheet=squash_score(bs * topic_damp, scale=2.7, max_abs=2.3),
        uncertainty_risk=squash_score(unc * topic_damp, scale=2.7, max_abs=2.3),
        word_count=total_words,
        policy_relevance=policy_relevance,
        live_signal_share=live_signal_share,
        source_quality=source_quality_score(clean),
        driving_snippets=driving,
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


def parse_index_items(year_soup: BeautifulSoup, event_type: str, year: int) -> List[Dict[str, Any]]:
    text_lines = [
        clean_text(line)
        for line in year_soup.get_text("\n", strip=True).split("\n")
        if clean_text(line)
    ]

    links: List[Tuple[str, str]] = []
    for a in year_soup.find_all("a", href=True):
        href = urljoin(BASE_URL, a["href"])
        href_low = href.lower()
        title = clean_text(a.get_text(" ", strip=True))
        if not title:
            continue
        if re.search(r"/newsevents/(speech|testimony)/.+\.htm$", href_low):
            links.append((title, href))

    dedup = {}
    for title, href in links:
        dedup[href] = title

    items: List[Dict[str, Any]] = []
    for full_url, title in dedup.items():
        title_idx = None
        for idx, line in enumerate(text_lines):
            if line == title:
                title_idx = idx
                break

        date_hint = None
        speaker_hint = None
        venue_hint = None

        if title_idx is not None:
            before = text_lines[max(0, title_idx - 4): title_idx]
            after = text_lines[title_idx + 1: title_idx + 8]

            for cand in reversed(before):
                maybe_date = parse_date_text(cand)
                if maybe_date:
                    date_hint = maybe_date
                    break

            for cand in after:
                low = cand.lower()
                if any(skip in low for skip in SKIP_LINE_PATTERNS):
                    continue

                role, speaker = canonicalize_speaker_name(cand)
                if speaker_hint is None and role is not None and speaker is not None:
                    speaker_hint = cand
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
                "venue_hint": venue_hint,
            }
        )

    return items


def extract_pdf_url(soup: BeautifulSoup, page_url: str) -> Optional[str]:
    for a in soup.find_all("a", href=True):
        href = clean_text(a.get("href", ""))
        label = clean_text(a.get_text(" ", strip=True)).lower()
        if href.lower().endswith(".pdf") or label == "pdf":
            return urljoin(page_url, href)
    return None


def extract_meta(soup: BeautifulSoup, page_url: str, hints: Dict[str, Any]) -> Dict[str, Optional[str]]:
    result = {
        "title": None,
        "date": None,
        "speaker": None,
        "role": None,
        "venue": None,
        "pdf_url": extract_pdf_url(soup, page_url),
    }

    title_candidates = []
    for node in soup.find_all(["h1", "h2", "h3"]):
        txt = clean_text(node.get_text(" ", strip=True))
        if txt and txt.lower() not in {"speech", "testimony", "speeches", "testimonies"}:
            title_candidates.append(txt)

    if title_candidates:
        result["title"] = title_candidates[0]

    lines = [
        clean_text(x)
        for x in soup.get_text("\n", strip=True).split("\n")
        if clean_text(x)
    ]

    for line in lines[:180]:
        maybe_date = parse_date_text(line)
        if maybe_date and result["date"] is None:
            result["date"] = maybe_date
            continue

        role, speaker = canonicalize_speaker_name(line)
        if result["speaker"] is None and role is not None and speaker is not None:
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

    if result["title"] is None:
        result["title"] = hints.get("title")

    if result["date"] is None:
        result["date"] = hints.get("date_hint")

    if result["speaker"] is None:
        hinted_role, hinted_speaker = canonicalize_speaker_name(hints.get("speaker_hint"))
        result["role"] = hinted_role
        result["speaker"] = hinted_speaker

    if result["venue"] is None:
        result["venue"] = hints.get("venue_hint")

    return result


def strip_known_junk(text: str) -> str:
    t = text or ""
    t = re.sub(r"\n{3,}", "\n\n", t)
    low = clean_text(t).lower()

    for pat in ACCESSIBILITY_JUNK_PATTERNS:
        if pat in low:
            idx = low.find(pat)
            next_start = re.search(
                r"(thank you|good morning|good afternoon|good evening|let me|today i|i would like)",
                low[idx:],
            )
            if next_start:
                cut = idx + next_start.start()
                t = t[cut:]
                low = clean_text(t).lower()
                break

    for pat in FOOTER_JUNK_PATTERNS:
        if pat in low:
            idx = low.find(pat)
            t = t[:idx]
            low = clean_text(t).lower()
            break

    last_update = re.search(r"last update:", low)
    if last_update:
        t = t[: last_update.start()]

    return clean_text(t)


def _candidate_body_from_container(container: BeautifulSoup) -> str:
    lines = [
        clean_text(x)
        for x in container.get_text("\n", strip=True).split("\n")
        if clean_text(x)
    ]

    if not lines:
        return ""

    def is_end_line(line: str) -> bool:
        low = line.lower()
        return (
            low.startswith("last update:")
            or low == "back to top"
            or low.startswith("board of governors of the federal reserve system")
            or low.startswith("stay connected")
            or low.startswith("tools and information")
            or low.startswith("connect with the board")
        )

    def is_skip_line(line: str) -> bool:
        low = line.lower().strip()
        if not low:
            return True
        if low in {"share", "pdf", "speech", "testimony", "speeches", "testimonies"}:
            return True
        if any(bad == low or bad in low for bad in SKIP_LINE_PATTERNS):
            return True
        if any(bad in low for bad in ACCESSIBILITY_JUNK_PATTERNS):
            return True
        if re.match(r"^\[?[a-z/ ]+\]?\s+(toggles|seeks|increase|decrease)", low):
            return True
        if re.match(r"^\d+\.$", low):
            return True
        if "twitter.com" in low or "facebook.com" in low or "linkedin.com" in low:
            return True
        return False

    def is_body_opening(line: str) -> bool:
        low = line.lower().strip()
        starts = (
            "thank you",
            "good morning",
            "good afternoon",
            "good evening",
            "let me",
            "today i",
            "this morning",
            "this afternoon",
            "this evening",
            "i would like",
            "i am pleased",
            "i'm pleased",
            "it is a pleasure",
            "it’s a pleasure",
            "it is an honor",
            "i appreciate",
            "chairman ",
            "chair ",
            "ranking member",
            "members of the committee",
            "madam chair",
            "mr. chairman",
            "the federal reserve",
            "as always",
            "in my remarks",
            "in these remarks",
        )
        return low.startswith(starts)

    def is_venue_line(line: str) -> bool:
        return (
            line.startswith("At ")
            or line.startswith("Before ")
            or line.startswith("At the ")
            or line.startswith("Before the ")
        )

    cut_lines = []
    for line in lines:
        if is_end_line(line):
            break
        cut_lines.append(line)

    lines = cut_lines

    start_idx = None
    for i, line in enumerate(lines):
        if is_body_opening(line):
            start_idx = i
            break

    if start_idx is None:
        venue_indices = [i for i, line in enumerate(lines) if is_venue_line(line)]
        if venue_indices:
            start_idx = venue_indices[-1] + 1

    if start_idx is None:
        # Last resort: start after the first speaker/date/title block.
        for i, line in enumerate(lines):
            role, speaker = canonicalize_speaker_name(line)
            if role and speaker:
                start_idx = i + 1
                break

    if start_idx is None:
        start_idx = 0

    body_lines = []
    for line in lines[start_idx:]:
        if is_skip_line(line):
            continue
        if parse_date_text(line):
            continue

        role, speaker = canonicalize_speaker_name(line)
        if role and speaker and len(line.split()) <= 8:
            continue

        if is_venue_line(line) and len(body_lines) == 0:
            continue

        body_lines.append(line)

    return strip_known_junk("\n\n".join(body_lines))


def extract_body_text(soup: BeautifulSoup) -> str:
    # Do not rely on the first generic "content" div. The Fed site has many navigation
    # containers before the article body, and selecting one of those is what caused the
    # one-word extraction failure.
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()

    candidates = []

    selectors = [
        "main",
        "article",
        "div.col-xs-12.col-sm-8.col-md-8",
        "div.col-sm-8",
        "div.col-md-8",
        "div.article",
        "div.article__body",
        "div#article",
        "div#content",
        "div.content",
    ]

    seen = set()
    for selector in selectors:
        try:
            for node in soup.select(selector):
                ident = id(node)
                if ident not in seen:
                    seen.add(ident)
                    candidates.append(node)
        except Exception:
            continue

    candidates.append(soup)

    best_body = ""
    best_score = -1.0

    for node in candidates:
        paragraphs = []
        for p in node.find_all(["p", "blockquote"]):
            txt = clean_text(p.get_text(" ", strip=True))
            low = txt.lower()
            if not txt or len(txt.split()) < 4:
                continue
            if any(bad in low for bad in SKIP_LINE_PATTERNS):
                continue
            if any(bad in low for bad in ACCESSIBILITY_JUNK_PATTERNS):
                continue
            paragraphs.append(txt)

        paragraph_body = strip_known_junk("\n\n".join(paragraphs))
        line_body = _candidate_body_from_container(node)

        for body in [paragraph_body, line_body]:
            wc = word_count(body)
            if wc <= 0:
                continue

            low = body.lower()
            junk_hits = sum(1 for pat in FOOTER_JUNK_PATTERNS + ACCESSIBILITY_JUNK_PATTERNS if pat in low)
            score = wc - 140 * junk_hits

            # Prefer the true body over the whole-page candidate with navigation.
            if "main menu toggle button" in low or "official websites use .gov" in low:
                score -= 500
            if "last update:" in low:
                score -= 75

            if score > best_score:
                best_score = score
                best_body = body

    return strip_known_junk(best_body)

def parse_document(session: requests.Session, item: Dict[str, Any]) -> Dict[str, Any]:
    soup = fetch_html(session, item["url"])
    meta = extract_meta(soup, item["url"], item)
    body_text = extract_body_text(soup)
    score = score_text(body_text)

    now = pd.Timestamp.utcnow().isoformat()
    role = meta.get("role")
    speaker = meta.get("speaker")

    if speaker:
        parsed_role, parsed_speaker = canonicalize_speaker_name(speaker)
        if parsed_speaker and parsed_role:
            role = role or parsed_role
            speaker = parsed_speaker

    return {
        "url": item["url"],
        "event_type": item["event_type"],
        "year": int(item["year"]),
        "date": meta.get("date") or item.get("date_hint"),
        "title": meta.get("title") or item.get("title"),
        "speaker": speaker,
        "role": role,
        "venue": meta.get("venue") or item.get("venue_hint"),
        "pdf_url": meta.get("pdf_url"),
        "body_text": body_text,
        "body_hash": body_hash(body_text),
        "word_count": score.word_count,
        "source_quality": score.source_quality,
        "scraped_at": now,
        "first_seen_at": now,
        "last_checked_at": now,
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
        "driving_snippets_json": json.dumps(score.driving_snippets, ensure_ascii=False),
    }


def upsert_document(conn: sqlite3.Connection, doc: Dict[str, Any]) -> None:
    init_db()

    old_first_seen = None
    old_body_hash = None
    cur = conn.cursor()
    cur.execute("SELECT first_seen_at, body_hash FROM documents WHERE url = ?", (doc["url"],))
    old = cur.fetchone()
    if old:
        old_first_seen, old_body_hash = old[0], old[1]

    if old_first_seen:
        doc["first_seen_at"] = old_first_seen

    columns = list(DOCUMENT_COLUMNS.keys())
    values = [doc.get(col) for col in columns]
    placeholders = ", ".join(["?"] * len(columns))
    update_cols = [col for col in columns if col != "url"]
    update_sql = ", ".join([f"{col}=excluded.{col}" for col in update_cols])

    cur.execute(
        f"""
        INSERT INTO documents ({", ".join(columns)})
        VALUES ({placeholders})
        ON CONFLICT(url) DO UPDATE SET {update_sql}
        """,
        values,
    )

    conn.commit()

    if old_body_hash and old_body_hash != doc.get("body_hash"):
        log_event("INFO", f"Body text changed and was rescored: {doc.get('url')}")


def get_existing_hashes(conn: sqlite3.Connection) -> Dict[str, Optional[str]]:
    init_db()
    cur = conn.cursor()
    cur.execute("SELECT url, body_hash FROM documents")
    return {row[0]: row[1] for row in cur.fetchall()}


@st.cache_data(show_spinner=False, ttl=300)
def load_documents() -> pd.DataFrame:
    init_db()
    conn = db_connection()
    df = pd.read_sql_query("SELECT * FROM documents", conn)
    conn.close()

    if df.empty:
        return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    num_cols = [
        "word_count", "source_quality", "hawkish_score", "dovish_score", "net_score",
        "inflation_concern", "labor_concern", "growth_concern", "financial_stability",
        "balance_sheet", "uncertainty_risk", "policy_relevance", "live_signal_share",
    ]

    for col in num_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["speaker", "role", "title", "event_type", "body_text", "url", "driving_snippets_json"]:
        if col not in df.columns:
            df[col] = ""

    df["speaker"] = df["speaker"].fillna("Unknown speaker")
    df["role"] = df["role"].fillna("")
    df["title"] = df["title"].fillna("Untitled")
    df["event_type"] = df["event_type"].fillna("")
    df["body_text"] = df["body_text"].fillna("")
    df["url"] = df["url"].fillna("")

    return df.sort_values(["date", "title"], ascending=[True, True]).reset_index(drop=True)


def refresh_corpus(
    max_years: int = 6,
    force_refresh: bool = False,
    progress_bar=None,
    status_box=None,
) -> Dict[str, int]:
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
            status_box.info("Loading Fed speeches and testimony index.")
        index_soup = fetch_html(session, INDEX_URL)
        year_links = parse_year_links(index_soup)

        if not year_links:
            raise RuntimeError("No yearly speech or testimony links were parsed from the Fed index.")

        by_type: Dict[str, List[Tuple[str, str, int]]] = {"speech": [], "testimony": []}
        for event_type, url, year in year_links:
            by_type[event_type].append((event_type, url, year))

        selected_year_links: List[Tuple[str, str, int]] = []
        for event_type in ["speech", "testimony"]:
            selected_year_links.extend(by_type[event_type][:max_years])

        selected_year_links = sorted(selected_year_links, key=lambda x: (x[2], x[0]), reverse=True)
        stats["year_pages_found"] = len(selected_year_links)

        existing_hashes = get_existing_hashes(conn)
        all_items: List[Dict[str, Any]] = []

        for idx, (event_type, year_url, year) in enumerate(selected_year_links, start=1):
            try:
                if status_box is not None:
                    status_box.info(f"Loading {event_type} index for {year} ({idx}/{len(selected_year_links)})")
                year_soup = fetch_html(session, year_url)
                items = parse_index_items(year_soup, event_type, year)
                all_items.extend(items)
                log_event("INFO", f"Parsed {len(items)} items from {year_url}")
                time.sleep(0.08)
            except Exception as exc:
                log_event("ERROR", f"Failed year index {year_url}: {repr(exc)}")

        dedup = {}
        for item in all_items:
            dedup[item["url"]] = item

        all_items = sorted(dedup.values(), key=lambda x: (x["year"], x["url"]), reverse=True)
        stats["index_items_found"] = len(all_items)

        if not all_items:
            raise RuntimeError("Parsed zero speech or testimony documents from the selected year pages.")

        for i, item in enumerate(all_items, start=1):
            stats["attempted_docs"] += 1

            if progress_bar is not None:
                progress_bar.progress(i / max(len(all_items), 1))

            if status_box is not None:
                status_box.info(f"Processing {i:,}/{len(all_items):,}: {item['title'][:100]}")

            if item["url"] in existing_hashes and not force_refresh:
                stats["skipped_existing"] += 1
                continue

            try:
                doc = parse_document(session, item)
                wc = int(doc.get("word_count") or 0)
                quality = float(doc.get("source_quality") or 0.0)

                if not doc.get("body_text") or wc < 35 or quality < 0.05:
                    stats["failed_docs"] += 1
                    log_event(
                        "WARN",
                        f"Skipped weak body: {item['url']} | words={wc} | quality={quality:.2f}",
                    )
                    continue

                upsert_document(conn, doc)
                stats["inserted_docs"] += 1

                log_event(
                    "INFO",
                    f"Stored {doc.get('date')} | {doc.get('speaker')} | {doc.get('title')} | "
                    f"words={wc} | relevance={float(doc.get('policy_relevance') or 0.0):.2f} | "
                    f"quality={quality:.2f}"
                )

                time.sleep(0.04)

            except Exception as exc:
                stats["failed_docs"] += 1
                log_event("ERROR", f"Failed document {item['url']}: {repr(exc)}")

    finally:
        conn.close()
        load_documents.clear()

    return stats


def safe_mean_std(series: pd.Series) -> Tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0, 1.0
    mean = float(s.mean())
    std = float(s.std(ddof=0))
    if std <= 1e-12:
        std = 1.0
    return mean, std


def bounded_z(value: float, mean: float, std: float, limit: float = 3.0) -> float:
    if pd.isna(value):
        return np.nan
    return float(np.clip((float(value) - mean) / std, -limit, limit))


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


def policy_implication(row: pd.Series) -> str:
    tone = row.get("tone_z_fed", np.nan)
    inflation = row.get("inflation_z_fed", np.nan)
    labor = row.get("labor_z_fed", np.nan)
    growth = row.get("growth_z_fed", np.nan)
    relevance = row.get("policy_relevance", np.nan)

    if pd.isna(tone):
        return "Insufficient signal"
    if pd.notna(relevance) and relevance < 0.25:
        return "Low policy signal"

    inflation = 0.0 if pd.isna(inflation) else float(inflation)
    labor = 0.0 if pd.isna(labor) else float(labor)
    growth = 0.0 if pd.isna(growth) else float(growth)

    if tone >= 0.75 and inflation >= max(labor, growth):
        return "Cuts delayed, inflation reaction function dominant"
    if tone >= 0.75:
        return "Restrictive bias reinforced"
    if tone <= -0.75 and labor >= inflation:
        return "Labor put strengthening"
    if tone <= -0.75 and growth >= inflation:
        return "Growth risk pulling cuts forward"
    if tone <= -0.75:
        return "Easing bias rising"
    if abs(tone) < 0.35 and max(inflation, labor, growth) >= 1.0:
        return "Topic pressure high, stance still balanced"
    return "No decisive rate-path shift"


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    for col in [
        "net_score", "inflation_concern", "labor_concern", "growth_concern",
        "financial_stability", "balance_sheet", "uncertainty_risk", "policy_relevance",
        "source_quality", "live_signal_share",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    metrics = {
        "tone_z_fed": "net_score",
        "inflation_z_fed": "inflation_concern",
        "labor_z_fed": "labor_concern",
        "growth_z_fed": "growth_concern",
        "fs_z_fed": "financial_stability",
        "balance_sheet_z_fed": "balance_sheet",
        "uncertainty_z_fed": "uncertainty_risk",
    }

    for z_col, raw_col in metrics.items():
        mean, std = safe_mean_std(out[raw_col])
        out[z_col] = out[raw_col].apply(lambda x: bounded_z(x, mean, std))

    out["tone_z_speaker"] = np.nan
    out["inflation_z_speaker"] = np.nan
    out["labor_z_speaker"] = np.nan
    out["growth_z_speaker"] = np.nan

    for speaker, idx in out.groupby("speaker", dropna=False).groups.items():
        subset = out.loc[idx]
        for z_col, raw_col in [
            ("tone_z_speaker", "net_score"),
            ("inflation_z_speaker", "inflation_concern"),
            ("labor_z_speaker", "labor_concern"),
            ("growth_z_speaker", "growth_concern"),
        ]:
            m, s = safe_mean_std(subset[raw_col])
            out.loc[idx, z_col] = subset[raw_col].apply(lambda x: bounded_z(x, m, s))

    out["speaker_weight"] = [
        speaker_weight(speaker, role)
        for speaker, role in zip(out["speaker"], out["role"])
    ]

    out["tone_percentile_fed"] = out["net_score"].apply(lambda x: percentile_rank(out["net_score"], x))
    out["inflation_percentile_fed"] = out["inflation_concern"].apply(lambda x: percentile_rank(out["inflation_concern"], x))
    out["labor_percentile_fed"] = out["labor_concern"].apply(lambda x: percentile_rank(out["labor_concern"], x))
    out["growth_percentile_fed"] = out["growth_concern"].apply(lambda x: percentile_rank(out["growth_concern"], x))

    out["stance"] = out["tone_z_fed"].apply(tone_bucket)
    out["policy_bucket"] = out["policy_relevance"].apply(policy_bucket)
    out["policy_implication"] = out.apply(policy_implication, axis=1)

    return out


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_market_history(start_date: str, end_date: str, tickers: Tuple[str, ...]) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()

    try:
        raw = yf.download(
            list(tickers),
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    close = None
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            close = raw["Adj Close"].copy()
        elif "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
    else:
        if "Adj Close" in raw.columns:
            close = raw[["Adj Close"]].copy()
            close.columns = [tickers[0]]
        elif "Close" in raw.columns:
            close = raw[["Close"]].copy()
            close.columns = [tickers[0]]

    if close is None or close.empty:
        return pd.DataFrame()

    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index()
    return close


def nearest_market_row(close: pd.DataFrame, date: pd.Timestamp, side: str) -> Optional[pd.Timestamp]:
    if close.empty or pd.isna(date):
        return None

    date = pd.Timestamp(date).normalize()
    idx = close.index

    if side == "on_or_before":
        eligible = idx[idx <= date]
        return eligible[-1] if len(eligible) else None

    if side == "after":
        eligible = idx[idx > date]
        return eligible[0] if len(eligible) else None

    return None


def compute_market_reactions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or yf is None:
        return pd.DataFrame()

    dated = df.dropna(subset=["date"]).copy()
    if dated.empty:
        return pd.DataFrame()

    start = (dated["date"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = (dated["date"].max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    tickers = tuple(MARKET_ASSETS.keys())
    close = load_market_history(start, end, tickers)

    if close.empty:
        return pd.DataFrame()

    rows = []
    for _, doc in dated.iterrows():
        event_day = nearest_market_row(close, doc["date"], "on_or_before")
        next_day = nearest_market_row(close, doc["date"], "after")
        if event_day is None:
            continue

        prev_candidates = close.index[close.index < event_day]
        prev_day = prev_candidates[-1] if len(prev_candidates) else None
        if prev_day is None:
            continue

        for ticker, meta in MARKET_ASSETS.items():
            if ticker not in close.columns:
                continue

            prev_px = close.loc[prev_day, ticker]
            event_px = close.loc[event_day, ticker]
            next_px = close.loc[next_day, ticker] if next_day is not None else np.nan

            if pd.isna(prev_px) or pd.isna(event_px) or float(prev_px) == 0:
                continue

            event_ret = (float(event_px) / float(prev_px) - 1.0) * 100.0

            if pd.isna(next_px) or float(event_px) == 0:
                next_ret = np.nan
                cumulative_ret = event_ret
            else:
                next_ret = (float(next_px) / float(event_px) - 1.0) * 100.0
                cumulative_ret = (float(next_px) / float(prev_px) - 1.0) * 100.0

            rows.append(
                {
                    "url": doc["url"],
                    "date": doc["date"],
                    "ticker": ticker,
                    "asset": meta["label"],
                    "event_day": event_day,
                    "next_day": next_day,
                    "event_return_pct": event_ret,
                    "next_return_pct": next_ret,
                    "two_day_return_pct": cumulative_ret,
                    "expected_if_hawkish": meta["expected_if_hawkish"],
                }
            )

    return pd.DataFrame(rows)


def attach_market_confirmation(df: pd.DataFrame, reactions: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["market_confirmation"] = np.nan
    out["market_confirmed_tone_z"] = out["tone_z_fed"]

    if reactions.empty:
        return out

    grouped = reactions.groupby("url")
    values = {}

    for url, sub in grouped:
        row = out[out["url"] == url]
        if row.empty:
            continue

        tone = float(row.iloc[0].get("tone_z_fed", np.nan))
        if pd.isna(tone) or abs(tone) < 0.35:
            values[url] = 0.0
            continue

        directional = np.sign(tone)
        components = []

        for _, r in sub.iterrows():
            ret = r.get("event_return_pct", np.nan)
            expected_if_hawkish = r.get("expected_if_hawkish", np.nan)
            if pd.isna(ret) or pd.isna(expected_if_hawkish):
                continue
            components.append(directional * float(expected_if_hawkish) * float(ret))

        if not components:
            values[url] = np.nan
        else:
            raw = float(np.nanmean(components))
            values[url] = float(np.clip(raw / 0.60, -1.0, 1.0))

    out["market_confirmation"] = out["url"].map(values)
    adjustment = out["market_confirmation"].fillna(0.0) * 0.45
    out["market_confirmed_tone_z"] = (out["tone_z_fed"] + adjustment).clip(-3.0, 3.0)

    return out


def aggregate_series(df: pd.DataFrame, freq: str = "30D") -> pd.DataFrame:
    if df.empty:
        return df

    frame = df.dropna(subset=["date"]).copy()
    if frame.empty:
        return pd.DataFrame()

    if freq == "30D":
        min_date = frame["date"].min().normalize()
        frame["bucket"] = min_date + (((frame["date"] - min_date).dt.days // 30) * pd.Timedelta(days=30))
    else:
        frame["bucket"] = frame["date"].dt.to_period(freq).dt.start_time

    def weighted_avg(x: pd.DataFrame, col: str) -> float:
        vals = pd.to_numeric(x[col], errors="coerce")
        weights = pd.to_numeric(x["speaker_weight"], errors="coerce").fillna(1.0)
        mask = vals.notna()
        if not mask.any():
            return np.nan
        return float(np.average(vals[mask], weights=weights[mask]))

    rows = []
    for bucket, x in frame.groupby("bucket"):
        rows.append(
            {
                "date": bucket,
                "tone_z_fed": weighted_avg(x, "tone_z_fed"),
                "market_confirmed_tone_z": weighted_avg(x, "market_confirmed_tone_z")
                if "market_confirmed_tone_z" in x.columns else np.nan,
                "inflation_z_fed": weighted_avg(x, "inflation_z_fed"),
                "labor_z_fed": weighted_avg(x, "labor_z_fed"),
                "growth_z_fed": weighted_avg(x, "growth_z_fed"),
                "fs_z_fed": weighted_avg(x, "fs_z_fed"),
                "uncertainty_z_fed": weighted_avg(x, "uncertainty_z_fed"),
                "policy_relevance": weighted_avg(x, "policy_relevance"),
                "source_quality": weighted_avg(x, "source_quality"),
                "documents": int(len(x)),
            }
        )

    grouped = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    for col in ["tone_z_fed", "market_confirmed_tone_z", "inflation_z_fed", "labor_z_fed", "growth_z_fed"]:
        if col in grouped.columns:
            grouped[f"{col}_smooth"] = grouped[col].ewm(span=4, adjust=False).mean()

    return grouped


def speaker_matrix(df: pd.DataFrame, top_n: int = 14) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    agg = (
        df.groupby(["speaker", "role"], dropna=False)
        .agg(
            docs=("url", "count"),
            tone_z_fed=("tone_z_fed", "mean"),
            tone_z_speaker=("tone_z_speaker", "mean"),
            inflation_z_fed=("inflation_z_fed", "mean"),
            labor_z_fed=("labor_z_fed", "mean"),
            policy_relevance=("policy_relevance", "mean"),
            source_quality=("source_quality", "mean"),
            last_date=("date", "max"),
            weight=("speaker_weight", "mean"),
        )
        .reset_index()
    )

    latest = (
        df.sort_values("date")
        .groupby("speaker", dropna=False)
        .tail(1)[["speaker", "tone_z_speaker", "tone_z_fed", "stance", "policy_implication"]]
        .rename(columns={"tone_z_speaker": "latest_vs_own", "tone_z_fed": "latest_vs_fed"})
    )

    agg = agg.merge(latest, on="speaker", how="left")
    agg["avg_stance"] = agg["tone_z_fed"].apply(tone_bucket)
    agg = agg.sort_values(["weight", "last_date", "docs"], ascending=[False, False, False]).head(top_n)
    return agg


def format_z(x: float) -> str:
    if pd.isna(x):
        return "n.a."
    return f"{float(x):+.2f}σ"


def format_pctile(x: float) -> str:
    if pd.isna(x):
        return "n.a."
    return f"{float(x):.0f}th %ile"


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "n.a."
    return f"{float(x):+.2f}%"


def stance_color(z: float) -> Tuple[str, str]:
    if pd.isna(z):
        return "#f0f1f3", "#4f5661"
    if z <= -0.35:
        return "#dcf7e6", "#166534"
    if z >= 0.35:
        return "#fde2e2", "#991b1b"
    return "#eef0f2", "#4b5563"


def make_badge(label: str, z: float) -> str:
    bg, fg = stance_color(z)
    return (
        f"<span style='display:inline-block;padding:0.28rem 0.62rem;border-radius:999px;"
        f"background:{bg};color:{fg};font-weight:650;font-size:0.88rem'>{html.escape(label)}</span>"
    )


def render_about() -> None:
    st.sidebar.markdown("## About This Tool")
    st.sidebar.markdown(
        """
        Primary-source Fed communication monitor.

        It scrapes official Federal Reserve speeches and testimony, scores live policy language, normalizes by speaker and Fed baseline, and optionally checks whether the market confirmed the communication shock through rates-sensitive assets.

        The useful output is the rate-path implication, not the raw hawkish/dovish label.
        """
    )


def render_diagnostics(log_df: pd.DataFrame, stats: Optional[Dict[str, int]] = None) -> None:
    st.subheader("Diagnostics")

    if stats:
        st.write(
            f"Year pages: {stats.get('year_pages_found', 0):,} | "
            f"Index items: {stats.get('index_items_found', 0):,} | "
            f"Attempted: {stats.get('attempted_docs', 0):,} | "
            f"Inserted: {stats.get('inserted_docs', 0):,} | "
            f"Skipped: {stats.get('skipped_existing', 0):,} | "
            f"Failed: {stats.get('failed_docs', 0):,}"
        )

    if log_df.empty:
        st.info("No scrape log yet.")
    else:
        st.dataframe(log_df, use_container_width=True, hide_index=True, height=260)


def tone_chart(series: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if series.empty:
        return fig

    y_cols = [c for c in ["tone_z_fed_smooth", "market_confirmed_tone_z_smooth"] if c in series.columns]
    y_vals = pd.concat([pd.to_numeric(series[c], errors="coerce") for c in y_cols], axis=0).dropna()

    if y_vals.empty:
        y_low, y_high = -1.5, 1.5
    else:
        y_min = float(y_vals.min())
        y_max = float(y_vals.max())
        pad = max(0.25, (y_max - y_min) * 0.22)
        y_low = max(-3.0, np.floor((y_min - pad) / 0.25) * 0.25)
        y_high = min(3.0, np.ceil((y_max + pad) / 0.25) * 0.25)
        if y_high - y_low < 1.5:
            mid = (y_high + y_low) / 2
            y_low, y_high = mid - 0.75, mid + 0.75

    def band(y0: float, y1: float, color: str, opacity: float) -> None:
        low = max(y0, y_low)
        high = min(y1, y_high)
        if high > low:
            fig.add_hrect(y0=low, y1=high, fillcolor=color, opacity=opacity, line_width=0)

    band(-3, -1.25, PASTEL_GREEN, 0.13)
    band(-1.25, -0.35, PASTEL_GREEN, 0.07)
    band(-0.35, 0.35, PASTEL_GREY, 0.06)
    band(0.35, 1.25, PASTEL_RED, 0.07)
    band(1.25, 3, PASTEL_RED, 0.13)

    fig.add_trace(
        go.Scatter(
            x=series["date"],
            y=series["tone_z_fed_smooth"],
            mode="lines+markers",
            name="Fed language tone",
            line=dict(width=2.4, color=SOFT_BLUE),
            marker=dict(size=6),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}σ<extra></extra>",
        )
    )

    if "market_confirmed_tone_z_smooth" in series.columns and series["market_confirmed_tone_z_smooth"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=series["date"],
                y=series["market_confirmed_tone_z_smooth"],
                mode="lines",
                name="Market-confirmed tone",
                line=dict(width=2.4, color=SOFT_PURPLE, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}σ<extra></extra>",
            )
        )

    fig.add_hline(y=0, line_width=1, line_color="#999999")
    fig.add_hline(y=0.35, line_width=1, line_dash="dot", line_color=PASTEL_RED)
    fig.add_hline(y=-0.35, line_width=1, line_dash="dot", line_color=PASTEL_GREEN)
    fig.add_hline(y=1.25, line_width=1, line_dash="dash", line_color=PASTEL_RED)
    fig.add_hline(y=-1.25, line_width=1, line_dash="dash", line_color=PASTEL_GREEN)

    fig.update_layout(
        height=470,
        margin=dict(l=20, r=20, t=64, b=28),
        title=dict(text="Fed communication tone", x=0.01, xanchor="left"),
        xaxis=dict(showgrid=True, tickformat="%Y", dtick="M12", automargin=True),
        yaxis=dict(range=[y_low, y_high], title="Z-score vs Fed baseline", zeroline=True, automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0.0),
    )

    return fig


def topic_chart(series: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if series.empty:
        return fig

    traces = [
        ("inflation_z_fed_smooth", "Inflation pressure", PASTEL_RED),
        ("labor_z_fed_smooth", "Labor pressure", PASTEL_GREEN),
        ("growth_z_fed_smooth", "Growth pressure", SOFT_BLUE),
    ]

    for col, label, color in traces:
        if col in series.columns and series[col].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=series["date"],
                    y=series[col],
                    mode="lines",
                    name=label,
                    line=dict(width=2.2, color=color),
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}σ<extra></extra>",
                )
            )

    fig.add_hline(y=0, line_width=1, line_color="#999999")
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=56, b=25),
        title=dict(text="Policy topic pressure", x=0.01, xanchor="left"),
        xaxis=dict(showgrid=True, tickformat="%Y", dtick="M12"),
        yaxis=dict(title="Z-score vs Fed baseline", range=[-3, 3]),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0.0),
    )
    return fig


def speaker_matrix_chart(matrix: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if matrix.empty:
        return fig

    matrix = matrix.sort_values("latest_vs_fed", ascending=True).copy()

    colors = []
    for z in matrix["latest_vs_own"]:
        if pd.isna(z):
            colors.append(PASTEL_GREY)
        elif z >= 0.35:
            colors.append(PASTEL_RED)
        elif z <= -0.35:
            colors.append(PASTEL_GREEN)
        else:
            colors.append(PASTEL_GREY)

    hover = []
    for _, row in matrix.iterrows():
        hover.append(
            f"{html.escape(str(row['speaker']))}<br>"
            f"Role: {html.escape(str(row.get('role', '')))}<br>"
            f"Latest vs Fed: {format_z(row.get('latest_vs_fed'))}<br>"
            f"Latest vs own: {format_z(row.get('latest_vs_own'))}<br>"
            f"Docs: {int(row.get('docs', 0))}<br>"
            f"Policy implication: {html.escape(str(row.get('policy_implication', '')))}"
        )

    fig.add_trace(
        go.Scatter(
            x=matrix["latest_vs_fed"],
            y=matrix["speaker"],
            mode="markers",
            marker=dict(
                size=np.clip(matrix["docs"].astype(float) * 3.0 + 8.0, 10, 34),
                color=colors,
                line=dict(width=1, color="white"),
            ),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            name="Speakers",
        )
    )

    fig.add_vline(x=0, line_width=1, line_color="#999999")
    fig.add_vline(x=0.35, line_width=1, line_dash="dot", line_color=PASTEL_RED)
    fig.add_vline(x=-0.35, line_width=1, line_dash="dot", line_color=PASTEL_GREEN)
    fig.update_layout(
        height=460,
        margin=dict(l=20, r=20, t=56, b=35),
        title=dict(text="Speaker matrix", x=0.01, xanchor="left"),
        xaxis=dict(title="Latest speech vs Fed baseline", range=[-3, 3]),
        yaxis=dict(title=""),
        showlegend=False,
    )

    return fig


def scorecard_dataframe(doc: pd.Series) -> pd.DataFrame:
    rows = [
        ("Overall tone", doc.get("tone_z_fed"), doc.get("tone_percentile_fed"), tone_bucket(doc.get("tone_z_fed"))),
        ("Vs own history", doc.get("tone_z_speaker"), np.nan, tone_bucket(doc.get("tone_z_speaker"))),
        ("Market-confirmed tone", doc.get("market_confirmed_tone_z"), np.nan, tone_bucket(doc.get("market_confirmed_tone_z"))),
        ("Policy relevance", (doc.get("policy_relevance") * 2.0 - 1.0) if pd.notna(doc.get("policy_relevance")) else np.nan, np.nan, policy_bucket(doc.get("policy_relevance"))),
        ("Source quality", (doc.get("source_quality") * 2.0 - 1.0) if pd.notna(doc.get("source_quality")) else np.nan, np.nan, emphasis_bucket((doc.get("source_quality") * 2.0 - 1.0) if pd.notna(doc.get("source_quality")) else np.nan)),
        ("Inflation salience", doc.get("inflation_z_fed"), doc.get("inflation_percentile_fed"), emphasis_bucket(doc.get("inflation_z_fed"))),
        ("Labor salience", doc.get("labor_z_fed"), doc.get("labor_percentile_fed"), emphasis_bucket(doc.get("labor_z_fed"))),
        ("Growth salience", doc.get("growth_z_fed"), doc.get("growth_percentile_fed"), emphasis_bucket(doc.get("growth_z_fed"))),
        ("Financial stability", doc.get("fs_z_fed"), np.nan, emphasis_bucket(doc.get("fs_z_fed"))),
        ("Uncertainty", doc.get("uncertainty_z_fed"), np.nan, emphasis_bucket(doc.get("uncertainty_z_fed"))),
    ]
    return pd.DataFrame(rows, columns=["dimension", "z", "percentile", "bucket"])


def scorecard_chart(scorecard: pd.DataFrame) -> go.Figure:
    plot_df = scorecard.copy()
    plot_df["z"] = pd.to_numeric(plot_df["z"], errors="coerce")
    plot_df = plot_df.sort_values("z", ascending=True)

    colors = []
    for _, row in plot_df.iterrows():
        z = row["z"]
        dim = row["dimension"]
        if pd.isna(z):
            colors.append(PASTEL_GREY)
        elif dim in {"Policy relevance", "Source quality"}:
            colors.append(SOFT_PURPLE)
        elif z >= 0.35:
            colors.append(PASTEL_RED)
        elif z <= -0.35:
            colors.append(PASTEL_GREEN)
        else:
            colors.append(PASTEL_GREY)

    fig = go.Figure(
        go.Bar(
            x=plot_df["z"],
            y=plot_df["dimension"],
            orientation="h",
            marker_color=colors,
            text=[format_z(z) if pd.notna(z) else "n.a." for z in plot_df["z"]],
            textposition="outside",
            hovertemplate="%{y}<br>%{x:.2f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="#999999")
    fig.add_vline(x=-0.35, line_dash="dot", line_width=1, line_color=PASTEL_GREEN)
    fig.add_vline(x=0.35, line_dash="dot", line_width=1, line_color=PASTEL_RED)
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=35, t=48, b=24),
        title=dict(text="Selected document scorecard", x=0.01, xanchor="left"),
        xaxis=dict(range=[-3, 3], title="Relative score"),
        yaxis=dict(title=""),
    )
    return fig


def latest_table(view: pd.DataFrame) -> pd.DataFrame:
    latest = view.sort_values("date", ascending=False).copy()
    latest["date"] = latest["date"].dt.strftime("%Y-%m-%d")
    latest["tone"] = latest["tone_z_fed"].apply(format_z)
    latest["market tone"] = latest["market_confirmed_tone_z"].apply(format_z) if "market_confirmed_tone_z" in latest.columns else "n.a."
    latest["vs own"] = latest["tone_z_speaker"].apply(format_z)
    latest["fed %ile"] = latest["tone_percentile_fed"].apply(format_pctile)
    latest["policy relevance"] = latest["policy_relevance"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "n.a.")
    latest["quality"] = latest["source_quality"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "n.a.")
    latest["source"] = latest["url"]

    cols = [
        "date", "speaker", "event_type", "stance", "tone", "market tone",
        "vs own", "fed %ile", "policy relevance", "quality",
        "policy_implication", "title", "source",
    ]
    available = [c for c in cols if c in latest.columns]
    return latest[available].rename(columns={"event_type": "type"})


def parse_driving_snippets(doc: pd.Series) -> List[Dict[str, Any]]:
    raw = doc.get("driving_snippets_json", "")
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def highlight_policy_terms(text: str) -> str:
    phrases = list(HAWKISH_PHRASES.keys()) + list(DOVISH_PHRASES.keys())
    escaped = html.escape(clean_text(text))

    for phrase in sorted(set(phrases), key=len, reverse=True):
        if len(phrase) < 4:
            continue
        pattern = re.compile(re.escape(html.escape(phrase)), re.IGNORECASE)
        escaped = pattern.sub(
            lambda m: f"<mark style='background-color:#fff2a8'>{m.group(0)}</mark>",
            escaped,
        )

    return escaped


def doc_header_html(doc: pd.Series) -> str:
    tone = doc.get("tone_z_fed", np.nan)
    own = doc.get("tone_z_speaker", np.nan)
    market_tone = doc.get("market_confirmed_tone_z", np.nan)

    tone_badge = make_badge(tone_bucket(tone), tone)
    own_badge = make_badge(f"Vs own: {tone_bucket(own)}", own)
    market_badge = make_badge(f"Market: {tone_bucket(market_tone)}", market_tone)
    inflation_badge = make_badge(f"Inflation: {emphasis_bucket(doc.get('inflation_z_fed'))}", doc.get("inflation_z_fed"))
    labor_badge = make_badge(f"Labor: {emphasis_bucket(doc.get('labor_z_fed'))}", doc.get("labor_z_fed"))
    growth_badge = make_badge(f"Growth: {emphasis_bucket(doc.get('growth_z_fed'))}", doc.get("growth_z_fed"))

    date_text = "Unknown date"
    if pd.notna(doc.get("date")):
        date_text = pd.Timestamp(doc.get("date")).strftime("%Y-%m-%d")

    policy_rel = doc.get("policy_relevance", np.nan)
    live_share = doc.get("live_signal_share", np.nan)
    quality = doc.get("source_quality", np.nan)

    return f"""
    <div style="padding:1rem 1.05rem;border:1px solid #e7e9ee;border-radius:0.85rem;background:#fbfbfc">
        <div style="font-size:1.08rem;font-weight:750;margin-bottom:0.40rem">
            {html.escape(str(doc.get("title", "Untitled")))}
        </div>
        <div style="margin-bottom:0.60rem;color:#5f6b7a">
            {html.escape(date_text)} | {html.escape(str(doc.get("speaker", "Unknown speaker")))} |
            {html.escape(str(doc.get("role", "")))} | {html.escape(str(doc.get("event_type", "")).title())}
        </div>
        <div style="display:flex;gap:0.42rem;flex-wrap:wrap;margin-bottom:0.70rem">
            {tone_badge}{own_badge}{market_badge}{inflation_badge}{labor_badge}{growth_badge}
        </div>
        <div style="font-size:0.92rem;color:#5f6b7a;line-height:1.55">
            Rate-path read: <b>{html.escape(str(doc.get("policy_implication", "n.a.")))}</b><br>
            Fed baseline: {format_z(tone)} | vs own history: {format_z(own)} |
            Fed percentile: {format_pctile(doc.get("tone_percentile_fed"))} |
            Policy relevance: {policy_rel:.2f} | Live-signal share: {live_share:.0%} | Source quality: {quality:.2f}
        </div>
    </div>
    """


def render_market_reaction_for_doc(doc: pd.Series, reactions: pd.DataFrame) -> None:
    if reactions.empty:
        st.info("Market confirmation is unavailable. Install yfinance or enable market reaction in the sidebar.")
        return

    sub = reactions[reactions["url"] == doc["url"]].copy()
    if sub.empty:
        st.info("No market reaction data found for this document date.")
        return

    sub = sub.sort_values("asset")
    display = sub[
        ["asset", "event_day", "event_return_pct", "next_return_pct", "two_day_return_pct"]
    ].copy()

    display["event_day"] = pd.to_datetime(display["event_day"]).dt.strftime("%Y-%m-%d")
    for col in ["event_return_pct", "next_return_pct", "two_day_return_pct"]:
        display[col] = display[col].apply(format_pct)

    display = display.rename(
        columns={
            "asset": "Asset",
            "event_day": "Market day",
            "event_return_pct": "Event-day move",
            "next_return_pct": "Next-day move",
            "two_day_return_pct": "Two-day move",
        }
    )

    st.dataframe(display, use_container_width=True, hide_index=True)


def render_snippets(doc: pd.Series) -> None:
    snippets = parse_driving_snippets(doc)

    if not snippets:
        body = clean_text(doc.get("body_text", ""))
        fallback = split_sentences(body)[:5]
        snippets = [{"sentence": s, "relevance": np.nan, "net": np.nan, "hawk": np.nan, "dove": np.nan} for s in fallback]

    for i, item in enumerate(snippets[:6], start=1):
        sentence = item.get("sentence", "")
        rel = item.get("relevance", np.nan)
        net = item.get("net", np.nan)
        st.markdown(
            f"""
            <div style="padding:0.75rem 0.85rem;border:1px solid #edf0f4;border-radius:0.65rem;margin-bottom:0.45rem;background:white">
                <div style="font-size:0.86rem;color:#6b7280;margin-bottom:0.30rem">
                    Signal sentence {i} | relevance {rel if pd.notna(rel) else "n.a."} | net {net if pd.notna(net) else "n.a."}
                </div>
                <div style="font-size:0.96rem;line-height:1.52">{highlight_policy_terms(sentence)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Official Fed speeches and testimony, scored for policy signal, speaker shift, and market confirmation.")

    init_db()
    render_about()

    with st.sidebar:
        st.subheader("Controls")
        years_to_pull = st.slider("Years to scan from latest backward", min_value=1, max_value=20, value=6)
        force_refresh = st.checkbox("Force refresh existing documents", value=False)
        rebuild_db = st.checkbox("Rebuild local corpus before refresh", value=False)
        include_market = st.checkbox("Add market confirmation layer", value=True)
        refresh_clicked = st.button("Refresh corpus", use_container_width=True)

    refresh_stats = None

    if refresh_clicked:
        if rebuild_db:
            clear_documents()
        progress_bar = st.sidebar.progress(0)
        status_box = st.sidebar.empty()
        with st.spinner("Refreshing Fed corpus."):
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

    raw_df = load_documents()
    log_df = load_log()

    if refresh_stats is not None:
        st.caption(
            f"Refresh complete. Inserted {refresh_stats['inserted_docs']:,} documents out of "
            f"{refresh_stats['attempted_docs']:,} attempted."
        )

    if raw_df.empty:
        st.warning("No local corpus found yet. Use Refresh corpus in the sidebar.")
        render_diagnostics(log_df, refresh_stats)
        st.stop()

    df = compute_features(raw_df)

    market_reactions = pd.DataFrame()
    if include_market:
        with st.spinner("Loading market confirmation layer."):
            market_reactions = compute_market_reactions(df)
        df = attach_market_confirmation(df, market_reactions)
    else:
        df["market_confirmation"] = np.nan
        df["market_confirmed_tone_z"] = df["tone_z_fed"]

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

        min_policy = st.slider("Minimum policy relevance", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        min_quality = st.slider("Minimum source quality", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        search = st.text_input("Search title or transcript")

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
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

    if min_quality > 0:
        view = view[view["source_quality"].fillna(0.0) >= min_quality]

    if search.strip():
        q = search.strip().lower()
        view = view[
            view["title"].fillna("").str.lower().str.contains(q, na=False)
            | view["speaker"].fillna("").str.lower().str.contains(q, na=False)
            | view["body_text"].fillna("").str.lower().str.contains(q, na=False)
        ]

    if view.empty:
        st.warning("No documents match the current filters.")
        render_diagnostics(log_df, refresh_stats)
        st.stop()

    st.caption(
        f"Loaded {len(df):,} documents. Current view: {len(view):,}. "
        f"Latest document date: {max_date_ts.strftime('%Y-%m-%d')}."
    )

    series = aggregate_series(view, freq="30D")
    matrix = speaker_matrix(view, top_n=14)

    left, right = st.columns([1.35, 1.0])
    with left:
        st.plotly_chart(tone_chart(series), use_container_width=True)
    with right:
        st.plotly_chart(speaker_matrix_chart(matrix), use_container_width=True)

    st.plotly_chart(topic_chart(series), use_container_width=True)

    st.subheader("Latest communications")
    st.dataframe(
        latest_table(view).head(100),
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config={
            "source": st.column_config.LinkColumn("source"),
        },
    )

    st.subheader("Selected communication underwrite")
    select_df = view.sort_values("date", ascending=False).copy()
    select_df["select_label"] = (
        select_df["date"].dt.strftime("%Y-%m-%d")
        + " | "
        + select_df["speaker"].fillna("Unknown speaker")
        + " | "
        + select_df["title"].fillna("Untitled").str.slice(0, 120)
    )

    selected_label = st.selectbox(
        "Select a speech or testimony",
        options=select_df["select_label"].tolist(),
        index=0,
    )

    selected_doc = select_df[select_df["select_label"] == selected_label].iloc[0]

    st.markdown(doc_header_html(selected_doc), unsafe_allow_html=True)

    c1, c2 = st.columns([0.95, 1.05])

    with c1:
        scorecard = scorecard_dataframe(selected_doc)
        st.plotly_chart(scorecard_chart(scorecard), use_container_width=True)

    with c2:
        st.markdown("#### Market reaction")
        render_market_reaction_for_doc(selected_doc, market_reactions)

    st.markdown("#### Sentences driving the score")
    render_snippets(selected_doc)

    with st.expander("Full extracted transcript"):
        body = selected_doc.get("body_text", "")
        st.text_area(
            "Transcript",
            value=body,
            height=420,
            label_visibility="collapsed",
        )

    with st.expander("Diagnostics"):
        render_diagnostics(log_df, refresh_stats)


if __name__ == "__main__":
    main()
