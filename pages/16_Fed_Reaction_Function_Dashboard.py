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

APP_TITLE = "RoW Central Bank Tone Underwriter"
DATA_DIR = Path("row_cb_tone_data")
DB_PATH = DATA_DIR / "row_cb_tone.sqlite"
REQUEST_TIMEOUT = 30
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
)

DATA_DIR.mkdir(parents=True, exist_ok=True)

BANK_CONFIGS: Dict[str, Dict[str, object]] = {
    "Bank of Canada": {
        "bank_code": "BOC",
        "country": "Canada",
        "index_url": "https://www.bankofcanada.ca/press/speeches/",
        "enabled": True,
        "speaker_weights": {
            "Tiff Macklem": 3.0,
            "Carolyn Rogers": 1.8,
            "Sharon Kozicki": 1.2,
            "Nicolas Vincent": 1.1,
            "Toni Gravelle": 1.1,
        },
    },
    "Bank of Japan": {
        "bank_code": "BOJ",
        "country": "Japan",
        "index_url": "https://www.boj.or.jp/en/announcements/press/",
        "enabled": True,
        "speaker_weights": {
            "UEDA Kazuo": 3.0,
            "HIMINO Ryozo": 1.8,
            "TAKATA Hajime": 1.2,
            "TAMURA Naoki": 1.2,
            "NAKAGAWA Junko": 1.1,
            "KOEDA Junko": 1.1,
        },
    },
    "Reserve Bank of Australia": {
        "bank_code": "RBA",
        "country": "Australia",
        "index_url": "https://www.rba.gov.au/speeches/",
        "enabled": True,
        "speaker_weights": {
            "Michele Bullock": 3.0,
            "Andrew Hauser": 1.8,
            "Sarah Hunter": 1.3,
            "Christopher Kent": 1.3,
            "Brad Jones": 1.1,
        },
    },
    "Norges Bank": {
        "bank_code": "NORGES",
        "country": "Norway",
        "index_url": "https://www.norges-bank.no/en/topics/news-events/speeches/",
        "enabled": True,
        "speaker_weights": {
            "Ida Wolden Bache": 3.0,
            "Pål Longva": 1.4,
            "Martin E. Nordhagen": 1.0,
            "Torbjørn Hægeland": 1.0,
        },
    },
}

BANK_WEIGHTS = {
    "Bank of Canada": 1.8,
    "Bank of Japan": 1.8,
    "Reserve Bank of Australia": 1.4,
    "Norges Bank": 1.0,
}

LEXICON = {
    "hawkish": [
        "higher for longer", "restrictive", "sufficiently restrictive", "upside risk",
        "inflation remains too high", "inflation is too high", "persistent inflation",
        "reaccelerat", "not yet done", "additional firming", "further tightening",
        "vigilant", "upward pressure on prices", "elevated inflation", "inflation pressure",
        "price stability", "tight labor market", "strong labour market", "strong labor market",
        "overheating", "policy restraint", "maintain restraint", "firming", "tightening",
        "hold rates higher", "risk of inflation", "unanchored inflation expectations",
        "still above target", "premature to ease", "not appropriate to cut",
        "more work to do", "upside risks to inflation", "inflation persistence",
        "restrictive stance", "need to remain", "longer than expected",
        "capacity pressures", "cost pressures", "wage pressures", "second-round effects",
        "upside risks remain", "inflationary pressures", "domestic inflation", "price pressures remain",
        "prepared to raise", "prepared to tighten", "monetary tightening", "underlying inflation remains high",
    ],
    "dovish": [
        "disinflation", "cooling labor market", "cooling labour market", "softening labor market",
        "softening labour market", "downside risk", "growth is slowing", "economic slowdown",
        "below-trend growth", "normalizing inflation", "further progress on inflation",
        "policy can respond", "room to ease", "easing", "rate cuts", "cut rates", "lower rates",
        "less restrictive", "downward path", "labour market is moderating", "labor market is moderating",
        "balanced risks", "two-sided risks", "weak demand", "headwinds", "slack",
        "unemployment is rising", "financial conditions tightened", "support the labor market",
        "support the labour market", "downside risks to employment", "can be patient",
        "can adjust policy", "could move lower", "normalisation", "normalize policy",
        "inflation has eased", "inflation is easing", "demand is softening", "growth has weakened",
        "slower activity", "underlying inflation has declined", "there is scope to reduce restraint",
    ],
    "inflation_concern": [
        "inflation", "prices", "price pressures", "services inflation", "core inflation",
        "underlying inflation", "trimmed mean inflation", "wage growth", "price stability",
        "inflation expectations", "sticky inflation", "import prices", "headline inflation",
    ],
    "labor_concern": [
        "labor market", "labour market", "employment", "unemployment", "job growth",
        "payroll", "wages", "hiring", "layoffs", "slack", "participation",
        "vacancies", "earnings", "household income",
    ],
    "growth_concern": [
        "growth", "activity", "demand", "consumer spending", "investment", "slowdown",
        "recession", "weakness", "output", "expansion", "exports", "productivity",
        "housing", "trade", "business investment",
    ],
    "financial_stability": [
        "financial stability", "banking", "banks", "stress", "liquidity", "funding",
        "market functioning", "credit conditions", "treasury market", "vulnerabilities",
        "private credit", "non-bank", "leverage", "macroprudential",
    ],
    "balance_sheet": [
        "balance sheet", "quantitative tightening", "qt", "runoff", "reserves",
        "securities holdings", "mbs", "bond purchases", "asset purchases",
        "yield curve control", "jgb purchases", "government bond purchases",
    ],
    "uncertainty_risk": [
        "uncertain", "uncertainty", "risk management", "careful", "monitor", "watching",
        "data dependent", "incoming data", "proceed carefully", "humble", "attentive",
        "uncertain outlook", "uncertain environment", "high uncertainty",
    ],
}

POLICY_RELEVANCE_TERMS = [
    "policy", "monetary policy", "committee", "policy board", "governing council", "board",
    "interest rate", "policy rate", "cash rate", "bank rate", "overnight rate", "target rate",
    "outlook", "forecast", "incoming data", "data dependent", "financial conditions",
    "today", "going forward", "at this time", "we expect", "we will", "we may", "appropriate",
    "balance of risks", "inflation remains", "economic outlook", "price stability",
    "restrictive", "easing", "tightening", "stance", "rate path", "demand and supply",
]

HISTORICAL_TERMS = [
    "in the early", "in the late", "history", "historical", "legacy", "years ago", "decades ago",
    "career", "served", "was appointed", "at that time", "in the 1980s", "in the 1970s",
]

CEREMONIAL_TERMS = [
    "award", "honor", "ceremony", "commencement", "tribute", "memorial", "anniversary",
    "public service", "integrity", "congratulations", "pleasure to be here", "thank you for inviting me",
    "i would like to begin by acknowledging", "traditional custodians", "elders past and present",
]

FORWARD_LOOKING_TERMS = [
    "will", "may", "could", "expect", "outlook", "going forward", "at this time",
    "in coming months", "we remain", "we are prepared", "if needed", "from here", "currently",
]

SKIP_PATTERNS = [
    "share this page", "share on x", "share on linkedin", "share on facebook",
    "for media enquiries", "for media inquiries", "watch video", "audio", "webcast",
    "download", "hansard transcript", "pdf", "return to text", "back to top",
    "skip to content", "subscribe", "copyright", "cookie", "privacy", "accessibility",
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
            bank_name TEXT,
            bank_code TEXT,
            country TEXT,
            event_type TEXT,
            date TEXT,
            title TEXT,
            speaker TEXT,
            role TEXT,
            venue TEXT,
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
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_phrase(phrase: str) -> str:
    return re.escape(phrase.lower())


def count_phrase_hits(text: str, phrases: List[str]) -> int:
    low = text.lower()
    hits = 0
    for phrase in phrases:
        hits += len(re.findall(normalize_phrase(phrase), low))
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

    if "policy rate" in low or "interest rate" in low or "cash rate" in low or "bank rate" in low:
        score += 0.35
    if "today" in low and ("inflation" in low or "policy" in low or "outlook" in low):
        score += 0.20

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

    net = 1.15 * hawk - 1.15 * dove + 0.05 * infl - 0.03 * labor - 0.03 * growth - 0.02 * fs

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

    damp = 0.35 + 0.65 * policy_relevance
    net = squash_score(net * damp, scale=2.2, max_abs=2.1)
    hawk = squash_score(hawk * damp, scale=2.0, max_abs=2.3)
    dove = squash_score(dove * damp, scale=2.0, max_abs=2.3)

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


def strip_known_junk(text: str) -> str:
    t = clean_text(text)
    low = t.lower()
    for pat in SKIP_PATTERNS:
        low = t.lower()
        if pat in low:
            idx = low.find(pat)
            t = t[:idx]
    return clean_text(t)


def extract_body_text_generic(soup: BeautifulSoup) -> str:
    paragraphs = []
    for p in soup.find_all(["p", "li"]):
        txt = clean_text(p.get_text(" ", strip=True))
        if not txt:
            continue
        low = txt.lower()
        if any(bad in low for bad in SKIP_PATTERNS):
            continue
        if len(txt.split()) < 6:
            continue
        paragraphs.append(txt)

    body = "\n\n".join(paragraphs)
    body = strip_known_junk(body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body


def parse_date_text(raw: str) -> Optional[str]:
    raw = clean_text(raw)
    if not raw:
        return None
    try:
        dt = pd.to_datetime(raw, errors="raise")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def detect_event_type(text: str) -> str:
    low = (text or "").lower()
    if "testimony" in low or "appearance" in low or "parliament" in low or "committee" in low:
        return "testimony"
    if "panel" in low:
        return "panel"
    if "press conference" in low or "media conference" in low:
        return "press_conference"
    if "statement" in low or "opening statement" in low:
        return "statement"
    return "speech"


def parse_boc_index(session: requests.Session, bank_name: str, cfg: Dict[str, object], max_items: int) -> List[Dict[str, Optional[str]]]:
    soup = fetch_html(session, str(cfg["index_url"]))
    items: List[Dict[str, Optional[str]]] = []

    for h in soup.find_all(["h2", "h3"]):
        title = clean_text(h.get_text(" ", strip=True))
        if not title or len(title.split()) < 2:
            continue
        parent = h.parent
        link = None
        for a in parent.find_all("a", href=True):
            href = urljoin(str(cfg["index_url"]), a["href"])
            if "/" in href and href.startswith("https://www.bankofcanada.ca/"):
                link = href
                break
        if not link:
            continue
        block_text = clean_text(parent.get_text(" ", strip=True))
        date = None
        m = re.search(r"([A-Z][a-z]+\.?\s+\d{1,2},\s+\d{4})", block_text)
        if m:
            date = parse_date_text(m.group(1))
        speaker = None
        role = None
        possible_speakers = [
            "Tiff Macklem", "Carolyn Rogers", "Sharon Kozicki", "Nicolas Vincent", "Toni Gravelle",
            "Carolyn A. Rogers",
        ]
        for sp in possible_speakers:
            if sp.lower() in block_text.lower():
                speaker = sp.replace(" A. ", " ")
                break
        items.append({
            "bank_name": bank_name,
            "bank_code": cfg["bank_code"],
            "country": cfg["country"],
            "event_type": detect_event_type(block_text),
            "date": date,
            "title": title,
            "speaker": speaker,
            "role": role,
            "venue": None,
            "url": link,
        })
        if len(items) >= max_items:
            break

    dedup = {}
    for item in items:
        dedup[item["url"]] = item
    return list(dedup.values())[:max_items]


def parse_boj_index(session: requests.Session, bank_name: str, cfg: Dict[str, object], max_items: int) -> List[Dict[str, Optional[str]]]:
    soup = fetch_html(session, str(cfg["index_url"]))
    items: List[Dict[str, Optional[str]]] = []

    for a in soup.find_all("a", href=True):
        href = urljoin(str(cfg["index_url"]), a.get("href", ""))
        title = clean_text(a.get_text(" ", strip=True))
        if not title:
            continue
        if "/en/announcements/press/" not in href and "/en/about/press/" not in href:
            continue
        if len(title.split()) < 3:
            continue
        row_text = clean_text(a.parent.get_text(" ", strip=True))
        if "speech" not in row_text.lower() and "monetary policy" not in title.lower() and "economic activity" not in title.lower():
            continue
        m = re.search(r"([A-Z][a-z]{2}\.\s+\d{1,2},\s+\d{4}|[A-Z][a-z]+\.?\s+\d{1,2},\s+\d{4})", row_text)
        date = parse_date_text(m.group(1)) if m else None

        speaker = None
        speaker_match = re.search(r"\b([A-Z][A-Z]+\s+[A-Za-z]+|UEDA Kazuo|HIMINO Ryozo|TAKATA Hajime|TAMURA Naoki|NAKAGAWA Junko|KOEDA Junko)\b", row_text)
        if speaker_match:
            speaker = clean_text(speaker_match.group(1))

        items.append({
            "bank_name": bank_name,
            "bank_code": cfg["bank_code"],
            "country": cfg["country"],
            "event_type": "speech",
            "date": date,
            "title": title,
            "speaker": speaker,
            "role": None,
            "venue": None,
            "url": href,
        })
        if len(items) >= max_items * 3:
            break

    dedup = {}
    for item in items:
        dedup[item["url"]] = item
    out = sorted(dedup.values(), key=lambda x: x.get("date") or "", reverse=True)
    return out[:max_items]


def parse_rba_index(session: requests.Session, bank_name: str, cfg: Dict[str, object], max_items: int) -> List[Dict[str, Optional[str]]]:
    soup = fetch_html(session, str(cfg["index_url"]))
    items: List[Dict[str, Optional[str]]] = []

    current_year = None
    for node in soup.find_all(["h2", "h3", "a", "p", "li"]):
        txt = clean_text(node.get_text(" ", strip=True))
        if re.fullmatch(r"20\d{2}", txt):
            current_year = txt
        if node.name == "a" and node.get("href"):
            href = urljoin(str(cfg["index_url"]), node["href"])
            title = txt
            if "/speeches/" not in href or href.endswith("/speeches/") or "/index.html" in href:
                continue
            if len(title.split()) < 3:
                continue
            parent_text = clean_text(node.parent.get_text(" ", strip=True))
            m = re.search(r"(\d{1,2}\s+[A-Z][a-z]+\s+20\d{2}|[A-Z][a-z]+\s+\d{1,2},\s+20\d{2})", parent_text)
            date = parse_date_text(m.group(1)) if m else None
            speaker = None
            for sp in ["Michele Bullock", "Andrew Hauser", "Sarah Hunter", "Christopher Kent", "Brad Jones", "Philip Lowe"]:
                if sp.lower() in parent_text.lower():
                    speaker = sp
                    break
            items.append({
                "bank_name": bank_name,
                "bank_code": cfg["bank_code"],
                "country": cfg["country"],
                "event_type": detect_event_type(parent_text),
                "date": date,
                "title": title,
                "speaker": speaker,
                "role": None,
                "venue": None,
                "url": href,
            })
        if len(items) >= max_items * 3:
            break

    dedup = {}
    for item in items:
        dedup[item["url"]] = item
    out = sorted(dedup.values(), key=lambda x: x.get("date") or "", reverse=True)
    return out[:max_items]


def parse_norges_index(session: requests.Session, bank_name: str, cfg: Dict[str, object], max_items: int) -> List[Dict[str, Optional[str]]]:
    soup = fetch_html(session, str(cfg["index_url"]))
    items: List[Dict[str, Optional[str]]] = []

    for a in soup.find_all("a", href=True):
        href = urljoin(str(cfg["index_url"]), a["href"])
        title = clean_text(a.get_text(" ", strip=True))
        if "/Speeches/" not in href and "/speeches/" not in href:
            continue
        if len(title.split()) < 3:
            continue
        parent_text = clean_text(a.parent.get_text(" ", strip=True))
        m = re.search(r"(\d{1,2}\s+[A-Z][a-z]+\s+20\d{2}|[A-Z][a-z]+\s+\d{4})", parent_text)
        date = parse_date_text(m.group(1)) if m else None
        speaker = None
        for sp in ["Ida Wolden Bache", "Pål Longva", "Martin E. Nordhagen", "Torbjørn Hægeland"]:
            if sp.lower() in parent_text.lower():
                speaker = sp
                break
        items.append({
            "bank_name": bank_name,
            "bank_code": cfg["bank_code"],
            "country": cfg["country"],
            "event_type": detect_event_type(parent_text),
            "date": date,
            "title": title,
            "speaker": speaker,
            "role": None,
            "venue": None,
            "url": href,
        })
        if len(items) >= max_items * 3:
            break

    dedup = {}
    for item in items:
        dedup[item["url"]] = item
    out = sorted(dedup.values(), key=lambda x: x.get("date") or "", reverse=True)
    return out[:max_items]


def parse_index_for_bank(session: requests.Session, bank_name: str, cfg: Dict[str, object], max_items: int) -> List[Dict[str, Optional[str]]]:
    code = str(cfg["bank_code"])
    if code == "BOC":
        return parse_boc_index(session, bank_name, cfg, max_items)
    if code == "BOJ":
        return parse_boj_index(session, bank_name, cfg, max_items)
    if code == "RBA":
        return parse_rba_index(session, bank_name, cfg, max_items)
    if code == "NORGES":
        return parse_norges_index(session, bank_name, cfg, max_items)
    return []


def extract_meta_generic(soup: BeautifulSoup, item: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    title = item.get("title")
    h1 = soup.find(["h1", "h2"])
    if h1:
        t = clean_text(h1.get_text(" ", strip=True))
        if len(t.split()) >= 2:
            title = t

    all_text = clean_text(soup.get_text(" \n ", strip=True))
    date = item.get("date")
    if not date:
        m = re.search(r"([A-Z][a-z]{2}\.\s+\d{1,2},\s+\d{4}|[A-Z][a-z]+\s+\d{1,2},\s+\d{4}|\d{1,2}\s+[A-Z][a-z]+\s+20\d{2})", all_text)
        if m:
            date = parse_date_text(m.group(1))

    speaker = item.get("speaker")
    role = item.get("role")
    venue = item.get("venue")

    for line in soup.get_text("\n", strip=True).split("\n")[:80]:
        line = clean_text(line)
        low = line.lower()
        if not speaker and any(x in low for x in ["governor", "deputy governor", "assistant governor", "senior deputy governor", "member of the policy board"]):
            if len(line.split()) <= 8:
                role = line if any(r in low for r in ["governor", "deputy governor", "assistant governor"]) else role
        if not venue and len(line.split()) >= 2 and (" sydney" in low or " tokyo" in low or " toronto" in low or " oslo" in low):
            venue = line

    return {
        "title": title,
        "date": date,
        "speaker": speaker,
        "role": role,
        "venue": venue,
    }


def parse_document(session: requests.Session, item: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    soup = fetch_html(session, str(item["url"]))
    meta = extract_meta_generic(soup, item)
    body_text = extract_body_text_generic(soup)
    score = score_text(body_text)

    return {
        "url": item["url"],
        "bank_name": item["bank_name"],
        "bank_code": item["bank_code"],
        "country": item["country"],
        "event_type": item["event_type"],
        "date": meta["date"],
        "title": meta["title"],
        "speaker": meta["speaker"],
        "role": meta["role"],
        "venue": meta["venue"],
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
            url, bank_name, bank_code, country, event_type, date, title, speaker, role, venue,
            body_text, word_count, scraped_at, hawkish_score, dovish_score, net_score,
            inflation_concern, labor_concern, growth_concern, financial_stability,
            balance_sheet, uncertainty_risk, policy_relevance, live_signal_share
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            bank_name=excluded.bank_name,
            bank_code=excluded.bank_code,
            country=excluded.country,
            event_type=excluded.event_type,
            date=excluded.date,
            title=excluded.title,
            speaker=excluded.speaker,
            role=excluded.role,
            venue=excluded.venue,
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
            doc["url"], doc["bank_name"], doc["bank_code"], doc["country"], doc["event_type"], doc["date"],
            doc["title"], doc["speaker"], doc["role"], doc["venue"], doc["body_text"], doc["word_count"],
            doc["scraped_at"], doc["hawkish_score"], doc["dovish_score"], doc["net_score"],
            doc["inflation_concern"], doc["labor_concern"], doc["growth_concern"], doc["financial_stability"],
            doc["balance_sheet"], doc["uncertainty_risk"], doc["policy_relevance"], doc["live_signal_share"],
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
    num_cols = [
        "word_count", "hawkish_score", "dovish_score", "net_score", "inflation_concern",
        "labor_concern", "growth_concern", "financial_stability", "balance_sheet",
        "uncertainty_risk", "policy_relevance", "live_signal_share",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(["date", "bank_name", "title"]).reset_index(drop=True)


def refresh_corpus(selected_banks: List[str], max_items_per_bank: int = 40, force_refresh: bool = False, progress_bar=None, status_box=None) -> Dict[str, int]:
    init_db()
    clear_log()

    stats = {
        "banks": 0,
        "index_items_found": 0,
        "attempted_docs": 0,
        "inserted_docs": 0,
        "skipped_existing": 0,
        "failed_docs": 0,
    }

    session = make_session()
    conn = db_connection()

    try:
        existing = get_existing_urls(conn)
        all_items: List[Dict[str, Optional[str]]] = []
        stats["banks"] = len(selected_banks)

        for i, bank_name in enumerate(selected_banks, start=1):
            cfg = BANK_CONFIGS[bank_name]
            if status_box is not None:
                status_box.info(f"Loading index: {bank_name} ({i}/{len(selected_banks)})")
            try:
                items = parse_index_for_bank(session, bank_name, cfg, max_items=max_items_per_bank)
                all_items.extend(items)
                log_event("INFO", f"Parsed {len(items)} index items for {bank_name}")
                time.sleep(0.2)
            except Exception as exc:
                log_event("ERROR", f"Failed index for {bank_name}: {repr(exc)}")

        dedup = {}
        for item in all_items:
            dedup[item["url"]] = item
        all_items = sorted(dedup.values(), key=lambda x: (x.get("date") or "", x["bank_name"]), reverse=True)
        stats["index_items_found"] = len(all_items)

        for i, item in enumerate(all_items, start=1):
            stats["attempted_docs"] += 1
            if progress_bar is not None:
                progress_bar.progress(i / max(len(all_items), 1))
            if status_box is not None:
                status_box.info(f"Processing {i:,}/{len(all_items):,}: {item['bank_name']} | {str(item['title'])[:100]}")

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
                    f"Stored {doc.get('bank_name')} | {doc.get('date')} | {doc.get('speaker')} | {doc.get('title')}"
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

    out["tone_z_global"] = out["net_score"].apply(lambda x: bounded_z(zscore_value(x, corpus_mean, corpus_std)))
    out["inflation_z_global"] = out["inflation_concern"].apply(lambda x: bounded_z(zscore_value(x, infl_mean, infl_std)))
    out["labor_z_global"] = out["labor_concern"].apply(lambda x: bounded_z(zscore_value(x, labor_mean, labor_std)))
    out["growth_z_global"] = out["growth_concern"].apply(lambda x: bounded_z(zscore_value(x, growth_mean, growth_std)))
    out["fs_z_global"] = out["financial_stability"].apply(lambda x: bounded_z(zscore_value(x, fs_mean, fs_std)))
    out["uncertainty_z_global"] = out["uncertainty_risk"].apply(lambda x: bounded_z(zscore_value(x, unc_mean, unc_std)))

    out["bank_weight"] = out["bank_name"].map(BANK_WEIGHTS).fillna(1.0)
    out["speaker_weight"] = out.apply(
        lambda r: BANK_CONFIGS.get(r["bank_name"], {}).get("speaker_weights", {}).get(r["speaker"], 1.0),
        axis=1,
    )
    out["combined_weight"] = out["bank_weight"] * out["speaker_weight"]

    out["tone_z_bank"] = np.nan
    out["tone_z_speaker"] = np.nan

    for bank_name, idx in out.groupby("bank_name", dropna=False).groups.items():
        subset = out.loc[idx]
        m, s = safe_mean_std(subset["net_score"])
        out.loc[idx, "tone_z_bank"] = subset["net_score"].apply(lambda x: bounded_z(zscore_value(x, m, s)))

    for speaker, idx in out.groupby(["bank_name", "speaker"], dropna=False).groups.items():
        subset = out.loc[idx]
        m, s = safe_mean_std(subset["net_score"])
        out.loc[idx, "tone_z_speaker"] = subset["net_score"].apply(lambda x: bounded_z(zscore_value(x, m, s)))

    out["tone_percentile_global"] = out["net_score"].apply(lambda x: percentile_rank(out["net_score"], x))
    out["stance"] = out["tone_z_global"].apply(tone_bucket)
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
        frame.groupby(["bucket", "bank_name"])
        .apply(
            lambda x: pd.Series(
                {
                    "tone_z_global": np.average(x["tone_z_global"], weights=x["combined_weight"]),
                    "inflation_z_global": np.average(x["inflation_z_global"], weights=x["combined_weight"]),
                    "labor_z_global": np.average(x["labor_z_global"], weights=x["combined_weight"]),
                    "growth_z_global": np.average(x["growth_z_global"], weights=x["combined_weight"]),
                    "documents": len(x),
                    "policy_relevance": np.average(x["policy_relevance"].fillna(0.0), weights=x["combined_weight"]),
                }
            )
        )
        .reset_index()
        .rename(columns={"bucket": "date"})
        .sort_values(["bank_name", "date"])
        .reset_index(drop=True)
    )

    grouped["tone_z_smooth"] = grouped.groupby("bank_name")["tone_z_global"].transform(lambda s: s.ewm(span=4, adjust=False).mean())
    return grouped


def aggregate_cross_bank(df: pd.DataFrame, freq: str = "30D") -> pd.DataFrame:
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
                    "tone_z_global": np.average(x["tone_z_global"], weights=x["combined_weight"]),
                    "inflation_z_global": np.average(x["inflation_z_global"], weights=x["combined_weight"]),
                    "labor_z_global": np.average(x["labor_z_global"], weights=x["combined_weight"]),
                    "growth_z_global": np.average(x["growth_z_global"], weights=x["combined_weight"]),
                    "documents": len(x),
                }
            )
        )
        .reset_index()
        .rename(columns={"bucket": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    grouped["tone_z_smooth"] = grouped["tone_z_global"].ewm(span=4, adjust=False).mean()
    grouped["tone_3m_ma"] = grouped["tone_z_smooth"].rolling(3, min_periods=1).mean()
    grouped["tone_6m_ma"] = grouped["tone_z_smooth"].rolling(6, min_periods=1).mean()
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
        highlighted = pattern.sub(lambda m: f"<mark style='background-color:#ffe58f'>{m.group(0)}</mark>", highlighted)
    return highlighted


def speaker_matrix(df: pd.DataFrame, top_n: int = 18) -> pd.DataFrame:
    if df.empty:
        return df

    agg = (
        df.groupby(["bank_name", "speaker"], dropna=False)
        .agg(
            docs=("url", "count"),
            tone_z_global=("tone_z_global", "mean"),
            tone_z_speaker=("tone_z_speaker", "mean"),
            policy_relevance=("policy_relevance", "mean"),
            last_date=("date", "max"),
        )
        .reset_index()
    )

    latest = (
        df.sort_values("date")
        .groupby(["bank_name", "speaker"], dropna=False)
        .tail(1)[["bank_name", "speaker", "tone_z_speaker", "tone_z_global", "stance"]]
        .rename(columns={"tone_z_speaker": "latest_vs_own", "tone_z_global": "latest_vs_global"})
    )

    agg = agg.merge(latest, on=["bank_name", "speaker"], how="left")
    agg["weight"] = agg["bank_name"].map(BANK_WEIGHTS).fillna(1.0)
    agg["label"] = agg["bank_name"] + " | " + agg["speaker"].fillna("Unknown")
    agg = agg.sort_values(["weight", "docs", "tone_z_global"], ascending=[False, False, False]).head(top_n)
    return agg


def render_about() -> None:
    st.sidebar.markdown("### About This Tool")
    st.sidebar.markdown(
        """
        <div style='padding:0.85rem 1rem;border:1px solid #d9d9d9;border-radius:0.7rem;background:#fafafa;line-height:1.65'>
        This dashboard extends the Fed tone underwriter framework to selected rest-of-world central banks using official speech pages, then scores each document on hawkish and dovish language after filtering out ceremonial noise and low-policy passages.
        <br><br>
        Version 1 prioritises scrapeable, transparent institutions: Bank of Canada, Bank of Japan, Reserve Bank of Australia and Norges Bank. The architecture is modular, so ECB, BoE, SNB and others can be added with dedicated adapters.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_diagnostics(log_df: pd.DataFrame, stats: Optional[Dict[str, int]] = None) -> None:
    st.subheader("Diagnostics")
    if stats:
        a, b, c, d, e, f = st.columns(6)
        a.metric("Banks", stats.get("banks", 0))
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

    y_series = pd.to_numeric(series["tone_z_smooth"], errors="coerce").dropna()
    combined_y = y_series.dropna()
    if combined_y.empty:
        y_low, y_high = -1.5, 1.5
    else:
        y_min = float(combined_y.min())
        y_max = float(combined_y.max())
        pad = max(0.18, (y_max - y_min) * 0.18)
        y_low = max(-3.0, np.floor((y_min - pad) / 0.25) * 0.25)
        y_high = min(3.0, np.ceil((y_max + pad) / 0.25) * 0.25)

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

    for bank_name, sub in series.groupby("bank_name"):
        fig.add_trace(
            go.Scatter(
                x=sub["date"],
                y=sub["tone_z_smooth"],
                mode="lines+markers",
                name=bank_name,
                line=dict(width=2.2),
                marker=dict(size=6),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}σ<extra></extra>",
            )
        )

    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=84, b=28),
        title=dict(text="RoW central bank tone over time", x=0.01, xanchor="left", y=0.98, yanchor="top"),
        yaxis_title="Z-score vs global baseline",
        xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0.0),
    )
    return fig


def cross_bank_figure(series: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series["date"],
            y=series["tone_z_smooth"],
            mode="lines+markers",
            name="Cross-bank tone",
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
        )
    )
    fig.add_trace(
        go.Scatter(
            x=series["date"],
            y=series["tone_6m_ma"],
            mode="lines",
            name="6-bucket mean",
            line=dict(dash="dash", width=2),
        )
    )
    fig.update_layout(
        height=440,
        margin=dict(l=20, r=20, t=70, b=28),
        title=dict(text="Cross-bank composite tone", x=0.01),
        yaxis_title="Z-score",
        xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    return fig


def speaker_matrix_figure(matrix: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        matrix,
        x="latest_vs_global",
        y="label",
        size="docs",
        color="latest_vs_own",
        hover_data=["bank_name", "speaker", "docs", "latest_vs_own", "latest_vs_global", "policy_relevance", "last_date"],
        title="Speaker matrix",
        height=520,
        color_continuous_scale="RdYlGn_r",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=68, b=25),
        xaxis_title="Latest speech vs global baseline (z-score)",
        yaxis_title="",
        coloraxis_colorbar_title="Vs own history",
    )
    return fig


def scorecard_dataframe(doc: pd.Series) -> pd.DataFrame:
    rows = [
        ("Overall tone", doc["tone_z_global"], doc["tone_percentile_global"], tone_bucket(doc["tone_z_global"])),
        ("Vs bank history", doc["tone_z_bank"], np.nan, tone_bucket(doc["tone_z_bank"])),
        ("Vs own history", doc["tone_z_speaker"], np.nan, tone_bucket(doc["tone_z_speaker"])),
        ("Policy relevance", doc["policy_relevance"] * 2.0 - 1.0 if pd.notna(doc["policy_relevance"]) else np.nan, np.nan, policy_bucket(doc["policy_relevance"])),
        ("Inflation salience", doc["inflation_z_global"], np.nan, emphasis_bucket(doc["inflation_z_global"])),
        ("Labor salience", doc["labor_z_global"], np.nan, emphasis_bucket(doc["labor_z_global"])),
        ("Growth salience", doc["growth_z_global"], np.nan, emphasis_bucket(doc["growth_z_global"])),
        ("Financial stability", doc["fs_z_global"], np.nan, emphasis_bucket(doc["fs_z_global"])),
        ("Uncertainty", doc["uncertainty_z_global"], np.nan, emphasis_bucket(doc["uncertainty_z_global"])),
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
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), title="Document scorecard", xaxis_title="Relative score", yaxis_title="", xaxis=dict(range=[-3, 3]))
    return fig


def doc_summary_html(doc: pd.Series) -> str:
    tone = make_badge(tone_bucket(doc["tone_z_global"]), doc["tone_z_global"])
    policy = make_badge(policy_bucket(doc["policy_relevance"]), doc["policy_relevance"])
    return f"""
    <div style='padding:1rem;border:1px solid #d9d9d9;border-radius:0.8rem;background:#fafafa'>
      <div style='font-size:1.05rem;font-weight:700;margin-bottom:0.35rem'>{html.escape(str(doc['title']))}</div>
      <div style='color:#555;margin-bottom:0.6rem'>{html.escape(str(doc['bank_name']))} | {html.escape(str(doc['speaker'] or 'Unknown'))} | {doc['date'].strftime('%Y-%m-%d') if pd.notna(doc['date']) else 'Unknown date'}</div>
      <div style='display:flex;gap:0.5rem;flex-wrap:wrap'>{tone}{policy}</div>
    </div>
    """


def latest_table(view: pd.DataFrame) -> pd.DataFrame:
    latest = view.sort_values("date", ascending=False).copy()
    latest["tone"] = latest["tone_z_global"].apply(format_z)
    latest["vs bank"] = latest["tone_z_bank"].apply(format_z)
    latest["vs own"] = latest["tone_z_speaker"].apply(format_z)
    latest["global %ile"] = latest["tone_percentile_global"].apply(format_pct)
    latest["inflation"] = latest["inflation_z_global"].apply(format_z)
    latest["labor"] = latest["labor_z_global"].apply(format_z)
    latest["growth"] = latest["growth_z_global"].apply(format_z)
    latest["date"] = latest["date"].dt.strftime("%Y-%m-%d")
    latest["source"] = latest["url"]
    return latest[["date", "bank_name", "speaker", "event_type", "stance", "tone", "vs bank", "vs own", "global %ile", "inflation", "labor", "growth", "source"]].rename(columns={"event_type": "type", "bank_name": "bank"})


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    render_about()
    init_db()

    with st.sidebar:
        st.markdown("### Controls")
        selected_banks = st.multiselect(
            "Banks",
            options=list(BANK_CONFIGS.keys()),
            default=[k for k, v in BANK_CONFIGS.items() if v.get("enabled")],
        )
        max_items_per_bank = st.slider("Max recent speeches per bank", min_value=10, max_value=80, value=40, step=5)
        force_refresh = st.checkbox("Force refresh existing speeches", value=False)
        refresh_now = st.button("Refresh corpus", use_container_width=True)
        freq = st.selectbox("Aggregation frequency", options=["30D", "M", "Q"], index=0)

    stats = None
    if refresh_now:
        progress_bar = st.progress(0)
        status_box = st.empty()
        stats = refresh_corpus(
            selected_banks=selected_banks,
            max_items_per_bank=max_items_per_bank,
            force_refresh=force_refresh,
            progress_bar=progress_bar,
            status_box=status_box,
        )
        progress_bar.empty()
        status_box.empty()
        st.success("Refresh complete.")

    df = load_documents()
    if df.empty:
        st.info("No speeches loaded yet. Pick banks in the sidebar and click Refresh corpus.")
        render_diagnostics(load_log(), stats)
        return

    feat = compute_features(df)
    feat = feat[feat["bank_name"].isin(selected_banks)].copy()

    if feat.empty:
        st.warning("No documents available for the selected bank set.")
        render_diagnostics(load_log(), stats)
        return

    series = aggregate_series(feat, freq=freq)
    cross = aggregate_cross_bank(feat, freq=freq)
    matrix = speaker_matrix(feat, top_n=18)

    latest_doc = feat.sort_values("date").tail(1).iloc[0]
    latest_date = latest_doc["date"].strftime("%Y-%m-%d") if pd.notna(latest_doc["date"]) else "Unknown"
    latest_stance = tone_bucket(latest_doc["tone_z_global"])

    a, b, c, d = st.columns(4)
    a.metric("Documents", f"{len(feat):,}")
    b.metric("Banks", feat["bank_name"].nunique())
    c.metric("Latest document", latest_date)
    d.metric("Latest stance", latest_stance)

    row1a, row1b = st.columns([1.45, 1.0])
    with row1a:
        st.plotly_chart(z_dashboard_figure(series), use_container_width=True)
    with row1b:
        st.plotly_chart(cross_bank_figure(cross), use_container_width=True)

    st.plotly_chart(speaker_matrix_figure(matrix), use_container_width=True)

    st.subheader("Latest documents")
    st.dataframe(latest_table(feat).head(40), use_container_width=True, hide_index=True, height=420)

    st.subheader("Document deep dive")
    doc_options = feat.sort_values("date", ascending=False).copy()
    doc_options["label"] = doc_options.apply(
        lambda r: f"{r['date'].strftime('%Y-%m-%d') if pd.notna(r['date']) else 'Unknown'} | {r['bank_name']} | {r['speaker'] or 'Unknown'} | {r['title']}",
        axis=1,
    )
    selected_label = st.selectbox("Select a document", options=doc_options["label"].tolist())
    doc = doc_options.loc[doc_options["label"] == selected_label].iloc[0]

    st.markdown(doc_summary_html(doc), unsafe_allow_html=True)

    deep_a, deep_b = st.columns([1.0, 1.1])
    with deep_a:
        st.plotly_chart(scorecard_figure(scorecard_dataframe(doc)), use_container_width=True)
    with deep_b:
        topic = st.selectbox("Snippet view", options=["hawkish", "dovish", "inflation_concern", "labor_concern", "growth_concern"], index=0)
        phrases = LEXICON[topic]
        snippets = find_snippets(str(doc["body_text"]), phrases, max_snippets=5)
        if not snippets:
            st.info("No matching snippets found for that theme.")
        else:
            for snip in snippets:
                st.markdown(
                    f"<div style='padding:0.75rem 0.9rem;border:1px solid #e5e5e5;border-radius:0.7rem;margin-bottom:0.55rem;background:#fff'>{highlight_terms(snip, phrases)}</div>",
                    unsafe_allow_html=True,
                )

    with st.expander("Show cleaned document text"):
        st.write(doc["body_text"])

    render_diagnostics(load_log(), stats)


if __name__ == "__main__":
    main()
