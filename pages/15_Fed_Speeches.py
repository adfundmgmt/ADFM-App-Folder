import html
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import numpy as np
import pandas as pd
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

APP_CSS = """
<style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }
    .sidebar-note {
        color: #475467;
        font-size: 0.92rem;
        line-height: 1.55;
    }
    .status-line {
        color: #475467;
        font-size: 0.95rem;
        line-height: 1.5;
        margin: 0.15rem 0 0.9rem 0;
    }
    .chart-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #101828;
        margin: 0.15rem 0 0.35rem 0;
    }
    .scroll-table-wrap {
        max-height: 460px;
        overflow-y: auto;
        border: 1px solid #e4e7ec;
        border-radius: 12px;
        background: #ffffff;
    }
    .scroll-table-wrap table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.93rem;
    }
    .scroll-table-wrap thead th {
        position: sticky;
        top: 0;
        z-index: 2;
        background: #f8fafc;
    }
    .scroll-table-wrap th,
    .scroll-table-wrap td {
        padding: 0.72rem 0.82rem;
        border-bottom: 1px solid #eef2f6;
        text-align: left;
        white-space: nowrap;
    }
    .scroll-table-wrap td:nth-child(2) {
        white-space: normal;
        min-width: 220px;
    }
    .scroll-table-wrap tbody tr:hover {
        background: #f9fafb;
    }
    .summary-card {
        padding: 0.95rem 1rem;
        border: 1px solid #e6e6e6;
        border-radius: 0.8rem;
        background: #fafafa;
    }
</style>
"""

ROLE_PREFIXES = [
    ("Vice Chair for Supervision ", "Vice Chair for Supervision"),
    ("Vice Chair ", "Vice Chair"),
    ("Chair ", "Chair"),
    ("Governor ", "Governor"),
    ("President ", "President"),
]

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
}

LEXICON = {
    "hawkish": [
        "higher for longer",
        "restrictive",
        "sufficiently restrictive",
        "upside risk",
        "inflation remains too high",
        "inflation is too high",
        "persistent inflation",
        "reaccelerat",
        "not yet done",
        "additional firming",
        "further tightening",
        "vigilant",
        "upward pressure on prices",
        "elevated inflation",
        "inflation pressure",
        "price stability",
        "tight labor market",
        "strong labor market",
        "overheating",
        "policy restraint",
        "maintain restraint",
        "firming",
        "tightening",
        "hold rates higher",
        "risk of inflation",
        "unanchored inflation expectations",
        "still above target",
        "premature to ease",
        "not appropriate to cut",
        "more work to do",
        "upside risks to inflation",
        "inflation persistence",
    ],
    "dovish": [
        "disinflation",
        "cooling labor market",
        "softening labor market",
        "downside risk",
        "growth is slowing",
        "economic slowdown",
        "below-trend growth",
        "normalizing inflation",
        "further progress on inflation",
        "policy can respond",
        "room to ease",
        "easing",
        "rate cuts",
        "cut rates",
        "lower rates",
        "less restrictive",
        "downward path",
        "labor market is moderating",
        "balanced risks",
        "two-sided risks",
        "weak demand",
        "headwinds",
        "slack",
        "unemployment is rising",
        "financial conditions tightened",
        "act as appropriate",
        "support the labor market",
        "downside risks to employment",
    ],
    "inflation_concern": [
        "inflation",
        "prices",
        "price pressures",
        "services inflation",
        "core inflation",
        "shelter inflation",
        "goods inflation",
        "inflation expectations",
        "price stability",
    ],
    "labor_concern": [
        "labor market",
        "employment",
        "unemployment",
        "job growth",
        "payroll",
        "wages",
        "hiring",
        "layoffs",
        "slack",
        "participation",
    ],
    "growth_concern": [
        "growth",
        "activity",
        "demand",
        "consumer spending",
        "investment",
        "slowdown",
        "recession",
        "weakness",
        "output",
        "expansion",
    ],
    "financial_stability": [
        "financial stability",
        "banking",
        "banks",
        "stress",
        "liquidity",
        "funding",
        "market functioning",
        "credit conditions",
        "treasury market",
        "vulnerabilities",
    ],
    "balance_sheet": [
        "balance sheet",
        "quantitative tightening",
        "qt",
        "runoff",
        "reserves",
        "securities holdings",
        "mbs",
        "treasury holdings",
    ],
    "uncertainty_risk": [
        "uncertain",
        "uncertainty",
        "risk management",
        "careful",
        "monitor",
        "watching",
        "data dependent",
        "incoming data",
        "proceed carefully",
        "humble",
        "attentive",
    ],
}

SNIPPET_PATTERNS = {
    "hawkish": LEXICON["hawkish"],
    "dovish": LEXICON["dovish"],
    "inflation": LEXICON["inflation_concern"],
    "labor": LEXICON["labor_concern"],
    "growth": LEXICON["growth_concern"],
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


def clean_text(text: Optional[str]) -> str:
    if text is None or pd.isna(text):
        return ""
    text = html.unescape(str(text))
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_role_speaker(raw_speaker_line: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    s = clean_text(raw_speaker_line)
    if not s:
        return None, None

    for prefix, role in ROLE_PREFIXES:
        if s.startswith(prefix):
            return role, clean_text(s[len(prefix):])

    return None, s


def normalize_speaker_name(raw_speaker_line: Optional[str]) -> Optional[str]:
    _, speaker = split_role_speaker(raw_speaker_line)
    return speaker


def normalize_role_name(raw_role_line: Optional[str]) -> Optional[str]:
    role, speaker = split_role_speaker(raw_role_line)
    if role:
        return role
    if clean_text(raw_role_line) and speaker != clean_text(raw_role_line):
        return clean_text(raw_role_line)
    return clean_text(raw_role_line) or None


def normalize_phrase(phrase: str) -> str:
    return re.escape(phrase.lower())


def count_phrase_hits(text: str, phrases: List[str]) -> int:
    low = text.lower()
    hits = 0
    for phrase in phrases:
        hits += len(re.findall(normalize_phrase(phrase), low))
    return hits


def count_distinct_phrase_hits(text: str, phrases: List[str]) -> int:
    low = text.lower()
    seen = 0
    for phrase in phrases:
        if re.search(normalize_phrase(phrase), low):
            seen += 1
    return seen


def smooth_phrase_intensity(total_hits: int, distinct_hits: int, word_count: int) -> float:
    if word_count <= 0:
        return 0.0
    length_scale = max(np.sqrt(word_count), 1.0)
    return float(((0.75 * np.log1p(total_hits)) + (0.25 * distinct_hits)) / length_scale * 100.0)


def safe_mean_std(series: pd.Series) -> Tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0, 1.0
    if len(s) >= 12:
        lower = s.quantile(0.05)
        upper = s.quantile(0.95)
        s = s.clip(lower=lower, upper=upper)
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


def score_text(text: str) -> ToneResult:
    low = clean_text(text).lower()
    words = re.findall(r"\b[a-z][a-z\-']+\b", low)
    word_count = max(len(words), 1)

    raw_hits = {k: count_phrase_hits(low, v) for k, v in LEXICON.items()}
    distinct_hits = {k: count_distinct_phrase_hits(low, v) for k, v in LEXICON.items()}
    scaled = {
        key: smooth_phrase_intensity(raw_hits[key], distinct_hits[key], word_count)
        for key in LEXICON
    }

    hawkish = scaled.get("hawkish", 0.0)
    dovish = scaled.get("dovish", 0.0)
    inflation_weight = scaled.get("inflation_concern", 0.0)
    labor_weight = scaled.get("labor_concern", 0.0)
    growth_weight = scaled.get("growth_concern", 0.0)
    fs_weight = scaled.get("financial_stability", 0.0)
    bs_weight = scaled.get("balance_sheet", 0.0)
    uncertainty_weight = scaled.get("uncertainty_risk", 0.0)

    document_reliability = min(1.0, 0.55 + 0.45 * np.tanh(word_count / 600.0))
    net = (
        hawkish
        - dovish
        + 0.08 * inflation_weight
        - 0.06 * labor_weight
        - 0.06 * growth_weight
        - 0.03 * fs_weight
    ) * document_reliability

    return ToneResult(
        net_score=net,
        hawkish_score=hawkish,
        dovish_score=dovish,
        inflation_concern=inflation_weight,
        labor_concern=labor_weight,
        growth_concern=growth_weight,
        financial_stability=fs_weight,
        balance_sheet=bs_weight,
        uncertainty_risk=uncertainty_weight,
        word_count=word_count,
    )


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
            uncertainty_risk REAL
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
                    low.startswith(prefix)
                    for prefix in [
                        "chair ",
                        "vice chair ",
                        "vice chair for supervision ",
                        "governor ",
                        "president ",
                    ]
                ):
                    speaker_hint = cand
                    role_hint, _ = split_role_speaker(cand)
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
        if result["speaker"] is None and any(
            low.startswith(prefix)
            for prefix in [
                "chair ",
                "vice chair ",
                "vice chair for supervision ",
                "governor ",
                "president ",
            ]
        ):
            role, speaker = split_role_speaker(line)
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
        hinted_role, hinted_speaker = split_role_speaker(hints.get("speaker_hint"))
        result["speaker"] = hinted_speaker
        if result["role"] is None:
            result["role"] = hinted_role
    if result["role"] is None:
        result["role"] = hints.get("role_hint")
    if result["venue"] is None:
        result["venue"] = hints.get("venue_hint")

    return result


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
                low.startswith(prefix)
                for prefix in [
                    "at ",
                    "before ",
                    "at the ",
                    "before the ",
                    "thank you",
                    "good morning",
                    "good afternoon",
                    "good evening",
                    "mr. chairman",
                    "chairman",
                    "madam chair",
                    "let me",
                    "today i",
                    "this evening",
                    "i would like",
                    "it is a pleasure",
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

    speaker = merge_preferred(meta["speaker"], item.get("speaker_hint"))
    role = merge_preferred(meta["role"], item.get("role_hint"))
    parsed_role, parsed_speaker = split_role_speaker(speaker)
    speaker = parsed_speaker or speaker
    role = merge_preferred(role, parsed_role)

    return {
        "url": item["url"],
        "event_type": item["event_type"],
        "year": item["year"],
        "date": merge_preferred(meta["date"], item.get("date_hint")),
        "title": merge_preferred(meta["title"], item.get("title")),
        "speaker": speaker,
        "role": role,
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
    }


def upsert_document(conn: sqlite3.Connection, doc: Dict[str, Optional[str]]) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO documents (
            url, event_type, year, date, title, speaker, role, venue, pdf_url, body_text,
            word_count, scraped_at, hawkish_score, dovish_score, net_score, inflation_concern,
            labor_concern, growth_concern, financial_stability, balance_sheet, uncertainty_risk
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            uncertainty_risk=excluded.uncertainty_risk
        """,
        (
            doc["url"],
            doc["event_type"],
            doc["year"],
            doc["date"],
            doc["title"],
            doc["speaker"],
            doc["role"],
            doc["venue"],
            doc["pdf_url"],
            doc["body_text"],
            doc["word_count"],
            doc["scraped_at"],
            doc["hawkish_score"],
            doc["dovish_score"],
            doc["net_score"],
            doc["inflation_concern"],
            doc["labor_concern"],
            doc["growth_concern"],
            doc["financial_stability"],
            doc["balance_sheet"],
            doc["uncertainty_risk"],
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
    original_speakers = df["speaker"].copy()
    df["speaker"] = original_speakers.apply(normalize_speaker_name)
    df["role"] = df["role"].apply(normalize_role_name)

    needs_role = df["role"].isna() | (df["role"].astype(str).str.strip() == "")
    inferred_roles = original_speakers.apply(lambda x: split_role_speaker(x)[0])
    df.loc[needs_role, "role"] = inferred_roles[needs_role]

    num_cols = [
        "word_count",
        "hawkish_score",
        "dovish_score",
        "net_score",
        "inflation_concern",
        "labor_concern",
        "growth_concern",
        "financial_stability",
        "balance_sheet",
        "uncertainty_risk",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["date", "title"], ascending=[True, True]).reset_index(drop=True)
    return df


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
                    f"Stored {doc.get('date')} | {doc.get('speaker')} | {doc.get('title')} | words={doc.get('word_count')}",
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

    for _, idx in out.groupby("speaker", dropna=False).groups.items():
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

    out["tone_regime"] = pd.cut(
        out["tone_z_fed"],
        bins=[-np.inf, -1.25, -0.35, 0.35, 1.25, np.inf],
        labels=["Very Dovish", "Dovish", "Neutral", "Hawkish", "Very Hawkish"],
    )

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
                }
            )
        )
        .reset_index()
        .rename(columns={"bucket": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    grouped["tone_signal"] = grouped["tone_z_fed"].ewm(span=2, adjust=False).mean()
    grouped["tone_3m_ma"] = grouped["tone_signal"].rolling(3, min_periods=1).mean()
    grouped["tone_6m_ma"] = grouped["tone_signal"].rolling(6, min_periods=1).mean()
    return grouped


def latest_snapshot(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "tone_signal": np.nan,
            "delta_30d": np.nan,
            "core_tone_z": np.nan,
            "tone_percentile_fed": np.nan,
        }

    window = aggregate_series(df, freq="30D")
    tone_col = "tone_signal" if "tone_signal" in window.columns else "tone_z_fed"
    tone = window[tone_col].iloc[-1] if not window.empty else np.nan

    delta_30d = np.nan
    if len(window) >= 2:
        delta_30d = tone - window[tone_col].iloc[-2]

    core = df[
        df["speaker"].isin(
            ["Jerome H. Powell", "Philip N. Jefferson", "John C. Williams", "Christopher J. Waller"]
        )
    ]
    core_tone = np.average(core["tone_z_fed"], weights=core["speaker_weight"]) if not core.empty else np.nan

    latest_doc = df.sort_values("date").iloc[-1]
    return {
        "tone_signal": tone,
        "delta_30d": delta_30d,
        "core_tone_z": core_tone,
        "tone_percentile_fed": float(latest_doc["tone_percentile_fed"]) if "tone_percentile_fed" in latest_doc else np.nan,
    }


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
            lambda m: f"<mark style='background-color:#fee4a6'>{m.group(0)}</mark>",
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
            inflation_z_fed=("inflation_z_fed", "mean"),
            labor_z_fed=("labor_z_fed", "mean"),
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
    agg["last_date_label"] = agg["last_date"].dt.strftime("%Y-%m-%d")
    return agg


def render_about() -> None:
    st.sidebar.header("About This Tool")
    st.sidebar.markdown(
        """
<div class="sidebar-note">
This dashboard scrapes Federal Reserve speeches and testimony directly from the Board website, scores each document on hawkish and dovish language, and shows how speaker tone is shifting relative to both the speaker's own history and the broader Fed baseline.
<br><br>
The focus is signal extraction from primary source text. The highlighted excerpt and passage list are meant to make the scoring easy to sanity-check quickly.
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
        st.dataframe(log_df, use_container_width=True, hide_index=True)


def format_z(x: float) -> str:
    if pd.isna(x):
        return "n.a."
    return f"{x:+.2f}σ"


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "n.a."
    return f"{x:.0f}th %ile"


def make_badge(label: str, z: float) -> str:
    color = "#eef2f6"
    text = "#475467"
    if not pd.isna(z):
        if z <= -0.35:
            color = "#dcfce7"
            text = "#166534"
        elif z >= 0.35:
            color = "#fee2e2"
            text = "#991b1b"
    return (
        f"<span style='display:inline-block;padding:0.30rem 0.64rem;border-radius:999px;"
        f"background:{color};color:{text};font-weight:600;font-size:0.86rem'>{html.escape(label)}</span>"
    )


def tone_color(z: float) -> str:
    if pd.isna(z):
        return "#98a2b3"
    if z <= -0.35:
        return "#16a34a"
    if z >= 0.35:
        return "#dc2626"
    return "#667085"


def z_dashboard_figure(series: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_hrect(y0=-3, y1=-1.25, fillcolor="rgba(34,197,94,0.13)", line_width=0)
    fig.add_hrect(y0=-1.25, y1=-0.35, fillcolor="rgba(34,197,94,0.07)", line_width=0)
    fig.add_hrect(y0=-0.35, y1=0.35, fillcolor="rgba(148,163,184,0.08)", line_width=0)
    fig.add_hrect(y0=0.35, y1=1.25, fillcolor="rgba(239,68,68,0.07)", line_width=0)
    fig.add_hrect(y0=1.25, y1=3, fillcolor="rgba(239,68,68,0.13)", line_width=0)

    fig.add_trace(
        go.Scatter(
            x=series["date"],
            y=series["tone_signal"],
            mode="lines",
            name="Smoothed tone signal",
            line=dict(color="#2563eb", width=2.8, shape="spline", smoothing=0.8),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}σ<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=series["date"],
            y=series["tone_3m_ma"],
            mode="lines",
            name="3-period moving average",
            line=dict(color="#60a5fa", width=2, dash="dot", shape="spline", smoothing=0.7),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}σ<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=series["date"],
            y=series["tone_6m_ma"],
            mode="lines",
            name="6-period moving average",
            line=dict(color="#ef4444", width=2, dash="dash", shape="spline", smoothing=0.7),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}σ<extra></extra>",
        )
    )

    fig.add_annotation(
        xref="paper",
        x=1.01,
        yref="y",
        y=1.9,
        text="Hawkish",
        showarrow=False,
        font=dict(color="#991b1b", size=11),
    )
    fig.add_annotation(
        xref="paper",
        x=1.01,
        yref="y",
        y=-1.9,
        text="Dovish",
        showarrow=False,
        font=dict(color="#166534", size=11),
    )

    fig.update_layout(
        height=430,
        margin=dict(l=18, r=24, t=10, b=12),
        yaxis_title="Dovish ← z-score → Hawkish",
        xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.update_yaxes(zeroline=True, zerolinecolor="rgba(71,85,105,0.25)", gridcolor="rgba(15,23,42,0.08)")
    fig.update_xaxes(gridcolor="rgba(15,23,42,0.05)")
    return fig


def scorecard_dataframe(doc: pd.Series) -> pd.DataFrame:
    rows = [
        ("Tone vs Fed baseline", doc["tone_z_fed"], doc["tone_percentile_fed"], tone_bucket(doc["tone_z_fed"])),
        ("Tone vs own history", doc["tone_z_speaker"], np.nan, tone_bucket(doc["tone_z_speaker"])),
        ("Inflation emphasis", doc["inflation_z_fed"], doc["inflation_percentile_fed"], emphasis_bucket(doc["inflation_z_fed"])),
        ("Labor emphasis", doc["labor_z_fed"], doc["labor_percentile_fed"], emphasis_bucket(doc["labor_z_fed"])),
        ("Growth emphasis", doc["growth_z_fed"], doc["growth_percentile_fed"], emphasis_bucket(doc["growth_z_fed"])),
        ("Financial stability", doc["fs_z_fed"], np.nan, emphasis_bucket(doc["fs_z_fed"])),
        ("Uncertainty / caution", doc["uncertainty_z_fed"], np.nan, emphasis_bucket(doc["uncertainty_z_fed"])),
    ]
    return pd.DataFrame(rows, columns=["dimension", "z", "percentile", "bucket"])


def scorecard_figure(scorecard: pd.DataFrame) -> go.Figure:
    plot_df = scorecard.copy().sort_values("z", ascending=True)
    colors = [tone_color(z) for z in plot_df["z"]]

    fig = go.Figure(
        go.Bar(
            x=plot_df["z"],
            y=plot_df["dimension"],
            orientation="h",
            marker=dict(color=colors),
            text=[f"{z:+.2f}σ" if pd.notna(z) else "n.a." for z in plot_df["z"]],
            textposition="outside",
            hovertemplate="%{y}<br>%{x:.2f}σ<extra></extra>",
        )
    )
    fig.add_vrect(x0=-3, x1=-0.35, fillcolor="rgba(34,197,94,0.10)", line_width=0)
    fig.add_vrect(x0=0.35, x1=3, fillcolor="rgba(239,68,68,0.10)", line_width=0)
    fig.add_vline(x=-0.35, line_dash="dot", line_width=1)
    fig.add_vline(x=0.35, line_dash="dot", line_width=1)
    fig.add_vline(x=-1.25, line_dash="dash", line_width=1)
    fig.add_vline(x=1.25, line_dash="dash", line_width=1)
    fig.update_layout(
        height=370,
        margin=dict(l=18, r=18, t=10, b=12),
        xaxis_title="More dovish ← z-score → More hawkish",
        yaxis_title="",
        xaxis=dict(range=[-3, 3]),
    )
    return fig


def speaker_matrix_figure(matrix: pd.DataFrame) -> go.Figure:
    if matrix.empty:
        return go.Figure()

    marker_sizes = 12 + np.sqrt(matrix["docs"].fillna(1).to_numpy()) * 8
    customdata = np.column_stack(
        [
            matrix["avg_stance"].fillna("Unknown"),
            matrix["last_date_label"].fillna("n.a."),
            matrix["docs"].fillna(0).astype(int),
            matrix["latest_vs_own"].round(2),
            matrix["latest_vs_fed"].round(2),
        ]
    )

    fig = go.Figure(
        go.Scatter(
            x=matrix["latest_vs_fed"],
            y=matrix["speaker"],
            mode="markers",
            customdata=customdata,
            marker=dict(
                size=marker_sizes,
                color=matrix["latest_vs_own"],
                colorscale=[
                    [0.0, "#16a34a"],
                    [0.5, "#e5e7eb"],
                    [1.0, "#dc2626"],
                ],
                cmin=-1.5,
                cmax=1.5,
                line=dict(color="rgba(15,23,42,0.18)", width=1),
                colorbar=dict(title="Vs own history"),
                opacity=0.92,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Latest vs Fed: %{x:.2f}σ<br>"
                "Latest vs own: %{marker.color:.2f}σ<br>"
                "Average stance: %{customdata[0]}<br>"
                "Latest date: %{customdata[1]}<br>"
                "Documents: %{customdata[2]}<extra></extra>"
            ),
        )
    )

    fig.add_vrect(x0=-3, x1=-0.35, fillcolor="rgba(34,197,94,0.08)", line_width=0)
    fig.add_vrect(x0=0.35, x1=3, fillcolor="rgba(239,68,68,0.08)", line_width=0)
    fig.add_vline(x=0, line_width=1, line_color="rgba(71,85,105,0.35)")

    fig.update_layout(
        height=430,
        margin=dict(l=18, r=18, t=10, b=12),
        xaxis_title="More dovish ← latest vs Fed baseline → More hawkish",
        yaxis_title="",
    )
    fig.update_xaxes(gridcolor="rgba(15,23,42,0.05)")
    fig.update_yaxes(categoryorder="array", categoryarray=matrix["speaker"][::-1].tolist())
    return fig


def clean_latest_table(view: pd.DataFrame) -> pd.DataFrame:
    latest = view.sort_values("date", ascending=False).copy()
    latest["date"] = latest["date"].dt.strftime("%Y-%m-%d")
    latest["open"] = latest["url"].apply(lambda x: f'<a href="{x}" target="_blank">open</a>')
    latest["tone"] = latest["tone_z_fed"].apply(format_z)
    latest["vs own"] = latest["tone_z_speaker"].apply(format_z)
    latest["inflation"] = latest["inflation_z_fed"].apply(format_z)
    latest["labor"] = latest["labor_z_fed"].apply(format_z)
    latest["growth"] = latest["growth_z_fed"].apply(format_z)
    latest["fed %ile"] = latest["tone_percentile_fed"].apply(format_pct)
    latest["stance"] = latest["tone_z_fed"].apply(lambda z: make_badge(tone_bucket(z), z))
    latest = latest[
        ["date", "speaker", "event_type", "stance", "tone", "vs own", "fed %ile", "inflation", "labor", "growth", "open"]
    ].rename(columns={"event_type": "type"})
    return latest


def render_latest_table(view: pd.DataFrame) -> None:
    latest_table = clean_latest_table(view)
    table_html = latest_table.to_html(index=False, escape=False)
    st.markdown(f"<div class='scroll-table-wrap'>{table_html}</div>", unsafe_allow_html=True)


def doc_summary_html(doc: pd.Series) -> str:
    tone = make_badge(f"Fed tone: {tone_bucket(doc['tone_z_fed'])}", doc["tone_z_fed"])
    own = make_badge(f"Vs own: {tone_bucket(doc['tone_z_speaker'])}", doc["tone_z_speaker"])
    infl = make_badge(f"Inflation: {emphasis_bucket(doc['inflation_z_fed'])}", doc["inflation_z_fed"])
    labor = make_badge(f"Labor: {emphasis_bucket(doc['labor_z_fed'])}", doc["labor_z_fed"])
    growth = make_badge(f"Growth: {emphasis_bucket(doc['growth_z_fed'])}", doc["growth_z_fed"])

    return f"""
    <div class="summary-card">
        <div style="font-size:1.05rem;font-weight:700;margin-bottom:0.45rem">{html.escape(doc['title'] or 'Untitled')}</div>
        <div style="margin-bottom:0.55rem;color:#555">
            {html.escape(str(doc['date'].date()) if pd.notna(doc['date']) else 'Unknown date')} |
            {html.escape(doc['speaker'] or 'Unknown speaker')} |
            {html.escape((doc['event_type'] or '').title())}
        </div>
        <div style="display:flex;gap:0.4rem;flex-wrap:wrap;margin-bottom:0.6rem">
            {tone}
            {own}
            {infl}
            {labor}
            {growth}
        </div>
        <div style="font-size:0.92rem;color:#555">
            Fed baseline: {format_z(doc['tone_z_fed'])} |
            vs own history: {format_z(doc['tone_z_speaker'])} |
            Fed percentile: {format_pct(doc['tone_percentile_fed'])}
        </div>
    </div>
    """


def build_status_line(df: pd.DataFrame, view: pd.DataFrame, refresh_stats: Optional[Dict[str, int]]) -> str:
    parts = []
    if refresh_stats is not None:
        parts.append(
            f"Refresh checked {refresh_stats['attempted_docs']:,} items and added {refresh_stats['inserted_docs']:,} new documents."
        )
    min_date = df["date"].dropna().min()
    max_date = df["date"].dropna().max()
    if pd.notna(min_date) and pd.notna(max_date):
        parts.append(f"Corpus: {len(df):,} documents from {min_date.date()} to {max_date.date()}.")
    parts.append(f"Current view: {len(view):,} documents.")
    return " ".join(parts)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(APP_CSS, unsafe_allow_html=True)
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
            finally:
                progress_bar.empty()
                status_box.empty()

    df = load_documents()
    log_df = load_log()

    if df.empty:
        st.warning("No local corpus found yet.")
        render_diagnostics(log_df, refresh_stats)
        st.stop()

    df = compute_features(df)

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

    st.markdown(f"<div class='status-line'>{build_status_line(df, view, refresh_stats)}</div>", unsafe_allow_html=True)

    series = aggregate_series(view, freq="30D")
    matrix = speaker_matrix(view, top_n=12)

    top_left, top_right = st.columns([1.7, 1.05])
    with top_left:
        st.markdown("<div class='chart-title'>Fed tone over time</div>", unsafe_allow_html=True)
        st.plotly_chart(z_dashboard_figure(series), use_container_width=True)
    with top_right:
        st.markdown("<div class='chart-title'>Speaker matrix</div>", unsafe_allow_html=True)
        if matrix.empty:
            st.info("No speaker matrix available.")
        else:
            st.plotly_chart(speaker_matrix_figure(matrix), use_container_width=True)

    st.subheader("Latest communication")
    st.caption("Showing the newest documents in a fixed-height box. Scroll inside the table for the rest.")
    render_latest_table(view)

    st.subheader("Transcript underwrite")
    options = view.sort_values("date", ascending=False).copy()
    options["label"] = options.apply(
        lambda r: f"{r['date'].date() if pd.notna(r['date']) else 'n.a.'} | {r['speaker'] or 'Unknown'} | {r['title']}",
        axis=1,
    )
    selected_label = st.selectbox("Choose a document", options["label"].tolist(), index=0)
    doc = options.loc[options["label"] == selected_label].iloc[0]

    st.markdown(doc_summary_html(doc), unsafe_allow_html=True)

    upper_left, upper_right = st.columns([1.5, 1.0])

    with upper_left:
        st.markdown("**Highlighted transcript excerpt**")
        excerpt = clean_text(doc["body_text"])[:6000]
        patterns = list({p for vals in SNIPPET_PATTERNS.values() for p in vals})
        st.markdown(
            f"<div class='summary-card' style='line-height:1.65;background:#fff'>{highlight_terms(excerpt, patterns)}</div>",
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
            <div class="summary-card" style="line-height:1.7">
                <div><strong>Title</strong>: {html.escape(doc['title'] or 'Unknown')}</div>
                <div><strong>Speaker</strong>: {html.escape(doc['speaker'] or 'Unknown')}</div>
                <div><strong>Role</strong>: {html.escape(doc['role'] or 'Unknown')}</div>
                <div><strong>Date</strong>: {html.escape(str(doc['date'].date()) if pd.notna(doc['date']) else 'Unknown')}</div>
                <div><strong>Type</strong>: {html.escape((doc['event_type'] or '').title())}</div>
                <div><strong>Fed baseline</strong>: {format_z(doc['tone_z_fed'])} ({tone_bucket(doc['tone_z_fed'])})</div>
                <div><strong>Vs own history</strong>: {format_z(doc['tone_z_speaker'])} ({tone_bucket(doc['tone_z_speaker'])})</div>
                <div><strong>Fed percentile</strong>: {format_pct(doc['tone_percentile_fed'])}</div>
                <div><strong>Source</strong>: <a href="{html.escape(doc['url'])}" target="_blank">open original</a></div>
                {"<div><strong>PDF</strong>: <a href='" + html.escape(doc['pdf_url']) + "' target='_blank'>open pdf</a></div>" if doc.get("pdf_url") else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Document scorecard**")
        scorecard = scorecard_dataframe(doc)
        st.plotly_chart(scorecard_figure(scorecard), use_container_width=True)

        scorecard_display = scorecard.copy()
        scorecard_display["z"] = scorecard_display["z"].apply(format_z)
        scorecard_display["percentile"] = scorecard_display["percentile"].apply(format_pct)
        st.dataframe(scorecard_display, use_container_width=True, hide_index=True)

    st.subheader("Latest speaker tape")
    speaker_latest = (
        view.sort_values("date", ascending=False)
        .groupby("speaker", dropna=False)
        .head(1)[["speaker", "tone_z_fed", "tone_z_speaker", "stance", "date"]]
        .sort_values("tone_z_fed", ascending=False)
        .rename(columns={"tone_z_fed": "vs Fed", "tone_z_speaker": "vs own"})
    )
    speaker_latest["date"] = speaker_latest["date"].dt.strftime("%Y-%m-%d")
    speaker_latest["vs Fed"] = speaker_latest["vs Fed"].apply(format_z)
    speaker_latest["vs own"] = speaker_latest["vs own"].apply(format_z)
    st.dataframe(speaker_latest, use_container_width=True, hide_index=True)

    st.subheader("Export")
    export_cols = [
        "date",
        "speaker",
        "role",
        "event_type",
        "title",
        "stance",
        "tone_z_fed",
        "tone_z_speaker",
        "tone_percentile_fed",
        "inflation_z_fed",
        "labor_z_fed",
        "growth_z_fed",
        "financial_stability",
        "balance_sheet",
        "uncertainty_risk",
        "url",
    ]
    csv_bytes = view.sort_values("date", ascending=False)[export_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered dataset as CSV",
        data=csv_bytes,
        file_name="fed_tone_filtered.csv",
        mime="text/csv",
    )

    render_diagnostics(log_df, refresh_stats)


if __name__ == "__main__":
    main()
