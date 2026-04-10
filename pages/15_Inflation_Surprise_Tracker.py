import re
import time
import math
import html
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
REQUEST_TIMEOUT = 25
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
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
    ],
    "dovish": [
        "disinflation", "cooling labor market", "softening labor market", "downside risk",
        "growth is slowing", "economic slowdown", "below-trend growth", "normalizing inflation",
        "further progress on inflation", "policy can respond", "room to ease", "easing",
        "rate cuts", "cut rates", "lower rates", "less restrictive", "downward path",
        "labor market is moderating", "balanced risks", "two-sided risks", "weak demand",
        "headwinds", "slack", "unemployment is rising", "financial conditions tightened",
        "act as appropriate", "support the labor market", "downside risks to employment",
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
    "watch live", "for media inquiries", "last update", "return to text", "pdf",
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
    conn.commit()
    conn.close()


def db_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\xa0", " ").strip()
    return text


def normalize_phrase(phrase: str) -> str:
    return re.escape(phrase.lower())


def count_phrase_hits(text: str, phrases: List[str]) -> int:
    low = text.lower()
    hits = 0
    for phrase in phrases:
        pattern = normalize_phrase(phrase)
        hits += len(re.findall(pattern, low))
    return hits


def score_text(text: str) -> ToneResult:
    low = clean_text(text).lower()
    words = re.findall(r"\b[a-z][a-z\-']+\b", low)
    word_count = max(len(words), 1)

    raw = {k: count_phrase_hits(low, v) for k, v in LEXICON.items()}
    scaled = {k: (v / word_count) * 1000.0 for k, v in raw.items()}

    hawkish = scaled.get("hawkish", 0.0)
    dovish = scaled.get("dovish", 0.0)

    inflation_weight = scaled.get("inflation_concern", 0.0)
    labor_weight = scaled.get("labor_concern", 0.0)
    growth_weight = scaled.get("growth_concern", 0.0)
    fs_weight = scaled.get("financial_stability", 0.0)
    bs_weight = scaled.get("balance_sheet", 0.0)
    uncertainty_weight = scaled.get("uncertainty_risk", 0.0)

    net = (
        hawkish
        - dovish
        + 0.15 * inflation_weight
        - 0.10 * labor_weight
        - 0.10 * growth_weight
        - 0.05 * fs_weight
    )

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


def fetch_html(session: requests.Session, url: str) -> BeautifulSoup:
    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def parse_year_links(index_soup: BeautifulSoup) -> List[Tuple[str, str, int]]:
    out = []
    heading_map = {"Speeches": "speech", "Testimony": "testimony"}
    for h4 in index_soup.find_all(["h3", "h4"]):
        section = clean_text(h4.get_text(" ", strip=True))
        if section not in heading_map:
            continue
        ul = h4.find_next(lambda tag: tag.name in ["ul", "div"])
        if ul is None:
            continue
        for a in ul.find_all("a", href=True):
            year_text = clean_text(a.get_text(" ", strip=True))
            if not year_text.isdigit():
                continue
            year = int(year_text)
            href = urljoin(BASE_URL, a["href"])
            out.append((heading_map[section], href, year))
    return out


def parse_index_items(year_soup: BeautifulSoup, event_type: str, year: int) -> List[Dict[str, str]]:
    items = []
    body_text = year_soup.get_text("\n", strip=True)
    lines = [clean_text(x) for x in body_text.split("\n") if clean_text(x)]

    anchors = []
    for a in year_soup.find_all("a", href=True):
        href = a.get("href", "")
        title = clean_text(a.get_text(" ", strip=True))
        if not href or not title:
            continue
        if "/newsevents/" not in href and not href.startswith("/"):
            continue
        if any(bad in title.lower() for bad in STOP_PHRASES):
            continue
        full_url = urljoin(BASE_URL, href)
        if re.search(r"/newsevents/(speech|testimony)/", full_url):
            anchors.append((title, full_url))

    if not anchors:
        return items

    for title, full_url in anchors:
        date_val = None
        speaker = None
        role = None
        venue = None

        title_idx = None
        for idx, line in enumerate(lines):
            if line == title:
                title_idx = idx
                break
        if title_idx is not None:
            context = lines[max(0, title_idx - 4): min(len(lines), title_idx + 8)]
            for ctx in context:
                if re.search(r"[A-Z][a-z]+ \d{1,2}, \d{4}", ctx):
                    date_val = re.search(r"[A-Z][a-z]+ \d{1,2}, \d{4}", ctx).group(0)
                if ctx.startswith("Federal Reserve") or "Board of Governors" in ctx:
                    role = ctx
                if ctx.startswith("At ") or ctx.startswith("Before ") or ctx.startswith("At the "):
                    venue = ctx

            for j in range(1, 5):
                if title_idx + j < len(lines):
                    maybe = lines[title_idx + j]
                    if not speaker and re.match(r"^[A-Z][A-Za-z\.\-\' ]{5,}$", maybe) and len(maybe.split()) <= 6:
                        if "federal reserve" not in maybe.lower() and "board of governors" not in maybe.lower():
                            speaker = maybe
                            break

        items.append(
            {
                "event_type": event_type,
                "year": year,
                "title": title,
                "url": full_url,
                "date_hint": date_val,
                "speaker_hint": speaker,
                "role_hint": role,
                "venue_hint": venue,
            }
        )

    dedup = {}
    for item in items:
        dedup[item["url"]] = item
    return list(dedup.values())


def extract_pdf_url(soup: BeautifulSoup, page_url: str) -> Optional[str]:
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        text = clean_text(a.get_text(" ", strip=True)).lower()
        if href.endswith(".pdf") or text == "pdf" or "pdf" in text:
            return urljoin(page_url, a["href"])
    return None


def extract_body_text(soup: BeautifulSoup) -> str:
    selectors = [
        "div.col-xs-12.col-sm-8.col-md-8",
        "div#article",
        "div.eventlist",
        "div.col-xs-12.col-md-8",
        "main",
        "article",
    ]
    chunks = []
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            paras = node.find_all(["p", "div"])
            for p in paras:
                txt = clean_text(p.get_text(" ", strip=True))
                if txt and len(txt.split()) >= 4 and not any(bad in txt.lower() for bad in STOP_PHRASES):
                    chunks.append(txt)
            if chunks:
                break

    if not chunks:
        paras = soup.find_all("p")
        for p in paras:
            txt = clean_text(p.get_text(" ", strip=True))
            if txt and len(txt.split()) >= 4 and not any(bad in txt.lower() for bad in STOP_PHRASES):
                chunks.append(txt)

    body = "\n\n".join(chunks)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body


def extract_meta_line(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    result = {"date": None, "speaker": None, "role": None, "venue": None, "title": None}
    title_node = soup.find(["h3", "h1"])
    if title_node:
        result["title"] = clean_text(title_node.get_text(" ", strip=True))

    text_lines = [clean_text(t) for t in soup.get_text("\n", strip=True).split("\n") if clean_text(t)]

    for idx, line in enumerate(text_lines[:80]):
        if result["date"] is None:
            m = re.search(r"([A-Z][a-z]+ \d{1,2}, \d{4})", line)
            if m:
                result["date"] = m.group(1)
        if result["speaker"] is None and re.match(r"^[A-Z][A-Za-z\.\-\' ]{5,}$", line):
            if 2 <= len(line.split()) <= 6 and "federal reserve" not in line.lower():
                prev = text_lines[max(0, idx - 2): idx]
                nxt = text_lines[idx + 1: idx + 3]
                around = " ".join(prev + nxt).lower()
                if "chair" in around or "governor" in around or "president" in around or "board of governors" in around:
                    result["speaker"] = line
        if result["role"] is None and (
            "board of governors" in line.lower()
            or line.lower().startswith("chair ")
            or line.lower().startswith("governor ")
            or "federal reserve bank of" in line.lower()
        ):
            result["role"] = line
        if result["venue"] is None and (
            line.startswith("At ")
            or line.startswith("Before ")
            or line.startswith("At the ")
            or line.startswith("Before the ")
        ):
            result["venue"] = line

    return result


def merge_preferred(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if v and str(v).strip():
            return clean_text(str(v))
    return None


def parse_document(session: requests.Session, item: Dict[str, str]) -> Dict[str, Optional[str]]:
    soup = fetch_html(session, item["url"])
    meta = extract_meta_line(soup)
    pdf_url = extract_pdf_url(soup, item["url"])
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
        "pdf_url": pdf_url,
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
            doc["url"], doc["event_type"], doc["year"], doc["date"], doc["title"], doc["speaker"],
            doc["role"], doc["venue"], doc["pdf_url"], doc["body_text"], doc["word_count"],
            doc["scraped_at"], doc["hawkish_score"], doc["dovish_score"], doc["net_score"],
            doc["inflation_concern"], doc["labor_concern"], doc["growth_concern"],
            doc["financial_stability"], doc["balance_sheet"], doc["uncertainty_risk"]
        ),
    )
    conn.commit()


def get_existing_urls(conn: sqlite3.Connection) -> set:
    cur = conn.cursor()
    cur.execute("SELECT url FROM documents")
    return {row[0] for row in cur.fetchall()}


@st.cache_data(show_spinner=False, ttl=3600)
def crawl_fed(max_years: int = 6, force_refresh: bool = False, progress: Optional[st.delta_generator.DeltaGenerator] = None) -> pd.DataFrame:
    init_db()
    session = make_session()
    index_soup = fetch_html(session, INDEX_URL)
    year_links = parse_year_links(index_soup)
    year_links = sorted(year_links, key=lambda x: x[2], reverse=True)[: max_years * 2]

    conn = db_connection()
    existing = get_existing_urls(conn)

    all_items = []
    for event_type, url, year in year_links:
        try:
            year_soup = fetch_html(session, url)
            items = parse_index_items(year_soup, event_type, year)
            all_items.extend(items)
            time.sleep(0.2)
        except Exception:
            continue

    all_items = sorted({item["url"]: item for item in all_items}.values(), key=lambda x: (x["year"], x["url"]), reverse=True)

    total = len(all_items)
    done = 0
    for item in all_items:
        done += 1
        if progress is not None:
            progress.info(f"Scanning {done:,}/{total:,}: {item['title'][:110]}")
        if item["url"] in existing and not force_refresh:
            continue
        try:
            doc = parse_document(session, item)
            if doc["body_text"] and doc["word_count"] >= 80:
                upsert_document(conn, doc)
            time.sleep(0.15)
        except Exception:
            continue

    conn.close()
    return load_documents()


@st.cache_data(show_spinner=False, ttl=600)
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
        "word_count", "hawkish_score", "dovish_score", "net_score", "inflation_concern",
        "labor_concern", "growth_concern", "financial_stability", "balance_sheet", "uncertainty_risk",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)
    return df


def classify_stance(net_score: float) -> str:
    if pd.isna(net_score):
        return "Unknown"
    if net_score >= 0.35:
        return "Hawkish"
    if net_score <= -0.35:
        return "Dovish"
    return "Neutral"


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["stance"] = out["net_score"].apply(classify_stance)

    speaker_mean = out.groupby("speaker", dropna=False)["net_score"].transform("mean")
    out["speaker_delta"] = out["net_score"] - speaker_mean

    out["speaker_weight"] = out["speaker"].map(SPEAKER_WEIGHTS).fillna(1.0)
    out["weighted_net"] = out["net_score"] * out["speaker_weight"]
    out["weighted_hawkish"] = out["hawkish_score"] * out["speaker_weight"]
    out["weighted_dovish"] = out["dovish_score"] * out["speaker_weight"]

    return out


def aggregate_series(df: pd.DataFrame, freq: str = "30D") -> pd.DataFrame:
    if df.empty:
        return df

    frame = df.copy().dropna(subset=["date"])
    if frame.empty:
        return frame

    frame["bucket"] = frame["date"].dt.to_period(freq).dt.start_time
    grouped = frame.groupby("bucket", as_index=False).apply(
        lambda x: pd.Series(
            {
                "tone_composite": np.average(x["net_score"], weights=x["speaker_weight"]),
                "hawkish_composite": np.average(x["hawkish_score"], weights=x["speaker_weight"]),
                "dovish_composite": np.average(x["dovish_score"], weights=x["speaker_weight"]),
                "documents": len(x),
            }
        )
    )
    grouped = grouped.rename(columns={"bucket": "date"}).reset_index(drop=True)
    return grouped


def latest_snapshot(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "tone": np.nan,
            "delta_30d": np.nan,
            "core_tone": np.nan,
        }

    window = aggregate_series(df, freq="30D")
    tone = window["tone_composite"].iloc[-1] if not window.empty else np.nan
    delta_30d = np.nan
    if len(window) >= 2:
        delta_30d = tone - window["tone_composite"].iloc[-2]

    core = df[df["speaker"].isin(["Jerome H. Powell", "Philip N. Jefferson", "John C. Williams", "Christopher J. Waller"])]
    if core.empty:
        core_tone = np.nan
    else:
        core_tone = np.average(core["net_score"], weights=core["speaker_weight"])

    return {"tone": tone, "delta_30d": delta_30d, "core_tone": core_tone}


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
            tone=("net_score", "mean"),
            hawkish=("hawkish_score", "mean"),
            dovish=("dovish_score", "mean"),
            last_date=("date", "max"),
        )
        .reset_index()
    )
    agg["stance"] = agg["tone"].apply(classify_stance)

    deltas = (
        df.sort_values("date")
          .groupby("speaker", dropna=False)
          .tail(1)[["speaker", "speaker_delta", "stance"]]
    )
    agg = agg.merge(deltas, on="speaker", how="left")
    agg["weight"] = agg["speaker"].map(SPEAKER_WEIGHTS).fillna(1.0)
    agg = agg.sort_values(["weight", "docs", "tone"], ascending=[False, False, False]).head(top_n)
    return agg


def render_about():
    with st.sidebar.expander("About This Tool", expanded=False):
        st.write(
            "This dashboard scrapes Federal Reserve speeches and testimony directly from the Board's website, "
            "scores each document across hawkish and dovish dimensions, and tracks how each speaker's tone shifts over time. "
            "The output is meant to help underwrite the reaction function, not replace reading the original text."
        )
        st.write(
            "Primary source: Federal Reserve speeches and testimony pages. Scoring is rules-based and transparent so you can inspect the passages driving the result."
        )


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Primary-source underwriting of Fed communication tone from speeches and testimony.")
    render_about()
    init_db()

    with st.sidebar:
        st.subheader("Controls")
        years_to_pull = st.slider("Years to scan from latest backward", min_value=1, max_value=20, value=6)
        force_refresh = st.checkbox("Force refresh existing documents", value=False)
        if st.button("Refresh corpus", use_container_width=True):
            progress_box = st.empty()
            with st.spinner("Refreshing Fed corpus..."):
                df = crawl_fed(max_years=years_to_pull, force_refresh=force_refresh, progress=progress_box)
            progress_box.empty()
            st.success(f"Loaded {len(df):,} documents.")

    df = load_documents()
    if df.empty:
        st.info("No local corpus found yet. Click 'Refresh corpus' in the sidebar to build the database.")
        st.stop()

    df = compute_features(df)

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    with st.sidebar:
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        speakers = sorted([x for x in df["speaker"].dropna().unique().tolist() if x])
        selected_speakers = st.multiselect("Speakers", options=speakers, default=[])
        event_types = st.multiselect("Event types", options=sorted(df["event_type"].dropna().unique()), default=sorted(df["event_type"].dropna().unique()))
        search = st.text_input("Search title or transcript")

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start_date, end_date = df["date"].min(), df["date"].max()

    view = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
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

    snapshot = latest_snapshot(view)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tone composite", f"{snapshot['tone']:.2f}" if pd.notna(snapshot["tone"]) else "n.a.")
    c2.metric("30-day change", f"{snapshot['delta_30d']:+.2f}" if pd.notna(snapshot["delta_30d"]) else "n.a.")
    c3.metric("Core speaker tone", f"{snapshot['core_tone']:.2f}" if pd.notna(snapshot["core_tone"]) else "n.a.")
    c4.metric("Documents in view", f"{len(view):,}")

    series = aggregate_series(view, freq="30D")
    left, right = st.columns([1.7, 1.1])

    with left:
        if not series.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series["date"], y=series["tone_composite"], mode="lines+markers", name="Tone composite"))
            fig.add_trace(go.Scatter(x=series["date"], y=series["hawkish_composite"], mode="lines", name="Hawkish"))
            fig.add_trace(go.Scatter(x=series["date"], y=series["dovish_composite"], mode="lines", name="Dovish"))
            fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), title="Fed tone over time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No time series available for the current filters.")

    with right:
        matrix = speaker_matrix(view, top_n=12)
        if not matrix.empty:
            heat = px.scatter(
                matrix,
                x="tone",
                y="speaker",
                size="docs",
                color="speaker_delta",
                hover_data=["stance", "last_date", "docs"],
                title="Speaker matrix",
                height=420,
            )
            heat.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(heat, use_container_width=True)
        else:
            st.info("No speaker matrix available.")

    st.subheader("Latest communication")
    latest_cols = [
        "date", "speaker", "event_type", "title", "stance", "net_score", "speaker_delta",
        "inflation_concern", "labor_concern", "growth_concern", "url"
    ]
    latest_view = view.sort_values("date", ascending=False)[latest_cols].copy()
    latest_view = latest_view.rename(columns={
        "net_score": "tone",
        "speaker_delta": "delta vs speaker baseline",
        "inflation_concern": "inflation",
        "labor_concern": "labor",
        "growth_concern": "growth",
    })
    st.dataframe(latest_view, use_container_width=True, hide_index=True)

    st.subheader("Transcript underwrite")
    options = view.sort_values("date", ascending=False).copy()
    options["label"] = options.apply(
        lambda r: f"{r['date'].date() if pd.notna(r['date']) else 'n.a.'} | {r['speaker'] or 'Unknown'} | {r['title']}", axis=1
    )
    if options.empty:
        st.info("No documents match the current filters.")
        st.stop()

    selected_label = st.selectbox("Choose a document", options["label"].tolist(), index=0)
    doc = options.loc[options["label"] == selected_label].iloc[0]

    d1, d2 = st.columns([1.2, 1.8])
    with d1:
        st.markdown(f"**Title**: {doc['title']}")
        st.markdown(f"**Speaker**: {doc['speaker'] or 'Unknown'}")
        st.markdown(f"**Role**: {doc['role'] or 'Unknown'}")
        st.markdown(f"**Date**: {doc['date'].date() if pd.notna(doc['date']) else 'Unknown'}")
        st.markdown(f"**Event type**: {doc['event_type'].title()}")
        if doc.get("venue"):
            st.markdown(f"**Venue**: {doc['venue']}")
        st.markdown(f"**Tone**: {doc['stance']} ({doc['net_score']:.2f})")
        st.markdown(f"**Delta vs speaker baseline**: {doc['speaker_delta']:+.2f}" if pd.notna(doc['speaker_delta']) else "**Delta vs speaker baseline**: n.a.")
        st.markdown(f"**Source**: [Open original]({doc['url']})")
        if doc.get("pdf_url"):
            st.markdown(f"**PDF**: [Open PDF]({doc['pdf_url']})")

        scorecard = pd.DataFrame(
            {
                "dimension": ["hawkish", "dovish", "inflation", "labor", "growth", "financial stability", "balance sheet", "uncertainty"],
                "score": [
                    doc["hawkish_score"], doc["dovish_score"], doc["inflation_concern"], doc["labor_concern"],
                    doc["growth_concern"], doc["financial_stability"], doc["balance_sheet"], doc["uncertainty_risk"]
                ],
            }
        )
        bar = px.bar(scorecard, x="score", y="dimension", orientation="h", height=320, title="Document scorecard")
        bar.update_layout(margin=dict(l=20, r=20, t=40, b=20), yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(bar, use_container_width=True)

    with d2:
        hawk_snips = find_snippets(doc["body_text"], SNIPPET_PATTERNS["hawkish"], max_snippets=4)
        dove_snips = find_snippets(doc["body_text"], SNIPPET_PATTERNS["dovish"], max_snippets=4)
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

        st.markdown("**Highlighted transcript excerpt**")
        excerpt = clean_text(doc["body_text"])[:6000]
        patterns = list({p for vals in SNIPPET_PATTERNS.values() for p in vals})
        st.markdown(
            f"<div style='line-height:1.6'>{highlight_terms(excerpt, patterns)}</div>",
            unsafe_allow_html=True,
        )

    st.subheader("Export")
    export_cols = [
        "date", "speaker", "role", "event_type", "title", "stance", "net_score", "speaker_delta",
        "hawkish_score", "dovish_score", "inflation_concern", "labor_concern", "growth_concern",
        "financial_stability", "balance_sheet", "uncertainty_risk", "url"
    ]
    csv_bytes = view.sort_values("date", ascending=False)[export_cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered dataset as CSV", data=csv_bytes, file_name="fed_tone_filtered.csv", mime="text/csv")


if __name__ == "__main__":
    main()
