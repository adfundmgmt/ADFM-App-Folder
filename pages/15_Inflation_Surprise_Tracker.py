bash -lc cat > /mnt/data/fed_tone_dashboard.py <<'PY'
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
            for back in range(max(0, title_idx - 3), title_idx):
                if re.match(r"\d{1,2}/\d{1,2}/\d{4}$", lines[back]):
                    date_val = pd.to_datetime(lines[back], errors="coerce")
                    if pd.notna(date_val):
                        date_val = date_val.strftime("%Y-%m-%d")
                        break
            for fwd in range(title_idx + 1, min(len(lines), title_idx + 6)):
                line = lines[fwd]
                if line.startswith(("Chair ", "Vice Chair", "Governor ", "President ")):
                    role = line
                    speaker = re.sub(r"^(Chair|Vice Chair(?: for Supervision)?|Governor|President)\s+", "", line).strip()
                    if speaker.startswith("for Supervision "):
                        speaker = speaker.replace("for Supervision ", "")
                elif role and venue is None and line != role:
                    venue = line
                    break

        items.append(
            {
                "url": full_url,
                "event_type": event_type,
                "year": year,
                "date": date_val,
                "title": title,
                "speaker": speaker,
                "role": role,
                "venue": venue,
            }
        )

    dedup = {x["url"]: x for x in items}
    return list(dedup.values())


def extract_body_text(page_soup: BeautifulSoup) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    title = None
    date_val = None
    speaker = None
    role = None
    pdf_url = None

    if page_soup.find("h3", class_="title"):
        title = clean_text(page_soup.find("h3", class_="title").get_text(" ", strip=True))
    elif page_soup.find(["h1", "h2", "h3"]):
        title = clean_text(page_soup.find(["h1", "h2", "h3"]).get_text(" ", strip=True))

    for p in page_soup.find_all(["p", "div"]):
        text = clean_text(p.get_text(" ", strip=True))
        if re.match(r"^[A-Z][a-z]+\s+\d{1,2},\s+\d{4}$", text):
            parsed = pd.to_datetime(text, errors="coerce")
            if pd.notna(parsed):
                date_val = parsed.strftime("%Y-%m-%d")
                break

    page_text = page_soup.get_text("\n", strip=True)
    lines = [clean_text(x) for x in page_text.split("\n") if clean_text(x)]
    for i, line in enumerate(lines[:50]):
        if line.startswith(("Chair ", "Vice Chair", "Governor ", "President ")):
            role = line
            speaker = re.sub(r"^(Chair|Vice Chair(?: for Supervision)?|Governor|President)\s+", "", line).strip()
            if speaker.startswith("for Supervision "):
                speaker = speaker.replace("for Supervision ", "")
            break

    pdf_anchor = page_soup.find("a", href=re.compile(r"\.pdf($|\?)", re.I))
    if pdf_anchor and pdf_anchor.get("href"):
        pdf_url = urljoin(BASE_URL, pdf_anchor["href"])

    content_candidates = []
    selectors = [
        "#article", ".article", ".col-xs-12.col-sm-8.col-md-8", ".col-md-8", "main",
    ]
    for sel in selectors:
        for node in page_soup.select(sel):
            txt = clean_text(node.get_text("\n", strip=True))
            if len(txt.split()) > 300:
                content_candidates.append(txt)
    if not content_candidates:
        paras = [clean_text(p.get_text(" ", strip=True)) for p in page_soup.find_all("p")]
        paras = [p for p in paras if len(p.split()) > 8]
        text = "\n\n".join(paras)
    else:
        text = max(content_candidates, key=len)

    text = re.sub(r"\b(Return to text|For media inquiries.*)$", "", text, flags=re.I | re.S)
    return clean_text(text), title, date_val, speaker, role, pdf_url


def upsert_documents(rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    conn = db_connection()
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO documents (
                url, event_type, year, date, title, speaker, role, venue, pdf_url, body_text,
                word_count, scraped_at, hawkish_score, dovish_score, net_score,
                inflation_concern, labor_concern, growth_concern, financial_stability,
                balance_sheet, uncertainty_risk
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
                r["url"], r["event_type"], r["year"], r["date"], r["title"], r["speaker"], r["role"],
                r["venue"], r["pdf_url"], r["body_text"], r["word_count"], r["scraped_at"],
                r["hawkish_score"], r["dovish_score"], r["net_score"], r["inflation_concern"],
                r["labor_concern"], r["growth_concern"], r["financial_stability"], r["balance_sheet"],
                r["uncertainty_risk"],
            ),
        )
    conn.commit()
    conn.close()


def load_documents() -> pd.DataFrame:
    conn = db_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM documents", conn)
    finally:
        conn.close()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["weight"] = df["speaker"].map(SPEAKER_WEIGHTS).fillna(1.0)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def crawl_fed(max_years: Optional[int] = None, force_refresh: bool = False, progress=None) -> pd.DataFrame:
    init_db()
    session = make_session()
    index_soup = fetch_html(session, INDEX_URL)
    year_links = parse_year_links(index_soup)
    year_links = sorted(year_links, key=lambda x: x[2], reverse=True)
    if max_years is not None:
        allowed_years = sorted(list({y for _, _, y in year_links}), reverse=True)[:max_years]
        year_links = [x for x in year_links if x[2] in allowed_years]

    existing = load_documents()
    existing_urls = set(existing["url"].tolist()) if not existing.empty else set()

    metadata_rows = []
    total = len(year_links)
    for idx, (event_type, year_url, year) in enumerate(year_links, start=1):
        if progress:
            progress.info(f"Scanning {event_type} index for {year} ({idx}/{total})")
        year_soup = fetch_html(session, year_url)
        metadata_rows.extend(parse_index_items(year_soup, event_type, year))
        time.sleep(0.15)

    rows_to_fetch = []
    for row in metadata_rows:
        if force_refresh or row["url"] not in existing_urls:
            rows_to_fetch.append(row)

    fetched_rows = []
    total_fetch = max(len(rows_to_fetch), 1)
    for idx, row in enumerate(rows_to_fetch, start=1):
        if progress:
            progress.info(f"Parsing document {idx}/{total_fetch}: {row['title'][:90]}")
        try:
            page_soup = fetch_html(session, row["url"])
            body_text, title, date_val, speaker, role, pdf_url = extract_body_text(page_soup)
            if len(body_text.split()) < 120:
                continue
            scored = score_text(body_text)
            row_out = {
                **row,
                "title": title or row.get("title"),
                "date": date_val or row.get("date"),
                "speaker": speaker or row.get("speaker"),
                "role": role or row.get("role"),
                "pdf_url": pdf_url,
                "body_text": body_text,
                "word_count": scored.word_count,
                "scraped_at": pd.Timestamp.utcnow().isoformat(),
                "hawkish_score": scored.hawkish_score,
                "dovish_score": scored.dovish_score,
                "net_score": scored.net_score,
                "inflation_concern": scored.inflation_concern,
                "labor_concern": scored.labor_concern,
                "growth_concern": scored.growth_concern,
                "financial_stability": scored.financial_stability,
                "balance_sheet": scored.balance_sheet,
                "uncertainty_risk": scored.uncertainty_risk,
            }
            fetched_rows.append(row_out)
            time.sleep(0.2)
        except Exception:
            continue

    if fetched_rows:
        upsert_documents(fetched_rows)
    return load_documents()


def classify_stance(net_score: float, hawkish: float, dovish: float) -> str:
    if net_score >= 0.6 or hawkish - dovish >= 0.4:
        return "Hawkish"
    if net_score <= -0.4 or dovish - hawkish >= 0.3:
        return "Dovish"
    return "Balanced"


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out = out.sort_values(["speaker", "date"])
    out["speaker_baseline"] = out.groupby("speaker")["net_score"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    out["speaker_delta"] = out["net_score"] - out["speaker_baseline"].fillna(0.0)
    out["stance"] = out.apply(lambda r: classify_stance(r["net_score"], r["hawkish_score"], r["dovish_score"]), axis=1)
    out["relevance_weight"] = out["weight"] * np.where(out["event_type"].eq("testimony"), 1.15, 1.0)
    out["weighted_net"] = out["net_score"] * out["relevance_weight"]
    out["weighted_hawkish"] = out["hawkish_score"] * out["relevance_weight"]
    out["weighted_dovish"] = out["dovish_score"] * out["relevance_weight"]
    out = out.sort_values("date").reset_index(drop=True)
    return out


def aggregate_series(df: pd.DataFrame, freq: str = "30D") -> pd.DataFrame:
    if df.empty:
        return df
    temp = df.copy()
    temp = temp.dropna(subset=["date"])
    temp = temp.set_index("date")
    grouped = temp.resample(freq).apply(
        {
            "weighted_net": "sum",
            "weighted_hawkish": "sum",
            "weighted_dovish": "sum",
            "relevance_weight": "sum",
            "inflation_concern": "mean",
            "labor_concern": "mean",
            "growth_concern": "mean",
            "financial_stability": "mean",
            "uncertainty_risk": "mean",
            "title": "count",
        }
    )
    grouped = grouped.rename(columns={"title": "doc_count"}).reset_index()
    for col in ["weighted_net", "weighted_hawkish", "weighted_dovish"]:
        grouped[col] = np.where(grouped["relevance_weight"] > 0, grouped[col] / grouped["relevance_weight"], np.nan)
    grouped = grouped.rename(
        columns={
            "weighted_net": "tone_composite",
            "weighted_hawkish": "hawkish_composite",
            "weighted_dovish": "dovish_composite",
        }
    )
    return grouped


def find_snippets(text: str, patterns: List[str], max_snippets: int = 3) -> List[str]:
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", clean_text(text))
    hits = []
    for sent in sentences:
        low = sent.lower()
        if any(p in low for p in patterns) and 30 <= len(sent) <= 400:
            hits.append(sent.strip())
    seen = []
    for h in hits:
        if h not in seen:
            seen.append(h)
    return seen[:max_snippets]


def highlight_terms(text: str, patterns: List[str]) -> str:
    escaped = text
    for phrase in sorted(patterns, key=len, reverse=True):
        escaped = re.sub(
            f"({re.escape(phrase)})",
            r"<mark>\1</mark>",
            escaped,
            flags=re.IGNORECASE,
        )
    return escaped


def latest_snapshot(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"tone": np.nan, "delta_30d": np.nan, "core_tone": np.nan}
    end = df["date"].max()
    start_30 = end - pd.Timedelta(days=30)
    current = df[df["date"] >= start_30]
    prior = df[(df["date"] < start_30) & (df["date"] >= start_30 - pd.Timedelta(days=30))]
    tone = np.average(current["net_score"], weights=current["relevance_weight"]) if not current.empty else np.nan
    prior_tone = np.average(prior["net_score"], weights=prior["relevance_weight"]) if not prior.empty else np.nan
    core = current[current["speaker"].isin(["Jerome H. Powell", "Philip N. Jefferson", "John C. Williams", "Christopher J. Waller"])]
    core_tone = np.average(core["net_score"], weights=core["relevance_weight"]) if not core.empty else np.nan
    return {
        "tone": tone,
        "delta_30d": tone - prior_tone if pd.notna(tone) and pd.notna(prior_tone) else np.nan,
        "core_tone": core_tone,
    }


def speaker_matrix(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    if df.empty:
        return df
    last_date = df["date"].max()
    recent = df[df["date"] >= last_date - pd.Timedelta(days=180)].copy()
    if recent.empty:
        recent = df.copy()
    agg = recent.groupby("speaker", dropna=False).agg(
        docs=("title", "count"),
        tone=("net_score", "mean"),
        hawkish=("hawkish_score", "mean"),
        dovish=("dovish_score", "mean"),
        last_date=("date", "max"),
    ).reset_index()
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
PY
python -m py_compile /mnt/data/fed_tone_dashboard.py
