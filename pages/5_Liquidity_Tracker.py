import io
import json
import time
import zipfile
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


HTTP_TIMEOUT = 20
HTTP_RETRIES = 5
HTTP_BACKOFF = 1.5


def get_http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=HTTP_RETRIES,
        connect=HTTP_RETRIES,
        read=HTTP_RETRIES,
        backoff_factor=HTTP_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
            "Connection": "keep-alive",
        }
    )
    return session


def clean_series(s: pd.Series, name: str) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()]
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.name = name
    return s


def fetch_csv_series(session: requests.Session, url: str, date_col: str, value_col: str, name: str, **read_csv_kwargs) -> pd.Series:
    r = session.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), **read_csv_kwargs)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    s = pd.Series(df[value_col].values, index=df[date_col], name=name)
    return clean_series(s, name)


def fetch_nfci_chicago_fed(session: requests.Session) -> pd.Series:
    page = session.get("https://www.chicagofed.org/research/data/nfci/current-data", timeout=HTTP_TIMEOUT)
    page.raise_for_status()

    soup = BeautifulSoup(page.text, "html.parser")
    csv_url = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(" ", strip=True).lower()
        if "nfci indexes" in text and href.lower().endswith(".csv"):
            csv_url = href
            break

    if not csv_url:
        raise RuntimeError("Could not find Chicago Fed NFCI CSV link")

    if csv_url.startswith("/"):
        csv_url = "https://www.chicagofed.org" + csv_url

    r = session.get(csv_url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))
    cols = {str(c).strip().lower(): c for c in df.columns}

    date_col = None
    value_col = None

    for c in df.columns:
        cl = str(c).strip().lower()
        if "date" in cl:
            date_col = c
        if cl == "nfci" or "national financial conditions index" in cl:
            value_col = c

    if date_col is None:
        date_col = df.columns[0]
    if value_col is None:
        # usually the main index is the second column
        value_col = df.columns[1]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    s = pd.Series(df[value_col].values, index=df[date_col], name="NFCI")
    return clean_series(s, "NFCI")


def fetch_tga_treasury(session: requests.Session) -> pd.Series:
    # Treasury Fiscal Data API, Daily Treasury Statement.
    # We use the operating cash balance / closing balance style field if available.
    base = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
    endpoint = "/v1/accounting/dts/dts_table_1"
    fields = ",".join(["record_date", "open_today_bal", "close_today_bal"])
    url = f"{base}{endpoint}?fields={fields}&page[size]=10000&sort=record_date"

    r = session.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    payload = r.json()

    data = payload.get("data", [])
    if not data:
        raise RuntimeError("Treasury API returned no data")

    df = pd.DataFrame(data)
    df["record_date"] = pd.to_datetime(df["record_date"], errors="coerce")

    # Prefer closing balance if present, otherwise opening balance.
    value_col = "close_today_bal" if "close_today_bal" in df.columns else "open_today_bal"
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # Treasury API amounts are typically reported in millions for DTS tables.
    s = pd.Series(df[value_col].values, index=df["record_date"], name="TGA_b")
    s = clean_series(s, "TGA_b") / 1000.0
    return s


def fetch_rrp_newyorkfed(session: requests.Session) -> pd.Series:
    # New York Fed markets API family exists, but the exact endpoint structure can vary.
    # The most stable low-maintenance route is to use the public operation results page data feed if available.
    # This fallback uses the FRED mirrored public csv only if NY Fed direct parsing is unavailable.
    #
    # If you want fully direct NY Fed parsing, wire the exact API endpoint once and leave the rest of this tool unchanged.
    try:
        api_url = "https://markets.newyorkfed.org/api/rp/reverserepo/all/results/search.json"
        r = session.get(api_url, timeout=HTTP_TIMEOUT)
        if r.ok:
            payload = r.json()
            results = payload.get("repo", {}).get("operations", []) or payload.get("operations", [])
            if results:
                rows = []
                for x in results:
                    dt = x.get("operationDate") or x.get("date")
                    amt = x.get("submitted") or x.get("accepted") or x.get("totalAmtAccepted")
                    rows.append((dt, amt))
                df = pd.DataFrame(rows, columns=["date", "value"])
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                s = pd.Series(df["value"].values, index=df["date"], name="RRP_b")
                return clean_series(s, "RRP_b")
    except Exception:
        pass

    # fallback: public mirrored series
    fred_csv = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=RRPONTSYD"
    return fetch_csv_series(session, fred_csv, "DATE", "RRPONTSYD", "RRP_b")


def fetch_walcl_fed(session: requests.Session) -> pd.Series:
    # Federal Reserve H.4.1 Data Download link exists on the release page.
    # In practice, the lowest-maintenance implementation is the public csv mirror for the time series.
    # If you later want the full Board DDP zip parser, you can swap just this function.
    fred_csv = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WALCL"
    s = fetch_csv_series(session, fred_csv, "DATE", "WALCL", "WALCL_b")
    return s / 1000.0  # millions -> billions


def build_liquidity_dataframe(start_date: Optional[str] = None) -> pd.DataFrame:
    session = get_http_session()

    walcl = fetch_walcl_fed(session)
    rrp = fetch_rrp_newyorkfed(session)
    tga = fetch_tga_treasury(session)
    nfci = fetch_nfci_chicago_fed(session)

    df = pd.concat([walcl, rrp, tga, nfci], axis=1).sort_index().ffill()

    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        df = df[df.index >= start_ts]

    df["NetLiq_b"] = df["WALCL_b"] - df["RRP_b"] - df["TGA_b"]
    return df
