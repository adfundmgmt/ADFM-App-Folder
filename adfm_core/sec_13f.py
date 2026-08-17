"""Official SEC Form 13F data loading and exposure-ranking helpers.

The SEC publishes one bulk archive for each quarterly filing window.  This
module turns the large tab-delimited archive into a small filing index plus a
compressed holdings parquet file, then applies Form 13F amendment semantics
before calculating manager-level portfolio weights.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Sequence
from urllib.parse import urljoin, urlparse
from zipfile import ZipFile

import pandas as pd
import pyarrow as pa
import pyarrow.csv as arrow_csv
import pyarrow.parquet as parquet
import requests
from bs4 import BeautifulSoup

SEC_13F_DATASETS_PAGE = (
    "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
)
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_ARCHIVES_ROOT = "https://www.sec.gov/Archives/edgar/data"
DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[1] / "data" / "13f"
DEFAULT_SEC_USER_AGENT = "ADFM Analytics 13F Browser/1.0 (public-data research)"
REQUIRED_ARCHIVE_FILES = (
    "SUBMISSION.TSV",
    "COVERPAGE.TSV",
    "SUMMARYPAGE.TSV",
    "INFOTABLE.TSV",
)

_COMPANY_SUFFIXES = {
    "CO",
    "COMPANY",
    "CORP",
    "CORPORATION",
    "INC",
    "INCORPORATED",
    "LIMITED",
    "LLC",
    "LTD",
    "NV",
    "PLC",
    "SA",
    "THE",
}


class Sec13FError(RuntimeError):
    """Raised when SEC data cannot be discovered, prepared, or interpreted."""


@dataclass(frozen=True)
class QuarterDataset:
    """One official SEC quarterly Form 13F bulk archive."""

    slug: str
    label: str
    url: str
    size_label: str = ""


@dataclass(frozen=True)
class PreparedDataset:
    """Local, query-ready representation of a Form 13F archive."""

    slug: str
    label: str
    source_url: str
    cache_dir: Path
    filings_path: Path
    holdings_path: Path
    securities_path: Path
    metadata_path: Path
    prepared_at: str
    holdings_rows: int

    @classmethod
    def from_metadata(cls, metadata_path: Path) -> "PreparedDataset":
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        cache_dir = metadata_path.parent
        return cls(
            slug=str(payload["slug"]),
            label=str(payload["label"]),
            source_url=str(payload["source_url"]),
            cache_dir=cache_dir,
            filings_path=cache_dir / str(payload["filings_file"]),
            holdings_path=cache_dir / str(payload["holdings_file"]),
            securities_path=cache_dir / str(payload["securities_file"]),
            metadata_path=metadata_path,
            prepared_at=str(payload["prepared_at"]),
            holdings_rows=int(payload["holdings_rows"]),
        )


def _cache_root(cache_root: Path | str | None = None) -> Path:
    configured = os.getenv("ADFM_13F_CACHE_DIR")
    root = Path(cache_root or configured or DEFAULT_CACHE_ROOT).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def sec_headers(user_agent: str | None = None) -> dict[str, str]:
    """Return a transparent SEC-compatible request header set."""

    return {
        "User-Agent": user_agent
        or os.getenv("ADFM_SEC_USER_AGENT", DEFAULT_SEC_USER_AGENT),
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }


def _request_get(
    session: requests.Session | None,
    url: str,
    *,
    stream: bool = False,
    timeout: tuple[int, int] = (8, 90),
):
    client = session or requests
    response = client.get(
        url,
        headers=sec_headers(),
        timeout=timeout,
        stream=stream,
    )
    response.raise_for_status()
    return response


def discover_quarter_datasets(
    session: requests.Session | None = None,
) -> list[QuarterDataset]:
    """Discover official quarterly archive links from the SEC landing page."""

    try:
        response = _request_get(session, SEC_13F_DATASETS_PAGE, timeout=(8, 30))
    except requests.RequestException as exc:
        raise Sec13FError(f"Could not reach the SEC Form 13F data page: {exc}") from exc

    soup = BeautifulSoup(response.text, "html.parser")
    datasets: list[QuarterDataset] = []
    seen: set[str] = set()
    for link in soup.find_all("a", href=True):
        url = urljoin(SEC_13F_DATASETS_PAGE, str(link["href"]))
        lower_url = url.lower()
        if not lower_url.endswith(".zip") or "form13f" not in lower_url:
            continue
        slug = re.sub(r"[^a-z0-9_-]+", "-", Path(urlparse(url).path).stem.lower())
        if not slug or slug in seen:
            continue
        seen.add(slug)
        row = link.find_parent("tr")
        cells = row.find_all("td") if row is not None else []
        size_label = cells[-1].get_text(" ", strip=True) if len(cells) >= 3 else ""
        label = link.get_text(" ", strip=True) or slug
        datasets.append(
            QuarterDataset(slug=slug, label=label, url=url, size_label=size_label)
        )

    if not datasets:
        raise Sec13FError("The SEC page did not expose any Form 13F ZIP archives.")
    return datasets


def _download_archive(
    dataset: QuarterDataset,
    target: Path,
    session: requests.Session | None,
    progress: Callable[[int, int | None], None] | None,
) -> None:
    try:
        response = _request_get(session, dataset.url, stream=True)
    except requests.RequestException as exc:
        raise Sec13FError(f"Could not download {dataset.label}: {exc}") from exc

    total = int(response.headers.get("content-length", 0)) or None
    downloaded = 0
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if progress is not None:
                    progress(downloaded, total)
        partial.replace(target)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise Sec13FError(f"Could not save the SEC archive: {exc}") from exc


def _extract_required_files(archive_path: Path, destination: Path) -> dict[str, Path]:
    extracted: dict[str, Path] = {}
    try:
        with ZipFile(archive_path) as archive:
            members = {
                Path(member.filename).name.upper(): member for member in archive.infolist()
            }
            missing = [name for name in REQUIRED_ARCHIVE_FILES if name not in members]
            if missing:
                raise Sec13FError(
                    "The SEC archive is missing required files: " + ", ".join(missing)
                )
            for name in REQUIRED_ARCHIVE_FILES:
                target = destination / name
                with archive.open(members[name]) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                extracted[name] = target
    except Sec13FError:
        raise
    except (OSError, ValueError) as exc:
        raise Sec13FError(f"Could not read the SEC ZIP archive: {exc}") from exc
    return extracted


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )


def _prepare_filings(files: dict[str, Path], target: Path) -> int:
    submissions = _read_tsv(files["SUBMISSION.TSV"])
    cover = _read_tsv(files["COVERPAGE.TSV"])
    summary = _read_tsv(files["SUMMARYPAGE.TSV"])
    filings = submissions.merge(cover, on="ACCESSION_NUMBER", how="left").merge(
        summary,
        on="ACCESSION_NUMBER",
        how="left",
    )
    filings = filings.loc[
        filings["SUBMISSIONTYPE"].isin(["13F-HR", "13F-HR/A"])
    ].copy()
    for column in ("FILING_DATE", "PERIODOFREPORT", "REPORTCALENDARORQUARTER"):
        if column in filings:
            filings[column] = pd.to_datetime(filings[column], errors="coerce")
    for column in ("AMENDMENTNO", "TABLEENTRYTOTAL", "TABLEVALUETOTAL"):
        if column in filings:
            filings[column] = pd.to_numeric(filings[column], errors="coerce")
    filings.to_parquet(target, index=False, compression="zstd")
    return len(filings)


def normalize_issuer_name(value: object) -> str:
    """Normalize an issuer/company name for transparent ticker-to-name matching."""

    text = re.sub(r"[^A-Z0-9]+", " ", str(value or "").upper()).strip()
    tokens = [token for token in text.split() if token not in _COMPANY_SUFFIXES]
    return " ".join(tokens)


def _prepare_holdings_and_catalog(
    source: Path,
    holdings_target: Path,
    securities_target: Path,
) -> int:
    read_options = arrow_csv.ReadOptions(block_size=16 * 1024 * 1024, use_threads=True)
    parse_options = arrow_csv.ParseOptions(delimiter="\t", newlines_in_values=False)
    convert_options = arrow_csv.ConvertOptions(
        include_columns=[
            "ACCESSION_NUMBER",
            "NAMEOFISSUER",
            "TITLEOFCLASS",
            "CUSIP",
            "VALUE",
            "SSHPRNAMT",
            "SSHPRNAMTTYPE",
            "PUTCALL",
        ],
        column_types={
            "ACCESSION_NUMBER": pa.string(),
            "NAMEOFISSUER": pa.string(),
            "TITLEOFCLASS": pa.string(),
            "CUSIP": pa.string(),
            "VALUE": pa.float64(),
            "SSHPRNAMT": pa.float64(),
            "SSHPRNAMTTYPE": pa.string(),
            "PUTCALL": pa.string(),
        },
        strings_can_be_null=True,
        null_values=["", "NULL"],
    )
    try:
        reader = arrow_csv.open_csv(
            source,
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        )
    except (pa.ArrowInvalid, OSError) as exc:
        raise Sec13FError(f"Could not open the SEC information table: {exc}") from exc

    writer: parquet.ParquetWriter | None = None
    security_rows: set[tuple[str, str, str, str]] = set()
    row_count = 0
    try:
        for batch in reader:
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = parquet.ParquetWriter(
                    holdings_target,
                    table.schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            writer.write_table(table)
            row_count += table.num_rows
            catalog_batch = table.select(
                ["NAMEOFISSUER", "TITLEOFCLASS", "CUSIP", "PUTCALL"]
            ).to_pandas()
            for row in catalog_batch.drop_duplicates().itertuples(index=False, name=None):
                issuer, title, cusip, put_call = row
                if pd.isna(cusip) or pd.isna(issuer) or not str(cusip).strip():
                    continue
                security_rows.add(
                    (
                        str(issuer).strip(),
                        "" if pd.isna(title) else str(title).strip(),
                        str(cusip).strip(),
                        "" if pd.isna(put_call) else str(put_call).strip().upper(),
                    )
                )
    except (pa.ArrowInvalid, OSError, ValueError) as exc:
        raise Sec13FError(f"Could not convert the SEC information table: {exc}") from exc
    finally:
        if writer is not None:
            writer.close()

    if writer is None or row_count == 0:
        raise Sec13FError("The SEC information table did not contain any holdings rows.")

    catalog = pd.DataFrame(
        sorted(security_rows),
        columns=["NAMEOFISSUER", "TITLEOFCLASS", "CUSIP", "PUTCALL"],
    )
    catalog["ISSUER_NORMALIZED"] = catalog["NAMEOFISSUER"].map(
        normalize_issuer_name
    )
    catalog.to_parquet(securities_target, index=False, compression="zstd")
    return row_count


def prepare_dataset(
    dataset: QuarterDataset,
    *,
    cache_root: Path | str | None = None,
    session: requests.Session | None = None,
    progress: Callable[[int, int | None], None] | None = None,
) -> PreparedDataset:
    """Download and prepare an SEC archive, or reuse a valid local preparation."""

    root = _cache_root(cache_root)
    target_dir = root / dataset.slug
    metadata_path = target_dir / "metadata.json"
    if metadata_path.is_file():
        prepared = PreparedDataset.from_metadata(metadata_path)
        if all(
            path.is_file()
            for path in (
                prepared.filings_path,
                prepared.holdings_path,
                prepared.securities_path,
            )
        ):
            return prepared

    target_dir.mkdir(parents=True, exist_ok=True)
    lock_path = target_dir / ".prepare.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(descriptor)
    except FileExistsError as exc:
        raise Sec13FError(
            "This SEC release is already being prepared by another app session."
        ) from exc

    archive_path = target_dir / f"{dataset.slug}.zip"
    try:
        if not archive_path.is_file():
            _download_archive(dataset, archive_path, session, progress)
        with tempfile.TemporaryDirectory(prefix="adfm-13f-", dir=root) as temporary:
            files = _extract_required_files(archive_path, Path(temporary))
            filings_path = target_dir / "filings.parquet"
            holdings_path = target_dir / "holdings.parquet"
            securities_path = target_dir / "securities.parquet"
            _prepare_filings(files, filings_path)
            holdings_rows = _prepare_holdings_and_catalog(
                files["INFOTABLE.TSV"],
                holdings_path,
                securities_path,
            )

        prepared_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "slug": dataset.slug,
            "label": dataset.label,
            "source_url": dataset.url,
            "prepared_at": prepared_at,
            "holdings_rows": holdings_rows,
            "filings_file": filings_path.name,
            "holdings_file": holdings_path.name,
            "securities_file": securities_path.name,
        }
        temporary_metadata = metadata_path.with_suffix(".json.part")
        temporary_metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary_metadata.replace(metadata_path)
        archive_path.unlink(missing_ok=True)
        return PreparedDataset.from_metadata(metadata_path)
    finally:
        lock_path.unlink(missing_ok=True)


def load_company_tickers(
    *,
    cache_root: Path | str | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Load the official SEC ticker-to-company-name mapping with a local fallback."""

    root = _cache_root(cache_root)
    path = root / "company_tickers.json"
    payload: dict[str, object] | None = None
    try:
        response = _request_get(session, SEC_COMPANY_TICKERS_URL, timeout=(8, 30))
        payload = response.json()
        partial = path.with_suffix(".json.part")
        partial.write_text(json.dumps(payload), encoding="utf-8")
        partial.replace(path)
    except (requests.RequestException, ValueError, OSError):
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
    if payload is None:
        raise Sec13FError("The SEC ticker directory is unavailable and no cache exists.")

    rows = list(payload.values())
    frame = pd.DataFrame(rows).rename(
        columns={"ticker": "TICKER", "title": "COMPANY_NAME", "cik_str": "CIK"}
    )
    required = {"TICKER", "COMPANY_NAME", "CIK"}
    if not required.issubset(frame.columns):
        raise Sec13FError("The SEC ticker directory has an unexpected format.")
    frame["TICKER"] = frame["TICKER"].astype(str).str.upper().str.strip()
    frame["COMPANY_NAME"] = frame["COMPANY_NAME"].astype(str).str.strip()
    frame["CIK"] = pd.to_numeric(frame["CIK"], errors="coerce").astype("Int64")
    return frame[["TICKER", "COMPANY_NAME", "CIK"]].sort_values("TICKER").reset_index(drop=True)


def load_security_catalog(prepared: PreparedDataset) -> pd.DataFrame:
    """Read the bounded security lookup table for a prepared release."""

    return pd.read_parquet(prepared.securities_path)


def _name_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 120.0
    if left in right or right in left:
        return 105.0 - min(abs(len(left) - len(right)), 20) * 0.25
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    sequence = SequenceMatcher(None, left, right).ratio()
    return 100.0 * (0.65 * sequence + 0.35 * jaccard)


def search_security_candidates(
    catalog: pd.DataFrame,
    ticker_directory: pd.DataFrame,
    query: str,
    *,
    position_kind: str = "Long holdings",
    limit: int = 20,
) -> pd.DataFrame:
    """Resolve a ticker, issuer name, or CUSIP to transparent 13F candidates."""

    cleaned_query = str(query or "").strip().upper()
    if not cleaned_query:
        return catalog.head(0).copy()

    candidates = catalog.copy()
    put_call = candidates["PUTCALL"].fillna("").astype(str).str.upper().str.strip()
    if position_kind == "Long holdings":
        candidates = candidates.loc[put_call.eq("")].copy()
    elif position_kind == "Call options":
        candidates = candidates.loc[put_call.eq("CALL")].copy()
    elif position_kind == "Put options":
        candidates = candidates.loc[put_call.eq("PUT")].copy()

    ticker_aliases = {cleaned_query, cleaned_query.replace(".", "-")}
    exact_ticker = ticker_directory.loc[
        ticker_directory["TICKER"].astype(str).str.upper().isin(ticker_aliases)
    ]
    resolved_company = (
        str(exact_ticker.iloc[0]["COMPANY_NAME"])
        if not exact_ticker.empty
        else str(query).strip()
    )
    target = normalize_issuer_name(resolved_company)
    exact_cusip = candidates["CUSIP"].astype(str).str.upper().eq(cleaned_query)
    candidates["MATCH_SCORE"] = candidates["ISSUER_NORMALIZED"].map(
        lambda value: _name_similarity(target, str(value))
    )
    candidates.loc[exact_cusip, "MATCH_SCORE"] = 150.0
    candidates["QUERY"] = str(query).strip()
    candidates["RESOLVED_COMPANY"] = resolved_company
    candidates = candidates.loc[
        exact_cusip | candidates["MATCH_SCORE"].ge(35.0)
    ].copy()
    return (
        candidates.sort_values(
            ["MATCH_SCORE", "NAMEOFISSUER", "TITLEOFCLASS", "CUSIP"],
            ascending=[False, True, True, True],
        )
        .drop_duplicates(["CUSIP", "PUTCALL", "TITLEOFCLASS"])
        .head(max(1, int(limit)))
        .reset_index(drop=True)
    )


def available_report_periods(prepared: PreparedDataset) -> list[pd.Timestamp]:
    """Return report dates represented in the selected SEC release."""

    filings = pd.read_parquet(prepared.filings_path, columns=["PERIODOFREPORT"])
    dates = pd.to_datetime(filings["PERIODOFREPORT"], errors="coerce").dropna()
    return sorted((pd.Timestamp(value) for value in dates.unique()), reverse=True)


def select_effective_filing_components(
    filings: pd.DataFrame,
    report_period: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Select the effective base filing plus subsequent new-holdings amendments.

    A restatement supersedes the earlier base and amendments.  An amendment that
    adds new holdings supplements the current base, so qualifying later add-ons
    are included in both the position numerator and portfolio denominator.
    """

    frame = filings.copy()
    frame["PERIODOFREPORT"] = pd.to_datetime(frame["PERIODOFREPORT"], errors="coerce")
    frame["FILING_DATE"] = pd.to_datetime(frame["FILING_DATE"], errors="coerce")
    frame = frame.loc[frame["SUBMISSIONTYPE"].isin(["13F-HR", "13F-HR/A"])]
    if frame.empty:
        return frame
    selected_period = (
        pd.Timestamp(report_period) if report_period is not None else frame["PERIODOFREPORT"].max()
    )
    frame = frame.loc[frame["PERIODOFREPORT"].eq(selected_period)].copy()
    frame["AMENDMENTTYPE"] = frame.get("AMENDMENTTYPE", "").fillna("").astype(str)
    frame["AMENDMENTNO"] = pd.to_numeric(
        frame.get("AMENDMENTNO", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    ).fillna(0)
    frame["TABLEVALUETOTAL"] = pd.to_numeric(
        frame.get("TABLEVALUETOTAL", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    )
    frame = frame.sort_values(
        ["CIK", "FILING_DATE", "AMENDMENTNO", "ACCESSION_NUMBER"]
    )

    selected_groups: list[pd.DataFrame] = []
    for _, group in frame.groupby("CIK", sort=False):
        amendment = group["AMENDMENTTYPE"].str.upper()
        restatements = group.loc[amendment.str.contains("RESTATEMENT", na=False)]
        initials = group.loc[group["SUBMISSIONTYPE"].eq("13F-HR")]
        base_candidates = restatements if not restatements.empty else initials
        if base_candidates.empty:
            continue
        base = base_candidates.iloc[-1]
        base_position = group.index.get_loc(base.name)
        subsequent = group.iloc[base_position + 1 :]
        additions = subsequent.loc[
            subsequent["AMENDMENTTYPE"]
            .str.upper()
            .str.contains("NEW HOLDINGS", na=False)
        ]
        components = pd.concat([base.to_frame().T, additions], ignore_index=True)
        components["COMPONENT_ROLE"] = ["Base"] + ["New holdings"] * len(additions)
        components["PRIMARY_ACCESSION_NUMBER"] = str(base["ACCESSION_NUMBER"])
        selected_groups.append(components)

    if not selected_groups:
        return frame.head(0)
    result = pd.concat(selected_groups, ignore_index=True)
    result["TABLEVALUETOTAL"] = pd.to_numeric(
        result["TABLEVALUETOTAL"], errors="coerce"
    )
    return result


def filing_url(cik: object, accession_number: object) -> str:
    """Build the canonical EDGAR filing index URL."""

    cik_text = str(cik).lstrip("0") or "0"
    accession = str(accession_number)
    accession_path = accession.replace("-", "")
    return f"{SEC_ARCHIVES_ROOT}/{cik_text}/{accession_path}/{accession}-index.html"


def _position_kind_mask(holdings: pd.DataFrame, position_kind: str) -> pd.Series:
    put_call = holdings["PUTCALL"].fillna("").astype(str).str.upper().str.strip()
    if position_kind == "Long holdings":
        return put_call.eq("")
    if position_kind == "Call options":
        return put_call.eq("CALL")
    if position_kind == "Put options":
        return put_call.eq("PUT")
    return pd.Series(True, index=holdings.index)


def rank_fund_exposure(
    prepared: PreparedDataset,
    cusips: Sequence[str],
    *,
    report_period: str | pd.Timestamp | None = None,
    position_kind: str = "Long holdings",
    minimum_portfolio_millions: float = 0.0,
) -> pd.DataFrame:
    """Rank managers holding selected CUSIPs by disclosed 13F portfolio weight."""

    selected_cusips = sorted({str(value).strip() for value in cusips if str(value).strip()})
    if not selected_cusips:
        return pd.DataFrame()

    filings = pd.read_parquet(prepared.filings_path)
    components = select_effective_filing_components(filings, report_period)
    if components.empty:
        return pd.DataFrame()

    try:
        holdings = pd.read_parquet(
            prepared.holdings_path,
            filters=[("CUSIP", "in", selected_cusips)],
        )
    except (TypeError, ValueError):
        holdings = pd.read_parquet(prepared.holdings_path)
        holdings = holdings.loc[holdings["CUSIP"].isin(selected_cusips)]
    holdings = holdings.loc[_position_kind_mask(holdings, position_kind)].copy()
    if holdings.empty:
        return pd.DataFrame()

    components = components.copy()
    components["CIK"] = components["CIK"].astype(str).str.zfill(10)
    components["TABLEVALUETOTAL"] = pd.to_numeric(
        components["TABLEVALUETOTAL"], errors="coerce"
    )
    manager_rows: list[dict[str, object]] = []
    for cik, group in components.groupby("CIK", sort=False):
        base = group.loc[group["COMPONENT_ROLE"].eq("Base")].iloc[-1]
        total = group["TABLEVALUETOTAL"].sum(min_count=1)
        if pd.isna(total) or total <= 0:
            continue
        manager_rows.append(
            {
                "CIK": cik,
                "MANAGER": str(base.get("FILINGMANAGER_NAME", "")).strip() or cik,
                "REPORT_PERIOD": pd.Timestamp(base["PERIODOFREPORT"]),
                "LATEST_FILING_DATE": pd.to_datetime(group["FILING_DATE"]).max(),
                "PORTFOLIO_VALUE_THOUSANDS": float(total),
                "COMPONENT_COUNT": len(group),
                "PRIMARY_ACCESSION_NUMBER": str(base["PRIMARY_ACCESSION_NUMBER"]),
            }
        )
    managers = pd.DataFrame(manager_rows)
    if managers.empty:
        return pd.DataFrame()
    managers = managers.loc[
        managers["PORTFOLIO_VALUE_THOUSANDS"].ge(
            max(0.0, float(minimum_portfolio_millions)) * 1_000.0
        )
    ]
    if managers.empty:
        return pd.DataFrame()

    accessions = components[["ACCESSION_NUMBER", "CIK", "FILING_DATE"]].drop_duplicates()
    holdings = holdings.merge(accessions, on="ACCESSION_NUMBER", how="inner")
    if holdings.empty:
        return pd.DataFrame()
    holdings["VALUE"] = pd.to_numeric(holdings["VALUE"], errors="coerce")
    holdings["SSHPRNAMT"] = pd.to_numeric(holdings["SSHPRNAMT"], errors="coerce")
    holdings["SHARES_ONLY"] = holdings["SSHPRNAMT"].where(
        holdings["SSHPRNAMTTYPE"].fillna("").astype(str).str.upper().eq("SH")
    )
    holdings = holdings.sort_values(["CIK", "FILING_DATE", "ACCESSION_NUMBER"])
    positions = (
        holdings.groupby("CIK", as_index=False, sort=False)
        .agg(
            POSITION_VALUE_THOUSANDS=("VALUE", "sum"),
            REPORTED_SHARES=(
                "SHARES_ONLY",
                lambda values: values.sum(min_count=1),
            ),
            POSITION_LINES=("ACCESSION_NUMBER", "size"),
            POSITION_ACCESSION_NUMBER=("ACCESSION_NUMBER", "last"),
        )
    )
    ranking = managers.merge(positions, on="CIK", how="inner")
    if ranking.empty:
        return ranking
    ranking["PORTFOLIO_WEIGHT_PCT"] = (
        ranking["POSITION_VALUE_THOUSANDS"]
        / ranking["PORTFOLIO_VALUE_THOUSANDS"]
        * 100.0
    )
    ranking["POSITION_VALUE_USD"] = ranking["POSITION_VALUE_THOUSANDS"] * 1_000.0
    ranking["PORTFOLIO_VALUE_USD"] = ranking["PORTFOLIO_VALUE_THOUSANDS"] * 1_000.0
    ranking["FILING_URL"] = ranking.apply(
        lambda row: filing_url(row["CIK"], row["POSITION_ACCESSION_NUMBER"]),
        axis=1,
    )
    ranking = ranking.sort_values(
        ["PORTFOLIO_WEIGHT_PCT", "POSITION_VALUE_USD", "MANAGER"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "RANK", range(1, len(ranking) + 1))
    return ranking


def prepared_dataset_summary(prepared: PreparedDataset) -> dict[str, object]:
    """Return serializable status fields for diagnostics and testing."""

    payload = asdict(prepared)
    return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}
