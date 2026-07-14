"""Data-integrity contract shared by ADFM analytical pages.

Raw provider observations are assessed before any alignment, filling, scoring,
or visualization. Pages can use the same report for eligibility decisions,
user-facing diagnostics, and a transparent data-through record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .market_data import benchmark_calendar, stale_session_count, unique_tickers


@dataclass(frozen=True)
class DataIntegrityPolicy:
    """Explicit eligibility rules for raw daily observations.

    Filling is deliberately excluded from this policy. Any alignment filling
    must be an explicit downstream ratio-analysis transformation.
    """

    min_valid_sessions: int = 252
    max_stale_sessions: int = 3
    required_fields: Tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")
    reject_invalid_ranges: bool = True
    reject_nonpositive_close: bool = True


@dataclass(frozen=True)
class DataQualityReport:
    """Raw-data validation result measured on one benchmark calendar."""

    benchmark: str
    data_through: Optional[pd.Timestamp]
    diagnostics: pd.DataFrame
    policy: DataIntegrityPolicy

    @property
    def benchmark_ready(self) -> bool:
        row = self.diagnostics.loc[self.diagnostics["Ticker"] == self.benchmark]
        return bool(not row.empty and row["Status"].iloc[0] == "OK")

    @property
    def eligible_tickers(self) -> Tuple[str, ...]:
        return tuple(self.diagnostics.loc[self.diagnostics["Status"] == "OK", "Ticker"])

    def reason_for(self, ticker: str) -> Optional[str]:
        row = self.diagnostics.loc[self.diagnostics["Ticker"] == ticker]
        if row.empty or row["Status"].iloc[0] == "OK":
            return None
        return str(row["Reason"].iloc[0])


def _empty_row(ticker: str, reason: str) -> dict[str, object]:
    return {"Ticker": ticker, "Status": "Excluded", "Reason": reason, "First Date": pd.NaT, "Latest Date": pd.NaT, "Valid Sessions": 0, "Missing Benchmark Sessions": np.nan, "Stale Sessions": np.nan, "Raw Last Close": np.nan}


def build_data_quality_report(
    raw_frames: Mapping[str, pd.DataFrame],
    benchmark: str,
    tickers: Optional[Sequence[str]] = None,
    policy: DataIntegrityPolicy = DataIntegrityPolicy(),
) -> DataQualityReport:
    """Validate raw series without filling, fabrication, or lookahead."""
    symbols = unique_tickers(tickers or tuple(raw_frames.keys()))
    if benchmark not in symbols:
        symbols = (benchmark,) + symbols
    sessions = benchmark_calendar(raw_frames, benchmark)
    data_through = pd.Timestamp(sessions[-1]) if len(sessions) else None
    rows: list[dict[str, object]] = []
    for ticker in symbols:
        frame = raw_frames.get(ticker)
        if frame is None or frame.empty:
            rows.append(_empty_row(ticker, "Missing raw data"))
            continue
        missing_fields = [field for field in policy.required_fields if field not in frame.columns]
        if missing_fields:
            rows.append(_empty_row(ticker, "Missing required fields: " + ", ".join(missing_fields)))
            continue
        observed = frame.copy()
        for field in policy.required_fields:
            observed[field] = pd.to_numeric(observed[field], errors="coerce")
        complete = observed.dropna(subset=list(policy.required_fields))
        if complete.empty:
            rows.append(_empty_row(ticker, "No complete raw observations"))
            continue
        reason: Optional[str] = None
        if len(complete) < policy.min_valid_sessions:
            reason = f"Thin history (<{policy.min_valid_sessions} complete sessions)"
        elif policy.reject_invalid_ranges and {"High", "Low"}.issubset(complete.columns) and (complete["High"] < complete["Low"]).any():
            reason = "Invalid high/low range"
        elif policy.reject_nonpositive_close and (complete["Close"] <= 0).any():
            reason = "Nonpositive close"
        stale = stale_session_count(complete, sessions)
        if reason is None and stale > policy.max_stale_sessions:
            reason = f"Stale (>{policy.max_stale_sessions} benchmark sessions)"
        active_sessions = sessions[sessions >= complete.index.min()]
        missing_sessions = int(complete["Close"].reindex(active_sessions).isna().sum()) if len(active_sessions) else np.nan
        rows.append({"Ticker": ticker, "Status": "OK" if reason is None else "Excluded", "Reason": "" if reason is None else reason, "First Date": pd.Timestamp(complete.index.min()).date(), "Latest Date": pd.Timestamp(complete.index.max()).date(), "Valid Sessions": int(len(complete)), "Missing Benchmark Sessions": missing_sessions, "Stale Sessions": stale, "Raw Last Close": float(complete["Close"].iloc[-1])})
    return DataQualityReport(benchmark=benchmark, data_through=data_through, diagnostics=pd.DataFrame(rows), policy=policy)


def report_caption(report: DataQualityReport) -> str:
    """Compact reusable disclosure for page headers and exports."""
    date_label = report.data_through.date().isoformat() if report.data_through is not None else "unavailable"
    eligible = int((report.diagnostics["Status"] == "OK").sum())
    return f"Data through: {date_label} | Benchmark: {report.benchmark} | Raw-data eligible: {eligible}/{len(report.diagnostics)}"
