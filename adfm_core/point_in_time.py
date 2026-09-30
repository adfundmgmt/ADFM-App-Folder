"""Historical availability helpers: strict reconstruction never guesses a release."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

CFTC_HISTORY_SOURCE = "https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm"
CFTC_SCHEDULE_SOURCE = "https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm"
# The historical announcement describes these as reports actually issued that day.
CFTC_ACTUAL_RELEASES = {
    "2020-12-21": "2020-12-28",
    "2023-01-31": "2023-02-24", "2023-02-07": "2023-03-03",
    "2023-02-14": "2023-03-08", "2023-02-21": "2023-03-10",
    "2023-02-28": "2023-03-14", "2023-03-07": "2023-03-16",
    "2023-03-14": "2023-03-21",
}
# Announcements saying "will release" and tables labelled "intended" are schedules.
CFTC_SCHEDULE_OVERRIDES = {
    "2021-06-15": "2021-06-21", "2025-01-07": "2025-01-13",
    "2025-09-30": "2025-11-19", "2025-10-07": "2025-11-21",
    "2025-10-14": "2025-11-25", "2025-10-21": "2025-12-02",
    "2025-10-28": "2025-12-05", "2025-11-04": "2025-12-09",
    "2025-11-10": "2025-12-10", "2025-11-18": "2025-12-12",
    "2025-11-25": "2025-12-15", "2025-12-02": "2025-12-17",
    "2025-12-09": "2025-12-19", "2025-12-16": "2025-12-23",
    "2025-12-23": "2025-12-29",
    # 2026 holiday dates from the published tentative calendar.
    "2025-12-30": "2026-01-05", "2026-06-16": "2026-06-22",
    "2026-06-30": "2026-07-06", "2026-11-10": "2026-11-16",
    "2026-11-24": "2026-11-30", "2026-12-22": "2026-12-28",
}


def cftc_publication(report_date, *, strict: bool = True, records: dict | None = None) -> tuple[pd.Timestamp, str]:
    """Use verified actual records in strict mode; return NaT for unknown dates.

    records may supply independently verified report-date -> actual-publication-date
    records. A current tentative schedule is never promoted to actual evidence.
    """
    date = pd.Timestamp(report_date).normalize()
    key = date.strftime("%Y-%m-%d")
    actual = {**CFTC_ACTUAL_RELEASES, **(records or {})}.get(key)
    if actual:
        return pd.Timestamp(actual), "verified actual publication record"
    if strict:
        return pd.NaT, "unknown actual publication date; excluded"
    if pd.Timestamp("2018-12-24") <= date <= pd.Timestamp("2019-02-26"):
        return pd.NaT, "unresolved shutdown backlog; excluded"
    scheduled = CFTC_SCHEDULE_OVERRIDES.get(key)
    if scheduled:
        return pd.Timestamp(scheduled), "announced schedule; actual publication unverified"
    # Tuesday observations are normally published Friday; other report weekdays
    # and Friday federal holidays require records rather than a guessed holiday lag.
    expected = date + pd.Timedelta(days=3)
    from pandas.tseries.holiday import USFederalHolidayCalendar
    holidays = USFederalHolidayCalendar().holidays(date, expected)
    if date.weekday() != 1 or expected in holidays:
        return pd.NaT, "unresolved exceptional or holiday date; excluded"
    return expected, "assumed ordinary Tuesday/Friday schedule; actual publication unverified"


def known_observations(records: pd.DataFrame, as_of) -> pd.Series:
    """Latest knowable version per observation; unknown release dates stay missing."""
    required = {"observation_date", "available_at", "vintage_date", "value"}
    if not required.issubset(records):
        raise ValueError(f"Availability records require {sorted(required)}")
    frame = records.copy()
    cutoff = pd.Timestamp(as_of)
    for column in ("observation_date", "available_at", "vintage_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame = frame.loc[frame["available_at"].le(cutoff) & frame["vintage_date"].le(cutoff) & frame["observation_date"].le(cutoff)]
    frame = frame.sort_values(["observation_date", "vintage_date", "available_at"]).drop_duplicates("observation_date", keep="last")
    return pd.Series(pd.to_numeric(frame["value"], errors="coerce").to_numpy(), index=pd.DatetimeIndex(frame["observation_date"]), name="value")


def _fred_vintage(symbol, start, end, *, vintage):
    from .fred_store import FredStore
    result = FredStore().get(symbol, start, end, vintage=vintage)
    return result.series, result.metadata


def fed_regimes_at_month_start(months: pd.PeriodIndex, *, loader: Callable | None = None) -> pd.DataFrame:
    """One ALFRED vintage per decision month; compute change within that vintage.

    ALFRED's as-of query supplies release availability, so an unpublished monthly
    observation is absent. No release lag is fabricated. USREC/NBER recession
    labels are deliberately unavailable for decision conditioning.
    """
    rows = []
    loader = loader or _fred_vintage
    for period in months:
        decision = period.start_time
        vintage = (decision - pd.Timedelta(days=1)).date().isoformat()
        row = {"fedfunds": np.nan, "fed_regime": "Unknown", "is_recession": np.nan,
               "regime_cycle": "Unknown", "vintage": vintage, "availability_basis": "ALFRED month-start vintage unavailable"}
        try:
            values, metadata = loader("FEDFUNDS", (decision - pd.DateOffset(months=8)).date().isoformat(), vintage, vintage=vintage)
            if metadata.get("vintage") != vintage:
                rows.append(row)
                continue
            values = pd.to_numeric(values, errors="coerce").dropna().sort_index()
            values.index = pd.to_datetime(values.index)
            # Monthly averages dated at the month's start require the entire
            # month to have ended; the vintage query further enforces release.
            values = values.loc[(values.index.to_period("M").end_time < decision)]
            if not values.empty:
                latest = values.index[-1].to_period("M")
                monthly = values.groupby(values.index.to_period("M")).last()
                prior = monthly.get(latest - 3, np.nan)
                current = float(monthly.iloc[-1])
                if latest >= period - 2 and pd.notna(prior):
                    delta = current - prior
                    row.update(fedfunds=current, fed_regime="Hiking" if delta > .05 else "Cutting" if delta < -.05 else "Steady",
                               availability_basis="verified ALFRED vintage at prior calendar day", observed_month=str(latest))
        except Exception:
            # Fail closed: no latest revised or guessed release-date fallback.
            pass
        rows.append(row)
    return pd.DataFrame(rows, index=months)


def fetch_fred_availability_records(symbol: str, start: str, end: str, *, key: str | None = None, transport=None) -> pd.DataFrame:
    """Retrieve ALFRED real-time periods, including initial releases and revisions.

    The earliest realtime_start avoids truncating historical availability dates to
    the query start. API output_type=1 is observations by real-time period. This
    makes one bounded batch request for a sample rather than one request per month.
    No unauthenticated latest-data fallback is used for historical reconstruction.
    """
    import requests

    from .fred_store import api_key
    credential = key if key is not None else api_key()
    if not credential:
        raise ValueError("ALFRED release histories require FRED_API_KEY")
    get = transport or requests.get
    rows = []
    for offset in range(0, 500000, 100000):
        response = get("https://api.stlouisfed.org/fred/series/observations", params={
            "series_id": symbol, "api_key": credential, "file_type": "json", "output_type": 1,
            "observation_start": start, "observation_end": end,
            "realtime_start": "1776-07-04", "realtime_end": end, "limit": 100000, "offset": offset,
        }, timeout=25)
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("observations", [])
        rows.extend(batch)
        if len(rows) >= int(payload.get("count", len(rows))):
            break
        if not batch:
            raise ValueError("ALFRED release-history pagination was incomplete")
    else:
        raise ValueError("ALFRED release history exceeds the bounded request limit")
    records = pd.DataFrame(rows)
    if records.empty:
        return pd.DataFrame(columns=["observation_date", "available_at", "vintage_date", "value"])
    if not {"date", "realtime_start", "value"}.issubset(records):
        raise ValueError("ALFRED omitted real-time availability fields")
    records = records.rename(columns={"date": "observation_date", "realtime_start": "available_at"})
    records["vintage_date"] = records["available_at"]
    records["value"] = pd.to_numeric(records["value"], errors="coerce")
    return records


def fed_regimes_from_availability(months: pd.PeriodIndex, records: pd.DataFrame) -> pd.DataFrame:
    """Classify Fed rates from official release/revision records known before month."""
    def loader(symbol, start, end, *, vintage):
        values = known_observations(records, vintage)
        return values.loc[start:end], {"vintage": vintage, "source": "ALFRED real-time periods"}
    return fed_regimes_at_month_start(months, loader=loader)
