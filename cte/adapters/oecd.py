"""OECD SDMX adapter — cross-country macro inputs from the OECD Data Explorer.

Two inputs, from the OECD Data Explorer SDMX API (SDMX-CSV):
  bcicp    Amplitude-adjusted business confidence for ALL 8 currencies — the
           standalone Pillar A leading indicator (a PMI-equivalent survey).
  cpi_yoy  Headline CPI year-on-year (%) for GBP/CAD/AUS/NZL.

Returns the same tidy long contract as the FRED adapter:
  [date, ccy, metric, value, source, fetched_at]

TIME_PERIOD arrives monthly ("2026-05") or quarterly ("2026-Q1"); both are parsed
to a period-end timestamp. Quarterly series (e.g. NZ CPI) are never interpolated.
"""
from __future__ import annotations

import io

import pandas as pd

from cte.adapters.base import make_session, utcnow
from cte.config import (
    HTTP_TIMEOUT, HTTP_UA, OECD_BCICP_DATAFLOW, OECD_BCICP_CCYS,
    OECD_CPI_CCYS, OECD_CPI_DATAFLOW, OECD_CSV_ACCEPT, OECD_REF_AREA,
    OECD_SDMX_BASE,
)

_START = "1990-01"


def _get_csv(dataflow: str, start: str = _START, session=None) -> pd.DataFrame:
    sess = session or make_session()
    url = (f"{OECD_SDMX_BASE}/{dataflow}/all"
           f"?startPeriod={start}&dimensionAtObservation=AllDimensions")
    r = sess.get(url, headers={"User-Agent": HTTP_UA, "Accept": OECD_CSV_ACCEPT},
                 timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))


def _parse_period(s: pd.Series) -> pd.Series:
    """OECD TIME_PERIOD -> period-end timestamp. Handles 'YYYY-MM' and 'YYYY-Qn'."""
    txt = s.astype(str)
    is_q = txt.str.contains("Q")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if is_q.any():
        out[is_q] = (pd.PeriodIndex(txt[is_q].str.replace("-", ""), freq="Q")
                     .to_timestamp(how="end").normalize())
    if (~is_q).any():
        out[~is_q] = (pd.to_datetime(txt[~is_q], format="%Y-%m")
                      + pd.offsets.MonthEnd(0))
    return out


def _area_to_ccy(ccys: tuple[str, ...]) -> dict[str, str]:
    return {OECD_REF_AREA[c]: c for c in ccys}


def fetch_oecd_bcicp(session=None) -> pd.DataFrame:
    raw = _get_csv(OECD_BCICP_DATAFLOW, session=session)
    a2c = _area_to_ccy(OECD_BCICP_CCYS)
    df = raw[(raw.MEASURE == "BCICP") & (raw.ADJUSTMENT == "AA")
             & (raw.TRANSFORMATION == "IX") & (raw.FREQ == "M")
             & (raw.REF_AREA.isin(a2c))].copy()
    df["ccy"] = df.REF_AREA.map(a2c)
    df["date"] = _parse_period(df.TIME_PERIOD)
    df["value"] = pd.to_numeric(df.OBS_VALUE, errors="coerce")
    df["metric"] = "bcicp"
    df["source"] = "oecd_bcicp"
    df["fetched_at"] = utcnow()
    return df.dropna(subset=["value"])[
        ["date", "ccy", "metric", "value", "source", "fetched_at"]]


def fetch_oecd_cpi(session=None) -> pd.DataFrame:
    raw = _get_csv(OECD_CPI_DATAFLOW, session=session)
    a2c = _area_to_ccy(OECD_CPI_CCYS)
    df = raw[(raw.EXPENDITURE == "_T") & (raw.MEASURE == "CPI")
             & (raw.TRANSFORMATION == "GY") & (raw.UNIT_MEASURE == "PA")
             & (raw.METHODOLOGY == "N")  # national basis (GBP also has HICP; use N for consistency)
             & (raw.FREQ.isin(["M", "Q"]))
             & (raw.REF_AREA.isin(a2c))].copy()
    # prefer monthly where a country publishes both M and Q
    df["date"] = _parse_period(df.TIME_PERIOD)
    df["ccy"] = df.REF_AREA.map(a2c)
    df["value"] = pd.to_numeric(df.OBS_VALUE, errors="coerce")
    df = df.dropna(subset=["value"])
    # Prefer monthly only where it has real depth (>=10y). Australia's monthly CPI
    # indicator is only ~1y long, so it falls back to its 30y+ quarterly series;
    # GBP/CAD keep their long monthly series. NZ is quarterly-only.
    keep = []
    for ccy, g in df.groupby("ccy"):
        m = g[g.FREQ == "M"]
        if len(m) and (m.date.max() - m.date.min()).days / 365.25 >= 10:
            keep.append(m)
        else:
            keep.append(g[g.FREQ == "Q"] if (g.FREQ == "Q").any() else g)
    df = pd.concat(keep, ignore_index=True)
    df["metric"] = "cpi_yoy"
    df["source"] = "oecd_cpi_headline"
    df["fetched_at"] = utcnow()
    return df[["date", "ccy", "metric", "value", "source", "fetched_at"]]


def fetch_oecd_gdp(session=None) -> pd.DataFrame:
    """Real GDP levels (chain-linked volume, SA, quarterly) for the 7 non-EUR legs."""
    from cte.config import OECD_GDP_CCYS, OECD_QNA_DATAFLOW
    raw = _get_csv(OECD_QNA_DATAFLOW, start="1990-Q1", session=session)
    a2c = _area_to_ccy(OECD_GDP_CCYS)
    df = raw[(raw.TRANSACTION == "B1GQ") & (raw.PRICE_BASE == "L")
             & (raw.ADJUSTMENT == "Y") & (raw.FREQ == "Q")
             & (raw.EXPENDITURE == "_Z") & (raw.REF_AREA.isin(a2c))].copy()
    df["ccy"] = df.REF_AREA.map(a2c)
    df["date"] = _parse_period(df.TIME_PERIOD)
    df["value"] = pd.to_numeric(df.OBS_VALUE, errors="coerce")
    df["metric"] = "gdp"
    df["source"] = "oecd_qna_gdp_real"
    df["fetched_at"] = utcnow()
    return (df.dropna(subset=["value"]).sort_values(["ccy", "date"])
            .reset_index(drop=True)[
                ["date", "ccy", "metric", "value", "source", "fetched_at"]])


def fetch_oecd_unemployment(session=None) -> pd.DataFrame:
    """Harmonised unemployment rate for the 6 non-US/EUR legs, fresher than FRED.
    Monthly where it has >=10y depth (JPY/GBP/CAD/AUD), else quarterly (CHF/NZD)."""
    from cte.config import OECD_UNEMP_CCYS, OECD_UNEMP_DATAFLOW
    raw = _get_csv(OECD_UNEMP_DATAFLOW, start="1990-01", session=session)
    a2c = _area_to_ccy(OECD_UNEMP_CCYS)
    df = raw[(raw.MEASURE == "UNE_LF_M") & (raw.UNIT_MEASURE == "PT_LF_SUB")
             & (raw.SEX == "_T") & (raw.ADJUSTMENT == "Y") & (raw.AGE == "Y_GE15")
             & (raw.FREQ.isin(["M", "Q"])) & (raw.REF_AREA.isin(a2c))].copy()
    df["date"] = _parse_period(df.TIME_PERIOD)
    df["ccy"] = df.REF_AREA.map(a2c)
    df["value"] = pd.to_numeric(df.OBS_VALUE, errors="coerce")
    df = df.dropna(subset=["value"])
    keep = []
    for ccy, g in df.groupby("ccy"):
        m = g[g.FREQ == "M"]
        if len(m) and (m.date.max() - m.date.min()).days / 365.25 >= 10:
            keep.append(m)
        else:
            keep.append(g[g.FREQ == "Q"] if (g.FREQ == "Q").any() else g)
    df = pd.concat(keep, ignore_index=True)
    df["metric"] = "unemp"
    df["source"] = "oecd_ialfs"
    df["fetched_at"] = utcnow()
    return (df.dropna(subset=["value"]).sort_values(["ccy", "date"])
            .reset_index(drop=True)[
                ["date", "ccy", "metric", "value", "source", "fetched_at"]])


def fetch_oecd_all() -> pd.DataFrame:
    sess = make_session()
    parts = [fetch_oecd_bcicp(session=sess), fetch_oecd_cpi(session=sess),
             fetch_oecd_gdp(session=sess), fetch_oecd_unemployment(session=sess)]
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["ccy", "metric", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    df = fetch_oecd_all()
    print(f"rows: {len(df)} | groups: {df.groupby(['ccy','metric']).ngroups}")
    latest = df.sort_values("date").groupby(["ccy", "metric"]).tail(1)
    print(latest.to_string(index=False))
