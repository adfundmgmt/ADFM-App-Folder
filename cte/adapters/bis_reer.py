"""BIS Real Effective Exchange Rate adapter — spec axis-2 (REER valuation, §8 G).

Monthly broad-basket REER index for the eight currencies, from the BIS Data
Portal SDMX REST API (v1). Key = FREQ.EER_TYPE.EER_BASKET.REF_AREA =
M (monthly) . R (real) . B (broad) . <area>. All eight areas come back in a
single batched request via the '+' OR-operator on the area dimension.

REER is the level a currency trades at versus a trade- and inflation-weighted
basket; a rich REER (high z vs its own history) is the spec's structural
'stretch' signal on Axis 2. Published monthly with a ~1-month lag — never
interpolated to daily (spec §13).

Tidy contract returned: [date, ccy, value, source, fetched_at]
  date  = month-end timestamp of the REER observation
  value = broad real EER index level
"""
from __future__ import annotations

import io

import pandas as pd

from cte.adapters.base import make_session, utcnow
from cte.config import BIS_EER_AREA, HTTP_UA, HTTP_TIMEOUT

_URL = ("https://stats.bis.org/api/v1/data/BIS,WS_EER,1.0/"
        "M.R.B.{areas}?detail=dataonly")
_CBPOL_URL = "https://stats.bis.org/api/v1/data/BIS,WS_CBPOL,1.0/M.{areas}"
_CSV_ACCEPT = "application/vnd.sdmx.data+csv"


def fetch_reer() -> pd.DataFrame:
    area_to_ccy = {v: k for k, v in BIS_EER_AREA.items()}
    areas = "+".join(BIS_EER_AREA.values())
    url = _URL.format(areas=areas)
    sess = make_session()
    resp = sess.get(url, headers={"User-Agent": HTTP_UA, "Accept": _CSV_ACCEPT},
                    timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    raw = pd.read_csv(io.BytesIO(resp.content))

    df = raw[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
    df["ccy"] = df["REF_AREA"].map(area_to_ccy)
    df = df.dropna(subset=["ccy"])
    # TIME_PERIOD is 'YYYY-MM'; anchor to month-end and never interpolate
    df["date"] = (pd.to_datetime(df["TIME_PERIOD"], format="%Y-%m")
                  + pd.offsets.MonthEnd(0))
    df["value"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["value"])
    df["source"] = "bis_eer_broad_real"
    df["fetched_at"] = utcnow()
    out = (df[["date", "ccy", "value", "source", "fetched_at"]]
           .sort_values(["ccy", "date"]).reset_index(drop=True))
    return out


def fetch_policy_rates() -> pd.DataFrame:
    """Actual central-bank policy rates for all 8, monthly, from BIS WS_CBPOL.
    One uniform source/definition (replaces the FRED actual + 3M-interbank mix).
    Tidy contract: [date, ccy, metric, value, source, fetched_at]."""
    area_to_ccy = {v: k for k, v in BIS_EER_AREA.items()}
    areas = "+".join(BIS_EER_AREA.values())
    sess = make_session()
    resp = sess.get(_CBPOL_URL.format(areas=areas),
                    headers={"User-Agent": HTTP_UA, "Accept": _CSV_ACCEPT},
                    timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    raw = pd.read_csv(io.BytesIO(resp.content))
    df = raw[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
    df["ccy"] = df["REF_AREA"].map(area_to_ccy)
    df = df.dropna(subset=["ccy"])
    df["date"] = (pd.to_datetime(df["TIME_PERIOD"], format="%Y-%m")
                  + pd.offsets.MonthEnd(0))
    df["value"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["value"])
    df["metric"] = "policy"
    df["source"] = "bis_cbpol"
    df["fetched_at"] = utcnow()
    return (df[["date", "ccy", "metric", "value", "source", "fetched_at"]]
            .sort_values(["ccy", "date"]).reset_index(drop=True))


if __name__ == "__main__":
    df = fetch_reer()
    print(f"rows: {len(df)} | currencies: {sorted(df.ccy.unique())}")
    print(f"date range: {df.date.min().date()} -> {df.date.max().date()}")
    latest = df.sort_values("date").groupby("ccy").tail(1)
    print(latest[["ccy", "date", "value"]].to_string(index=False))
