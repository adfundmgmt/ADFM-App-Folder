"""External-sector adapter (Pillar C) — current account as % of GDP.

Primary external-pressure signal for the tension engine: the current-account flow,
normalized by GDP, from the OECD balance-of-payments dataflow. A surplus currency
has natural funding support; a persistent deficit depends on capital inflows.

Direct % of GDP (UNIT_MEASURE PT_B1GQ), balance entry (B), quarterly, vs World.
Returns the tidy contract: [date, ccy, metric, value, source, fetched_at]

NIIP (the accumulated stock) is a planned light structural companion — see
config.OECD_IIP_DATAFLOW; it needs a nominal-GDP denominator and a euro-area source.
"""
from __future__ import annotations

import io

import pandas as pd

from cte.adapters.base import make_session, utcnow
from cte.adapters.oecd import _parse_period
from cte.config import (
    HTTP_TIMEOUT, HTTP_UA, OECD_BOP_DATAFLOW, OECD_CSV_ACCEPT, OECD_REF_AREA,
    OECD_SDMX_BASE,
)


def fetch_current_account(session=None) -> pd.DataFrame:
    sess = session or make_session()
    url = (f"{OECD_SDMX_BASE}/{OECD_BOP_DATAFLOW}/all"
           f"?startPeriod=1990-Q1&dimensionAtObservation=AllDimensions")
    r = sess.get(url, headers={"User-Agent": HTTP_UA, "Accept": OECD_CSV_ACCEPT},
                 timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    raw = pd.read_csv(io.BytesIO(r.content))

    a2c = {OECD_REF_AREA[c]: c for c in OECD_REF_AREA}
    df = raw[(raw.MEASURE == "CA") & (raw.UNIT_MEASURE == "PT_B1GQ")
             & (raw.ACCOUNTING_ENTRY == "B") & (raw.FREQ == "Q")
             & (raw.REF_AREA.isin(a2c))].copy()
    if "COUNTERPART_AREA" in df.columns and (df.COUNTERPART_AREA == "W").any():
        df = df[df.COUNTERPART_AREA == "W"]

    df["ccy"] = df.REF_AREA.map(a2c)
    df["date"] = _parse_period(df.TIME_PERIOD)
    df["value"] = pd.to_numeric(df.OBS_VALUE, errors="coerce")
    df["metric"] = "current_account"
    df["source"] = "oecd_bop_ca_pct_gdp"
    df["fetched_at"] = utcnow()
    return (df.dropna(subset=["value"]).sort_values(["ccy", "date"])
            .reset_index(drop=True)[
                ["date", "ccy", "metric", "value", "source", "fetched_at"]])


def fetch_niip(session=None) -> pd.DataFrame:
    """Net international investment position as % of GDP (Pillar C structural stock).

    For the 7 non-EUR legs: OECD net IIP (XDC) / annualized nominal GDP (trailing 4
    quarters, XDC) from the QNA. For the euro area: Eurostat's ready-made NIIP%GDP
    (tipsii40), since the euro-area aggregate isn't in the OECD IIP dataflow.
    Quarterly. Tidy contract: [date, ccy, metric, value, source, fetched_at].
    """
    from cte.config import (OECD_BOP_DATAFLOW, OECD_IIP_DATAFLOW,
                            OECD_QNA_DATAFLOW)
    sess = session or make_session()
    ccys = ("USD", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD")
    a2c = {OECD_REF_AREA[c]: c for c in ccys}

    def _oecd(dataflow, extra=""):
        url = (f"{OECD_SDMX_BASE}/{dataflow}/all"
               f"?startPeriod=1995-Q1&dimensionAtObservation=AllDimensions")
        return pd.read_csv(io.BytesIO(sess.get(
            url, headers={"User-Agent": HTTP_UA, "Accept": OECD_CSV_ACCEPT},
            timeout=HTTP_TIMEOUT).content))

    # net IIP (financial account net position), domestic currency
    iip = _oecd(OECD_IIP_DATAFLOW)
    iip = iip[(iip.MEASURE == "FA") & (iip.ACCOUNTING_ENTRY == "N")
              & (iip.UNIT_MEASURE == "XDC") & (iip.FREQ == "Q")
              & (iip.FS_ENTRY == "LE")           # LE = closing stock (the NIIP itself);
              & (iip.REF_AREA.isin(a2c))].copy()  # exclude K*/KA revaluation-flow entries
    iip["date"] = _parse_period(iip.TIME_PERIOD)
    iip["ccy"] = iip.REF_AREA.map(a2c)
    iip["niip"] = pd.to_numeric(iip.OBS_VALUE, errors="coerce")

    # nominal GDP (current prices), domestic currency, quarterly -> annualized (4Q sum)
    gdp = _oecd(OECD_QNA_DATAFLOW)
    gdp = gdp[(gdp.TRANSACTION == "B1GQ") & (gdp.PRICE_BASE == "V")
              & (gdp.ADJUSTMENT == "Y") & (gdp.FREQ == "Q")
              & (gdp.EXPENDITURE == "_Z") & (gdp.REF_AREA.isin(a2c))].copy()
    gdp["date"] = _parse_period(gdp.TIME_PERIOD)
    gdp["ccy"] = gdp.REF_AREA.map(a2c)
    gdp["ngdp"] = pd.to_numeric(gdp.OBS_VALUE, errors="coerce")
    gdp = gdp.sort_values(["ccy", "date"])
    gdp["ann_gdp"] = (gdp.groupby("ccy")["ngdp"]
                      .rolling(4).sum().reset_index(level=0, drop=True))

    m = iip.merge(gdp[["ccy", "date", "ann_gdp"]], on=["ccy", "date"], how="inner")
    m["value"] = (m["niip"] / m["ann_gdp"]) * 100.0
    out7 = m.dropna(subset=["value"])[["date", "ccy", "value"]].copy()
    out7["source"] = "oecd_iip_over_qna_gdp"

    # euro area: Eurostat NIIP%GDP directly (tipsii40)
    eur = _fetch_eurostat_niip(sess)

    out = pd.concat([out7, eur], ignore_index=True)
    out["metric"] = "niip"
    out["fetched_at"] = utcnow()
    return (out.dropna(subset=["value"]).sort_values(["ccy", "date"])
            .reset_index(drop=True)[
                ["date", "ccy", "metric", "value", "source", "fetched_at"]])


def _fetch_eurostat_niip(sess) -> pd.DataFrame:
    url = ("https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
           "tipsii40/?format=SDMX-CSV&sinceTimePeriod=1995")
    raw = pd.read_csv(io.BytesIO(sess.get(
        url, headers={"User-Agent": HTTP_UA}, timeout=HTTP_TIMEOUT).content))
    d = raw[raw.geo == "EA20"].copy()
    d["date"] = pd.PeriodIndex(d.TIME_PERIOD.str.replace("-", ""),
                               freq="Q").to_timestamp(how="end").normalize()
    d["ccy"] = "EUR"
    d["value"] = pd.to_numeric(d.OBS_VALUE, errors="coerce")
    d["source"] = "eurostat_tipsii40"
    return d[["date", "ccy", "value", "source"]]


if __name__ == "__main__":
    df = fetch_current_account()
    print(f"current account: {len(df)} rows | currencies: {df.ccy.nunique()}")
    latest = df.sort_values("date").groupby("ccy").tail(1)
    for _, r in latest.iterrows():
        print(f"  {r.ccy}: {r.value:+.1f}% of GDP ({r.date.date()})")
    print()
    n = fetch_niip()
    print(f"NIIP: {len(n)} rows | currencies: {n.ccy.nunique()}")
    nlatest = n.sort_values("date").groupby("ccy").tail(1)
    for _, r in nlatest.iterrows():
        print(f"  {r.ccy}: {r.value:+.0f}% of GDP ({r.date.date()})  [{r.source}]")
