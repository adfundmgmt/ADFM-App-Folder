"""FRED adapter — macro/fundamental backbone + US real yields (spec §3, §14).

Serves the cross-country fundamental inputs for all eight legs — GDP,
unemployment, core CPI, policy rate — plus US TIPS constant-maturity real yields.
Returns *raw levels* in a tidy long contract; all derived statistics (CPI YoY,
GDP momentum, unemployment 3-month change, real policy rate, dual-horizon
z-scores) are computed later in the transform layer, not here.

The API key is read from the environment variable named by FRED_API_KEY_ENV and
is never logged or committed. Get a free key at https://fred.stlouisfed.org.

Tidy contract: [date, ccy, metric, value, source, fetched_at]
  metric ∈ {gdp, unemp, core_cpi, policy, real_2y, real_10y}
"""
from __future__ import annotations

import os

import pandas as pd

from cte.adapters.base import make_session, utcnow
from cte.config import (
    FRED_API_KEY_ENV, FRED_BASE, FRED_SERIES, FRED_US_REAL,
    HTTP_TIMEOUT, HTTP_UA,
)

_OBSERVATIONS = FRED_BASE + "/series/observations"


class FredKeyMissing(RuntimeError):
    pass


def get_api_key() -> str:
    key = os.environ.get(FRED_API_KEY_ENV, "").strip()
    if not key:
        raise FredKeyMissing(
            f"Set {FRED_API_KEY_ENV} in the environment (free key at "
            f"https://fred.stlouisfed.org/docs/api/api_key.html)."
        )
    return key


def fetch_series(series_id: str, *, observation_start: str = "1990-01-01",
                 session=None, api_key: str | None = None) -> pd.DataFrame:
    """Raw observations for one FRED series → DataFrame[date, value]."""
    sess = session or make_session()
    key = api_key or get_api_key()
    params = {
        "series_id": series_id, "api_key": key, "file_type": "json",
        "observation_start": observation_start,
    }
    r = sess.get(_OBSERVATIONS, params=params,
                 headers={"User-Agent": HTTP_UA, "Accept": "application/json"},
                 timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    rows = []
    for o in obs:
        v = o.get("value")
        if v in (None, ".", ""):  # FRED marks missing as "."
            continue
        rows.append({"date": pd.to_datetime(o["date"]), "value": float(v)})
    return pd.DataFrame(rows)


def _tidy(frames: list[pd.DataFrame]) -> pd.DataFrame:
    cols = ["date", "ccy", "metric", "value", "source", "fetched_at"]
    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    return df[cols].sort_values(["ccy", "metric", "date"]).reset_index(drop=True)


def fetch_fred_macro(observation_start: str = "1990-01-01") -> pd.DataFrame:
    """Cross-country macro backbone + US real yields, one tidy long frame.

    Failures on individual series are collected and reported rather than aborting
    the whole pull — a discontinued OECD ID shouldn't take down the other 40.
    """
    sess = make_session()
    key = get_api_key()
    fetched = utcnow()
    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str, str]] = []

    def _grab(ccy: str, metric: str, sid: str) -> None:
        try:
            d = fetch_series(sid, observation_start=observation_start,
                             session=sess, api_key=key)
            if d.empty:
                failures.append((ccy, metric, f"{sid}: empty"))
                return
            d["ccy"] = ccy
            d["metric"] = metric
            d["source"] = f"fred:{sid}"
            d["fetched_at"] = fetched
            frames.append(d)
        except Exception as e:  # noqa: BLE001 — surface, don't abort
            failures.append((ccy, metric, f"{sid}: {type(e).__name__}"))

    for ccy, metrics in FRED_SERIES.items():
        for metric, sid in metrics.items():
            _grab(ccy, metric, sid)
    for metric, sid in FRED_US_REAL.items():
        _grab("USD", metric, sid)

    out = _tidy(frames)
    # FRED labels months/quarters at the period START; OECD uses period END.
    # Snap FRED dates to period-end per series so cross-source dates align.
    if not out.empty:
        parts = []
        for _, g in out.groupby(["ccy", "metric"], sort=False):
            g = g.sort_values("date").copy()
            gap = g["date"].diff().dt.days.median()
            if gap and gap > 45:          # quarterly
                g["date"] = g["date"] + pd.offsets.QuarterEnd(0)
            elif gap and gap > 4:         # monthly
                g["date"] = g["date"] + pd.offsets.MonthEnd(0)
            parts.append(g)
        out = pd.concat(parts, ignore_index=True)
    out.attrs["failures"] = failures
    return out


if __name__ == "__main__":
    try:
        df = fetch_fred_macro()
    except FredKeyMissing as e:
        raise SystemExit(str(e))
    fails = df.attrs.get("failures", [])
    print(f"rows: {len(df)} | series ok: "
          f"{df.groupby(['ccy','metric']).ngroups} | failures: {len(fails)}")
    for ccy, metric, why in fails:
        print(f"  FAIL {ccy}/{metric}: {why}")
