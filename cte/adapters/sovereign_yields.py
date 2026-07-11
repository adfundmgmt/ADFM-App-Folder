"""
Sovereign-yield adapters — spec §7. Nine official daily endpoints for eight
currencies (EUR carries two legs: ECB euro-area composite = primary, German
Bund = secondary cross-reference).

Each per-source parser returns the common tidy_yields contract
[date, ccy, tenor, value, source, fetched_at] with tenor in {2Y, 3Y, 10Y}.
`resolve_carry_tenor` selects the 2Y carry /
priced-path leg (every currency exposes a live 2Y today).

Source-specific notes captured during build:
  USD  US Treasury par-curve CSV, one file per year.
  EUR  ECB Data Portal SDMX, csvdata + startPeriod (the lastNObservations+
       jsondata combo silently drops longer maturities). AAA curve G_N_A.
  DE   Bundesbank BBSIS SDMX-JSON, residual-maturity keys R02XX/R03XX/R10XX.
  JPY  MoF JGB CSV, Shift-JIS, Japanese-era dates; merge full-history file
       (lags ~1mo) with current-year file for the fresh tail.
  GBP  BoE GLC nominal spot curve (the "/UK DMO nominal gilt curve" of §7),
       "latest" zip → "4. spot curve" sheet, maturity columns in years.
  CHF  SNB data-portal spot-rate curve (chart 'rendeidglfzch', 2Y/5Y/10Y/20Y),
       fetched from the portal's JSON chart endpoint. Replaced the discontinued
       rendoblid cube (frozen 2025-07-31); no 3Y node, but 2Y and 10Y are live.
  CAD  Bank of Canada Valet API, benchmark bond yields.
  AUD  RBA table F2, series FCMYGBAG{2,3,10}D.
  NZD  RBNZ table B2 (xlsx), secondary-market govt closing yields; no 3Y node.
"""
from __future__ import annotations

import datetime as dt
import io
import re
import zipfile

import pandas as pd

from cte.adapters.base import http_get, make_session, tidy_yields, utcnow
from cte.config import CARRY_TENOR, HTTP_TIMEOUT, HTTP_UA, TENORS_WANTED

# ----------------------------------------------------------------- USD
def fetch_us(years_back: int = 10) -> pd.DataFrame:
    this_year = dt.date.today().year
    rows = []
    for yr in range(this_year - years_back, this_year + 1):
        url = (f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
               f"daily-treasury-rates.csv/{yr}/all?type=daily_treasury_yield_curve"
               f"&field_tdr_date_value={yr}&page&_format=csv")
        try:
            df = pd.read_csv(io.BytesIO(http_get(url).content))
        except Exception:
            continue
        for tenor, col in (("2Y", "2 Yr"), ("3Y", "3 Yr"), ("10Y", "10 Yr")):
            if col in df.columns:
                for _, r in df[["Date", col]].dropna().iterrows():
                    rows.append({"date": r["Date"], "ccy": "USD", "tenor": tenor, "value": r[col]})
    return tidy_yields(rows, "us_treasury")


# ----------------------------------------------------------------- EUR (ECB) + DE (Bundesbank)
def fetch_ecb(years_back: int = 10) -> pd.DataFrame:
    start = (dt.date.today() - dt.timedelta(days=365 * years_back)).isoformat()
    rows = []
    for tenor in ("2Y", "3Y", "10Y"):
        key = f"B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{tenor}"
        url = f"https://data-api.ecb.europa.eu/service/data/YC/{key}?startPeriod={start}&format=csvdata"
        df = pd.read_csv(io.BytesIO(http_get(url).content))
        for _, r in df[["TIME_PERIOD", "OBS_VALUE"]].dropna().iterrows():
            rows.append({"date": r["TIME_PERIOD"], "ccy": "EUR", "tenor": tenor, "value": r["OBS_VALUE"]})
    return tidy_yields(rows, "ecb_yc")


def fetch_bundesbank(years_back: int = 10) -> pd.DataFrame:
    rows = []
    for tenor, rc in (("2Y", "R02XX"), ("3Y", "R03XX"), ("10Y", "R10XX")):
        key = f"D.I.ZAR.ZI.EUR.S1311.B.A604.{rc}.R.A.A._Z._Z.A"
        url = f"https://api.statistiken.bundesbank.de/rest/data/BBSIS/{key}?format=json"
        j = http_get(url).json()
        ser = j["data"]["dataSets"][0]["series"]
        skey = next(iter(ser))
        obs = ser[skey]["observations"]
        times = j["data"]["structure"]["dimensions"]["observation"][0]["values"]
        for idx, val in obs.items():
            d = times[int(idx)]["id"]
            rows.append({"date": d, "ccy": "EUR", "tenor": tenor, "value": val[0]})
    return tidy_yields(rows, "bundesbank")


# ----------------------------------------------------------------- JPY
_ERA = {"S": 1925, "H": 1988, "R": 2018, "M": 1867, "T": 1911}

def _parse_jp_date(s: str):
    s = str(s).strip()
    m = re.match(r"^([SHRMT])\.?(\d{1,2})\.(\d{1,2})\.(\d{1,2})$", s)
    if m:
        era, y, mo, d = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        return dt.date(_ERA[era] + y, mo, d)
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


def _parse_jgb_csv(content: bytes) -> list[dict]:
    txt = content.decode("shift_jis", errors="replace")
    rows = []
    for line in txt.splitlines():
        parts = line.split(",")
        if not parts:
            continue
        d = _parse_jp_date(parts[0])
        if d is None:
            continue  # header/footer/era-less line
        # column order: date,1Y,2Y,3Y,4Y,5Y,6Y,7Y,8Y,9Y,10Y,...
        def cell(i):
            try:
                return float(parts[i])
            except Exception:
                return None
        for tenor, i in (("2Y", 2), ("3Y", 3), ("10Y", 10)):
            v = cell(i)
            if v is not None:
                rows.append({"date": d, "ccy": "JPY", "tenor": tenor, "value": v})
    return rows


def fetch_jp() -> pd.DataFrame:
    rows = []
    for url in (
        "https://www.mof.go.jp/jgbs/reference/interest_rate/data/jgbcm_all.csv",  # history
        "https://www.mof.go.jp/jgbs/reference/interest_rate/jgbcm.csv",            # fresh tail
    ):
        try:
            rows += _parse_jgb_csv(http_get(url).content)
        except Exception:
            pass
    df = tidy_yields(rows, "jp_mof")
    return df.drop_duplicates(subset=["date", "ccy", "tenor"]).reset_index(drop=True)


# ----------------------------------------------------------------- GBP (BoE GLC)
def _boe_parse_spot(xlsx_bytes: bytes) -> list[dict]:
    xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    # sheet name varies by vintage: "4. spot curve" / "4. nominal spot curve";
    # pick the full spot curve, excluding the "short end" sheet.
    sheet = next(s for s in xls.sheet_names
                 if "spot curve" in s.lower() and "short end" not in s.lower())
    df = xls.parse(sheet_name=sheet, header=3)
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    mat_cols = [c for c in df.columns if isinstance(c, (int, float))]
    rows = []
    for tenor, target in (("2Y", 2.0), ("3Y", 3.0), ("10Y", 10.0)):
        col = min(mat_cols, key=lambda c: abs(float(c) - target))
        for _, r in df[["date", col]].dropna().iterrows():
            rows.append({"date": r["date"], "ccy": "GBP", "tenor": tenor,
                         "value": r[col]})
    return rows


def fetch_boe(history: bool = False) -> pd.DataFrame:
    """`latest` zip = fresh current-month tail (daily use). history=True pulls the
    full GLC archive (38MB, nominal spot curve back to 1979) for the cold-start
    backfill; the daily merge then appends the current tail from `latest`."""
    base = ("https://www.bankofengland.co.uk/-/media/boe/files/statistics/"
            "yield-curves/")
    url = base + ("glcnominalddata.zip" if history else "latest-yield-curve-data.zip")
    z = zipfile.ZipFile(io.BytesIO(http_get(url).content))
    names = [n for n in z.namelist()
             if "Nominal" in n and n.lower().endswith("xlsx")]
    rows = []
    for name in names:
        rows.extend(_boe_parse_spot(z.read(name)))
    return tidy_yields(rows, "boe_glc")


# ----------------------------------------------------------------- CHF (SNB, discontinued)
_SNB_BASE = "https://data.snb.ch/json"
# Live daily Swiss Confederation spot-rate curve. The old rendoblid cube froze
# on 2025-07-31; SNB moved the series to the data-portal chart 'rendeidglfzch'
# ("Spot interest rates ... for selected maturities": 2Y/5Y/10Y/20Y). The portal
# is a JS SPA, but its chart data is served by a JSON endpoint that resolves a
# chartId to its cube config + observations. We hit that endpoint directly.
_SNB_CHART_ID = "rendeidglfzch"


def _snb_page_view_time(sess) -> str:
    """SNB's chart endpoint requires the app's pageViewTime query param, which
    the SPA reads from /json/application/properties on load."""
    r = sess.get(f"{_SNB_BASE}/application/properties",
                 headers={"User-Agent": HTTP_UA, "Accept": "application/json"},
                 timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()["pageViewTime"]


def fetch_snb() -> pd.DataFrame:
    sess = make_session()
    pvt = _snb_page_view_time(sess)
    resp = sess.post(
        f"{_SNB_BASE}/chart/getAirchartConfigAndData",
        params={"lang": "en", "pageViewTime": pvt},
        json={"chartId": _SNB_CHART_ID, "maxZoomOut": False},
        headers={"User-Agent": HTTP_UA,
                 "Accept": "application/json, text/plain, */*",
                 "Content-Type": "application/json"},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()

    # Build series-key -> tenor from the plot labels ("N years" -> "NY") so we
    # never hard-code pk1..pk4; if SNB reorders keys or adds a node this adapts.
    # The plot config nests {key,label} pairs at varying depth, so walk it.
    want = set(TENORS_WANTED)  # {2Y, 3Y, 10Y}
    key_to_tenor: dict[str, str] = {}

    def _harvest(node) -> None:
        if isinstance(node, dict):
            if "key" in node and isinstance(node.get("label"), dict):
                label = node["label"].get("en", "")
                m = re.match(r"\s*(\d+)\s*year", label, re.IGNORECASE)
                if m:
                    tenor = f"{int(m.group(1))}Y"
                    if tenor in want:
                        key_to_tenor[node["key"]] = tenor
            for v in node.values():
                _harvest(v)
        elif isinstance(node, list):
            for v in node:
                _harvest(v)

    _harvest(payload["config"].get("plots"))

    rows = []
    for date, key, value in payload["data"]["data"]:
        tenor = key_to_tenor.get(key)
        if tenor is not None:
            rows.append({"date": date, "ccy": "CHF", "tenor": tenor, "value": value})
    return tidy_yields(rows, "snb_airchart_spot")


# ----------------------------------------------------------------- CAD (BoC Valet)
def fetch_boc() -> pd.DataFrame:
    rows = []
    for tenor, sc in (("2Y", "BD.CDN.2YR.DQ.YLD"), ("3Y", "BD.CDN.3YR.DQ.YLD"),
                      ("10Y", "BD.CDN.10YR.DQ.YLD")):
        url = f"https://www.bankofcanada.ca/valet/observations/{sc}/json"
        j = http_get(url).json()
        for o in j.get("observations", []):
            v = o.get(sc, {}).get("v")
            if v not in (None, ""):
                rows.append({"date": o["d"], "ccy": "CAD", "tenor": tenor, "value": v})
    return tidy_yields(rows, "boc_valet")


# ----------------------------------------------------------------- AUD (RBA F2)
def fetch_rba() -> pd.DataFrame:
    url = "https://www.rba.gov.au/statistics/tables/csv/f2-data.csv"
    raw = http_get(url).content.decode("utf-8", errors="replace").splitlines()
    # find the "Series ID" row and the header it implies
    sid_row = next(i for i, l in enumerate(raw) if l.startswith("Series ID"))
    ids = raw[sid_row].split(",")
    col_for = {}
    for tenor, sid in (("2Y", "FCMYGBAG2D"), ("3Y", "FCMYGBAG3D"), ("10Y", "FCMYGBAG10D")):
        if sid in ids:
            col_for[tenor] = ids.index(sid)
    rows = []
    for line in raw[sid_row + 1:]:
        cells = line.split(",")
        if not cells or not re.match(r"\d{1,2}-\w{3}-\d{4}", cells[0]):
            continue
        d = pd.to_datetime(cells[0], format="%d-%b-%Y", errors="coerce")
        if pd.isna(d):
            continue
        for tenor, ci in col_for.items():
            if ci < len(cells) and cells[ci]:
                rows.append({"date": d, "ccy": "AUD", "tenor": tenor, "value": cells[ci]})
    return tidy_yields(rows, "rba_f2")


# ----------------------------------------------------------------- NZD (RBNZ B2)
def fetch_rbnz() -> pd.DataFrame:
    # RBNZ migrated B2 from Refinitiv mid-rates (hb2-daily.csv, frozen 2025-08-22)
    # to NZFMA end-of-day closing rates (hb2-daily-close.xlsx) in Aug 2025.
    # Series IDs are unchanged; values are now closing rather than 11:10am mids.
    url = "https://www.rbnz.govt.nz/-/media/project/sites/rbnz/files/statistics/series/b/b2/hb2-daily-close.xlsx"
    # RBNZ rejects requests without a Referer (bot protection); send browser headers.
    content = http_get(url, headers={
        "Referer": "https://www.rbnz.govt.nz/statistics/series/"
                   "exchange-and-interest-rates",
        "Accept": "application/vnd.openxmlformats-officedocument."
                  "spreadsheetml.sheet,*/*",
        "Accept-Language": "en-US,en;q=0.9"}).content
    data = pd.read_excel(io.BytesIO(content), sheet_name="Data", header=None)
    # locate the row carrying Series Ids by scanning for the known codes
    want = {"2Y": "INM.DG102.NZZCF", "10Y": "INM.DG110.NZZCF"}  # no 3Y node in B2
    id_row = None
    for i in range(min(12, len(data))):
        rowvals = data.iloc[i].astype(str).tolist()
        if any(code in rowvals for code in want.values()):
            id_row = i
            break
    rows = []
    if id_row is not None:
        header = data.iloc[id_row].astype(str).tolist()
        col_for = {t: header.index(c) for t, c in want.items() if c in header}
        for i in range(id_row + 1, len(data)):
            d = pd.to_datetime(data.iloc[i, 0], errors="coerce")
            if pd.isna(d):
                continue
            for tenor, ci in col_for.items():
                v = data.iloc[i, ci]
                if pd.notna(v):
                    rows.append({"date": d, "ccy": "NZD", "tenor": tenor, "value": v})
    return tidy_yields(rows, "rbnz_b2")


# ----------------------------------------------------------------- registry + fallback
PRIMARY = {
    "USD": fetch_us, "EUR": fetch_ecb, "JPY": fetch_jp, "GBP": fetch_boe,
    "CHF": fetch_snb, "CAD": fetch_boc, "AUD": fetch_rba, "NZD": fetch_rbnz,
}
SECONDARY = {"EUR_DE": fetch_bundesbank}


def resolve_carry_tenor(df_ccy: pd.DataFrame) -> str | None:
    """Carry / priced-path leg is the 2Y for every currency in the universe.

    Returns "2Y" when present (always, on current data), else None so callers
    can surface a genuine gap rather than silently mis-selecting a tenor.
    """
    return CARRY_TENOR if CARRY_TENOR in set(df_ccy["tenor"].unique()) else None


def fetch_all_yields(primary_only: bool = True,
                     full_history: bool = False) -> pd.DataFrame:
    frames = []
    reg = dict(PRIMARY)
    if not primary_only:
        reg.update(SECONDARY)
    # deep-history overrides for the cold-start backfill
    overrides = {}
    if full_history:
        overrides = {"GBP": lambda: fetch_boe(history=True),
                     "EUR": lambda: fetch_ecb(years_back=30)}
    for name, fn in reg.items():
        call = overrides.get(name, fn)
        try:
            frames.append(call())
        except Exception as e:  # keep going; report at verify time
            print(f"[warn] {name} failed: {type(e).__name__}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
