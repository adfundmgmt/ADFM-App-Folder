# FRED reliability release

The application now reads validated macro snapshots before making provider
requests. Twenty-seven government and Federal Reserve series are refreshed by
`refresh-fred.yml` three times each weekday. A provider outage cannot remove a
previously validated snapshot. Each page shows observation dates separately
from download dates, expected cadence, units, and available history.

## Request and storage behavior

- FRED API requests use `FRED_API_KEY` from the environment or Streamlit secrets.
  API metadata must match registered units and frequency. Requests use natural
  units and paginate observations. Credentials and request URLs never enter
  public diagnostics or snapshot files.
- With no key configured, the service uses FRED's CSV endpoint. This is one
  transport, not two supposedly independent CSV fallbacks. Scheduled stored
  data protects the page from endpoint failures.
- Known, validated snapshots are served without network requests, even when a
  refresh is overdue. Overdue observation dates are marked stale. A scheduled
  failure retains the prior file and is reported as a failed GitHub job.
- Unregistered or vendor series use a local last-good cache with a six-hour
  refresh interval. Transient errors have bounded retries and a five-minute
  failure cooldown. At most three downloads run in a process at once.
- Cache writes are atomic. Invalid dates, duplicate dates, invalid numbers,
  implausible registered units, unexpectedly truncated government histories,
  and older replacement datasets are rejected. Missing observations remain
  missing. Accepted revisions replace the requested historical window rather
  than mixing old and new vintages.
- The public snapshot allowlist excludes vendor credit and equity indices;
  those observations remain in the application's runtime cache. Runtime
  caches can be lost on a Streamlit restart. Longer licensed histories require
  an appropriate source and are never reconstructed from invented values.

## Page changes

| Page | Result |
|---|---|
| Yield Curve | Official curve preferred; 2Y and 2s10s added; 5Y/10Y real yields and breakevens added. Yahoo is a separate nominal fallback, never spliced into the official curve. |
| Liquidity | Shared snapshot recovery replaces duplicated calls to the same CSV endpoint; failed/stale source status is visible. |
| Credit | Shared source service; ICE ranks show actual available history instead of incorrectly labeling shorter windows 5Y. Sovereign FRED history shares the same cache. |
| Global Macro | Shared diagnostics and extra official inflation, unemployment, payroll, production, and claims context. Existing regime thresholds are unchanged. |
| Market Stress | Weekly NFCI and STLFSI4 comparisons, separated from daily composite and hedge thresholds. |
| Market Memory | Official rates and financial-conditions context; existing calendar-year analog rankings are unchanged. |
| Seasonality | Shared FRED history and explicit retrospective-revision/recession-label disclosure. |
| Currency adapter | Shared source recovery and preserved original fetch timestamps. The public Currency page still consumes its separately generated engine snapshot. |

## Historical interpretation

FRED observation dates describe the measured period, not when it became
available. Monthly and quarterly freshness limits account for period-start
labels and publication lag. Current FRED histories can contain revisions.
Seasonality's recession filters remain retrospective.

The service supports explicit ALFRED vintage-date requests through the API.
Vintage caches are isolated, and a missing key or failed vintage request never
falls back to today's data. This does not turn existing historical pages into
publication-by-publication point-in-time backtests; that requires a separate
vintage-aware analytical design.

## Configuration and operations

Set `FRED_API_KEY` as a GitHub Actions repository secret for authenticated
scheduled pulls. A Streamlit secret of the same name enables authenticated
runtime requests. Do not put a key in source, a commit, or a chat message.
The application functions through CSV and saved snapshots until a key exists.

Run `python -m scripts.refresh_fred_snapshot` from the repository root. Inspect
`data/fred/refresh_status.json` for per-series results. The workflow commits
successful validated updates before reporting any partial failure and retries
safe pushes when another update reaches main first.

## Validation

Regression tests cover cache survival, offline rendering, revisions, truncated
history, stale observations, corrupt files, safe error messages, natural units,
API pagination, retries, and isolated historical vintages. A Streamlit fixture
checks the expanded Treasury page and its optional source tables.

With downloads disabled, all 27 saved series loaded in 0.566 seconds locally.
Actual-provider local page checks completed without exceptions or error banners:
Treasury 2.65s, Liquidity 41.46s, Macro 6.36s, Credit 6.18s, Stress 3.05s,
Memory 17.41s, Seasonality 5.59s. These single-run local timings include other
providers and are not production latency guarantees or before/after benchmarks.

## Primary references

- [FRED observations API](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)
- [FRED series metadata](https://fred.stlouisfed.org/docs/api/fred/series.html)
- [ALFRED vintage data](https://fred.stlouisfed.org/docs/api/fred/alfred.html)
- [ICE high-yield history restriction](https://fred.stlouisfed.org/series/BAMLH0A0HYM2)
