# ADFM Tools data, research, and risk upgrade

The user approved implementation of the recommendations made after reviewing all 25 Streamlit page files on 2026-09-29. Deliver the full scope in the existing application, preserve routes and calculations except where a regression test establishes the intended correction, and publish only after the repository checks pass.

## Data and performance

Centralize daily Yahoo transport, retries, bounded caches, observation-date handling and last-good recovery. Preserve yfinance-compatible layouts for legacy pages through `download_market_data(tickers, **kwargs) -> DataFrame`; accepted arguments include start, end, period, interval, auto_adjust, group_by, progress and threads. Daily completed-session policy is explicit, and intraday overlays remain separate. Bridge at most two interior observations for documented ratio/calendar alignment; never extend trailing prices or OHLCV. Current signals require sufficiently recent observed inputs. Retain per-symbol source/observed dates. Record bounded provider/page timings, cache delivery, failures, and process memory, with developer diagnostics collapsed and no manual refresh controls.

## Bonds

Add official daily 10-year sovereign observations where a validated source exists. Keep source/tenor/curve-basis labels exact; do not splice monthly averages into daily histories or confuse spot curves with benchmark yields. Preserve long OECD monthly studies as a separate frequency. Add fixed-level failed-breakout confirmation, regime-aware non-overlapping control samples, independent sample counts, confidence intervals, forward adverse/favorable excursions and chronological holdout summaries to bond event studies. Use month-end availability for monthly data and suppress stale current signals. Keep one main sortable monitor table and a selected instrument chart; Max remains default.

## Historical information and contracts

Preserve every captured signal version idempotently with locking and atomic writes; provide a configurable persistent path and a scheduled public-data capture mechanism. Macro-conditioned historical studies explicitly distinguish contemporaneous/revised values; use historical vintages and known release availability when reconstructing a trading decision, otherwise exclude unavailable inputs rather than invent their publication dates. CFTC holiday/exception handling uses verified publication records where available and clearly identifies unresolved historical release timing. Futures studies expose exact contract specifications and provider roll limitations, detect discontinuity candidates, and label continuous-series price studies separately from realizable contract P&L. Known contract multipliers support exact scenario calculations.

## Portfolio risk

Extend the existing Position Sizing page with a private, session-only dated CSV holdings input and a single portfolio stress table. Support shares, FX spot, futures, direct DV01 exposures and listed equity/ETF calls and puts. Validate direction, quantity, multiplier, underlying marks, option strike/expiry/volatility and valuation date; never assume historical fund holdings/NAV. Apply equity, yield, FX, commodity and volatility shocks together. Reprice options with a clearly labeled European approximation, use supplied DV01/convexity for bond exposures, and show dollar/NAV P&L, scenario NAV, gross factor exposure, cash/margin requirement estimates and missing-input limitations. American exercise/assignment and broker margin cannot be presented as exact. Uploads and account positions never enter git or public telemetry.

## Layout and maintainability

Preserve consolidated tables, header sorting, white background, compact rows and all existing routes. Make detailed charts selected-on-demand where supported; collapse methodology and secondary outputs, removing unnecessary tabs and KPI-card rendering on modified pages without deleting analytical capability. Extract basket/chart calculations or transport helpers into focused modules with equivalence tests; avoid a broad visual redesign.

## Acceptance

Behavior tests cover partial provider failures, stale endpoints, holidays, monthly availability, independent event samples, concurrent/idempotent signal capture, futures signs/multipliers, invalid portfolio rows, option repricing and combined fund stress. Run the existing suite, coverage >=45%, dependency checks, compilation, required lint and meaningful Streamlit smoke checks. Reconcile representative new daily sources against official records; unavailable sources must fail visibly and leave other instruments usable.
