"""Native ADFM HTTP boundary. Production imports only adfm_engine."""
from __future__ import annotations

import hmac
import logging
import os
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from adfm_engine.data.market import configure_yfinance_cache
from adfm_engine.services import DataUnavailable, load_rate_of_change, overview
from adfm_engine.leadership_service import load_leadership
from adfm_engine.volatility_service import load_volatility
from adfm_engine.ratio_service import load_ratios
from adfm_engine.macro_service import load_macro_regime
from adfm_engine.yield_service import load_yields
from adfm_engine.liquidity_service import load_liquidity
from adfm_engine.credit_service import load_credit
from adfm_engine.cftc_service import load_cftc
from adfm_engine.options_service import load_options
from adfm_engine.underwriter_service import load_underwriter
from adfm_engine.hedge_service import load_hedge
from adfm_engine.stress_service import load_stress
from adfm_engine.calendar_service import load_calendar
from adfm_engine.sec13f_service import load_sec13f, release_list
from adfm_engine.jobs import JobQueue
from adfm_engine.baskets_service import load_baskets
from adfm_engine.baskets_legacy_math import CATEGORIES
from adfm_engine.commodity_service import load_commodity_event_study
from adfm_engine.sector_rotation_service import load_sector_rotation
from adfm_engine.volume_sentiment_service import load_volume_sentiment
from adfm_engine.etf_flow_service import load_etf_flow
from adfm_engine.position_sizing_service import load_position_sizing
from adfm_engine.currency_tension_service import load_currency_tension, DEFAULT_WEIGHTS, PILLAR_AXIS
from adfm_engine.seasonality_service import load_monthly_seasonality
from adfm_engine.sector_rotation_config import UNIVERSE_SCOPES, BENCHMARKS, ROTATION_MODES, WINDOW_PRESETS, TRAIL_OPTIONS, LABEL_MODES, SECTOR_GROUP_COLORS
from pathlib import Path

logger = logging.getLogger("adfm.api")


class BasketParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preset: Literal["YTD", "1W", "1M", "3M", "6M", "1Y", "3Y", "5Y"] = "YTD"
    categories: list[str] | None = Field(default=None, max_length=20)
    market_cap_filter: bool = False
    stale_days: int = Field(default=30, ge=10, le=90, multiple_of=5)

    @field_validator("categories")
    @classmethod
    def valid_categories(cls, value):
        if value is not None and (len(value) != len(set(value)) or any(item not in CATEGORIES for item in value)):
            raise ValueError("Unknown or duplicate basket category.")
        return value


class CommodityParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    symbol: str = Field(default="CL=F", min_length=2, max_length=20, pattern=r"^[A-Z0-9^=._-]+$")
    signal_type: Literal["Return threshold", "52-week breakout", "RSI extreme", "200D trend stretch"] = "Return threshold"
    direction: Literal["Rally", "Selloff", "High", "Low", "Overbought", "Oversold", "Above", "Below"] = "Rally"
    return_window: Literal["1M", "2M", "3M", "6M", "12M"] = "3M"
    threshold: float = Field(default=25.0, ge=1, le=300)
    rsi_period: int = Field(default=14, ge=5, le=50)
    spacing: Literal["1M", "2M", "3M", "6M", "12M"] = "3M"
    lookback: Literal["Max", "10Y", "25Y", "50Y"] = "Max"

    @model_validator(mode="after")
    def valid_signal(self):
        valid = {"Return threshold": ("Rally", "Selloff"),
                 "52-week breakout": ("High", "Low"),
                 "RSI extreme": ("Overbought", "Oversold"),
                 "200D trend stretch": ("Above", "Below")}
        if self.direction not in valid[self.signal_type]:
            raise ValueError("Direction must match the event signal.")
        if self.signal_type == "RSI extreme" and self.threshold > 99:
            raise ValueError("RSI threshold cannot exceed 99.")
        if self.signal_type == "200D trend stretch" and self.threshold > 200:
            raise ValueError("Trend stretch cannot exceed 200%.")
        return self


class SectorRotationParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    universe: Literal["Major sectors only", "Core subsectors", "Core + thematic subsectors"] = "Core subsectors"
    benchmark: Literal["SPY", "RSP", "QQQ", "IWM", "DIA", "TLT", "IEF", "UUP"] = "SPY"
    mode: Literal["Benchmark-relative rotation", "Absolute sector/subsector rotation"] = "Benchmark-relative rotation"
    window: Literal["Fast (1M vs 3M)", "Intermediate (3M vs 6M)", "Trend (6M vs 12M)"] = "Fast (1M vs 3M)"
    trail: Literal["None", "4 weeks", "8 weeks", "12 weeks"] = "4 weeks"
    groups: list[str] | None = Field(default=None, max_length=12)
    selected_ticker: str = Field(default="SMH", max_length=12, pattern=r"^[A-Z0-9^=._-]+$")
    label_mode: Literal["Top ranked only", "All tickers", "No labels"] = "Top ranked only"

    @field_validator("groups")
    @classmethod
    def valid_groups(cls, value):
        if value is not None and (len(value) != len(set(value)) or
                                  any(group not in SECTOR_GROUP_COLORS for group in value)):
            raise ValueError("Unknown or duplicate sector group.")
        return value


class VolumeSentimentParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    symbol: str = Field(default="QQQ", min_length=1, max_length=15, pattern=r"^[A-Z0-9^=._-]+$")
    lookback_months: int = Field(default=18, ge=6, le=48)
    volume_mode: Literal["Dollar volume", "Raw volume", "Turnover %"] = "Dollar volume"
    percentile_window: int = Field(default=126, ge=60, le=252)
    smooth_window: int = Field(default=20, ge=10, le=80)
    high_cutoff: int = Field(default=90, ge=75, le=99)
    low_cutoff: int = Field(default=10, ge=1, le=25)
    show_price_mas: bool = True
    event_filter: Literal["All extremes", "Heavy only", "Quiet only"] = "All extremes"
    max_event_rows: int = Field(default=12, ge=5, le=25)


class ETFFlowParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    period_label: Literal["1 Month", "3 Months", "6 Months", "12 Months", "YTD"] = "1 Month"


class PositionSizingParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    ticker: str = Field(default="AAPL", min_length=1, max_length=15, pattern=r"^[A-Z0-9^=._-]+$")
    direction: Literal["Long", "Short"] = "Long"
    conviction: int = Field(default=3, ge=1, le=5)
    horizon_label: Literal["1 month", "3 months", "1 year", "5 years"] = "3 months"
    portfolio_nav: float = Field(default=5_000_000, ge=1000, le=1e12)
    max_loss_pct: float = Field(default=1.25, ge=.1, le=10)
    hold_earnings: bool = True
    participation: int = Field(default=10, ge=1, le=25)
    liquidation_days: int = Field(default=3, ge=1, le=10)
    entry: float | None = Field(default=None, gt=0, le=1e8)
    target: float | None = Field(default=None, gt=0, le=1e8)
    stop: float | None = Field(default=None, gt=0, le=1e8)
    sampling_mode: Literal["Random historical blocks", "Recent-regime weighted blocks", "Chronological regime replay"] = "Random historical blocks"
    simulation_position_pct: float | None = Field(default=None, ge=.5, le=25)
    starting_balance: float | None = Field(default=None, ge=1000, le=1e12)
    seed: int | None = Field(default=None, ge=0, le=1_000_000_000)

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, value):
        return value.upper()

    @model_validator(mode="after")
    def valid_trade(self):
        if self.entry is not None and self.stop is not None:
            if (self.direction == "Long" and self.stop >= self.entry) or (self.direction == "Short" and self.stop <= self.entry):
                raise ValueError("Invalidation must be on the adverse side of entry.")
        if self.simulation_position_pct is not None and self.simulation_position_pct > self.conviction * 5:
            raise ValueError("Simulation exposure cannot exceed the conviction ceiling.")
        return self


class CurrencyTensionParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    horizon: Literal["struct", "regime", "secular"] = "struct"
    trail: int = Field(default=6, ge=0, le=12)
    asof: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    weights: dict[str, float] | None = None

    @field_validator("weights")
    @classmethod
    def valid_weights(cls, value):
        if value is None:
            return value
        if set(value) != set(DEFAULT_WEIGHTS) or any(not 0 <= weight <= 3 or round(weight * 4) != weight * 4 for weight in value.values()):
            raise ValueError("Supply all six pillar weights between 0 and 3 in quarter-point steps.")
        for axis in set(PILLAR_AXIS.values()):
            if not any(value[key] > 0 for key, group in PILLAR_AXIS.items() if group == axis):
                raise ValueError("Keep at least one pillar on each axis.")
        return value


class HedgeParameters(BaseModel):
    model_config=ConfigDict(extra="forbid")
    chart_years: Literal[1,2,3,5,10]=1

class StressParameters(BaseModel):
    model_config=ConfigDict(extra="forbid")
    lookback_years: Literal[1,2,3,5,10,25,50]=5
    target_mode: Literal["Auto","S&P 500","Nasdaq Composite"]="Auto"
    z_window_years: int=Field(default=3,ge=1,le=5)
    smoothing_mode: Literal["Fast - 3D","Base - 5D","Slow - 10D","21D","63D"]="Slow - 10D"

class CalendarParameters(BaseModel):
    model_config=ConfigDict(extra="forbid")
    horizon_days: Literal[14,30,60,90,120,180]=90
    include_macro: bool=True
    include_fed: bool=True
    hide_low: bool=False
    custom_text: str=Field(default="",max_length=24000)

class SEC13FParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    search_mode: Literal["Security","Manager"] = "Security"
    query: str = Field(default="INTC", min_length=1, max_length=200)
    release_slug: str = Field(default="",max_length=100,pattern=r"^[A-Za-z0-9_-]*$")
    position_kind: Literal["Long holdings","Call options","Put options","All reported"] = "Long holdings"
    minimum_portfolio_millions: float = Field(default=1000,ge=0,le=1e9)
    sort_label: Literal["Portfolio weight","Reported market value","Reported shares"] = "Portfolio weight"
    top_n: Literal[10,15,20,25,30,35,40,45,50] = 25
    candidate: int = Field(default=0,ge=0,le=24)
    manager_cik: str = Field(default="",pattern=r"^[0-9]{0,10}$")
    manager_filter: str = Field(default="",max_length=200)
    detail_columns: list[Literal["PORTFOLIO_WEIGHT_PCT","POSITION_VALUE_USD","REPORTED_SHARES","PORTFOLIO_VALUE_USD","LATEST_FILING_DATE","CIK","COMPONENT_COUNT","FILING_URL"]] | None = None
    portfolio_filter: str = Field(default="",max_length=200)
    portfolio_kind: Literal["All","Long","Call","Put"] = "All"

class JobParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[0-9a-f]{32}$")

class SeasonalityParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    symbol: str = Field(default="^SPX", min_length=1, max_length=20, pattern=r"^[A-Z0-9^=._-]+$")
    lookback: Literal["5Y","10Y","20Y","All","Custom"] = "10Y"
    start_year: int | None = Field(default=None, ge=1900, le=2100)
    end_year: int | None = Field(default=None, ge=1900, le=2100)
    cycle: Literal["All years","Election years","Midterm years","Pre-election years","Post-election years"] = "All years"
    complete_only: bool = True
    fed: Literal["All Fed regimes","Hiking","Cutting","Steady"] = "All Fed regimes"
    vix: Literal["All VIX regimes","VIX <15","VIX 15-20","VIX 20-25","VIX >25"] = "All VIX regimes"
    teny: Literal["All 10Y regimes","10Y rising","10Y falling","10Y flat"] = "All 10Y regimes"
    dxy: Literal["All dollar regimes","Dollar rising","Dollar falling","Dollar flat"] = "All dollar regimes"
    month: int | None = Field(default=None, ge=1, le=12)
    year: int | None = Field(default=None, ge=1900, le=2100)

    @model_validator(mode="after")
    def valid_window(self):
        if self.lookback == "Custom" and self.start_year and self.end_year and self.start_year > self.end_year:
            raise ValueError("Start year must precede end year.")
        return self

class UnderwriterParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(default="AAPL", min_length=1, max_length=200)


class ROCParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str = Field(default="^SPX", min_length=1, max_length=32, pattern=r"^[A-Z0-9^=._-]+$")
    window: Literal["3M", "6M", "1Y", "3Y", "5Y", "10Y", "25Y", "Max"] = "3Y"
    roc: Literal["10D", "20D", "63D", "126D", "252D"] = "63D"
    view: Literal["Candlestick", "Line"] = "Candlestick"
    inflections: bool = True

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, value):
        return value.strip().upper() if isinstance(value, str) else value


class LeadershipParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    families: list[Literal["S&P 500 Sector Leadership", "China / U.S. Leadership", "Breadth / Alternative Weighting", "Inter-Sector Leadership"]] | None = None
    states: list[Literal["Leading", "Improving", "Weakening", "Lagging"]] | None = None
    history: Literal["6 Months", "1 Year", "3 Years", "5 Years"] = "3 Years"


class VolatilityParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    primary: str = Field(default="^NDX", min_length=1, max_length=32, pattern=r"^[A-Z0-9^=._-]+$")
    comparison: str = Field(default="^GSPC", min_length=1, max_length=32, pattern=r"^[A-Z0-9^=._-]+$")
    primary_implied: str = Field(default="^VXN", max_length=32, pattern=r"^[A-Z0-9^=._-]*$")
    comparison_implied: str = Field(default="^VIX", max_length=32, pattern=r"^[A-Z0-9^=._-]*$")
    history: Literal["1y", "2y", "3y", "5y", "10y", "max"] = "5y"
    rvol_window: Literal[5, 10, 21, 42, 63, 126, 252] = 21
    normalization_window: Literal[21, 63, 126, 252, 504, 1260] = 252

    @field_validator("primary", "comparison", "primary_implied", "comparison_implied", mode="before")
    @classmethod
    def normalize(cls, value):
        return value.strip().upper() if isinstance(value, str) else value


class RatioParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    families: list[Literal["Duration / Crisis Hedges", "Commodities / Equity Indices", "Credit / Funding", "Financial Intermediaries"]] | None = None
    history: Literal["3 Months", "6 Months", "9 Months", "YTD", "1 Year", "3 Years", "5 Years", "10 Years", "20 Years"] = "3 Years"
    rsi_window: int = Field(default=14, ge=5, le=30)
    show_rsi: bool = False
    show_signal_strip: bool = True
    moving_averages: list[Literal[8, 21, 50, 100, 200]] | None = None
    custom: str = Field(default="", max_length=8192)


class YieldParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    history: Literal["6M", "1Y", "2Y", "3Y", "5Y", "10Y"] = "5Y"
    regime_period: Literal["Today", "1W", "1M", "3M", "YTD"] = "1M"
    selected_curve: Literal["3m10y", "5s10s", "10s30s", "5s30s"] = "3m10y"
    curve_compare: Literal["1W", "1M", "3M", "YTD"] = "1M"


class LiquidityParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lookback: Literal["6m", "1y", "2y", "3y", "5y", "10y", "max"] = "5y"
    z_window: int = Field(default=756, ge=252, le=1260)
    min_periods: int = Field(default=252, ge=126, le=756)
    smoothing: int = Field(default=3, ge=1, le=21)
    show_fcig: bool = True

    @model_validator(mode="after")
    def valid_windows(self):
        if self.min_periods > self.z_window:
            raise ValueError("Minimum observations must not exceed the score lookback.")
        return self


class CreditParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    focus_window: Literal["5D", "1M", "3M", "YTD", "1Y"] = "1M"
    global_window: Literal["5D", "1M", "YTD", "1Y", "3Y", "5Y"] = "1Y"
    history: Literal["1 Year", "3 Years", "5 Years", "10 Years"] = "3 Years"


class CFTCParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lookback: Literal["1Y", "2Y", "3Y", "5Y"] = "3Y"
    tff_cohort: Literal["Asset Managers", "Leveraged Funds", "Asset Managers + Leveraged Funds", "Dealers", "Other Reportables"] = "Asset Managers + Leveraged Funds"
    disagg_cohort: Literal["Managed Money", "Producer / Merchant", "Swap Dealers", "Other Reportables"] = "Managed Money"
    selected: str | None = Field(default=None, max_length=64, pattern=r"^(TFF|Disaggregated)\|[A-Za-z0-9]+$")
    assets: list[str] | None = Field(default=None, max_length=32)
    sort: Literal["Most crowded shorts", "Most crowded longs", "Largest 1W shift", "Largest 4W contract change", "Largest absolute z-score"] = "Most crowded shorts"


class OptionsParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selected: str = Field(default="QQQ", min_length=1, max_length=32, pattern=r"^[A-Z0-9^=._-]+$")
    universe_text: str = Field(default="SPY, QQQ, IWM, DIA, TLT, GLD, USO, SMH, EEM, HYG, LQD", max_length=8192)
    target_dte: int = Field(default=45, ge=14, le=120)
    term_count: int = Field(default=6, ge=3, le=10)
    risk_free_rate: float = Field(default=0.04, ge=0, le=0.20)

    @field_validator("selected", mode="before")
    @classmethod
    def normalize(cls, value):
        return value.strip().upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def valid_universe(self):
        from adfm_engine.analytics.options import parse_universe
        if len(parse_universe(self.universe_text, self.selected)) < 2:
            raise ValueError("Add at least one comparison ticker.")
        return self


def require_gateway(authorization: Annotated[str | None, Header()] = None):
    token = os.getenv("ADFM_GATEWAY_TOKEN", "")
    if os.getenv("ADFM_ENV", "production") == "development" and not token:
        return
    if not token:
        raise HTTPException(503, "Analytics authentication is not configured.")
    if not authorization or not hmac.compare_digest(authorization, f"Bearer {token}"):
        raise HTTPException(401, "Unauthorized")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("ADFM_ENV", "production") != "development" and len(os.getenv("ADFM_GATEWAY_TOKEN", "")) < 32:
        raise RuntimeError("Set a random ADFM_GATEWAY_TOKEN of at least 32 characters before production startup.")
    configure_yfinance_cache()
    app.state.jobs=JobQueue(Path(os.getenv("ADFM_DATA_DIR","/tmp/adfm-data"))/"jobs.sqlite", {"sec13f":load_sec13f,"baskets":load_baskets,"commodity":load_commodity_event_study,"sector_rotation":load_sector_rotation,"volume_sentiment":load_volume_sentiment,"etf_flow":load_etf_flow,"position_sizing":load_position_sizing,"currency_tension":load_currency_tension,"seasonality":load_monthly_seasonality})
    try:
        yield
    finally:
        app.state.jobs.close()


def create_app() -> FastAPI:
    app = FastAPI(title="ADFM Analytics", version="1.0.0", lifespan=lifespan, docs_url="/docs" if os.getenv("ADFM_ENV") == "development" else None, redoc_url=None, openapi_url="/openapi.json" if os.getenv("ADFM_ENV") == "development" else None)
    app.add_middleware(GZipMiddleware, minimum_size=1500)

    @app.middleware("http")
    async def response_metadata(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Request-ID"] = uuid.uuid4().hex
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(DataUnavailable)
    async def unavailable(request, exc):
        return JSONResponse(status_code=502, content={"detail": str(exc), "code": "provider_unavailable", "diagnostics": exc.diagnostics})

    @app.get("/health/live")
    def health():
        return {"status": "ok"}

    @app.get("/v1/rate-of-change", dependencies=[Depends(require_gateway)])
    def roc(parameters: Annotated[ROCParameters, Query()]):
        return load_rate_of_change(**parameters.model_dump())

    @app.get("/v1/overview", dependencies=[Depends(require_gateway)])
    def home():
        return overview()

    @app.post("/v1/leadership", dependencies=[Depends(require_gateway)])
    def equity_leadership(parameters: LeadershipParameters):
        return load_leadership(**parameters.model_dump())

    @app.post("/v1/baskets", dependencies=[Depends(require_gateway)])
    def public_baskets(parameters: BasketParameters, request: Request):
        # A date in the fingerprint prevents yesterday's completed job being served as today's.
        payload = parameters.model_dump()
        payload["session_date"] = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        return request.app.state.jobs.submit("baskets", payload)

    @app.get("/v1/basket-definitions", dependencies=[Depends(require_gateway)])
    def public_basket_definitions():
        # Raw membership is independent of price downloads and quality filters.
        return {"categories": CATEGORIES}

    @app.post("/v1/baskets-job", dependencies=[Depends(require_gateway)])
    def public_baskets_job(parameters: JobParameters, request: Request):
        return request.app.state.jobs.get(parameters.id, kind="baskets")

    @app.post("/v1/commodity-event-study", dependencies=[Depends(require_gateway)])
    def commodity_event_study(parameters: CommodityParameters, request: Request):
        payload = parameters.model_dump()
        payload["session_hour"] = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d-%H")
        return request.app.state.jobs.submit("commodity", payload)

    @app.post("/v1/commodity-event-study-job", dependencies=[Depends(require_gateway)])
    def commodity_event_study_job(parameters: JobParameters, request: Request):
        return request.app.state.jobs.get(parameters.id, kind="commodity")

    @app.post("/v1/sector-breadth-and-rotation", dependencies=[Depends(require_gateway)])
    def sector_breadth_and_rotation(parameters: SectorRotationParameters, request: Request):
        payload = parameters.model_dump()
        payload["session_hour"] = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d-%H")
        return request.app.state.jobs.submit("sector_rotation", payload)

    @app.post("/v1/sector-breadth-and-rotation-job", dependencies=[Depends(require_gateway)])
    def sector_breadth_and_rotation_job(parameters: JobParameters, request: Request):
        return request.app.state.jobs.get(parameters.id, kind="sector_rotation")

    @app.post("/v1/volume-based-sentiment-indicator", dependencies=[Depends(require_gateway)])
    def volume_based_sentiment_indicator(parameters: VolumeSentimentParameters, request: Request):
        return request.app.state.jobs.submit("volume_sentiment", parameters.model_dump())

    @app.post("/v1/volume-based-sentiment-indicator-job", dependencies=[Depends(require_gateway)])
    def volume_based_sentiment_indicator_job(parameters: JobParameters, request: Request):
        return request.app.state.jobs.get(parameters.id, kind="volume_sentiment")

    @app.post("/v1/etf-flow-pressure-proxy", dependencies=[Depends(require_gateway)])
    def etf_flow_pressure_proxy(parameters: ETFFlowParameters, request: Request):
        payload = parameters.model_dump()
        payload["session_hour"] = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d-%H")
        return request.app.state.jobs.submit("etf_flow", payload)

    @app.post("/v1/etf-flow-pressure-proxy-job", dependencies=[Depends(require_gateway)])
    def etf_flow_pressure_proxy_job(parameters: JobParameters, request: Request):
        return request.app.state.jobs.get(parameters.id, kind="etf_flow")

    @app.post("/v1/position-sizing-lab", dependencies=[Depends(require_gateway)])
    def position_sizing_lab(parameters: PositionSizingParameters, request: Request):
        payload = parameters.model_dump()
        payload["session_hour"] = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d-%H")
        return request.app.state.jobs.submit("position_sizing", payload)

    @app.post("/v1/position-sizing-lab-job", dependencies=[Depends(require_gateway)])
    def position_sizing_lab_job(parameters: JobParameters, request: Request):
        return request.app.state.jobs.get(parameters.id, kind="position_sizing")

    @app.post("/v1/currency-tension-engine", dependencies=[Depends(require_gateway)])
    def currency_tension_engine(parameters: CurrencyTensionParameters, request: Request):
        payload = parameters.model_dump()
        payload["session_slot"] = int(datetime.now(ZoneInfo("America/New_York")).timestamp() // 600)
        return request.app.state.jobs.submit("currency_tension", payload)

    @app.post("/v1/currency-tension-engine-job", dependencies=[Depends(require_gateway)])
    def currency_tension_engine_job(parameters: JobParameters, request: Request):
        return request.app.state.jobs.get(parameters.id, kind="currency_tension")

    @app.post("/v1/monthly-seasonality-explorer", dependencies=[Depends(require_gateway)])
    def monthly_seasonality(parameters: SeasonalityParameters, request: Request):
        payload = parameters.model_dump()
        payload["session_hour"] = int(datetime.now(ZoneInfo("America/New_York")).timestamp() // 3600)
        return request.app.state.jobs.submit("seasonality", payload)

    @app.post("/v1/monthly-seasonality-explorer-job", dependencies=[Depends(require_gateway)])
    def monthly_seasonality_job(parameters: JobParameters, request: Request):
        return request.app.state.jobs.get(parameters.id, kind="seasonality")

    @app.post("/v1/relative-volatility", dependencies=[Depends(require_gateway)])
    def relative_volatility(parameters: VolatilityParameters):
        return load_volatility(**parameters.model_dump())

    @app.post("/v1/ratios", dependencies=[Depends(require_gateway)])
    def ratio_chartbook(parameters: RatioParameters):
        return load_ratios(**parameters.model_dump())

    @app.get("/v1/macro-regime", dependencies=[Depends(require_gateway)])
    def global_macro_regime():
        return load_macro_regime()

    @app.post("/v1/yields", dependencies=[Depends(require_gateway)])
    def yield_curve(parameters: YieldParameters):
        return load_yields(**parameters.model_dump())

    @app.post("/v1/liquidity", dependencies=[Depends(require_gateway)])
    def liquidity_conditions(parameters: LiquidityParameters):
        return load_liquidity(**parameters.model_dump())

    @app.post("/v1/credit", dependencies=[Depends(require_gateway)])
    def credit_conditions(parameters: CreditParameters):
        return load_credit(**parameters.model_dump())

    @app.post("/v1/cftc", dependencies=[Depends(require_gateway)])
    def cftc_positioning(parameters: CFTCParameters):
        return load_cftc(**parameters.model_dump())

    @app.post("/v1/options", dependencies=[Depends(require_gateway)])
    def options_compass(parameters: OptionsParameters):
        return load_options(**parameters.model_dump())

    @app.post("/v1/underwriter", dependencies=[Depends(require_gateway)])
    def issuer_underwrite(parameters: UnderwriterParameters):
        return load_underwriter(**parameters.model_dump())

    @app.get("/v1/sec13f-releases", dependencies=[Depends(require_gateway)])
    def sec_releases():return release_list()

    @app.post("/v1/sec13f", dependencies=[Depends(require_gateway)])
    def sec_screen(parameters:SEC13FParameters,request:Request):
        return request.app.state.jobs.submit("sec13f",parameters.model_dump())

    @app.post("/v1/sec13f-job", dependencies=[Depends(require_gateway)])
    def sec_job(parameters:JobParameters,request:Request):
        return request.app.state.jobs.get(parameters.id)

    @app.post("/v1/calendar", dependencies=[Depends(require_gateway)])
    def catalysts(parameters:CalendarParameters):
        return load_calendar(**parameters.model_dump())

    @app.post("/v1/stress", dependencies=[Depends(require_gateway)])
    def market_stress(parameters:StressParameters):
        return load_stress(**parameters.model_dump())

    @app.post("/v1/hedge", dependencies=[Depends(require_gateway)])
    def hedge_timer(parameters:HedgeParameters):
        return load_hedge(**parameters.model_dump())

    return app


app = create_app()
