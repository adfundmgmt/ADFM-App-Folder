"""Native ADFM HTTP boundary. Production imports only adfm_engine."""
from __future__ import annotations

import hmac
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Literal

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
from adfm_engine.sec13f_service import load_sec13f, release_list
from adfm_engine.jobs import JobQueue
from pathlib import Path

logger = logging.getLogger("adfm.api")


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
    app.state.jobs=JobQueue(Path(os.getenv("ADFM_DATA_DIR","/tmp/adfm-data"))/"jobs.sqlite", {"sec13f":load_sec13f})
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

    return app


app = create_app()
