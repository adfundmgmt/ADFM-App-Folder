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
from pydantic import BaseModel, ConfigDict, Field, field_validator

from adfm_engine.data.market import configure_yfinance_cache
from adfm_engine.services import DataUnavailable, load_rate_of_change, overview
from adfm_engine.leadership_service import load_leadership
from adfm_engine.volatility_service import load_volatility

logger = logging.getLogger("adfm.api")


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
    yield


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
        return JSONResponse(status_code=502, content={"detail": str(exc), "code": "provider_unavailable"})

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

    return app


app = create_app()
