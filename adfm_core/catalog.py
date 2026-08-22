"""Typed source of truth for the ADFM Analytics Platform tool catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class ToolDefinition:
    """Stable metadata for one Streamlit page exposed from Home."""

    number: int
    title: str
    page_filename: str
    group: str
    description: str
    primary_inputs: str
    owner: str = "ADFM Analytics"


TOOL_CATALOG: Final[tuple[ToolDefinition, ...]] = (
    ToolDefinition(1, "ADFM Public Equities Baskets", "1_ADFM_Public_Equities_Baskets.py", "Equity Discovery", "Compares ADFM equity baskets across leadership, trend strength, dispersion, and benchmark-relative performance.", "Internal basket definitions; Yahoo Finance market data"),
    ToolDefinition(2, "Global Macro Regime Dashboard", "2_Global_Macro_Regime_Dashboard.py", "Macro Regime", "Combines growth, inflation, policy, financial conditions, and market signals into a broad macro-regime read.", "Yahoo Finance market proxies"),
    ToolDefinition(3, "Liquidity Conditions Monitor", "3_Liquidity_Tracker.py", "Macro Regime", "Separates the level and marginal impulse of system liquidity across Federal Reserve balance-sheet plumbing, overnight funding, credit transmission, and a capped market-confirmation sleeve.", "Federal Reserve H.4.1; New York Fed rates and RRP; ICE BofA OAS via FRED; broad dollar; real yields; Yahoo Finance confirmation proxies"),
    ToolDefinition(4, "Yield Curve + Rates Regime Monitor", "4_Yield_Curve_Rates_Regime_Monitor.py", "Macro Regime", "Tracks the Treasury curve, real yields, breakevens, and bull/bear steepener or flattener regimes.", "Yahoo Finance rate and market proxies"),
    ToolDefinition(5, "Credit Conditions Dashboard", "5_Credit_Conditions_Monitor.py", "Macro Regime", "Monitors credit spreads, credit ETF ratios, regional banks, loans, EM debt, and financial conditions.", "Yahoo Finance market proxies"),
    ToolDefinition(6, "Currency Tension Engine", "6_Currency_Tension_Dashboard.py", "Macro Regime", "Maps currencies across trajectory and valuation-policy stretch, with carry, pillar scores, overlays, and daily risk flags.", "Persisted Currency Tension Engine snapshot and configured adapters"),
    ToolDefinition(7, "Sector Breadth and Rotation", "7_Sector_Breadth_and_Rotation.py", "Equity Leadership", "Measures participation and sector rotation to identify where equity strength is broadening or narrowing.", "Yahoo Finance sector and subsector ETFs"),
    ToolDefinition(8, "Equity Leadership & Rotation", "8_Equity_Leadership_and_Rotation.py", "Equity Leadership", "Ranks 25 equity relationships across four horizons to identify established leadership, positive inflections, weakening trends, and persistent laggards.", "Yahoo Finance adjusted ETF prices"),
    ToolDefinition(9, "ADFM Underwriter", "9_ADFM_Underwriter.py", "Fundamental Research", "Calculates filing-driven valuation, per-share growth, margins, returns, liquidity, capital structure, issuer-credit ratios, debt maturities, and recent SEC events.", "SEC EDGAR Company Facts and submissions; Yahoo Finance completed-session close and price history"),
    ToolDefinition(10, "Technical Chart Explorer", "10_ADFM_Chart_Terminal.py", "Technical Confirmation", "Explores multi-timeframe chart structure, trend, momentum, volatility bands, and key moving averages.", "Yahoo Finance OHLCV"),
    ToolDefinition(11, "Cross-Asset Ratio Chartbook", "11_Cross_Asset_Ratio_Chartbook.py", "Technical Confirmation", "Provides focused and grouped historical inspection of 50 cross-asset, macro, thematic, and single-stock ratios, plus custom relationships.", "Yahoo Finance adjusted close history"),
    ToolDefinition(12, "Rate of Change Dashboard", "12_Rate_of_Change_Dashboard.py", "Technical Confirmation", "Tracks multi-horizon rate-of-change regimes for fast reads on momentum, acceleration, and trend pressure.", "Yahoo Finance daily OHLCV"),
    ToolDefinition(13, "Relative Volatility Lab", "13_Relative_Volatility_Lab.py", "Technical Confirmation", "Decomposes selectable realized-volatility ratios and compares them with implied volatility, acceleration, downside, semiconductor, and breadth diagnostics.", "Yahoo Finance adjusted close history; implied-volatility indexes and ETF proxies where available"),
    ToolDefinition(14, "ETF Flows Dashboard", "14_ETF_Flows_Dashboard.py", "Positioning + Flows", "Tracks ETF flow-pressure proxies to monitor allocation shifts across macro, equity, and thematic exposures.", "Yahoo Finance OHLCV"),
    ToolDefinition(15, "Volume Based Sentiment Indicator", "15_Volume_Based_Sentiment_Indicator.py", "Positioning + Flows", "Reads conviction, participation, and sentiment using volume-regime signals across major liquid assets.", "Yahoo Finance adjusted OHLCV; provider fallback where available"),
    ToolDefinition(16, "Options Positioning Compass", "16_Options_Positioning_Compass.py", "Positioning + Flows", "Maps current implied-volatility richness, downside skew, term structure, and aggregate option activity across a selected universe.", "Yahoo Finance current option chains and adjusted close history"),
    ToolDefinition(17, "Market Stress Composite", "17_Market_Stress_Composite.py", "Risk + Execution", "Builds a cross-asset stress score across equities, credit, commodities, FX, rates, breadth, and dispersion.", "Yahoo Finance; local last-good cache on provider failure"),
    ToolDefinition(18, "Event Risk + Catalyst Calendar", "18_Event_Risk_Catalyst_Calendar.py", "Risk + Execution", "Maps upcoming macro catalysts, options windows, Treasury supply, earnings season, and custom event risks.", "Yahoo Finance market proxies; configured calendar data"),
    ToolDefinition(19, "Hedge Timer", "19_Hedge_Timer.py", "Risk + Execution", "Provides tactical timing cues for adding, holding, reducing, or rolling portfolio hedges.", "Yahoo Finance; FRED regime inputs"),
    ToolDefinition(20, "Position Sizing Lab", "20_Position_Sizing_Lab.py", "Risk + Execution", "Runs an interactive bankroll simulation using real historical holding-period outcomes, then pressure-tests conviction-based exposure against volatility, invalidation, event, tail, and liquidity risk.", "Yahoo Finance adjusted OHLCV, earnings dates, and liquid cross-asset proxies"),
    ToolDefinition(21, "Market Memory Explorer", "21_Market_Memory_Explorer.py", "Historical Context", "Surfaces historical analogs to contextualize the current tape against prior return paths and regimes.", "Yahoo Finance market history"),
    ToolDefinition(22, "Monthly Seasonality Explorer", "22_Monthly_Seasonality_Explorer.py", "Historical Context", "Shows recurring monthly return and volatility patterns by asset, index, sector, or commodity.", "Yahoo Finance; FRED for selected series and regime tags"),
    ToolDefinition(23, "SEC 13F Exposure Browser", "23_SEC_13F_Exposure_Browser.py", "Positioning + Flows", "Ranks institutional managers by a selected security's share of their disclosed Form 13F portfolio.", "SEC Form 13F bulk data sets; SEC company ticker directory"),
    ToolDefinition(24, "CFTC Positioning Monitor", "24_CFTC_Positioning_Monitor.py", "Positioning + Flows", "Scans CFTC futures positioning for crowded longs, crowded shorts, and sharp weekly shifts across financial and physical futures, with historical percentile and z-score context.", "CFTC Public Reporting Environment; Yahoo Finance price overlays for mapped contracts"),
)

GROUP_ORDER: Final[tuple[str, ...]] = (
    "Equity Discovery",
    "Macro Regime",
    "Equity Leadership",
    "Fundamental Research",
    "Technical Confirmation",
    "Positioning + Flows",
    "Risk + Execution",
    "Historical Context",
)


def tool_order() -> list[str]:
    """Return Home's stable navigation order."""
    return [tool.title for tool in TOOL_CATALOG]


def tool_definitions() -> list[ToolDefinition]:
    """Return the ordered catalog for navigation and governance checks."""
    return list(TOOL_CATALOG)


def tool_groups() -> dict[str, list[str]]:
    """Return Home navigation groups while retaining catalog order."""
    groups = {"All tools": tool_order()}
    for group in GROUP_ORDER:
        groups[group] = [tool.title for tool in TOOL_CATALOG if tool.group == group]
    return groups


def tool_descriptions() -> dict[str, str]:
    """Return the Home-card description keyed by tool title."""
    return {tool.title: tool.description for tool in TOOL_CATALOG}


def tool_for_page(page_filename: str) -> ToolDefinition | None:
    """Resolve catalog metadata from a Streamlit page filename."""
    normalized = page_filename.replace("\\", "/").rsplit("/", 1)[-1]
    return next((tool for tool in TOOL_CATALOG if tool.page_filename == normalized), None)
