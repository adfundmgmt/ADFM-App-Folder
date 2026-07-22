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
    ToolDefinition(1, "ADFM Public Equities Baskets", "1_ADFM_Public_Equities_Baskets.py", "Equity Leadership", "Compares ADFM equity baskets across leadership, trend strength, dispersion, and benchmark-relative performance.", "Internal basket definitions; Yahoo Finance market data"),
    ToolDefinition(2, "Sector Breadth and Rotation", "2_Sector_Breadth_and_Rotation.py", "Equity Leadership", "Measures participation and sector rotation to identify where equity strength is broadening or narrowing.", "Yahoo Finance sector and subsector ETFs"),
    ToolDefinition(3, "Factor Momentum Leadership", "3_Factor_Momentum_Leadership.py", "Equity Leadership", "Ranks factor momentum to highlight which styles are leading, fading, or inflecting.", "Yahoo Finance ETF prices"),
    ToolDefinition(4, "Rate of Change Dashboard", "4_Rate_of_Change_Dashboard.py", "Equity Leadership", "Tracks multi-horizon rate-of-change regimes for fast reads on momentum, acceleration, and trend pressure.", "Yahoo Finance daily OHLCV"),
    ToolDefinition(5, "Technical Chart Explorer", "5_ADFM_Chart_Terminal.py", "Technicals + Analogs", "Explores multi-timeframe chart structure, trend, momentum, volatility bands, and key moving averages.", "Yahoo Finance OHLCV"),
    ToolDefinition(6, "Ratio Charts", "6_Ratio_Charts.py", "Technicals + Analogs", "Uses relative-strength ratios to compare assets, sectors, credit, factors, and risk appetite proxies.", "Yahoo Finance adjusted close history"),
    ToolDefinition(7, "Market Memory Explorer", "7_Market_Memory_Explorer.py", "Technicals + Analogs", "Surfaces historical analogs to contextualize the current tape against prior return paths and regimes.", "Yahoo Finance market history"),
    ToolDefinition(8, "Monthly Seasonality Explorer", "8_Monthly_Seasonality_Explorer.py", "Technicals + Analogs", "Shows recurring monthly return and volatility patterns by asset, index, sector, or commodity.", "Yahoo Finance; FRED for selected series and regime tags"),
    ToolDefinition(9, "Volume Based Sentiment Indicator", "9_Volume_Based_Sentiment_Indicator.py", "Flows + Sentiment", "Reads conviction, participation, and sentiment using volume-regime signals across major liquid assets.", "Yahoo Finance adjusted OHLCV; provider fallback where available"),
    ToolDefinition(10, "ETF Flows Dashboard", "10_ETF_Flows_Dashboard.py", "Flows + Sentiment", "Tracks ETF flow-pressure proxies to monitor allocation shifts across macro, equity, and thematic exposures.", "Yahoo Finance OHLCV"),
    ToolDefinition(11, "Global Macro Regime Dashboard", "11_Global_Macro_Regime_Dashboard.py", "Macro + Rates", "Combines growth, inflation, policy, financial conditions, and market signals into a broad macro-regime read.", "Yahoo Finance market proxies"),
    ToolDefinition(12, "Yield Curve + Rates Regime Monitor", "12_Yield_Curve_Rates_Regime_Monitor.py", "Macro + Rates", "Tracks the Treasury curve, real yields, breakevens, and bull/bear steepener or flattener regimes.", "Yahoo Finance rate and market proxies"),
    ToolDefinition(13, "Credit Conditions Dashboard", "13_Credit_Conditions_Monitor.py", "Macro + Rates", "Monitors credit spreads, credit ETF ratios, regional banks, loans, EM debt, and financial conditions.", "Yahoo Finance market proxies"),
    ToolDefinition(14, "Liquidity Tracker", "14_Liquidity_Tracker.py", "Macro + Rates", "Monitors major liquidity drivers including the Fed balance sheet, RRP, TGA, policy rates, and financial conditions.", "Yahoo Finance; Federal Reserve FCI-G data"),
    ToolDefinition(15, "Market Stress Composite", "15_Market_Stress_Composite.py", "Risk + Catalysts", "Builds a cross-asset stress score across equities, credit, commodities, FX, rates, breadth, and dispersion.", "Yahoo Finance; local last-good cache on provider failure"),
    ToolDefinition(16, "Event Risk + Catalyst Calendar", "16_Event_Risk_Catalyst_Calendar.py", "Risk + Catalysts", "Maps upcoming macro catalysts, options windows, Treasury supply, earnings season, and custom event risks.", "Yahoo Finance market proxies; configured calendar data"),
    ToolDefinition(17, "Hedge Timer", "17_Hedge_Timer.py", "Risk + Catalysts", "Provides tactical timing cues for adding, holding, reducing, or rolling portfolio hedges.", "Yahoo Finance; FRED regime inputs"),
    ToolDefinition(18, "Currency Tension Engine", "18_Currency_Tension_Dashboard.py", "Macro + Rates", "Maps currencies across trajectory and valuation-policy stretch, with carry, pillar scores, overlays, and daily risk flags.", "Persisted Currency Tension Engine snapshot and configured adapters"),
    ToolDefinition(19, "ADFM Momentum Expansion Scanner", "19_ADFM_Momentum_Expansion_Scanner.py", "Equity Leadership", "Ranks liquid equities with deterministic momentum-expansion setups and maps sector and subsector leadership.", "Yahoo Finance daily OHLCV; existing sector and subsector universe"),
)

GROUP_ORDER: Final[tuple[str, ...]] = (
    "Equity Leadership",
    "Technicals + Analogs",
    "Flows + Sentiment",
    "Macro + Rates",
    "Risk + Catalysts",
)


def tool_order() -> list[str]:
    """Return Home's stable navigation order."""
    return [tool.title for tool in TOOL_CATALOG]


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
