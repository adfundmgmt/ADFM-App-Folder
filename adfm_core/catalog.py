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


@dataclass(frozen=True)
class SidebarGuide:
    """Concise, page-specific reading sequence for the shared sidebar guide."""

    read_order: tuple[str, ...]
    caveat: str | None = None


TOOL_CATALOG: Final[tuple[ToolDefinition, ...]] = (
    ToolDefinition(1, "ADFM Public Equities Baskets", "1_ADFM_Public_Equities_Baskets.py", "Equity Discovery", "Compares ADFM equity baskets across leadership, trend strength, dispersion, and benchmark-relative performance.", "Internal basket definitions; Yahoo Finance market data"),
    ToolDefinition(2, "Global Macro Regime", "2_Global_Macro_Regime.py", "Macro Regime", "Separates growth, inflation, rates, liquidity, and risk confirmation into a transparent cross-asset macro read, then highlights where those signals agree or conflict.", "Federal Reserve / FRED primary macro series; Yahoo Finance market proxies"),
    ToolDefinition(3, "Liquidity Conditions Monitor", "3_Liquidity_Conditions_Monitor.py", "Macro Regime", "Separates the level and marginal impulse of system liquidity across Federal Reserve balance-sheet plumbing, overnight funding, credit transmission, and a capped market-confirmation sleeve.", "Federal Reserve H.4.1; New York Fed rates and RRP; ICE BofA OAS via FRED; broad dollar; real yields; Yahoo Finance confirmation proxies"),
    ToolDefinition(4, "Yield Curve Rates Regime Monitor", "4_Yield_Curve_Rates_Regime_Monitor.py", "Macro Regime", "Tracks outright U.S. Treasury yield levels, curve shape, and bull/bear steepener or flattener regimes across multiple horizons.", "Yahoo Finance Treasury yield symbols: ^IRX, ^FVX, ^TNX, ^TYX"),
    ToolDefinition(5, "Credit Conditions Monitor", "5_Credit_Conditions_Monitor.py", "Macro Regime", "Separates credit-spread stress from outright funding-cost pressure, then checks banks, loans, volatility, and global benchmark 10-year government-yield repricing across flexible horizons.", "ICE BofA corporate OAS and U.S. Treasury yields via Federal Reserve FRED; Yahoo Finance market confirmation; Trading Economics or fresh Stooq sovereign yields with OECD/FRED structural fallback"),
    ToolDefinition(6, "Currency Tension Engine", "6_Currency_Tension_Engine.py", "Macro Regime", "Maps currencies across trajectory and valuation-policy stretch, with carry, pillar scores, overlays, and daily risk flags.", "Persisted Currency Tension Engine snapshot and configured adapters"),
    ToolDefinition(7, "Sector Breadth and Rotation", "7_Sector_Breadth_and_Rotation.py", "Equity Leadership", "Measures participation and sector rotation to identify where equity strength is broadening or narrowing.", "Yahoo Finance sector and subsector ETFs"),
    ToolDefinition(8, "Equity Leadership & Rotation", "8_Equity_Leadership_&_Rotation.py", "Equity Leadership", "Ranks all 11 S&P 500 sectors versus SPY, five China/U.S. relationships, three breadth and alternative-weighting ratios, and six inter-sector relationships.", "Yahoo Finance adjusted ETF and index prices"),
    ToolDefinition(9, "ADFM Underwriter", "9_ADFM_Underwriter.py", "Fundamental Research", "Calculates filing-driven valuation, per-share growth, margins, returns, liquidity, capital structure, issuer-credit ratios, debt maturities, and recent SEC events.", "SEC EDGAR Company Facts and submissions; Yahoo Finance completed-session close and price history"),
    ToolDefinition(10, "ADFM Chart Terminal", "10_ADFM_Chart_Terminal.py", "Technical Confirmation", "Explores multi-timeframe chart structure, trend, momentum, volatility bands, and key moving averages.", "Yahoo Finance OHLCV"),
    ToolDefinition(11, "Cross-Asset Ratio Chartbook", "11_Cross-Asset_Ratio_Chartbook.py", "Technical Confirmation", "Provides grouped historical inspection of 38 duration, crisis-hedge, commodity, credit, funding, and financial-intermediary ratios, plus custom relationships.", "Yahoo Finance adjusted close history"),
    ToolDefinition(12, "Rate of Change Regime Explorer", "12_Rate_of_Change_Regime_Explorer.py", "Technical Confirmation", "Tracks multi-horizon rate-of-change regimes for fast reads on momentum, acceleration, and trend pressure.", "Yahoo Finance daily OHLCV"),
    ToolDefinition(13, "Relative Volatility Lab", "13_Relative_Volatility_Lab.py", "Technical Confirmation", "Decomposes selectable realized-volatility ratios and compares them with implied volatility, acceleration, downside, semiconductor, and breadth diagnostics.", "Yahoo Finance adjusted close history; implied-volatility indexes and ETF proxies where available"),
    ToolDefinition(14, "ETF Flow Pressure Proxy", "14_ETF_Flow_Pressure_Proxy.py", "Positioning + Flows", "Tracks ETF flow-pressure proxies to monitor allocation shifts across macro, equity, and thematic exposures.", "Yahoo Finance OHLCV"),
    ToolDefinition(15, "Volume Based Sentiment Indicator", "15_Volume_Based_Sentiment_Indicator.py", "Positioning + Flows", "Reads conviction, participation, and sentiment using volume-regime signals across major liquid assets.", "Yahoo Finance adjusted OHLCV; provider fallback where available"),
    ToolDefinition(16, "Options Positioning Compass", "16_Options_Positioning_Compass.py", "Positioning + Flows", "Maps current implied-volatility richness, downside skew, term structure, and aggregate option activity across a selected universe.", "Yahoo Finance current option chains and adjusted close history"),
    ToolDefinition(17, "SEC 13F Exposure Browser", "17_SEC_13F_Exposure_Browser.py", "Positioning + Flows", "Ranks institutional managers by a selected security's share of their disclosed Form 13F portfolio.", "SEC Form 13F bulk data sets; SEC company ticker directory"),
    ToolDefinition(18, "CFTC Positioning Monitor", "18_CFTC_Positioning_Monitor.py", "Positioning + Flows", "Scans CFTC futures positioning for crowded longs, crowded shorts, and sharp weekly shifts across financial and physical futures, with historical percentile and z-score context.", "CFTC Public Reporting Environment; Yahoo Finance price overlays for mapped contracts"),
    ToolDefinition(19, "Market Stress Composite", "19_Market_Stress_Composite.py", "Risk + Execution", "Builds a cross-asset stress score across equities, credit, commodities, FX, rates, breadth, and dispersion.", "Yahoo Finance; local last-good cache on provider failure"),
    ToolDefinition(20, "Catalyst Calendar", "20_Catalyst_Calendar.py", "Risk + Execution", "Maps upcoming macro catalysts, options windows, Treasury supply, earnings season, and custom event risks.", "Official agency calendars; recurring market-calendar rules; Yahoo Finance market proxies"),
    ToolDefinition(21, "Hedge Timer", "21_Hedge_Timer.py", "Risk + Execution", "Provides tactical timing cues for adding, holding, reducing, or rolling portfolio hedges.", "Yahoo Finance; FRED regime inputs"),
    ToolDefinition(22, "Position Sizing Lab", "22_Position_Sizing_Lab.py", "Risk + Execution", "Runs an interactive bankroll simulation using real historical holding-period outcomes, then pressure-tests conviction-based exposure against volatility, invalidation, event, tail, and liquidity risk.", "Yahoo Finance adjusted OHLCV, earnings dates, and liquid cross-asset proxies"),
    ToolDefinition(23, "Market Memory Explorer", "23_Market_Memory_Explorer.py", "Historical Context", "Surfaces historical analogs to contextualize the current tape against prior return paths and regimes.", "Yahoo Finance market history"),
    ToolDefinition(24, "Monthly Seasonality Explorer", "24_Monthly_Seasonality_Explorer.py", "Historical Context", "Shows recurring monthly return and volatility patterns by asset, index, sector, or commodity.", "Yahoo Finance; FRED for selected series and regime tags"),
    ToolDefinition(25, "Commodity Event Study", "25_Commodity_Event_Study.py", "Historical Context", "Identifies historically extended commodity moves, waits for exhaustion or reversal confirmation, and measures how often those signals marked durable tops.", "Yahoo Finance daily continuous-futures history; CFTC Disaggregated Managed Money positioning where mapped"),
)


SIDEBAR_GUIDES: Final[dict[str, SidebarGuide]] = {
    "1_ADFM_Public_Equities_Baskets.py": SidebarGuide(
        (
            "Choose the basket family and benchmark that match the research question.",
            "Compare leadership, trend strength, dispersion, and benchmark-relative performance.",
            "Open the composition and chart detail before treating a basket signal as actionable.",
        )
    ),
    "2_Global_Macro_Regime.py": SidebarGuide(
        (
            "Read growth, inflation, rates, liquidity, and risk confirmation as separate sleeves.",
            "Use the tension notes to find signals that disagree with the headline regime.",
            "Confirm the narrative in the cross-asset performance and primary-series tables.",
        ),
        "The page is a transparent dashboard of sleeves, not a hidden weighted macro score.",
    ),
    "3_Liquidity_Conditions_Monitor.py": SidebarGuide(
        (
            "Start with the overall liquidity level and marginal impulse.",
            "Compare balance-sheet, funding, transmission, and market-confirmation sleeves.",
            "Check source status before relying on a sleeve with partial coverage.",
        ),
        "Changing the display window does not change the fixed-history scoring formula.",
    ),
    "4_Yield_Curve_Rates_Regime_Monitor.py": SidebarGuide(
        (
            "Start with outright Treasury yield levels and their direction.",
            "Read curve spreads next to classify steepening or flattening.",
            "Compare horizons to separate a short-lived move from a persistent rates regime.",
        ),
        "This page isolates U.S. rates and curve structure; cross-asset confirmation belongs in the other macro tools.",
    ),
    "5_Credit_Conditions_Monitor.py": SidebarGuide(
        (
            "Separate spread stress from the level of risk-free funding costs.",
            "Confirm the move through high yield, loans, banks, emerging-market debt, and volatility.",
            "Use the global 10-year table to locate sovereign-rate repricing.",
        )
    ),
    "6_Currency_Tension_Engine.py": SidebarGuide(
        (
            "Choose the scoring horizon before comparing currencies.",
            "Read the map as trajectory on the horizontal axis and valuation-policy stretch on the vertical axis.",
            "Open the pillars, carry, positioning, and flags to understand why a currency moved.",
        ),
        "Lower-right is the cleanest cheap-and-improving quadrant; rings and notes flag crowding or data caveats.",
    ),
    "7_Sector_Breadth_and_Rotation.py": SidebarGuide(
        (
            "Choose major sectors for a top-down read or subsectors for more detail.",
            "Use the rotation map to identify direction and persistence.",
            "Confirm the move with breadth, relative strength, and underlying coverage.",
        )
    ),
    "8_Equity_Leadership_&_Rotation.py": SidebarGuide(
        (
            "Scan the four leadership states for established leaders, laggards, and transitions.",
            "Use acceleration to compare short-horizon ranks with the 3- and 6-month trend.",
            "Open the related chartbook when a ranked relationship needs full historical context.",
        ),
        "A positive score means the numerator ranks in the stronger half of the 25-ratio universe.",
    ),
    "9_ADFM_Underwriter.py": SidebarGuide(
        (
            "Search the issuer and verify the filing period and coverage status.",
            "Review valuation, per-share growth, margins, returns, and liquidity together.",
            "Finish with capital structure, debt service, maturities, and recent SEC events.",
        ),
        "Banks, insurers, foreign private issuers, partnerships, and custom-tag-heavy filers can require issuer-specific adjustments.",
    ),
    "10_ADFM_Chart_Terminal.py": SidebarGuide(
        (
            "Set the symbol, window, and interval for the decision horizon.",
            "Read price, return, drawdown, and volatility context before the indicators.",
            "Use the signal matrix to confirm trend, momentum, volatility, structure, and invalidation levels.",
        )
    ),
    "11_Cross-Asset_Ratio_Chartbook.py": SidebarGuide(
        (
            "Choose the relationship families and lookback that match the thesis.",
            "Read a rising ratio as outperformance by the first ticker versus the second.",
            "Use the signal line for trend and stale-data context, then compare related charts.",
        ),
        "Ratios are rebased to 100 at the selected lookback start.",
    ),
    "12_Rate_of_Change_Regime_Explorer.py": SidebarGuide(
        (
            "Anchor on price versus the 21-, 50-, 100-, and 200-day moving averages.",
            "Read rate of change for momentum direction and magnitude.",
            "Use acceleration and zero-line inflections to identify transitions.",
        ),
        "Trading sessions share one observation index, so weekends and holidays are compressed.",
    ),
    "13_Relative_Volatility_Lab.py": SidebarGuide(
        (
            "Choose the numerator, denominator, and realized-volatility window.",
            "Compare each instrument's volatility before reading the ratio and its percentile.",
            "Use implied volatility and fixed stress diagnostics to confirm or challenge the ratio signal.",
        ),
        "Missing observations remain unavailable rather than being filled with fabricated values.",
    ),
    "14_ETF_Flow_Pressure_Proxy.py": SidebarGuide(
        (
            "Choose the lookback and ranking metric.",
            "Scan sign and magnitude to find the strongest positive and negative pressure.",
            "Compare nearby sectors, factors, regions, rates, credit, commodities, and FX exposures.",
        ),
        "This is a directional price-volume pressure proxy, not official ETF creation and redemption data.",
    ),
    "15_Volume_Based_Sentiment_Indicator.py": SidebarGuide(
        (
            "Choose the symbol and percentile window.",
            "Classify current participation as heavy, normal, or quiet.",
            "Use setup labels and matured forward returns to judge how similar signals behaved.",
        )
    ),
    "16_Options_Positioning_Compass.py": SidebarGuide(
        (
            "Set the focus ticker and a relevant comparison universe.",
            "Compare ATM volatility, downside skew, term structure, and activity ranks.",
            "Inspect the selected ticker's expirations, surface, open interest, and largest activity.",
        ),
        "Public chains do not reveal trade direction, spread IDs, dealer positioning, or a complete historical chain record.",
    ),
    "17_SEC_13F_Exposure_Browser.py": SidebarGuide(
        (
            "Choose a security search or a manager search and select the SEC release.",
            "Rank holders by portfolio weight, reported value, or shares when screening a security.",
            "Open a manager to inspect the effective disclosed portfolio and position detail.",
        ),
        "Form 13F is a delayed quarterly disclosure, not a real-time holdings or trade feed.",
    ),
    "18_CFTC_Positioning_Monitor.py": SidebarGuide(
        (
            "Scan crowded longs, crowded shorts, and the largest weekly changes.",
            "Choose one contract for historical percentile, z-score, and price context.",
            "Change cohorts or the crowding lookback only when the research question requires it.",
        ),
        "COT is a Tuesday position snapshot normally released Friday; it is not a real-time flow feed.",
    ),
    "19_Market_Stress_Composite.py": SidebarGuide(
        (
            "Read directional Risk-Off separately from direction-agnostic Dislocation.",
            "Identify the regions and asset groups contributing most to the signal.",
            "Use the U.S. overlay and forward-drawdown history to frame transmission risk.",
        )
    ),
    "20_Catalyst_Calendar.py": SidebarGuide(
        (
            "Choose the event horizon and scan the dated catalyst sequence.",
            "Prioritize events by timing, risk score, and the market setup going into them.",
            "Add mandate-specific events with the custom-event template when needed.",
        ),
        "Confirm agency schedules before trading directly around a release; recurring market dates can be rule-based.",
    ),
    "21_Hedge_Timer.py": SidebarGuide(
        (
            "Start with the composite hedge state and its gating conditions.",
            "Compare trend, stress, and drawdown evidence across horizons.",
            "Use the post-2020 sanity check to understand historical misses and false positives.",
        ),
        "Treat the output as timing evidence within a hedge plan, not as a standalone trade instruction.",
    ),
    "22_Position_Sizing_Lab.py": SidebarGuide(
        (
            "Define direction, conviction, holding period, loss budget, and liquidity constraints.",
            "Review the volatility, invalidation, event, tail, and liquidity caps.",
            "Run multiple historical paths and size from the binding risk cap, not the best simulation outcome.",
        )
    ),
    "23_Market_Memory_Explorer.py": SidebarGuide(
        (
            "Choose the ticker and historical sample before ranking analog years.",
            "Compare correlation, endpoint gap, volatility, drawdown, and slope across the top matches.",
            "Keep unconditional base rates and the full distribution beside any highlighted analog.",
        ),
        "A similar historical year is context, not a forecast.",
    ),
    "24_Monthly_Seasonality_Explorer.py": SidebarGuide(
        (
            "Set the global lookback; it controls every output on the page.",
            "Select a month for the return distribution and a year for the path overlay.",
            "Apply regime filters, then verify that the conditional sample remains large enough to interpret.",
        )
    ),
    "25_Commodity_Event_Study.py": SidebarGuide(
        (
            "Choose the commodity, signal profile, and history used to define an extreme.",
            "Confirm whether price extension has been followed by actual reversal evidence.",
            "Read forward returns, drawdowns, hit rates, and sample size across horizons.",
        ),
        "This is a top study: negative post-signal returns favor the signal. Continuous-futures roll construction can affect history.",
    ),
}

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


def sidebar_guide_for_page(page_filename: str) -> SidebarGuide | None:
    """Resolve the shared sidebar reading guide for a cataloged page."""

    normalized = page_filename.replace("\\", "/").rsplit("/", 1)[-1]
    return SIDEBAR_GUIDES.get(normalized)
