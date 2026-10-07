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
    ToolDefinition(1, 'Equity Baskets', '1_ADFM_Public_Equities_Baskets.py', 'Equity Research', 'Compares thematic equity baskets by performance, trend, leadership and dispersion against a selected benchmark.', 'Internal basket definitions; Yahoo Finance market data'),
    ToolDefinition(7, 'Sector Breadth and Rotation', '7_Sector_Breadth_and_Rotation.py', 'Equity Research', 'Shows where equity participation is broadening or narrowing and which sectors are gaining or losing leadership.', 'Yahoo Finance sector and subsector ETFs'),
    ToolDefinition(8, 'Equity Relative Strength', '8_Equity_Leadership_&_Rotation.py', 'Equity Research', 'Ranks sector, regional and breadth relationships to show which equity exposures are outperforming their counterparts.', 'Yahoo Finance adjusted ETF and index prices'),
    ToolDefinition(9, 'Equity Underwriter', '9_ADFM_Underwriter.py', 'Equity Research', 'Examines company valuation, growth, profitability, balance-sheet strength and debt using financial statements and recent market prices.', 'SEC EDGAR Company Facts and submissions; Yahoo Finance completed-session close and price history'),
    ToolDefinition(2, 'Bond Cycle Monitor', '2_Global_Macro_Regime.py', 'Macro Regime', 'Examines yield exhaustion, reversal signals and historical outcomes across Treasury yields, credit spreads and global sovereign rates.', 'Federal Reserve and ICE BofA via FRED; OECD monthly sovereign rates via FRED'),
    ToolDefinition(3, 'Liquidity Conditions', '3_Liquidity_Conditions_Monitor.py', 'Macro Regime', 'Tracks changes in central-bank liquidity, funding conditions and credit transmission, alongside market confirmation and financial conditions.', 'Federal Reserve H.4.1; New York Fed rates and RRP; ICE BofA OAS via FRED; broad dollar; real yields; Yahoo Finance confirmation proxies'),
    ToolDefinition(4, 'Rates and Yield Curve', '4_Yield_Curve_Rates_Regime_Monitor.py', 'Macro Regime', 'Tracks Treasury yields, real yields, inflation expectations and changes in the shape of the yield curve.', 'Federal Reserve / FRED nominal and real Treasury yields and inflation breakevens; Yahoo nominal-curve fallback'),
    ToolDefinition(5, 'Credit Conditions', '5_Credit_Conditions_Monitor.py', 'Macro Regime', 'Compares global government-yield moves, corporate credit spreads and funding costs, with market indicators of financial stress.', 'ICE BofA corporate OAS and U.S. Treasury yields via Federal Reserve FRED; Yahoo Finance market confirmation; Trading Economics or fresh Stooq sovereign yields with OECD/FRED structural fallback'),
    ToolDefinition(6, 'FX Valuation and Trend', '6_Currency_Tension_Engine.py', 'Macro Regime', 'Compares currency trends with valuation, policy and carry to identify stretched or changing market conditions.', 'Persisted Currency Tension Engine snapshot and configured adapters'),
    ToolDefinition(10, 'Chart Terminal', '10_ADFM_Chart_Terminal.py', 'Technical Confirmation', 'Explores price history, trend, momentum, volatility and chart structure across assets and multiple time horizons.', 'Yahoo Finance OHLCV'),
    ToolDefinition(11, 'Cross-Asset Ratios', '11_Cross-Asset_Ratio_Chartbook.py', 'Technical Confirmation', 'Charts relative performance across rates, equities, commodities, credit and currencies, including relationships selected by the user.', 'Yahoo Finance adjusted close history'),
    ToolDefinition(12, 'Momentum', '12_Rate_of_Change_Regime_Explorer.py', 'Technical Confirmation', 'Compares price trends, returns and acceleration across several horizons to identify strengthening or weakening market momentum.', 'Yahoo Finance daily OHLCV'),
    ToolDefinition(13, 'Relative Volatility', '13_Relative_Volatility_Lab.py', 'Technical Confirmation', "Compares two assets' realized volatility, its historical distribution and optional implied-volatility measures over a selected window.", 'Yahoo Finance adjusted close history; implied-volatility indexes and ETF proxies where available'),
    ToolDefinition(14, 'ETF Trading Pressure', '14_ETF_Flow_Pressure_Proxy.py', 'Positioning and Flows', 'Ranks dollar-weighted ETF trading pressure across equities, rates, credit, commodities and other tactical market exposures.', 'Yahoo Finance OHLCV'),
    ToolDefinition(15, 'Volume Participation', '15_Volume_Based_Sentiment_Indicator.py', 'Positioning and Flows', 'Shows unusually heavy or quiet trading participation alongside price trends and the historical outcomes of similar sessions.', 'Yahoo Finance adjusted OHLCV; provider fallback where available'),
    ToolDefinition(16, 'Options Relative Value', '16_Options_Positioning_Compass.py', 'Positioning and Flows', 'Compares price trends with implied versus realized volatility to identify relatively rich or cheap option premiums.', 'Yahoo Finance current option chains and adjusted close history; Cboe delayed option-chain fallback'),
    ToolDefinition(18, 'CFTC Positioning', '18_CFTC_Positioning_Monitor.py', 'Positioning and Flows', 'Tracks futures positioning, historical crowding and weekly changes across major financial and commodity contracts using CFTC reports.', 'CFTC Public Reporting Environment; Yahoo Finance price overlays for mapped contracts'),
    ToolDefinition(19, 'Market Stress', '19_Market_Stress_Composite.py', 'Risk and Catalysts', 'Tracks stress across equities, credit, rates, currencies and commodities to identify broader changes in market conditions.', 'Yahoo Finance; local last-good cache on provider failure'),
    ToolDefinition(20, 'Catalyst Calendar', '20_Catalyst_Calendar.py', 'Risk and Catalysts', 'Charts upcoming economic releases, central-bank decisions and market-calendar events, with dates, categories and sources in one table.', 'Official agency calendars; recurring market-calendar rules; Yahoo Finance market proxies'),
    ToolDefinition(21, 'Drawdown Risk', '21_Hedge_Timer.py', 'Risk and Catalysts', 'Monitors price, breadth, volatility and credit conditions for drawdown warnings in the S&P 500 and Nasdaq-100.', 'Yahoo Finance adjusted closes for S&P 500, Nasdaq-100, SPY, RSP, IWM, HYG, LQD, all 11 S&P 500 sector ETFs, VIX, VIX9D, VIX3M, and VVIX; index daily highs and lows'),
    ToolDefinition(23, 'Historical Analogs', '23_Market_Memory_Explorer.py', 'Historical Context', 'Finds historical return paths resembling the current market and compares what followed across different periods and regimes.', 'Yahoo Finance market history'),
    ToolDefinition(24, 'Seasonality', '24_Monthly_Seasonality_Explorer.py', 'Historical Context', 'Compares recurring monthly return and volatility patterns across assets, with historical distributions and optional regime filters.', 'Yahoo Finance; FRED for selected series and regime tags'),
    ToolDefinition(25, 'Commodity Exhaustion', '25_Commodity_Event_Study.py', 'Historical Context', 'Studies extended commodity moves, reversal confirmation and subsequent returns to assess whether potential tops held historically.', 'Yahoo Finance daily continuous-futures history; CFTC Disaggregated Managed Money positioning where mapped'),
)


SIDEBAR_GUIDES: Final[dict[str, SidebarGuide]] = {
    "1_ADFM_Public_Equities_Baskets.py": SidebarGuide(("Choose the basket family and benchmark that match the research question.", "Compare leadership, trend strength, dispersion, and benchmark-relative performance.", "Open the composition and chart detail before treating a basket signal as actionable.")),
    "2_Global_Macro_Regime.py": SidebarGuide(("Select a yield or spread and an exhaustion profile.", "Read the current watch or signal and its historical markers.", "Compare forward yield changes, baseline outcomes and independent sample sizes."), "Daily official sovereign curves and monthly OECD histories are separate; curve bases and observation dates differ."),
    "3_Liquidity_Conditions_Monitor.py": SidebarGuide(("Start with the overall liquidity level and marginal impulse.", "Compare balance-sheet, funding, transmission, and market-confirmation sleeves.", "Check source status before relying on a sleeve with partial coverage."), "Changing the display window does not change the fixed-history scoring formula."),
    "4_Yield_Curve_Rates_Regime_Monitor.py": SidebarGuide(("Start with outright Treasury yield levels and their direction.", "Read curve spreads next to classify steepening or flattening.", "Compare horizons to separate a short-lived move from a persistent rates regime."), "This page isolates U.S. rates and curve structure; cross-asset confirmation belongs in the other macro tools."),
    "5_Credit_Conditions_Monitor.py": SidebarGuide(("Separate spread stress from the level of risk-free funding costs.", "Confirm the move through high yield, loans, banks, emerging-market debt, and volatility.", "Use the global 10-year table to locate sovereign-rate repricing.")),
    "6_Currency_Tension_Engine.py": SidebarGuide(("Choose the scoring horizon before comparing currencies.", "Read the map as trajectory on the horizontal axis and valuation-policy stretch on the vertical axis.", "Open the pillars, carry, positioning, and flags to understand why a currency moved."), "Lower-right is the cleanest cheap-and-improving quadrant; rings and notes flag crowding or data caveats."),
    "7_Sector_Breadth_and_Rotation.py": SidebarGuide(("Choose major sectors for a top-down read or subsectors for more detail.", "Use the rotation map to identify direction and persistence.", "Confirm the move with breadth, relative strength, and underlying coverage.")),
    "8_Equity_Leadership_&_Rotation.py": SidebarGuide(("Scan the four leadership states for established leaders, laggards, and transitions.", "Use acceleration to compare short-horizon ranks with the 3- and 6-month trend.", "Open the related chartbook when a ranked relationship needs full historical context."), "A positive score means the numerator ranks in the stronger half of the 25-ratio universe."),
    "9_ADFM_Underwriter.py": SidebarGuide(("Search the issuer and verify the filing period and coverage status.", "Review valuation, per-share growth, margins, returns, and liquidity together.", "Finish with capital structure, debt service, maturities, and recent SEC events."), "Banks, insurers, foreign private issuers, partnerships, and custom-tag-heavy filers can require issuer-specific adjustments."),
    "10_ADFM_Chart_Terminal.py": SidebarGuide(("Set the symbol, window, and interval for the decision horizon.", "Read price, return, drawdown, and volatility context before the indicators.", "Use the signal matrix to confirm trend, momentum, volatility, structure, and invalidation levels.")),
    "11_Cross-Asset_Ratio_Chartbook.py": SidebarGuide(("Choose the relationship families and lookback that match the thesis.", "Read a rising ratio as outperformance by the first ticker versus the second.", "Use the signal line for trend and stale-data context, then compare related charts."), "Ratios are rebased to 100 at the selected lookback start."),
    "12_Rate_of_Change_Regime_Explorer.py": SidebarGuide(("Anchor on price versus the 21-, 50-, 100-, and 200-day moving averages.", "Read rate of change for momentum direction and magnitude.", "Use acceleration and zero-line inflections to identify transitions."), "Trading sessions share one observation index, so weekends and holidays are compressed."),
    "13_Relative_Volatility_Lab.py": SidebarGuide(("Choose the numerator, denominator, and realized-volatility window.", "Compare each instrument's volatility before reading the ratio and its percentile.", "Use implied volatility and fixed stress diagnostics to confirm or challenge the ratio signal."), "Missing observations remain unavailable rather than being filled with fabricated values."),
    "14_ETF_Flow_Pressure_Proxy.py": SidebarGuide(("Choose the trading window and keep the full cross-asset ETF universe in view.", "Read positive dollar pressure as trading concentrated toward session highs and negative pressure as concentration toward session lows.", "Use pressure intensity, returns, weekly pressure and dollar volume in the detail table to distinguish scale from signal strength."), "This is a dollar-weighted price-volume pressure proxy from observed market trading, not reported ETF creations or redemptions."),
    "15_Volume_Based_Sentiment_Indicator.py": SidebarGuide(("Choose the symbol and percentile window.", "Classify current participation as heavy, normal, or quiet.", "Use setup labels and matured forward returns to judge how similar signals behaved.")),
    "16_Options_Positioning_Compass.py": SidebarGuide(("Choose a liquid comparison universe and highlight ticker.", "Read price trend on the horizontal axis and IV richness on the vertical axis.", "Use the compact table to compare return, ATM IV, realized volatility, and the IV-RV spread."), "IV rich/cheap is a current cross-sectional classification based on ATM implied volatility minus 21-day realized volatility. It is not historical IV rank or a trading recommendation."),
    "18_CFTC_Positioning_Monitor.py": SidebarGuide(("Scan crowded longs, crowded shorts, and the largest weekly changes.", "Choose one contract for historical percentile, z-score, and price context.", "Change cohorts or the crowding lookback only when the research question requires it."), "COT is a Tuesday position snapshot normally released Friday; it is not a real-time flow feed."),
    "19_Market_Stress_Composite.py": SidebarGuide(("Read directional Risk-Off separately from direction-agnostic Dislocation.", "Identify the regions and asset groups contributing most to the signal.", "Use the U.S. overlay and forward-drawdown history to frame transmission risk.")),
    "20_Catalyst_Calendar.py": SidebarGuide(("Choose the event horizon and scan the dated catalyst sequence.", "Read the catalyst chart, then confirm event dates, types and sources in the table.", "Add mandate-specific events with the custom-event template when needed."), "Confirm agency schedules before trading directly around a release; recurring market dates can be rule-based."),
    "21_Hedge_Timer.py": SidebarGuide(("Read the red hedge alerts for SPX or NDX; each marks a new alert. The status remains active until sustained recovery.", "Use RSI and the 63-session drawdown gates to decide whether a fresh directional short is allowed.", "Browse every 10%+ local-peak drawdown since 2020, including intraday ranges; compare actual alert dates, early captures, misses, late calls, and false alarms."), "SPX-only calibration prioritizes callouts before the first 3% loss, then fewer false and repeated events. NDX uses the frozen SPX rules and does not achieve full recall. Historical coverage is an in-sample fit, not a guarantee of future warnings."),
    "23_Market_Memory_Explorer.py": SidebarGuide(("Choose the ticker and historical sample before ranking analog years.", "Compare correlation, endpoint gap, volatility, drawdown, and slope across the top matches.", "Keep unconditional base rates and the full distribution beside any highlighted analog."), "A similar historical year is context, not a forecast."),
    "24_Monthly_Seasonality_Explorer.py": SidebarGuide(("Set the global lookback; it controls every output on the page.", "Select a month for the return distribution and a year for the path overlay.", "Apply regime filters, then verify that the conditional sample remains large enough to interpret.")),
    "25_Commodity_Event_Study.py": SidebarGuide(("Choose the commodity, signal profile, and history used to define an extreme.", "Confirm whether price extension has been followed by actual reversal evidence.", "Read forward returns, drawdowns, hit rates, and sample size across horizons."), "This is a top study: negative post-signal returns favor the signal. Continuous-futures roll construction can affect history."),
}

GROUP_ORDER: Final[tuple[str, ...]] = (
    'Equity Research',
    'Macro Regime',
    'Technical Confirmation',
    'Positioning and Flows',
    'Risk and Catalysts',
    'Historical Context',
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


def tool_definition_for_page(page_filename: str) -> ToolDefinition | None:
    """Backward-compatible alias for resolving catalog metadata by page filename."""
    return tool_for_page(page_filename)


def sidebar_guide_for_page(page_filename: str) -> SidebarGuide | None:
    """Resolve the shared sidebar reading guide for a cataloged page."""
    normalized = page_filename.replace("\\", "/").rsplit("/", 1)[-1]
    return SIDEBAR_GUIDES.get(normalized)
