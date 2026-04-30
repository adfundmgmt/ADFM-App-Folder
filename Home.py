import streamlit as st


st.set_page_config(
    page_title="AD Fund Management",
    layout="wide",
    initial_sidebar_state="collapsed",
)


TOOL_ORDER = [
    "ADFM Public Equities Baskets",
    "Unusual Options Flows",
    "Weekly Cross - Asset Compass",
    "Market Stress Composite",
    "Liquidity Tracker",
    "ETF Flows Dashboard",
    "Factor Momentum Leadership",
    "VIX Spike Deep Dive",
    "Hedge Timer",
    "Sector Breadth and Rotation",
    "Breakout Scanner",
    "Technical Chart Explorer",
    "Ratio Charts",
    "Cross Asset Volatility Surface",
    "Fed Speeches Tone Underwriter",
    "RoW Central Bank Tone Underwriter",
    "Market Memory Explorer",
    "Monthly Seasonality Explorer",
    "Home Value to Rent Ratio",
    "Volume Based Sentiment Indicator",
]


TOOL_GROUPS = {
    "All tools": TOOL_ORDER,
    "Equities + Positioning": [
        "ADFM Public Equities Baskets",
        "Factor Momentum Leadership",
        "Sector Breadth and Rotation",
        "Breakout Scanner",
        "Technical Chart Explorer",
        "Ratio Charts",
        "Market Memory Explorer",
        "Volume Based Sentiment Indicator",
    ],
    "Flows + Derivatives": [
        "Unusual Options Flows",
        "ETF Flows Dashboard",
        "Cross Asset Volatility Surface",
        "Hedge Timer",
    ],
    "Macro + Risk Regime": [
        "Weekly Cross - Asset Compass",
        "Market Stress Composite",
        "Liquidity Tracker",
        "VIX Spike Deep Dive",
    ],
    "Rates + Macro Tone": [
        "Fed Speeches Tone Underwriter",
        "RoW Central Bank Tone Underwriter",
        "Monthly Seasonality Explorer",
        "Home Value to Rent Ratio",
    ],
}


TOOL_DESCRIPTIONS = {
    "ADFM Public Equities Baskets": "Compares ADFM equity sleeves to spot leadership, trend strength, and dispersion.",
    "Unusual Options Flows": "Flags outsized options activity to reveal potential positioning and intent.",
    "Weekly Cross - Asset Compass": "Summarizes cross-asset moves to classify the current market regime.",
    "Market Stress Composite": "Tracks a blended stress score across volatility, credit, and funding conditions.",
    "Liquidity Tracker": "Monitors major liquidity drivers like Fed balance sheet, RRP, and TGA flows.",
    "ETF Flows Dashboard": "Tracks ETF creations and redemptions to monitor real-time allocation shifts.",
    "Factor Momentum Leadership": "Ranks factor momentum to highlight which styles are leading or fading.",
    "VIX Spike Deep Dive": "Breaks down volatility spikes to show trigger, magnitude, and persistence.",
    "Hedge Timer": "Provides timing cues for adding, reducing, or rolling portfolio hedges.",
    "Sector Breadth and Rotation": "Measures participation and rotation to identify where equity strength is broadening or narrowing.",
    "Breakout Scanner": "Finds names and groups breaking above key technical levels with confirmation.",
    "Technical Chart Explorer": "Explores multi-timeframe chart structure, trend, and momentum in one view.",
    "Ratio Charts": "Uses relative-strength ratios to compare assets, sectors, and factors versus benchmarks.",
    "Cross Asset Volatility Surface": "Maps implied volatility across assets and tenors to locate stress pockets.",
    "Fed Speeches Tone Underwriter": "Scores Federal Reserve communication tone for policy bias and risk impact.",
    "RoW Central Bank Tone Underwriter": "Analyzes non-U.S. central bank tone to gauge global policy direction.",
    "Market Memory Explorer": "Surfaces historical analogs to contextualize current market behavior.",
    "Monthly Seasonality Explorer": "Shows recurring monthly return and volatility patterns by asset and sector.",
    "Home Value to Rent Ratio": "Tracks housing valuation pressure through the home price-to-rent relationship.",
    "Volume Based Sentiment Indicator": "Reads conviction and sentiment using volume trends and participation signals.",
}


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1180px;
            padding-top: 3.75rem;
            padding-bottom: 2.5rem;
        }

        .hero {
            padding: 2.1rem 2rem 2.2rem 2rem;
            border: 1px solid rgba(120, 120, 120, 0.2);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.06), rgba(2, 132, 199, 0.03));
            margin-top: 0.35rem;
            margin-bottom: 1.4rem;
            overflow: visible;
        }

        .eyebrow {
            font-size: 0.74rem;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            font-weight: 700;
            color: #64748b;
            margin-bottom: 0.45rem;
        }

        .title {
            font-size: clamp(2.4rem, 5vw, 3.35rem);
            line-height: 1.05;
            letter-spacing: -0.045em;
            font-weight: 850;
            color: #0f172a;
            margin-bottom: 0.65rem;
        }

        .subtitle {
            max-width: 900px;
            font-size: 1.03rem;
            line-height: 1.65;
            color: #475569;
        }

        .chip-row {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-top: 1.15rem;
        }

        .chip {
            border: 1px solid rgba(100, 116, 139, 0.28);
            padding: 0.28rem 0.72rem;
            border-radius: 999px;
            font-size: 0.78rem;
            line-height: 1;
            color: #475569;
            background: rgba(248, 250, 252, 0.74);
        }

        .section-title {
            font-size: 1.08rem;
            font-weight: 750;
            letter-spacing: -0.02em;
            color: #0f172a;
            margin-top: 0.25rem;
            margin-bottom: 0.9rem;
        }

        .tool-card {
            border: 1px solid rgba(120, 120, 120, 0.2);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            background: rgba(255, 255, 255, 0.72);
            margin-bottom: 0.9rem;
            min-height: 104px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        }

        .tool-title {
            font-size: 0.97rem;
            font-weight: 760;
            color: #0f172a;
            margin-bottom: 0.32rem;
        }

        .tool-copy {
            font-size: 0.89rem;
            line-height: 1.5;
            color: #64748b;
        }

        div[data-testid="stTextInput"] {
            margin-bottom: 0.9rem;
        }

        div[data-testid="stTextInput"] input {
            border-radius: 999px;
        }

        @media (prefers-color-scheme: dark) {
            .title,
            .section-title,
            .tool-title {
                color: #f8fafc;
            }

            .subtitle,
            .tool-copy,
            .chip {
                color: #cbd5e1;
            }

            .eyebrow {
                color: #94a3b8;
            }

            .hero,
            .tool-card,
            .chip {
                background: rgba(15, 23, 42, 0.52);
                border-color: rgba(148, 163, 184, 0.24);
            }

            .tool-card {
                box-shadow: none;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">ADFM Analytics Platform</div>
        <div class="title">AD Fund Management</div>
        <div class="subtitle">
            A centralized dashboard for regime analysis, liquidity monitoring, positioning,
            volatility, flows, and technical market structure.
        </div>
        <div class="chip-row">
            <span class="chip">Regime</span>
            <span class="chip">Liquidity</span>
            <span class="chip">Flows</span>
            <span class="chip">Volatility</span>
            <span class="chip">Technicals</span>
            <span class="chip">Risk</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)


if hasattr(st, "segmented_control"):
    selected_group = st.segmented_control(
        "Filter by group",
        options=list(TOOL_GROUPS.keys()),
        default="All tools",
        label_visibility="collapsed",
    )
else:
    selected_group = st.radio(
        "Filter by group",
        options=list(TOOL_GROUPS.keys()),
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )


query = st.text_input(
    "Search tools",
    placeholder="Try: liquidity, breakout, sentiment...",
    label_visibility="collapsed",
)


filtered_tools = TOOL_GROUPS[selected_group]


if query:
    q = query.lower().strip()
    filtered_tools = [
        tool
        for tool in filtered_tools
        if q in tool.lower() or q in TOOL_DESCRIPTIONS.get(tool, "").lower()
    ]


if filtered_tools:
    quick_cols = st.columns(2)

    for idx, tool in enumerate(filtered_tools):
        with quick_cols[idx % 2]:
            st.markdown(
                f"""
                <div class="tool-card">
                    <div class="tool-title">{tool}</div>
                    <div class="tool-copy">{TOOL_DESCRIPTIONS.get(tool, "Description coming soon.")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.info("No tools matched your search. Try a shorter keyword.")
