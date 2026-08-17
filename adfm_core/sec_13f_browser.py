"""Streamlit renderer for the ADFM SEC 13F Exposure Browser."""

from __future__ import annotations

import html
import os
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from adfm_core.palette import PASTEL
from adfm_core.sec_13f_corrected import (
    PreparedDataset,
    QuarterDataset,
    Sec13FError,
    available_report_periods,
    discover_quarter_datasets,
    load_company_tickers,
    load_security_catalog,
    manager_portfolio,
    prepare_dataset,
    rank_fund_exposure,
    search_security_candidates,
)
from adfm_core.ui import (
    PageHeader,
    dataframe_download,
    inject_explorer_style,
    render_kpi_cards,
    render_page_header,
    render_section_header,
    render_selection_note,
    render_status_line,
)

TITLE = "SEC 13F Exposure Browser"
POSITION_KINDS = ("Long holdings", "Call options", "Put options", "All reported")
SORT_OPTIONS = {
    "Portfolio weight": "PORTFOLIO_WEIGHT_PCT",
    "Reported market value": "POSITION_VALUE_USD",
    "Reported shares": "REPORTED_SHARES",
}
DETAIL_COLUMN_LABELS = {
    "PORTFOLIO_WEIGHT_PCT": "Portfolio weight",
    "POSITION_VALUE_USD": "Position value",
    "REPORTED_SHARES": "Reported shares",
    "PORTFOLIO_VALUE_USD": "13F portfolio",
    "LATEST_FILING_DATE": "Latest filing",
    "CIK": "Manager CIK",
    "COMPONENT_COUNT": "Filing components",
    "FILING_URL": "EDGAR filing",
}
DEFAULT_DETAIL_COLUMNS = [
    "PORTFOLIO_WEIGHT_PCT",
    "POSITION_VALUE_USD",
    "REPORTED_SHARES",
    "PORTFOLIO_VALUE_USD",
    "LATEST_FILING_DATE",
    "FILING_URL",
]

OFFICIAL_RELEASE_FALLBACKS = (
    QuarterDataset(
        slug="01mar2026-31may2026_form13f",
        label="2026 March April May 13F",
        url="https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01mar2026-31may2026_form13f.zip",
        size_label="94.81 MB",
    ),
    QuarterDataset(
        slug="01dec2025-28feb2026_form13f",
        label="2025 December 2026 January February 13F",
        url="https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01dec2025-28feb2026_form13f.zip",
        size_label="86.08 MB",
    ),
    QuarterDataset(
        slug="01sep2025-30nov2025_form13f",
        label="2025 September October November 13F",
        url="https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01sep2025-30nov2025_form13f.zip",
        size_label="81.65 MB",
    ),
    QuarterDataset(
        slug="01jun2025-31aug2025_form13f",
        label="2025 June July August 13F",
        url="https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01jun2025-31aug2025_form13f.zip",
        size_label="82.3 MB",
    ),
    QuarterDataset(
        slug="01mar2025-31may2025_form13f",
        label="2025 March April May 13F",
        url="https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01mar2025-31may2025_form13f.zip",
        size_label="84.18 MB",
    ),
    QuarterDataset(
        slug="01dec2024-28feb2025_form13f",
        label="2024 December 2025 January February 13F",
        url="https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01dec2024-28feb2025_form13f.zip",
        size_label="82.53 MB",
    ),
)

os.environ.setdefault(
    "ADFM_SEC_USER_AGENT", "AD Fund Management LP aryadeniz@adfundmgmt.com"
)


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def cached_releases() -> list[QuarterDataset]:
    try:
        releases = discover_quarter_datasets()
        return releases or list(OFFICIAL_RELEASE_FALLBACKS)
    except Sec13FError:
        return list(OFFICIAL_RELEASE_FALLBACKS)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def cached_ticker_directory() -> pd.DataFrame:
    return load_company_tickers()


@st.cache_data(show_spinner=False)
def cached_security_catalog(prepared: PreparedDataset) -> pd.DataFrame:
    return load_security_catalog(prepared)


@st.cache_data(show_spinner=False)
def cached_ranking(
    prepared: PreparedDataset,
    cusips: tuple[str, ...],
    report_period: str,
    position_kind: str,
    minimum_portfolio_millions: float,
) -> pd.DataFrame:
    return rank_fund_exposure(
        prepared,
        cusips,
        report_period=report_period,
        position_kind=position_kind,
        minimum_portfolio_millions=minimum_portfolio_millions,
    )


@st.cache_data(show_spinner=False)
def cached_manager_portfolio(
    prepared: PreparedDataset, cik: str, report_period: str
) -> tuple[dict[str, object], pd.DataFrame]:
    return manager_portfolio(prepared, cik, report_period)


def money_label(value: float) -> str:
    if not pd.notna(value):
        return "N/A"
    magnitude = abs(float(value))
    if magnitude >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if magnitude >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value:,.0f}"


def candidate_label(row: pd.Series) -> str:
    put_call = row.get("PUTCALL", "")
    instrument = "Long" if pd.isna(put_call) or not str(put_call).strip() else str(put_call)
    return (
        f"{row['NAMEOFISSUER']} | {row['TITLEOFCLASS']} | "
        f"CUSIP {row['CUSIP']} | {instrument.title()}"
    )


def inject_13f_style() -> None:
    st.markdown(
        """
        <style>
        .adfm-13f-dossier {
            border: 1px solid #d8d8d8;
            border-left: 5px solid #000000;
            margin: .4rem 0 .8rem;
            padding: 1rem 1.15rem;
        }
        .adfm-13f-eyebrow {
            color: #555; font-size: .68rem; font-weight: 800;
            letter-spacing: .09em; text-transform: uppercase;
        }
        .adfm-13f-title {
            color: #000; font-family: Georgia, "Times New Roman", serif;
            font-size: 1.5rem; font-weight: 700; line-height: 1.2;
            margin-top: .18rem;
        }
        .adfm-13f-facts {
            display: grid; gap: .65rem 1.2rem;
            grid-template-columns: repeat(4, minmax(120px, 1fr));
            margin-top: .8rem;
        }
        .adfm-13f-fact-label {
            color: #666; display: block; font-size: .65rem;
            font-weight: 800; letter-spacing: .05em; text-transform: uppercase;
        }
        .adfm-13f-fact-value {
            color: #111; display: block; font-size: .86rem;
            font-weight: 600; margin-top: .1rem;
        }
        .adfm-13f-note {
            border-left: 3px solid #000; color: #333; font-size: .78rem;
            margin: .35rem 0 .75rem; padding-left: .65rem;
        }
        @media (max-width: 850px) {
            .adfm-13f-facts { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(releases: list[QuarterDataset]) -> tuple[dict[str, object], bool]:
    with st.sidebar:
        st.header("Security search")
        with st.form("sec_13f_screen"):
            query = st.text_input(
                "Ticker, issuer, or CUSIP",
                value="INTC",
                help="Ticker symbols are matched to the exact filed issuer and CUSIP.",
            )
            release_label = st.selectbox(
                "SEC data release", [release.label for release in releases], index=0
            )
            position_kind = st.selectbox("Position type", POSITION_KINDS, index=0)
            minimum_portfolio_billions = st.number_input(
                "Minimum disclosed portfolio ($B)",
                min_value=0.0,
                value=1.0,
                step=0.25,
            )
            sort_label = st.selectbox("Rank funds by", list(SORT_OPTIONS), index=0)
            top_n = st.slider("Funds in overview", 10, 50, 25, 5)
            submitted = st.form_submit_button("Run 13F screen", width="stretch")

        st.caption(
            "The first run prepares the official SEC bulk release. Later searches reuse the cache."
        )
        st.markdown("---")
        st.header("About This Tool")
        st.markdown(
            """
            **Workflow:** search a security, rank institutional holders, then click a manager row to inspect that manager's complete effective 13F portfolio.

            **Calculation:** portfolio weight uses the sum of the manager's effective SEC information-table holdings as the denominator. This avoids filer summary-page unit inconsistencies.
            """
        )

    selected_release = next(r for r in releases if r.label == release_label)
    return {
        "query": query.strip(),
        "release_slug": selected_release.slug,
        "position_kind": position_kind,
        "minimum_portfolio_millions": float(minimum_portfolio_billions) * 1_000.0,
        "sort_label": sort_label,
        "top_n": int(top_n),
    }, submitted


def exposure_chart(ranking: pd.DataFrame, sort_label: str, top_n: int) -> go.Figure:
    sort_column = SORT_OPTIONS[sort_label]
    plot = ranking.sort_values(sort_column, ascending=False).head(top_n).copy()
    plot = plot.sort_values(sort_column)
    if sort_column == "PORTFOLIO_WEIGHT_PCT":
        axis_title, suffix = "Share of disclosed 13F portfolio", "%"
    elif sort_column == "POSITION_VALUE_USD":
        axis_title, suffix = "Reported market value ($)", ""
    else:
        axis_title, suffix = "Reported shares", ""
    fig = go.Figure(
        go.Bar(
            x=plot[sort_column],
            y=plot["MANAGER"],
            orientation="h",
            marker=dict(color=PASTEL["blue"]),
            customdata=plot[
                ["PORTFOLIO_WEIGHT_PCT", "POSITION_VALUE_USD", "PORTFOLIO_VALUE_USD"]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>Portfolio weight: %{customdata[0]:.2f}%"
                "<br>Position value: $%{customdata[1]:,.0f}"
                "<br>13F portfolio: $%{customdata[2]:,.0f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=max(440, 32 * len(plot) + 115),
        margin=dict(l=10, r=25, t=30, b=45),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        xaxis=dict(title=axis_title, ticksuffix=suffix, gridcolor="#e5e5e5"),
        yaxis=dict(title=None, automargin=True),
    )
    return fig


def render_manager_profile(
    prepared: PreparedDataset,
    cik: str,
    report_period: pd.Timestamp,
    request: dict[str, object],
) -> None:
    summary, portfolio = cached_manager_portfolio(
        prepared, cik, report_period.date().isoformat()
    )
    if not summary or portfolio.empty:
        st.error("The selected manager portfolio could not be reconstructed.")
        return

    if st.button(f"← Back to {str(request['query']).upper()} holders"):
        st.query_params.pop("manager", None)
        st.rerun()
    st.link_button("Open manager on SEC EDGAR", str(summary["FILER_URL"]))

    st.markdown(
        f"""
        <section class="adfm-13f-dossier">
          <div class="adfm-13f-eyebrow">13F manager profile</div>
          <div class="adfm-13f-title">{html.escape(str(summary['MANAGER']))}</div>
          <div class="adfm-13f-facts">
            <div><span class="adfm-13f-fact-label">CIK</span><span class="adfm-13f-fact-value">{summary['CIK']}</span></div>
            <div><span class="adfm-13f-fact-label">13F portfolio</span><span class="adfm-13f-fact-value">{money_label(float(summary['PORTFOLIO_VALUE_USD']))}</span></div>
            <div><span class="adfm-13f-fact-label">Positions</span><span class="adfm-13f-fact-value">{int(summary['POSITION_COUNT']):,}</span></div>
            <div><span class="adfm-13f-fact-label">Top 10 concentration</span><span class="adfm-13f-fact-value">{float(summary['TOP_TEN_PCT']):.1f}%</span></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    render_section_header(
        "Manager portfolio",
        "Effective holdings for the selected report date. Values are as filed, not current marks.",
    )
    filter_col, type_col = st.columns([1.3, .7])
    with filter_col:
        query = st.text_input("Filter portfolio", placeholder="Issuer, CUSIP, or class")
    with type_col:
        options = ["All", *sorted(portfolio["POSITION_TYPE"].dropna().unique())]
        kind = st.selectbox("Position type", options)
    filtered = portfolio.copy()
    if query.strip():
        needle = query.strip()
        filtered = filtered.loc[
            filtered["NAMEOFISSUER"].str.contains(needle, case=False, na=False, regex=False)
            | filtered["CUSIP"].astype(str).str.contains(needle, case=False, na=False, regex=False)
            | filtered["TITLEOFCLASS"].astype(str).str.contains(needle, case=False, na=False, regex=False)
        ]
    if kind != "All":
        filtered = filtered.loc[filtered["POSITION_TYPE"].eq(kind)]

    display = filtered[
        [
            "RANK", "NAMEOFISSUER", "TITLEOFCLASS", "POSITION_TYPE", "CUSIP",
            "PORTFOLIO_WEIGHT_PCT", "POSITION_VALUE_USD", "REPORTED_AMOUNT",
            "SSHPRNAMTTYPE", "SOURCE_FILING_DATE", "FILING_URL",
        ]
    ].head(750)
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        height=600,
        column_config={
            "RANK": st.column_config.NumberColumn("Rank", format="%,d"),
            "NAMEOFISSUER": "Issuer",
            "TITLEOFCLASS": "Class",
            "POSITION_TYPE": "Type",
            "CUSIP": "CUSIP",
            "PORTFOLIO_WEIGHT_PCT": st.column_config.NumberColumn("% of portfolio", format="%.2f%%"),
            "POSITION_VALUE_USD": st.column_config.NumberColumn("Market value", format="$%,.0f"),
            "REPORTED_AMOUNT": st.column_config.NumberColumn("Shares / principal", format="%,.0f"),
            "SSHPRNAMTTYPE": "Amount type",
            "SOURCE_FILING_DATE": st.column_config.DateColumn("Filing date", format="MMM D, YYYY"),
            "FILING_URL": st.column_config.LinkColumn("Source", display_text="Open EDGAR"),
        },
    )
    dataframe_download(
        "Download manager portfolio CSV",
        filtered,
        f"sec_13f_manager_{re.sub(r'[^A-Za-z0-9_-]+', '_', str(summary['MANAGER']))}_{report_period.date().isoformat()}.csv",
    )


def render_screen(releases: list[QuarterDataset], request: dict[str, object]) -> None:
    release = next(r for r in releases if r.slug == str(request["release_slug"]))
    with st.spinner("Loading the SEC release..."):
        prepared = prepare_dataset(release)
        periods = available_report_periods(prepared)
        ticker_directory = cached_ticker_directory()
        catalog = cached_security_catalog(prepared)
    if not periods:
        st.error("The selected release has no usable report period.")
        return
    report_period = periods[0]
    candidates = search_security_candidates(
        catalog,
        ticker_directory,
        str(request["query"]),
        position_kind=str(request["position_kind"]),
    )
    if candidates.empty:
        st.error("No matching filed security was found.")
        return
    if len(candidates) > 1:
        labels = [candidate_label(row) for _, row in candidates.iterrows()]
        label = st.selectbox("Matched filed security", labels, index=0)
        selected = candidates.iloc[labels.index(label)]
    else:
        selected = candidates.iloc[0]

    with st.spinner("Reconciling holder portfolios from SEC information tables..."):
        ranking = cached_ranking(
            prepared,
            (str(selected["CUSIP"]),),
            report_period.date().isoformat(),
            str(request["position_kind"]),
            float(request["minimum_portfolio_millions"]),
        )
    if ranking.empty:
        st.warning("No effective 13F filings met the current filters.")
        return

    manager_cik = str(st.query_params.get("manager", "") or "").strip()
    if manager_cik:
        render_manager_profile(prepared, manager_cik, report_period, request)
        return

    sort_label = str(request["sort_label"])
    ranked = ranking.sort_values(SORT_OPTIONS[sort_label], ascending=False).reset_index(drop=True)
    ranked["RANK"] = range(1, len(ranked) + 1)

    st.markdown(
        f"""
        <section class="adfm-13f-dossier">
          <div class="adfm-13f-eyebrow">SEC 13F security match</div>
          <div class="adfm-13f-title">{html.escape(str(selected['NAMEOFISSUER']))}</div>
          <div class="adfm-13f-facts">
            <div><span class="adfm-13f-fact-label">CUSIP</span><span class="adfm-13f-fact-value">{selected['CUSIP']}</span></div>
            <div><span class="adfm-13f-fact-label">Report period</span><span class="adfm-13f-fact-value">{report_period:%b. %d, %Y}</span></div>
            <div><span class="adfm-13f-fact-label">Position type</span><span class="adfm-13f-fact-value">{request['position_kind']}</span></div>
            <div><span class="adfm-13f-fact-label">Managers</span><span class="adfm-13f-fact-value">{len(ranked):,}</span></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    render_status_line(
        report_period=report_period.date().isoformat(),
        sec_release=release.label,
        position_type=request["position_kind"],
    )
    highest = ranked.iloc[0]
    render_kpi_cards(
        [
            ("Managers reporting", f"{len(ranked):,}", "After active filters"),
            ("Highest allocation", f"{highest['PORTFOLIO_WEIGHT_PCT']:.2f}%", str(highest["MANAGER"])),
            ("Largest position", money_label(ranked["POSITION_VALUE_USD"].max()), "Reported market value"),
            ("Aggregate reported value", money_label(ranked["POSITION_VALUE_USD"].sum()), "Across matching managers"),
            ("Median allocation", f"{ranked['PORTFOLIO_WEIGHT_PCT'].median():.2f}%", "Across matching managers"),
        ]
    )

    overview_tab, data_tab, methodology_tab = st.tabs(
        ["Overview", "Fund holdings", "Methodology"]
    )
    with overview_tab:
        render_section_header(
            f"Top managers by {sort_label.lower()}",
            "Portfolio weights reconcile to each manager's effective SEC information table.",
        )
        st.plotly_chart(
            exposure_chart(ranked, sort_label, int(request["top_n"])),
            width="stretch",
            config={"displaylogo": False},
        )

    with data_tab:
        render_section_header(
            "Fund holdings",
            "Search the holder list and click a manager row to open the complete 13F portfolio.",
        )
        st.markdown(
            '<div class="adfm-13f-note"><strong>Values:</strong> current Form 13F information-table values are displayed in dollars with thousands separators. Portfolio weights use the sum of effective holding lines, not the filer summary total.</div>',
            unsafe_allow_html=True,
        )
        manager_col, columns_col = st.columns([.8, 1.4])
        with manager_col:
            manager_query = st.text_input("Filter managers", placeholder="Search manager name")
        with columns_col:
            optional_columns = st.multiselect(
                "Customize columns",
                list(DETAIL_COLUMN_LABELS),
                default=DEFAULT_DETAIL_COLUMNS,
                format_func=lambda c: DETAIL_COLUMN_LABELS[c],
            )
        filtered = ranked
        if manager_query.strip():
            filtered = ranked.loc[
                ranked["MANAGER"].str.contains(manager_query.strip(), case=False, na=False, regex=False)
            ]
        display = filtered[["RANK", "MANAGER", *optional_columns]].head(500).reset_index(drop=True)
        st.markdown(f"**Results ({len(filtered):,} of {len(ranked):,} funds)**")
        selection = st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            height=520,
            on_select="rerun",
            selection_mode="single-row",
            key="sec_13f_holder_table",
            column_config={
                "RANK": st.column_config.NumberColumn("Rank", format="%,d"),
                "MANAGER": "Manager — click row to open",
                "PORTFOLIO_WEIGHT_PCT": st.column_config.NumberColumn("Portfolio weight", format="%.2f%%"),
                "POSITION_VALUE_USD": st.column_config.NumberColumn("Position value", format="$%,.0f"),
                "REPORTED_SHARES": st.column_config.NumberColumn("Reported shares", format="%,.0f"),
                "PORTFOLIO_VALUE_USD": st.column_config.NumberColumn("13F portfolio", format="$%,.0f"),
                "LATEST_FILING_DATE": st.column_config.DateColumn("Latest filing", format="MMM D, YYYY"),
                "CIK": "Manager CIK",
                "COMPONENT_COUNT": st.column_config.NumberColumn("Filing components", format="%,d"),
                "FILING_URL": st.column_config.LinkColumn("Filing", display_text="Open EDGAR"),
            },
        )
        rows = list(selection.selection.rows)
        if rows:
            manager_name = str(display.iloc[int(rows[0])]["MANAGER"])
            manager_row = ranked.loc[ranked["MANAGER"].eq(manager_name)].iloc[0]
            st.query_params["manager"] = str(manager_row["CIK"])
            st.rerun()
        dataframe_download(
            "Download filtered holdings CSV",
            filtered,
            f"sec_13f_exposure_{re.sub(r'[^A-Za-z0-9_-]+', '_', str(request['query']))}_{report_period.date().isoformat()}.csv",
        )

    with methodology_tab:
        st.markdown(
            """
            **Portfolio-weight correction**

            The SEC changed Form 13F information-table value reporting from thousands of dollars to the nearest dollar beginning January 3, 2023. Some filers still submit summary-page totals that behave like the former $000 convention. A numerator in dollars divided by a legacy-style summary denominator can create impossible weights above 100%.

            ADFM now calculates each manager's denominator directly from the sum of the effective information-table holding values for that same report date. Restatements supersede prior bases; subsequent new-holdings amendments are included. This makes the position value and portfolio value use the same underlying units and forces the portfolio arithmetic to reconcile.
            """
        )


def render_browser() -> None:
    inject_explorer_style(max_width_px=1560)
    inject_13f_style()
    render_page_header(
        PageHeader(
            title=TITLE,
            description="Search a ticker, rank institutional holders, and drill into any manager's complete disclosed Form 13F portfolio.",
            eyebrow="ADFM Positioning + Flows",
            source_note="Official SEC Form 13F filings and bulk data",
        )
    )
    releases = cached_releases()
    request, submitted = render_sidebar(releases)
    if submitted:
        st.session_state["sec_13f_request"] = request
        st.query_params.pop("manager", None)
    active = st.session_state.get("sec_13f_request")
    if not active:
        render_selection_note(
            "Start with a security",
            "INTC is prefilled. Run the screen to rank institutional holders and open manager portfolios.",
        )
        return
    try:
        render_screen(releases, active)
    except (Sec13FError, OSError, ValueError) as exc:
        st.error(f"The 13F screen could not be completed: {exc}")
