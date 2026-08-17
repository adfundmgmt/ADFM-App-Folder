"""SEC Form 13F institutional exposure and manager browser."""

from __future__ import annotations

import html
import os
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from adfm_core.palette import PASTEL
from adfm_core.sec_13f import (
    PreparedDataset,
    QuarterDataset,
    Sec13FError,
    available_report_periods,
    discover_quarter_datasets,
    filing_url,
    load_company_tickers,
    load_security_catalog,
    prepare_dataset,
    rank_fund_exposure,
    search_security_candidates,
    select_effective_filing_components,
)
from adfm_core.ui import (
    PageHeader,
    dataframe_download,
    inject_explorer_style,
    render_footer,
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
    "ADFM_SEC_USER_AGENT",
    "AD Fund Management LP aryadeniz@adfundmgmt.com",
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
    prepared: PreparedDataset,
    cik: str,
    report_period: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Return one manager's effective filing summary and full reported portfolio."""

    filings = pd.read_parquet(prepared.filings_path)
    components = select_effective_filing_components(filings, report_period)
    if components.empty:
        return {}, pd.DataFrame()

    target_cik = str(cik).strip().zfill(10)
    components = components.copy()
    components["CIK"] = components["CIK"].astype(str).str.zfill(10)
    manager_components = components.loc[components["CIK"].eq(target_cik)].copy()
    if manager_components.empty:
        return {}, pd.DataFrame()

    manager_components["TABLEVALUETOTAL"] = pd.to_numeric(
        manager_components["TABLEVALUETOTAL"], errors="coerce"
    )
    total_thousands = manager_components["TABLEVALUETOTAL"].sum(min_count=1)
    if pd.isna(total_thousands) or total_thousands <= 0:
        return {}, pd.DataFrame()

    accessions = manager_components["ACCESSION_NUMBER"].astype(str).tolist()
    try:
        holdings = pd.read_parquet(
            prepared.holdings_path,
            filters=[("ACCESSION_NUMBER", "in", accessions)],
        )
    except (TypeError, ValueError):
        holdings = pd.read_parquet(prepared.holdings_path)
        holdings = holdings.loc[holdings["ACCESSION_NUMBER"].astype(str).isin(accessions)]
    if holdings.empty:
        return {}, pd.DataFrame()

    filing_dates = manager_components[["ACCESSION_NUMBER", "FILING_DATE"]].copy()
    filing_dates["ACCESSION_NUMBER"] = filing_dates["ACCESSION_NUMBER"].astype(str)
    holdings["ACCESSION_NUMBER"] = holdings["ACCESSION_NUMBER"].astype(str)
    holdings = holdings.merge(filing_dates, on="ACCESSION_NUMBER", how="left")
    holdings["VALUE"] = pd.to_numeric(holdings["VALUE"], errors="coerce")
    holdings["SSHPRNAMT"] = pd.to_numeric(holdings["SSHPRNAMT"], errors="coerce")
    holdings["PUTCALL"] = holdings["PUTCALL"].fillna("").astype(str).str.upper().str.strip()
    holdings["TITLEOFCLASS"] = holdings["TITLEOFCLASS"].fillna("").astype(str)
    holdings["SSHPRNAMTTYPE"] = holdings["SSHPRNAMTTYPE"].fillna("").astype(str)
    holdings = holdings.sort_values(["FILING_DATE", "ACCESSION_NUMBER"])

    portfolio = (
        holdings.groupby(
            ["NAMEOFISSUER", "TITLEOFCLASS", "CUSIP", "PUTCALL", "SSHPRNAMTTYPE"],
            as_index=False,
            dropna=False,
        )
        .agg(
            POSITION_VALUE_THOUSANDS=("VALUE", "sum"),
            REPORTED_AMOUNT=("SSHPRNAMT", lambda values: values.sum(min_count=1)),
            SOURCE_ACCESSION_NUMBER=("ACCESSION_NUMBER", "last"),
            SOURCE_FILING_DATE=("FILING_DATE", "max"),
            LINES=("ACCESSION_NUMBER", "size"),
        )
    )
    portfolio["POSITION_VALUE_USD"] = portfolio["POSITION_VALUE_THOUSANDS"] * 1_000.0
    portfolio["PORTFOLIO_WEIGHT_PCT"] = (
        portfolio["POSITION_VALUE_THOUSANDS"] / float(total_thousands) * 100.0
    )
    portfolio["POSITION_TYPE"] = portfolio["PUTCALL"].replace(
        {"": "Long", "CALL": "Call", "PUT": "Put"}
    )
    portfolio["FILING_URL"] = portfolio["SOURCE_ACCESSION_NUMBER"].map(
        lambda accession: filing_url(target_cik, accession)
    )
    portfolio = portfolio.sort_values(
        ["PORTFOLIO_WEIGHT_PCT", "POSITION_VALUE_USD", "NAMEOFISSUER"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    portfolio.insert(0, "RANK", range(1, len(portfolio) + 1))

    base_rows = manager_components.loc[manager_components["COMPONENT_ROLE"].eq("Base")]
    base = base_rows.iloc[-1] if not base_rows.empty else manager_components.iloc[-1]
    manager_name = str(base.get("FILINGMANAGER_NAME", "")).strip() or target_cik
    report_date = pd.Timestamp(base["PERIODOFREPORT"])
    latest_filing = pd.to_datetime(manager_components["FILING_DATE"], errors="coerce").max()
    top_ten = float(portfolio.head(10)["PORTFOLIO_WEIGHT_PCT"].sum())
    summary = {
        "CIK": target_cik,
        "MANAGER": manager_name,
        "REPORT_PERIOD": report_date,
        "LATEST_FILING_DATE": latest_filing,
        "PORTFOLIO_VALUE_USD": float(total_thousands) * 1_000.0,
        "POSITION_COUNT": len(portfolio),
        "TOP_TEN_PCT": top_ten,
        "COMPONENT_COUNT": len(manager_components),
        "FILER_URL": (
            "https://www.sec.gov/edgar/browse/?CIK="
            f"{target_cik}&owner=exclude&action=getcompany&type=13F-HR"
        ),
    }
    return summary, portfolio


def money_label(value: float) -> str:
    if not pd.notna(value):
        return "N/A"
    magnitude = abs(float(value))
    if magnitude >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if magnitude >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value:,.0f}"


def portfolio_threshold_label(millions: float) -> str:
    if millions >= 1_000:
        return f"${millions / 1_000:g}B"
    return f"${millions:,.0f}M"


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
            background: #ffffff;
            border: 1px solid #d8d8d8;
            border-left: 5px solid #000000;
            margin: 0.4rem 0 0.75rem;
            padding: 1rem 1.15rem;
        }
        .adfm-13f-eyebrow {
            color: #585858;
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            margin-bottom: 0.22rem;
            text-transform: uppercase;
        }
        .adfm-13f-title-row {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
        }
        .adfm-13f-title {
            color: #000000;
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.55rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }
        .adfm-13f-query {
            border: 1px solid #000000;
            color: #000000;
            font-size: 0.75rem;
            font-weight: 750;
            letter-spacing: 0.04em;
            padding: 0.18rem 0.46rem;
        }
        .adfm-13f-facts {
            display: grid;
            gap: 0.65rem 1.3rem;
            grid-template-columns: repeat(4, minmax(120px, 1fr));
            margin-top: 0.85rem;
        }
        .adfm-13f-fact-label {
            color: #666666;
            display: block;
            font-size: 0.66rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .adfm-13f-fact-value {
            color: #111111;
            display: block;
            font-size: 0.88rem;
            font-weight: 600;
            margin-top: 0.1rem;
        }
        .adfm-13f-filter-strip {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 0 0 0.95rem;
        }
        .adfm-13f-filter-label {
            color: #555555;
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            margin-right: 0.1rem;
            text-transform: uppercase;
        }
        .adfm-13f-chip {
            border: 1px solid #cfcfcf;
            color: #202020;
            font-size: 0.74rem;
            line-height: 1.2;
            padding: 0.27rem 0.55rem;
        }
        .adfm-13f-results-count {
            color: #333333;
            font-size: 0.8rem;
            font-weight: 650;
            margin: 0.15rem 0 0.6rem;
        }
        .adfm-13f-drill-note {
            border-left: 3px solid #000000;
            color: #333333;
            font-size: 0.78rem;
            margin: 0.45rem 0 0.75rem;
            padding-left: 0.65rem;
        }
        @media (max-width: 850px) {
            .adfm-13f-facts { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_security_dossier(
    selected: pd.Series,
    request: dict[str, object],
    report_period: pd.Timestamp,
) -> None:
    query = html.escape(str(request["query"]).upper())
    issuer = html.escape(str(selected["NAMEOFISSUER"]))
    security_class = html.escape(str(selected["TITLEOFCLASS"]))
    cusip = html.escape(str(selected["CUSIP"]))
    position_kind = html.escape(str(request["position_kind"]))
    st.markdown(
        f"""
        <section class="adfm-13f-dossier">
            <div class="adfm-13f-eyebrow">SEC 13F security match</div>
            <div class="adfm-13f-title-row">
                <div class="adfm-13f-title">{issuer}</div>
                <span class="adfm-13f-query">{query}</span>
            </div>
            <div class="adfm-13f-facts">
                <div><span class="adfm-13f-fact-label">Security class</span><span class="adfm-13f-fact-value">{security_class}</span></div>
                <div><span class="adfm-13f-fact-label">CUSIP</span><span class="adfm-13f-fact-value">{cusip}</span></div>
                <div><span class="adfm-13f-fact-label">Reporting quarter</span><span class="adfm-13f-fact-value">{report_period:%b. %d, %Y}</span></div>
                <div><span class="adfm-13f-fact-label">Position view</span><span class="adfm-13f-fact-value">{position_kind}</span></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_active_filters(request: dict[str, object], report_period: pd.Timestamp) -> None:
    quarter = f"{report_period.year} Q{(report_period.month - 1) // 3 + 1}"
    filters = [
        ("Quarter", quarter),
        ("Position", str(request["position_kind"])),
        (
            "Minimum 13F portfolio",
            portfolio_threshold_label(float(request["minimum_portfolio_millions"])),
        ),
        ("Rank", str(request["sort_label"])),
    ]
    chips = "".join(
        f'<span class="adfm-13f-chip"><strong>{html.escape(label)}:</strong> '
        f"{html.escape(value)}</span>"
        for label, value in filters
    )
    st.markdown(
        f'<div class="adfm-13f-filter-strip"><span class="adfm-13f-filter-label">Active filters</span>{chips}</div>',
        unsafe_allow_html=True,
    )


def exposure_chart(ranking: pd.DataFrame, sort_label: str, top_n: int) -> go.Figure:
    sort_column = SORT_OPTIONS[sort_label]
    plot = ranking.sort_values(sort_column, ascending=False).head(top_n).copy()
    plot = plot.sort_values(sort_column, ascending=True)
    if sort_column == "PORTFOLIO_WEIGHT_PCT":
        x = plot[sort_column]
        axis_title = "Share of disclosed 13F portfolio"
        hover_value = "%{x:.2f}%"
        tick_suffix = "%"
    elif sort_column == "POSITION_VALUE_USD":
        x = plot[sort_column]
        axis_title = "Reported market value ($)"
        hover_value = "$%{x:,.0f}"
        tick_suffix = ""
    else:
        x = plot[sort_column]
        axis_title = "Reported shares"
        hover_value = "%{x:,.0f}"
        tick_suffix = ""

    custom = plot[
        ["PORTFOLIO_WEIGHT_PCT", "POSITION_VALUE_USD", "PORTFOLIO_VALUE_USD"]
    ].to_numpy()
    fig = go.Figure(
        go.Bar(
            x=x,
            y=plot["MANAGER"],
            orientation="h",
            marker=dict(color=PASTEL["blue"]),
            customdata=custom,
            hovertemplate=(
                "<b>%{y}</b><br>"
                + sort_label
                + ": "
                + hover_value
                + "<br>Portfolio weight: %{customdata[0]:.2f}%"
                + "<br>Position value: $%{customdata[1]:,.0f}"
                + "<br>13F portfolio: $%{customdata[2]:,.0f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=max(440, 32 * len(plot) + 115),
        margin=dict(l=10, r=25, t=30, b=45),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        font=dict(family="Arial", color="#171717"),
        xaxis=dict(
            title=axis_title,
            ticksuffix=tick_suffix,
            gridcolor="#e5e5e5",
            zeroline=False,
        ),
        yaxis=dict(title=None, automargin=True),
    )
    return fig


def render_sidebar(releases: list[QuarterDataset]) -> tuple[dict[str, object], bool]:
    with st.sidebar:
        st.header("Security search")
        with st.form("sec_13f_screen"):
            query = st.text_input(
                "Ticker, issuer, or CUSIP",
                value="INTC",
                help="Ticker symbols are resolved through the SEC company directory, then matched to the exact filed security and CUSIP.",
            )
            release_label = st.selectbox(
                "SEC data release",
                options=[release.label for release in releases],
                index=0,
                help="Each SEC bulk release contains the preceding three months of Form 13F filings and amendments.",
            )
            position_kind = st.selectbox("Position type", POSITION_KINDS, index=0)
            minimum_portfolio_billions = st.number_input(
                "Minimum disclosed portfolio ($B)",
                min_value=0.0,
                value=1.0,
                step=0.25,
            )
            sort_label = st.selectbox(
                "Rank funds by",
                options=list(SORT_OPTIONS),
                index=0,
            )
            top_n = st.slider(
                "Funds in overview", min_value=10, max_value=50, value=25, step=5
            )
            submitted = st.form_submit_button("Run 13F screen", width="stretch")

        st.caption(
            "The first run for a release downloads and prepares the official SEC bulk file. "
            "Later searches reuse the local cache."
        )
        st.markdown("---")
        st.header("About This Tool")
        st.markdown(
            """
            **Security → holders → manager portfolio**

            1. Search a ticker or issuer.
            2. Rank managers by portfolio weight, value, or shares.
            3. Open **Fund holdings** and click any manager row.
            4. The browser drills into that manager's complete effective 13F portfolio.

            **Primary source:** SEC Form 13F filings and official bulk data sets.
            """
        )

    selected_release = next(
        release for release in releases if release.label == release_label
    )
    request = {
        "query": query.strip(),
        "release_slug": selected_release.slug,
        "position_kind": position_kind,
        "minimum_portfolio_millions": float(minimum_portfolio_billions) * 1_000.0,
        "sort_label": sort_label,
        "top_n": int(top_n),
    }
    return request, submitted


def render_manager_profile(
    prepared: PreparedDataset,
    cik: str,
    report_period: pd.Timestamp,
    security: pd.Series,
    request: dict[str, object],
) -> None:
    summary, portfolio = cached_manager_portfolio(
        prepared,
        cik,
        report_period.date().isoformat(),
    )
    if not summary or portfolio.empty:
        st.error("The selected manager's effective 13F portfolio could not be reconstructed.")
        if st.button("Back to holders"):
            if "manager" in st.query_params:
                del st.query_params["manager"]
            st.rerun()
        return

    back_col, source_col = st.columns([1, 1], vertical_alignment="center")
    with back_col:
        if st.button(
            f"← Back to {str(request['query']).upper()} holders",
            type="secondary",
        ):
            if "manager" in st.query_params:
                del st.query_params["manager"]
            st.rerun()
    with source_col:
        st.link_button("Open manager on SEC EDGAR", str(summary["FILER_URL"]))

    manager = html.escape(str(summary["MANAGER"]))
    target_security = html.escape(str(security["NAMEOFISSUER"]))
    st.markdown(
        f"""
        <section class="adfm-13f-dossier">
            <div class="adfm-13f-eyebrow">13F manager profile · opened from {target_security}</div>
            <div class="adfm-13f-title-row">
                <div class="adfm-13f-title">{manager}</div>
                <span class="adfm-13f-query">CIK {html.escape(str(summary['CIK']))}</span>
            </div>
            <div class="adfm-13f-facts">
                <div><span class="adfm-13f-fact-label">Report period</span><span class="adfm-13f-fact-value">{pd.Timestamp(summary['REPORT_PERIOD']):%b. %d, %Y}</span></div>
                <div><span class="adfm-13f-fact-label">Latest filing</span><span class="adfm-13f-fact-value">{pd.Timestamp(summary['LATEST_FILING_DATE']):%b. %d, %Y}</span></div>
                <div><span class="adfm-13f-fact-label">13F portfolio</span><span class="adfm-13f-fact-value">{money_label(float(summary['PORTFOLIO_VALUE_USD']))}</span></div>
                <div><span class="adfm-13f-fact-label">Reported positions</span><span class="adfm-13f-fact-value">{int(summary['POSITION_COUNT']):,}</span></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    render_kpi_cards(
        [
            (
                "13F portfolio",
                money_label(float(summary["PORTFOLIO_VALUE_USD"])),
                "Effective disclosed market value",
            ),
            (
                "Reported positions",
                f"{int(summary['POSITION_COUNT']):,}",
                "Security / option lines after consolidation",
            ),
            (
                "Top 10 concentration",
                f"{float(summary['TOP_TEN_PCT']):.1f}%",
                "Share of disclosed portfolio in ten largest positions",
            ),
            (
                "Filing components",
                f"{int(summary['COMPONENT_COUNT']):,}",
                "Base filing plus effective new-holdings amendments",
            ),
        ]
    )

    render_section_header(
        "Manager portfolio",
        "Full effective 13F holdings for the selected reporting date. Values are as filed, not current marks.",
    )
    filter_col, type_col = st.columns([1.3, 0.7])
    with filter_col:
        issuer_filter = st.text_input(
            "Filter portfolio",
            placeholder="Search issuer, CUSIP, or share class",
            key="manager_portfolio_filter",
        )
    with type_col:
        type_options = ["All", *sorted(portfolio["POSITION_TYPE"].dropna().unique().tolist())]
        portfolio_type = st.selectbox(
            "Position type",
            type_options,
            key="manager_portfolio_type",
        )

    filtered = portfolio.copy()
    if issuer_filter.strip():
        needle = issuer_filter.strip()
        mask = (
            filtered["NAMEOFISSUER"].str.contains(needle, case=False, na=False, regex=False)
            | filtered["CUSIP"].astype(str).str.contains(needle, case=False, na=False, regex=False)
            | filtered["TITLEOFCLASS"].astype(str).str.contains(needle, case=False, na=False, regex=False)
        )
        filtered = filtered.loc[mask]
    if portfolio_type != "All":
        filtered = filtered.loc[filtered["POSITION_TYPE"].eq(portfolio_type)]

    manager_display = filtered[
        [
            "RANK",
            "NAMEOFISSUER",
            "TITLEOFCLASS",
            "POSITION_TYPE",
            "CUSIP",
            "PORTFOLIO_WEIGHT_PCT",
            "POSITION_VALUE_USD",
            "REPORTED_AMOUNT",
            "SSHPRNAMTTYPE",
            "SOURCE_FILING_DATE",
            "FILING_URL",
        ]
    ].head(750)
    st.markdown(
        f'<div class="adfm-13f-results-count">Results ({len(filtered):,} positions)</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        manager_display,
        hide_index=True,
        width="stretch",
        height=600,
        column_config={
            "RANK": st.column_config.NumberColumn("Rank", format="%d"),
            "NAMEOFISSUER": "Issuer",
            "TITLEOFCLASS": "Class",
            "POSITION_TYPE": "Type",
            "CUSIP": "CUSIP",
            "PORTFOLIO_WEIGHT_PCT": st.column_config.NumberColumn(
                "% of portfolio", format="%.2f%%"
            ),
            "POSITION_VALUE_USD": st.column_config.NumberColumn(
                "Market value", format="$%.0f"
            ),
            "REPORTED_AMOUNT": st.column_config.NumberColumn(
                "Shares / principal", format="%.0f"
            ),
            "SSHPRNAMTTYPE": "Amount type",
            "SOURCE_FILING_DATE": st.column_config.DateColumn(
                "Filing date", format="MMM D, YYYY"
            ),
            "FILING_URL": st.column_config.LinkColumn(
                "Source", display_text="Open EDGAR"
            ),
        },
    )
    export_manager = re.sub(r"[^A-Za-z0-9_-]+", "_", str(summary["MANAGER"]))
    dataframe_download(
        "Download manager portfolio CSV",
        filtered,
        f"sec_13f_manager_{export_manager}_{report_period.date().isoformat()}.csv",
    )


def render_methodology(
    selected: pd.Series,
    prepared: PreparedDataset,
    position_kind: str,
) -> None:
    st.markdown(
        f"""
        **Exposure definition**

        - Portfolio weight equals the selected security's reported market value divided by the manager's total reported Form 13F market value for the same reporting date.
        - SEC values are reported in thousands of dollars. This page converts them to dollars for display; the percentage calculation is unchanged.
        - The active security is **{selected['NAMEOFISSUER']} — {selected['TITLEOFCLASS']}**, CUSIP `{selected['CUSIP']}`. The active position view is **{position_kind.lower()}**.

        **Manager drill-through**

        - Clicking a fund row reconstructs that manager's complete effective portfolio from the same SEC release and reporting date.
        - The manager view combines the effective base filing with subsequent amendments that add new holdings and exposes direct EDGAR links for the underlying filing components.
        - Top-ten concentration is calculated from the manager's as-filed 13F market values, not current prices.

        **Filing scope**

        - A later restatement supersedes the original filing. Amendments that add new holdings are added only when they follow the effective base filing.
        - Form 13F is filed after quarter-end and can be up to 45 days stale when published. It does not disclose short positions, and option values are not delta-adjusted.
        - Reported market values are point-in-time filing values, not current marks. Confidential treatment, manager aggregation, shared discretion, and filer errors can affect comparability.
        - Tickers are not included in the 13F information table. The app uses the SEC ticker directory to resolve the company name and exposes the matched issuer, class, and CUSIP for review.

        **Source and cache**

        - [SEC Form 13F data release]({prepared.source_url})
        - Prepared locally at {prepared.prepared_at}; {prepared.holdings_rows:,} as-filed holding rows are indexed for this release.
        """
    )


def render_screen(releases: list[QuarterDataset], request: dict[str, object]) -> None:
    release = next(
        item for item in releases if item.slug == str(request["release_slug"])
    )
    with st.spinner(
        "Loading the SEC release. A first-time preparation can take several minutes..."
    ):
        prepared = prepare_dataset(release)
        periods = available_report_periods(prepared)
        ticker_directory = cached_ticker_directory()
        catalog = cached_security_catalog(prepared)

    if not periods:
        st.error("The selected SEC release does not contain a usable 13F report date.")
        return
    report_period = periods[0]
    candidates = search_security_candidates(
        catalog,
        ticker_directory,
        str(request["query"]),
        position_kind=str(request["position_kind"]),
    )
    if candidates.empty:
        st.error(
            "No matching filed security was found. Try the issuer name or a nine-character CUSIP."
        )
        return

    if len(candidates) > 1:
        labels = [candidate_label(row) for _, row in candidates.iterrows()]
        selected_label = st.selectbox(
            "Matched filed security",
            labels,
            index=0,
            help="Review the exact share class and CUSIP before using the ranking.",
        )
        selected = candidates.iloc[labels.index(selected_label)]
    else:
        selected = candidates.iloc[0]

    with st.spinner("Calculating manager allocations..."):
        ranking = cached_ranking(
            prepared,
            (str(selected["CUSIP"]),),
            report_period.date().isoformat(),
            str(request["position_kind"]),
            float(request["minimum_portfolio_millions"]),
        )
    if ranking.empty:
        st.warning(
            "No effective 13F filings met the current security, position-type, and portfolio-size filters."
        )
        return

    manager_cik = str(st.query_params.get("manager", "") or "").strip()
    if manager_cik:
        render_manager_profile(prepared, manager_cik, report_period, selected, request)
        return

    sort_label = str(request["sort_label"])
    sort_column = SORT_OPTIONS[sort_label]
    ranked = ranking.sort_values(sort_column, ascending=False).reset_index(drop=True)
    ranked["RANK"] = range(1, len(ranked) + 1)

    render_security_dossier(selected, request, report_period)
    render_active_filters(request, report_period)
    render_status_line(
        report_period=report_period.date().isoformat(),
        sec_release=release.label,
        position_type=request["position_kind"],
        minimum_portfolio=portfolio_threshold_label(
            float(request["minimum_portfolio_millions"])
        ),
    )
    render_selection_note(
        "Matched security",
        (
            f"{request['query']} resolved to {selected['NAMEOFISSUER']} — "
            f"{selected['TITLEOFCLASS']} (CUSIP {selected['CUSIP']})."
        ),
    )

    highest = ranking.loc[ranking["PORTFOLIO_WEIGHT_PCT"].idxmax()]
    render_kpi_cards(
        [
            (
                "Managers reporting",
                f"{len(ranking):,}",
                "After the portfolio-size and filing filters",
            ),
            (
                "Highest allocation",
                f"{highest['PORTFOLIO_WEIGHT_PCT']:.2f}%",
                str(highest["MANAGER"]),
            ),
            (
                "Largest position",
                money_label(ranking["POSITION_VALUE_USD"].max()),
                "Maximum reported market value",
            ),
            (
                "Aggregate reported value",
                money_label(ranking["POSITION_VALUE_USD"].sum()),
                "Sum across matching managers; not ownership-adjusted",
            ),
            (
                "Median allocation",
                f"{ranking['PORTFOLIO_WEIGHT_PCT'].median():.2f}%",
                "Median among matching managers",
            ),
        ]
    )

    ranking_tab, data_tab, methodology_tab = st.tabs(
        ["Overview", "Fund holdings", "Methodology"]
    )
    with ranking_tab:
        render_section_header(
            f"Top managers by {sort_label.lower()}",
            (
                f"Showing up to {int(request['top_n'])} managers for the {report_period:%b. %d, %Y} "
                "reporting date. Hover for allocation and value context."
            ),
        )
        st.markdown(
            f'<div class="adfm-13f-results-count">Results ({len(ranked):,} funds) · charting the top {min(int(request["top_n"]), len(ranked)):,}</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            exposure_chart(ranked, sort_label, int(request["top_n"])),
            width="stretch",
            config={"displaylogo": False, "scrollZoom": False},
        )

    with data_tab:
        render_section_header(
            "Fund holdings",
            "Search the holder list, then click any manager row to open that fund's complete 13F portfolio.",
        )
        st.markdown(
            '<div class="adfm-13f-drill-note"><strong>Drill-through:</strong> select one fund row below and ADFM will open the manager profile using the same reporting quarter.</div>',
            unsafe_allow_html=True,
        )
        manager_col, columns_col = st.columns([0.8, 1.4])
        with manager_col:
            manager_query = st.text_input(
                "Filter managers",
                placeholder="Search manager name",
            )
        with columns_col:
            optional_columns = st.multiselect(
                "Customize columns",
                options=list(DETAIL_COLUMN_LABELS),
                default=DEFAULT_DETAIL_COLUMNS,
                format_func=lambda column: DETAIL_COLUMN_LABELS[column],
            )

        filtered = ranked
        if manager_query.strip():
            filtered = ranked[
                ranked["MANAGER"].str.contains(
                    manager_query.strip(), case=False, na=False, regex=False
                )
            ]
        display_columns = ["RANK", "MANAGER", *optional_columns]
        display = filtered[display_columns].head(500).copy().reset_index(drop=True)
        st.markdown(
            f'<div class="adfm-13f-results-count">Results ({len(filtered):,} of {len(ranked):,} funds)</div>',
            unsafe_allow_html=True,
        )
        selection = st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            height=520,
            on_select="rerun",
            selection_mode="single-row",
            key="sec_13f_holder_table",
            column_config={
                "RANK": st.column_config.NumberColumn("Rank", format="%d"),
                "MANAGER": "Manager — click row to open",
                "PORTFOLIO_WEIGHT_PCT": st.column_config.NumberColumn(
                    "Portfolio weight", format="%.2f%%"
                ),
                "POSITION_VALUE_USD": st.column_config.NumberColumn(
                    "Position value", format="$%.0f"
                ),
                "REPORTED_SHARES": st.column_config.NumberColumn(
                    "Reported shares", format="%.0f"
                ),
                "PORTFOLIO_VALUE_USD": st.column_config.NumberColumn(
                    "13F portfolio", format="$%.0f"
                ),
                "LATEST_FILING_DATE": st.column_config.DateColumn(
                    "Latest filing", format="MMM D, YYYY"
                ),
                "CIK": "Manager CIK",
                "COMPONENT_COUNT": st.column_config.NumberColumn(
                    "Filing components", format="%d"
                ),
                "FILING_URL": st.column_config.LinkColumn(
                    "Filing", display_text="Open EDGAR"
                ),
            },
        )
        selected_rows = list(selection.selection.rows)
        if selected_rows:
            row = display.iloc[int(selected_rows[0])]
            manager_name = str(row["MANAGER"])
            manager_match = ranked.loc[ranked["MANAGER"].eq(manager_name)].iloc[0]
            st.query_params["manager"] = str(manager_match["CIK"])
            st.rerun()

        export_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(request["query"]))
        dataframe_download(
            "Download filtered holdings CSV",
            filtered,
            f"sec_13f_exposure_{export_name}_{report_period.date().isoformat()}.csv",
        )

    with methodology_tab:
        render_methodology(selected, prepared, str(request["position_kind"]))


def render_page() -> None:
    render_page_header(
        PageHeader(
            title=TITLE,
            description=(
                "Search a ticker, rank institutional holders, then drill into any manager's complete disclosed Form 13F portfolio."
            ),
            eyebrow="ADFM Positioning + Flows",
            source_note="Official SEC Form 13F filings and bulk data",
        )
    )
    releases = cached_releases()
    if not releases:
        st.error("No official SEC Form 13F data releases are configured.")
        return

    request, submitted = render_sidebar(releases)
    if submitted:
        st.session_state["sec_13f_request"] = request
        if "manager" in st.query_params:
            del st.query_params["manager"]
    active_request = st.session_state.get("sec_13f_request")
    if not active_request:
        render_selection_note(
            "Start with a security",
            "Enter a ticker such as VST, NVDA, or MSFT. Run the screen to rank institutional holders, then click a fund row to inspect that manager's full 13F portfolio.",
        )
        st.info(
            "No SEC archive is downloaded until you run a screen. The first preparation can take several minutes; later searches reuse it."
        )
        return

    try:
        render_screen(releases, active_request)
    except (Sec13FError, OSError, ValueError) as exc:
        st.error(f"The 13F screen could not be completed: {exc}")


st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")
inject_explorer_style(max_width_px=1560)
inject_13f_style()
render_page()
render_footer(
    data_note="Primary inputs: official SEC Form 13F filings and bulk data sets; SEC company ticker directory."
)
