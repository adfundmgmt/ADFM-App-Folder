"""Regression tests for the public-equities basket taxonomy."""

import ast
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "pages" / "1_ADFM_Public_Equities_Baskets.py"


def _literal_assignment(name: str):
    tree = ast.parse(PAGE.read_text(encoding="utf-8"), filename=str(PAGE))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} was not found in {PAGE.name}")


def test_geography_is_organized_by_continent() -> None:
    categories = _literal_assignment("CATEGORIES")
    expected = {
        "Global and Cross-Regional",
        "North America",
        "South America",
        "Europe",
        "Asia",
        "Oceania",
        "Middle East and Africa",
    }

    assert expected.issubset(categories)
    assert "Countries and Regions" not in categories


def test_global_leadership_and_financial_baskets_remain_present() -> None:
    categories = _literal_assignment("CATEGORIES")
    expected_locations = {
        "Custody and Trust Banks": "Financials",
        "Business Development Companies": "Financials",
        "Specialty P&C and E&S Insurance": "Financials",
        "Life and Annuity Platforms": "Financials",
        "Mechanical and Electrical Contractors": "Industrials",
        "Energy Royalties and Mineral Rights": "Energy",
        "Canadian Banks": "North America",
        "South American Banks": "South America",
        "European Banks": "Europe",
        "European Defense Leaders": "Europe",
        "European Grid Equipment and Cables": "Europe",
        "Japan Banks and Diversified Financials": "Asia",
        "Japanese Trading Houses": "Asia",
        "Japan Semiconductor Equipment": "Asia",
        "Taiwan AI ODM and Thermal": "Asia",
        "Korea Power Equipment": "Asia",
        "India Defense and Capital Goods": "Asia",
        "India Power and Grid Equipment": "Asia",
    }

    for basket, category in expected_locations.items():
        assert basket in categories[category]

    basket_names = [name for baskets in categories.values() for name in baskets]
    assert not [name for name, count in Counter(basket_names).items() if count > 1]


def test_obsolete_symbols_are_not_reintroduced() -> None:
    categories = _literal_assignment("CATEGORIES")
    tickers = {
        ticker
        for baskets in categories.values()
        for members in baskets.values()
        for ticker in members
    }

    assert {"NYCB", "DFS", "FM", "GXG"}.isdisjoint(tickers)
    assert {"FLG", "BNY", "COLO", "MFG", "MUFG", "SMFG"}.issubset(tickers)


def test_local_listing_suffixes_have_fx_conversion_rules() -> None:
    categories = _literal_assignment("CATEGORIES")
    suffix_rules = dict(_literal_assignment("FX_SUFFIX_CONVERSIONS"))
    tickers = {
        ticker
        for baskets in categories.values()
        for members in baskets.values()
        for ticker in members
    }

    local_tickers = {ticker for ticker in tickers if "." in ticker}
    uncovered = {
        ticker
        for ticker in local_tickers
        if not any(ticker.endswith(suffix) for suffix in suffix_rules)
    }
    assert not uncovered
