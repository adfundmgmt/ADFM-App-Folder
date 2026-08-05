from __future__ import annotations

import ast
import pprint
from pathlib import Path
from typing import Iterable

TARGET = Path("pages/1_ADFM_Public_Equities_Baskets.py")


def unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        ticker = str(item).upper().strip()
        if ticker and ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
    return out


def replace_at(mapping: dict[str, list[str]], old_names: list[str], new_name: str, members: Iterable[str]) -> None:
    missing = [name for name in old_names if name not in mapping]
    if missing:
        raise KeyError(f"Missing baskets in replacement: {missing}")
    items = list(mapping.items())
    first_index = min(i for i, (name, _) in enumerate(items) if name in old_names)
    items = [(name, values) for name, values in items if name not in old_names]
    items.insert(first_index, (new_name, unique(members)))
    mapping.clear()
    mapping.update(items)


def collapse(mapping: dict[str, list[str]], old_names: list[str], new_name: str, extras: Iterable[str] = ()) -> None:
    members: list[str] = []
    for name in old_names:
        if name not in mapping:
            raise KeyError(f"Missing basket: {name}")
        members.extend(mapping[name])
    members.extend(extras)
    replace_at(mapping, old_names, new_name, members)


def add_members(mapping: dict[str, list[str]], name: str, members: Iterable[str]) -> None:
    if name not in mapping:
        raise KeyError(f"Missing basket: {name}")
    mapping[name] = unique([*mapping[name], *members])


def insert_after(mapping: dict[str, list[str]], after_name: str, additions: list[tuple[str, list[str]]]) -> None:
    if after_name not in mapping:
        raise KeyError(f"Missing insertion anchor: {after_name}")
    addition_names = {name for name, _ in additions}
    for name in addition_names:
        mapping.pop(name, None)
    items = list(mapping.items())
    index = next(i for i, (name, _) in enumerate(items) if name == after_name) + 1
    normalized = [(name, unique(values)) for name, values in additions]
    items[index:index] = normalized
    mapping.clear()
    mapping.update(items)


def find_categories_assignment(tree: ast.Module) -> ast.AnnAssign | ast.Assign:
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "CATEGORIES":
            return node
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "CATEGORIES" for target in node.targets):
            return node
    raise RuntimeError("CATEGORIES assignment not found")


def main() -> None:
    source = TARGET.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignment = find_categories_assignment(tree)
    value_node = assignment.value
    categories: dict[str, dict[str, list[str]]] = eval(
        compile(ast.Expression(value_node), str(TARGET), "eval"),
        {
            "MAGNIFICENT_SEVEN": "Magnificent Seven",
            "MAGNIFICENT_SEVEN_TICKERS": ["NVDA", "MSFT", "GOOGL", "AMZN", "AAPL", "META", "TSLA"],
            "FAANG": "FAANG",
            "FAANG_TICKERS": ["META", "AAPL", "AMZN", "NFLX", "GOOGL"],
        },
    )

    if len(categories) != 14:
        raise AssertionError(f"Expected 14 groups before migration, found {len(categories)}")

    tech = categories["Information Technology"]
    financials = categories["Financials"]
    healthcare = categories["Health Care"]
    discretionary = categories["Consumer Discretionary"]
    communications = categories["Communication Services"]
    industrials = categories["Industrials"]
    staples = categories["Consumer Staples"]
    energy = categories["Energy"]
    utilities = categories["Utilities"]
    real_estate = categories["Real Estate"]
    thematic = categories["Thematic Cross-Sector Baskets"]
    macro = categories["Macro Dislocation and Special Situations"]

    # Core AI and semiconductor architecture.
    replace_at(
        tech,
        ["AI Compute and Accelerators"],
        "AI Compute and Silicon",
        ["NVDA", "AMD", "AVGO", "MRVL", "ARM", "TSM"],
    )
    replace_at(
        tech,
        ["Semiconductor Equipment"],
        "Semiconductor Manufacturing and Equipment",
        ["TSM", "ASML", "AMAT", "LRCX", "KLAC", "ONTO"],
    )
    collapse(
        tech,
        ["Analog and Mixed Signal", "Power Semiconductors"],
        "Analog, Mixed Signal and Power Semiconductors",
    )
    collapse(
        tech,
        ["Servers and AI Hardware", "Enterprise Storage"],
        "Servers, AI Hardware and Enterprise Storage",
    )
    tech.pop("Data Center Networking")
    add_members(tech, "Application Software", ["FIG"])
    add_members(tech, "Vertical Software", ["NAVN"])

    # Financial infrastructure and liquidity-sensitive cohorts.
    add_members(financials, "Brokers and Trading Platforms", ["ETOR"])
    add_members(financials, "Fintech Lenders and Platforms", ["CHYM", "KLAR"])
    replace_at(
        financials,
        ["Crypto and Tokenization Proxies"],
        "Crypto Transaction and Fintech Platforms",
        ["COIN", "HOOD", "MSTR", "PYPL", "XYZ", "IBKR", "CRCL"],
    )
    insert_after(
        financials,
        "Crypto Transaction and Fintech Platforms",
        [
            (
                "Stablecoin and Tokenized Cash Infrastructure",
                ["CRCL", "COIN", "HOOD", "PYPL", "IBKR", "CME", "CBOE"],
            ),
            (
                "Digital Banks and Consumer Fintech",
                ["CHYM", "KLAR", "SOFI", "NU", "AFRM", "UPST", "HOOD", "ETOR"],
            ),
            (
                "Retail Brokerage and Trading Activity",
                ["HOOD", "IBKR", "ETOR", "SCHW", "COIN", "VIRT", "CBOE", "NDAQ"],
            ),
        ],
    )

    # Healthcare consolidations and newer public-company cohorts.
    add_members(healthcare, "Diagnostics", ["TEM", "GH", "NTRA"])
    add_members(healthcare, "Healthcare IT", ["TEM"])
    collapse(
        healthcare,
        ["Healthcare Services", "Drug Distributors"],
        "Healthcare Services and Distribution",
        ["MDLN"],
    )
    insert_after(
        healthcare,
        "Healthcare Services and Distribution",
        [
            (
                "AI Precision Medicine and Intelligent Diagnostics",
                ["TEM", "GH", "NTRA", "EXAS", "ILMN", "RXRX", "SDGR"],
            ),
            (
                "Genomics and Gene Editing",
                ["CRSP", "NTLA", "BEAM", "EDIT", "VERV", "PACB", "ILMN"],
            ),
            (
                "Radiopharma and Nuclear Medicine",
                ["LNTH", "NVS", "LLY", "CATX", "ATNM", "CLRB"],
            ),
            (
                "Medical Distribution and Supplies",
                ["MDLN", "MCK", "COR", "CAH", "BDX", "HSIC"],
            ),
        ],
    )

    # Consumer and communications consolidations.
    beauty_members = [*discretionary["Beauty Retail and Cosmetics"], *staples["Personal Care"]]
    replace_at(
        discretionary,
        ["Beauty Retail and Cosmetics"],
        "Beauty and Personal Care",
        beauty_members,
    )
    staples.pop("Personal Care")
    collapse(
        communications,
        ["Video Games", "Interactive Entertainment"],
        "Video Games and Interactive Entertainment",
    )
    add_members(communications, "Live Entertainment and Sports", ["STUB"])
    insert_after(
        communications,
        "Live Entertainment and Sports",
        [
            (
                "Ticketing and Live Event Marketplaces",
                ["STUB", "LYV", "SEAT", "TKO", "MSGS"],
            )
        ],
    )
    insert_after(
        discretionary,
        "Casinos and Gaming",
        [
            (
                "Sports Betting and iGaming",
                ["DKNG", "FLUT", "RSI", "PENN", "MGM", "CZR"],
            ),
            (
                "Local Commerce and Delivery Platforms",
                ["UBER", "DASH", "CART", "TOST", "AMZN", "MELI"],
            ),
        ],
    )

    # Industrials, defense and physical-economy exposures.
    collapse(
        industrials,
        ["Engineering and Construction", "Building and Construction Services"],
        "Engineering and Construction Services",
    )
    insert_after(
        industrials,
        "Drones and Autonomous Defense",
        [
            (
                "Counter-UAS and Missile Defense",
                ["RTX", "NOC", "LMT", "GD", "LHX", "AVAV", "KTOS"],
            ),
            (
                "Naval Shipbuilding and Undersea Systems",
                ["HII", "GD", "BWXT", "LHX", "NOC"],
            ),
        ],
    )
    insert_after(
        industrials,
        "Industrial Automation",
        [
            (
                "Warehouse Automation and Physical AI",
                ["SYM", "GXO", "ROK", "CGNX", "TER", "AMZN"],
            ),
            (
                "Nuclear Components and Services",
                ["BWXT", "GEV", "LEU", "CCJ", "OKLO", "SMR", "CEG"],
            ),
            (
                "Water Infrastructure",
                [
                    "XYL", "WTS", "PNR", "MWA", "BMI", "FELE",
                    "AWK", "WTRG", "AWR", "CWT", "MSEX", "SJW",
                ],
            ),
        ],
    )

    # Energy, utilities and real estate clean-up.
    energy["Coal"] = ["BTU", "CNR", "AMR", "HCC", "METC", "WHC.AX"]
    uranium_energy = energy.pop("Uranium Miners and Fuel Cycle")
    collapse(
        utilities,
        ["Independent Power Producers", "Merchant Power"],
        "Merchant and Independent Power",
    )
    data_center_power = utilities.pop("Data Center Power Beneficiaries")
    add_members(real_estate, "Data Center REITs", ["IRM"])

    # Collapse generic AI-capex baskets into three core layers, then add specific bottlenecks.
    for obsolete in ["AI Data Center Capex", "AI Hardware Supply Chain", "Sovereign AI Infrastructure"]:
        thematic.pop(obsolete)
    thematic["AI Power Demand"] = unique([*thematic["AI Power Demand"], *data_center_power])
    insert_after(
        thematic,
        "FAANG",
        [
            (
                "AI Infrastructure Buildout",
                ["ANET", "VRT", "ETN", "GEV", "SMCI", "DELL", "PWR"],
            ),
            (
                "AI Cloud and GPU Infrastructure",
                ["CRWV", "NBIS", "ORCL", "MSFT", "AMZN", "GOOGL"],
            ),
            (
                "AI Scale-Up Connectivity and Fabrics",
                ["ALAB", "CRDO", "AVGO", "MRVL", "ANET", "APH", "TEL", "COHR", "LITE"],
            ),
            (
                "Data Center Cooling and Thermal Management",
                ["VRT", "MOD", "AAON", "TT", "CARR", "JCI", "NVT"],
            ),
            (
                "Data Center Electrical Balance of Plant",
                ["ETN", "GEV", "HUBB", "POWL", "NVT", "ATKR", "VRT", "PWR", "MYRG"],
            ),
            (
                "AI Data Center Water Infrastructure",
                ["XYL", "WTS", "PNR", "MWA", "BMI", "FELE"],
            ),
        ],
    )
    replace_at(
        thematic,
        ["Crypto and Tokenization Proxies"],
        "Crypto Beta, Miners and ETFs",
        ["COIN", "MSTR", "HOOD", "MARA", "RIOT", "CLSK", "IBIT", "ETHA"],
    )
    thematic.pop("Sports Live Events and Experiences")
    thematic["Humanoid Robotics"] = [ticker for ticker in thematic["Humanoid Robotics"] if ticker != "BOT"]
    insert_after(
        thematic,
        "Humanoid Robotics",
        [
            ("Private Robotics Access Vehicles", ["BOT"]),
            (
                "Recent IPO and New-Issue Beta",
                ["CRWV", "CRCL", "CHYM", "ETOR", "KLAR", "NAVN", "FIG", "MDLN", "STUB"],
            ),
            (
                "Pet Care and Premiumization",
                ["CHWY", "FRPT", "WOOF", "ZTS", "IDXX", "ELAN"],
            ),
        ],
    )

    # Special-situation consolidations.
    collapse(
        macro,
        ["Tanker Shipping", "Shipping and Geopolitical Tonne-Mile Risk"],
        "Tanker Shipping and Tonne-Mile Risk",
    )
    macro.pop("Critical Minerals: Tungsten and Antimony")
    collapse(
        macro,
        ["eVTOL and Urban Air Mobility", "Speculative Aviation and Autonomy"],
        "Speculative Aviation and Autonomy",
    )
    uranium_macro = macro.pop("Nuclear Fuel, Enrichment and Services")
    insert_after(
        macro,
        "Coal and Baseload Scarcity",
        [
            (
                "Uranium Mining, Fuel and Enrichment",
                unique([*uranium_energy, *uranium_macro]),
            )
        ],
    )

    required_exact = {
        ("Information Technology", "AI Compute and Silicon"): ["NVDA", "AMD", "AVGO", "MRVL", "ARM", "TSM"],
        ("Information Technology", "Semiconductor Manufacturing and Equipment"): ["TSM", "ASML", "AMAT", "LRCX", "KLAC", "ONTO"],
        ("Thematic Cross-Sector Baskets", "AI Infrastructure Buildout"): ["ANET", "VRT", "ETN", "GEV", "SMCI", "DELL", "PWR"],
        ("Energy", "Coal"): ["BTU", "CNR", "AMR", "HCC", "METC", "WHC.AX"],
    }
    for (group, basket), expected in required_exact.items():
        actual = categories[group][basket]
        if actual != expected:
            raise AssertionError(f"{group} / {basket}: expected {expected}, found {actual}")

    removed_names = {
        "AI Compute and Accelerators",
        "Semiconductor Equipment",
        "Data Center Networking",
        "Servers and AI Hardware",
        "Enterprise Storage",
        "Analog and Mixed Signal",
        "Power Semiconductors",
        "Healthcare Services",
        "Drug Distributors",
        "Video Games",
        "Interactive Entertainment",
        "Engineering and Construction",
        "Building and Construction Services",
        "Independent Power Producers",
        "Merchant Power",
        "Data Center Power Beneficiaries",
        "AI Data Center Capex",
        "AI Hardware Supply Chain",
        "Sovereign AI Infrastructure",
        "Sports Live Events and Experiences",
        "Tanker Shipping",
        "Shipping and Geopolitical Tonne-Mile Risk",
        "Critical Minerals: Tungsten and Antimony",
        "eVTOL and Urban Air Mobility",
        "Nuclear Fuel, Enrichment and Services",
        "Uranium Miners and Fuel Cycle",
    }
    surviving = {
        name
        for baskets in categories.values()
        for name in baskets
        if name in removed_names
    }
    if surviving:
        raise AssertionError(f"Obsolete basket names survived: {sorted(surviving)}")

    # Exact duplicate definitions usually indicate an accidental reintroduction of redundancy.
    definitions: dict[tuple[str, ...], list[str]] = {}
    duplicate_definitions: list[tuple[str, str]] = []
    for group, baskets in categories.items():
        for name, tickers in baskets.items():
            signature = tuple(tickers)
            label = f"{group} / {name}"
            if signature in definitions:
                duplicate_definitions.append((definitions[signature][0], label))
            definitions.setdefault(signature, []).append(label)
    allowed_duplicate_pairs = {
        frozenset({
            "Countries and Regions / Saudi Arabia",
            "Countries and Regions / Middle East Broad",
        }),
    }
    unapproved = [
        pair for pair in duplicate_definitions
        if frozenset(pair) not in allowed_duplicate_pairs
    ]
    # Report rather than fail because one-member country baskets can legitimately overlap.
    if unapproved:
        print("Remaining exact-definition overlaps:")
        for left, right in unapproved:
            print(f"  - {left} == {right}")

    replacement = (
        "CATEGORIES: Dict[str, Dict[str, List[str]]] = "
        + pprint.pformat(categories, width=120, sort_dicts=False)
    )
    lines = source.splitlines(keepends=True)
    start = assignment.lineno - 1
    end = assignment.end_lineno
    updated = "".join(lines[:start]) + replacement + "\n" + "".join(lines[end:])
    updated = updated.replace(
        "# 275 basket definitions across 14 groups. Internal keys include category names,",
        "# Consolidated basket definitions across 14 groups. Internal keys include category names,",
        1,
    )

    ast.parse(updated)
    TARGET.write_text(updated, encoding="utf-8")

    basket_count = sum(len(baskets) for baskets in categories.values())
    ticker_count = len({ticker for baskets in categories.values() for members in baskets.values() for ticker in members})
    print(f"Updated {basket_count} baskets across {len(categories)} groups and {ticker_count} unique tickers.")


if __name__ == "__main__":
    main()
