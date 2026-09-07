"""Shared Excel-style output palette for ADFM analytical charts and signals.

The application shell remains black and white. These colors are reserved for
data, series, regimes, risk states, and categorical distinctions inside outputs.
The stronger Office-style RGB values are designed for legibility on a white
canvas and in screenshots, exports, and investment-committee materials.
"""

from __future__ import annotations

from typing import Final

EXCEL: Final[dict[str, str]] = {
    "blue": "#4472C4",
    "coral": "#ED7D31",
    "sage": "#70AD47",
    "amber": "#FFC000",
    "lavender": "#8064A2",
    "teal": "#4BACC6",
    "rose": "#C0504D",
    "periwinkle": "#5B9BD5",
    "olive": "#9BBB59",
    "mauve": "#A64D79",
    "sky": "#00B0F0",
    "apricot": "#F79646",
    "mint": "#00B050",
    "salmon": "#E26B6A",
    "cornflower": "#2F5597",
    "plum": "#7030A0",
    "seafoam": "#008C95",
    "sand": "#A67C00",
    "slate_blue": "#7F8C8D",
    "clay": "#C65911",
}

EXCEL_20: Final[tuple[str, ...]] = tuple(EXCEL.values())

# Compatibility aliases keep every existing page on the centralized palette
# without a broad rename touching analytical code.
PASTEL: Final[dict[str, str]] = EXCEL
PASTEL_20: Final[tuple[str, ...]] = EXCEL_20

# Negative or adverse values sit at the low end; positive or constructive
# values sit at the high end. The neutral midpoint stays close to the canvas.
PASTEL_DIVERGING_SCALE: Final[list[list[float | str]]] = [
    [0.0, PASTEL["rose"]],
    [0.5, "#FBFBF8"],
    [1.0, PASTEL["sage"]],
]

# Rate-pressure matrices use the inverse interpretation: falling yields are
# constructive and rising yields are adverse.
PASTEL_RATES_SCALE: Final[list[list[float | str]]] = [
    [0.0, PASTEL["sage"]],
    [0.5, "#FBFBF8"],
    [1.0, PASTEL["rose"]],
]


def pastel(index: int) -> str:
    """Return a stable Excel palette color, cycling after all 20 are used."""

    return PASTEL_20[index % len(PASTEL_20)]
