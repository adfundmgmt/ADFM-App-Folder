"""Shared output palette for ADFM analytical charts and signal readouts.

The application shell remains black and white. These colors are reserved for
data, series, regimes, risk states, and categorical distinctions inside outputs.
"""

from __future__ import annotations

from typing import Final


PASTEL: Final[dict[str, str]] = {
    "blue": "#7FA7D8",
    "coral": "#E79A78",
    "sage": "#8FBF9F",
    "amber": "#D7AE62",
    "lavender": "#A995C8",
    "teal": "#72B7B2",
    "rose": "#D98C8C",
    "periwinkle": "#8F9ED1",
    "olive": "#A8B77A",
    "mauve": "#C38FB5",
    "sky": "#8FC6DF",
    "apricot": "#E7B98A",
    "mint": "#9BC9B3",
    "salmon": "#DF8F9D",
    "cornflower": "#779ECB",
    "plum": "#B185A7",
    "seafoam": "#82BEB0",
    "sand": "#C9B37E",
    "slate_blue": "#879BBE",
    "clay": "#C79274",
}

PASTEL_20: Final[tuple[str, ...]] = tuple(PASTEL.values())

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
    """Return a stable palette color, cycling only after all 20 are used."""

    return PASTEL_20[index % len(PASTEL_20)]
