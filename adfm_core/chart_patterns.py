"""Heuristic, volatility-aware price-pattern detection for ADFM charts.

The detector intentionally returns a small ranked set of explainable candidates.
It is not a trading signal engine: a pattern is only marked confirmed after the
latest close clears its inferred breakout boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

SUPPORTED_PATTERN_NAMES = (
    "Double Bottom",
    "Double Top",
    "Inverse Head and Shoulders",
    "Head and Shoulders Top",
    "Triple Bottom",
    "Triple Top",
    "Rounding Bottom",
    "Rounding Top",
    "Cup and Handle",
    "Inverse Cup and Handle",
    "Diamond Bottom",
    "Diamond Top",
    "V-Bottom",
    "V-Top",
    "Ascending Triangle",
    "Descending Triangle",
    "Symmetrical Triangle",
    "Bull Flag",
    "Bear Flag",
    "Bull Pennant",
    "Bear Pennant",
    "Rectangle",
    "Rising Wedge",
    "Falling Wedge",
    "Megaphone / Broadening Formation",
)


@dataclass(frozen=True)
class PatternPoint:
    pos: int
    price: float


@dataclass(frozen=True)
class PatternLine:
    start_pos: int
    end_pos: int
    start_price: float
    end_price: float
    role: str


@dataclass(frozen=True)
class PatternDetection:
    name: str
    family: str
    bias: str
    status: str
    confidence: float
    start_pos: int
    end_pos: int
    points: tuple[PatternPoint, ...]
    lines: tuple[PatternLine, ...]
    breakout_level: float | None
    target_level: float | None
    rationale: str


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _price_frame(df: pd.DataFrame, max_bars: int | None = None) -> pd.DataFrame:
    if df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
        return pd.DataFrame()

    out = df.tail(max_bars).copy() if max_bars is not None else df.copy()
    for column in ("Open", "High", "Low", "Close"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=["High", "Low", "Close"])


def _atr_value(df: pd.DataFrame) -> float:
    if "ATR14" in df.columns:
        atr_values = _finite(pd.to_numeric(df["ATR14"], errors="coerce").tail(60))
        if atr_values.size:
            return float(np.nanmedian(atr_values))

    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    previous = np.r_[close[0], close[:-1]]
    true_range = np.maximum(high - low, np.maximum(np.abs(high - previous), np.abs(low - previous)))
    tail = _finite(true_range[-60:])
    return float(np.nanmedian(tail)) if tail.size else 0.0


def _line_value(line: PatternLine, pos: int) -> float:
    span = line.end_pos - line.start_pos
    if span == 0:
        return float(line.end_price)
    fraction = (pos - line.start_pos) / span
    return float(line.start_price + fraction * (line.end_price - line.start_price))


def _status_and_target(
    bias: str,
    latest_close: float,
    breakout: float | None,
    height: float | None,
) -> tuple[str, float | None]:
    if breakout is None or not np.isfinite(breakout):
        return "Developing", None

    buffer = max(abs(breakout) * 0.0015, 1e-9)
    if bias == "bullish":
        confirmed = latest_close > breakout + buffer
        target = breakout + abs(height or 0.0) if confirmed and height else None
    elif bias == "bearish":
        confirmed = latest_close < breakout - buffer
        target = breakout - abs(height or 0.0) if confirmed and height else None
    else:
        confirmed = False
        target = None
    return ("Confirmed" if confirmed else "Developing"), target


def _regression(points: Sequence[PatternPoint]) -> tuple[float, float, float]:
    x = np.asarray([point.pos for point in points], dtype=float)
    y = np.asarray([point.price for point in points], dtype=float)
    if len(points) < 2 or np.ptp(x) == 0:
        return 0.0, float(np.nanmean(y)), 0.0
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    total = float(np.sum((y - np.mean(y)) ** 2))
    residual = float(np.sum((y - fitted) ** 2))
    r2 = 1.0 - residual / total if total > 1e-12 else 1.0
    return float(slope), float(intercept), _clamp(r2, 0.0, 1.0)


def _trend_change(close: np.ndarray, end_pos: int, lookback: int = 35) -> float:
    start = max(0, end_pos - lookback)
    if end_pos <= start or close[start] == 0:
        return 0.0
    return float(close[end_pos] / close[start] - 1.0)


def find_price_pivots(df: pd.DataFrame, max_pivots: int = 32) -> list[dict[str, float | int | str]]:
    """Return alternating, noise-filtered high/low pivots in visible-bar coordinates."""
    data = _price_frame(df)
    n = len(data)
    if n < 18:
        return []

    highs = data["High"].to_numpy(dtype=float)
    lows = data["Low"].to_numpy(dtype=float)
    close = data["Close"].to_numpy(dtype=float)
    window = max(2, min(9, int(round(n / 48))))
    atr = _atr_value(data)
    reference = float(np.nanmedian(close))
    min_move = max(atr * 1.05, abs(reference) * 0.010)
    candidates: list[dict[str, float | int | str]] = []

    for pos in range(window, n - window):
        high_slice = highs[pos - window : pos + window + 1]
        low_slice = lows[pos - window : pos + window + 1]
        if np.isfinite(highs[pos]) and highs[pos] >= np.nanmax(high_slice):
            candidates.append({"pos": pos, "price": float(highs[pos]), "kind": "H"})
        if np.isfinite(lows[pos]) and lows[pos] <= np.nanmin(low_slice):
            candidates.append({"pos": pos, "price": float(lows[pos]), "kind": "L"})

    candidates.sort(key=lambda item: int(item["pos"]))
    pivots: list[dict[str, float | int | str]] = []
    for candidate in candidates:
        if not pivots:
            pivots.append(candidate)
            continue
        previous = pivots[-1]
        if candidate["kind"] == previous["kind"]:
            more_extreme = (
                candidate["kind"] == "H" and float(candidate["price"]) > float(previous["price"])
            ) or (
                candidate["kind"] == "L" and float(candidate["price"]) < float(previous["price"])
            )
            if more_extreme:
                pivots[-1] = candidate
            continue
        if abs(float(candidate["price"]) - float(previous["price"])) >= min_move:
            pivots.append(candidate)

    return pivots[-max_pivots:]


def _point(pivot: dict[str, float | int | str]) -> PatternPoint:
    return PatternPoint(int(pivot["pos"]), float(pivot["price"]))


def _horizontal_line(points: Sequence[PatternPoint], role: str, end_pos: int | None = None) -> PatternLine:
    price = float(np.mean([point.price for point in points]))
    return PatternLine(points[0].pos, end_pos if end_pos is not None else points[-1].pos, price, price, role)


def _reversal_patterns(
    data: pd.DataFrame,
    pivots: list[dict[str, float | int | str]],
    atr: float,
) -> list[PatternDetection]:
    detections: list[PatternDetection] = []
    n = len(data)
    close = data["Close"].to_numpy(dtype=float)
    latest = float(close[-1])
    recent_cutoff = max(0, n - max(140, int(n * 0.65)))
    tolerance = max(atr * 1.65, abs(latest) * 0.022)

    def add_double(sequence: Sequence[dict[str, float | int | str]], bottom: bool) -> None:
        first, middle, last = map(_point, sequence)
        if first.pos < recent_cutoff or last.pos - first.pos < 8 or last.pos < n - max(75, n // 3):
            return
        outer_error = abs(first.price - last.price)
        if outer_error > tolerance:
            return
        support = (first.price + last.price) / 2
        height = (middle.price - support) if bottom else (support - middle.price)
        if height < max(atr * 2.2, abs(latest) * 0.028):
            return
        prior = _trend_change(close, first.pos)
        if bottom:
            name, bias, breakout = "Double Bottom", "bullish", middle.price
            role, desired_prior = "support", prior <= 0.03
            rationale = "Two volatility-matched lows form a W base; confirmation requires a close above the intervening high."
        else:
            name, bias, breakout = "Double Top", "bearish", middle.price
            role, desired_prior = "resistance", prior >= -0.03
            rationale = "Two volatility-matched highs form an M peak; confirmation requires a close below the intervening low."
        status, target = _status_and_target(bias, latest, breakout, height)
        similarity = 1.0 - outer_error / max(tolerance, 1e-9)
        recency = 1.0 - (n - 1 - last.pos) / max(n, 1)
        confidence = 61 + 17 * similarity + 8 * recency + (5 if desired_prior else 0) + (4 if status == "Confirmed" else 0)
        detections.append(
            PatternDetection(
                name, "reversal", bias, status, _clamp(confidence, 0, 96), first.pos, last.pos,
                (first, middle, last),
                (_horizontal_line((first, last), role, end_pos=n - 1),
                 PatternLine(first.pos, n - 1, breakout, breakout, "breakout")),
                breakout, target, rationale,
            )
        )

    def add_triple(sequence: Sequence[dict[str, float | int | str]], bottom: bool) -> None:
        points = tuple(map(_point, sequence))
        tests = points[0::2]
        reactions = points[1::2]
        level = float(np.mean([point.price for point in tests]))
        if points[0].pos < recent_cutoff or points[-1].pos < n - max(80, n // 3):
            return
        if max(abs(point.price - level) for point in tests) > tolerance:
            return
        breakout = max(point.price for point in reactions) if bottom else min(point.price for point in reactions)
        height = breakout - level if bottom else level - breakout
        if height < max(atr * 2.0, abs(latest) * 0.025):
            return
        name = "Triple Bottom" if bottom else "Triple Top"
        bias = "bullish" if bottom else "bearish"
        role = "support" if bottom else "resistance"
        status, target = _status_and_target(bias, latest, breakout, height)
        spread = max(point.price for point in tests) - min(point.price for point in tests)
        confidence = 75 + 16 * (1 - spread / max(tolerance * 2, 1e-9)) + (5 if status == "Confirmed" else 0)
        detections.append(
            PatternDetection(
                name, "reversal", bias, status, _clamp(confidence, 0, 97), points[0].pos, points[-1].pos,
                points,
                (_horizontal_line(tests, role, end_pos=n - 1),
                 PatternLine(points[0].pos, n - 1, breakout, breakout, "breakout")),
                breakout, target,
                "Three tests of a common price zone; the opposite boundary is the confirmation level.",
            )
        )

    def add_shoulders(sequence: Sequence[dict[str, float | int | str]], inverse: bool) -> None:
        points = tuple(map(_point, sequence))
        left, neck1, head, neck2, right = points
        shoulder_level = (left.price + right.price) / 2
        if left.pos < recent_cutoff or right.pos < n - max(85, n // 3):
            return
        if abs(left.price - right.price) > tolerance * 1.25:
            return
        prominence = (shoulder_level - head.price) if inverse else (head.price - shoulder_level)
        if prominence < max(atr * 1.35, abs(latest) * 0.018):
            return
        neckline = PatternLine(neck1.pos, neck2.pos, neck1.price, neck2.price, "breakout")
        breakout = _line_value(neckline, n - 1)
        bias = "bullish" if inverse else "bearish"
        status, target = _status_and_target(bias, latest, breakout, prominence + abs(breakout - shoulder_level))
        name = "Inverse Head and Shoulders" if inverse else "Head and Shoulders Top"
        symmetry = 1 - abs((head.pos - left.pos) - (right.pos - head.pos)) / max(right.pos - left.pos, 1)
        confidence = 66 + 12 * _clamp(symmetry, 0, 1) + 10 * (1 - abs(left.price - right.price) / max(tolerance * 1.25, 1e-9))
        if status == "Confirmed":
            confidence += 5
        detections.append(
            PatternDetection(
                name, "reversal", bias, status, _clamp(confidence, 0, 97), left.pos, right.pos, points,
                (PatternLine(neck1.pos, n - 1, neck1.price, _line_value(neckline, n - 1), "breakout"),),
                breakout, target,
                "Three alternating shoulders with a more extreme head; the neckline controls confirmation.",
            )
        )

    recent = pivots[-18:]
    for index in range(len(recent) - 2):
        sequence = recent[index : index + 3]
        kinds = "".join(str(item["kind"]) for item in sequence)
        if kinds == "LHL":
            add_double(sequence, True)
        elif kinds == "HLH":
            add_double(sequence, False)

    for index in range(len(recent) - 4):
        sequence = recent[index : index + 5]
        kinds = "".join(str(item["kind"]) for item in sequence)
        if kinds == "LHLHL":
            add_triple(sequence, True)
            add_shoulders(sequence, True)
        elif kinds == "HLHLH":
            add_triple(sequence, False)
            add_shoulders(sequence, False)
    return detections


def _envelope_patterns(
    data: pd.DataFrame,
    pivots: list[dict[str, float | int | str]],
    atr: float,
) -> list[PatternDetection]:
    n = len(data)
    if n < 30:
        return []
    latest = float(data["Close"].iloc[-1])
    lookback = min(max(80, int(n * 0.70)), 180)
    start = n - lookback
    highs = [_point(item) for item in pivots if item["kind"] == "H" and int(item["pos"]) >= start]
    lows = [_point(item) for item in pivots if item["kind"] == "L" and int(item["pos"]) >= start]
    if len(highs) < 2 or len(lows) < 2:
        return []
    highs, lows = highs[-5:], lows[-5:]
    upper_slope, upper_intercept, upper_r2 = _regression(highs)
    lower_slope, lower_intercept, lower_r2 = _regression(lows)
    line_start = max(start, min(highs[0].pos, lows[0].pos))
    line_end = n - 1
    upper_start = upper_slope * line_start + upper_intercept
    upper_end = upper_slope * line_end + upper_intercept
    lower_start = lower_slope * line_start + lower_intercept
    lower_end = lower_slope * line_end + lower_intercept
    width_start = upper_start - lower_start
    width_end = upper_end - lower_end
    if width_start <= 0 or width_end <= 0:
        return []

    span = max(line_end - line_start, 1)
    scale = max(abs(latest), 1e-9)
    upper_move = upper_slope * span / scale
    lower_move = lower_slope * span / scale
    flat = max(atr * 1.2 / scale, 0.012)
    converging = width_end < width_start * 0.78
    broadening = width_end > width_start * 1.20
    flat_error = max(atr * 1.8, max(width_start, width_end) * 0.16)
    upper_spread = max(point.price for point in highs) - min(point.price for point in highs)
    lower_spread = max(point.price for point in lows) - min(point.price for point in lows)
    upper_flat = abs(upper_move) <= flat and upper_spread <= flat_error
    lower_flat = abs(lower_move) <= flat and lower_spread <= flat_error
    fit_quality = (upper_r2 + lower_r2) / 2

    name: str | None = None
    bias = "bilateral"
    breakout: float | None = None
    height = max(width_start, width_end)
    rationale = ""
    if upper_flat and lower_move > flat and converging:
        name, bias, breakout = "Ascending Triangle", "bullish", upper_end
        rationale = "Flat resistance and rising lows compress price toward an upside decision level."
    elif lower_flat and upper_move < -flat and converging:
        name, bias, breakout = "Descending Triangle", "bearish", lower_end
        rationale = "Flat support and falling highs compress price toward a downside decision level."
    elif upper_move < -flat and lower_move > flat and converging:
        name, bias = "Symmetrical Triangle", "bilateral"
        if latest > upper_end:
            bias, breakout = "bullish", upper_end
        elif latest < lower_end:
            bias, breakout = "bearish", lower_end
        rationale = "Lower highs and higher lows form a bilateral compression; direction follows the break."
    elif upper_flat and lower_flat and width_end > max(atr * 2.2, scale * 0.025):
        name = "Rectangle"
        if latest > upper_end:
            bias, breakout = "bullish", upper_end
        elif latest < lower_end:
            bias, breakout = "bearish", lower_end
        rationale = "Repeated horizontal support and resistance contain price until either boundary breaks."
    elif upper_move > flat and lower_move > flat and converging and lower_slope > upper_slope:
        name, bias, breakout = "Rising Wedge", "bearish", lower_end
        rationale = "Rising but converging boundaries show waning upside efficiency; support is the bearish trigger."
    elif upper_move < -flat and lower_move < -flat and converging and upper_slope < lower_slope:
        name, bias, breakout = "Falling Wedge", "bullish", upper_end
        rationale = "Falling but converging boundaries show waning downside efficiency; resistance is the bullish trigger."
    elif upper_move > flat and lower_move < -flat and broadening:
        name, bias = "Megaphone / Broadening Formation", "bilateral"
        rationale = "Higher highs and lower lows mark expanding volatility and unstable price discovery."

    if name is None:
        return []
    status, target = _status_and_target(bias, latest, breakout, height)
    compression = abs(width_start - width_end) / max(width_start, width_end)
    confidence = 59 + 18 * fit_quality + 13 * _clamp(compression, 0, 1) + (4 if status == "Confirmed" else 0)
    lines = (
        PatternLine(line_start, line_end, upper_start, upper_end, "resistance"),
        PatternLine(line_start, line_end, lower_start, lower_end, "support"),
    )
    return [
        PatternDetection(
            name, "breakout", bias, status, _clamp(confidence, 0, 95), line_start, line_end,
            tuple(sorted(highs + lows, key=lambda point: point.pos)), lines, breakout, target, rationale,
        )
    ]


def _flag_or_pennant(data: pd.DataFrame, atr: float) -> list[PatternDetection]:
    n = len(data)
    if n < 36:
        return []
    high = data["High"].to_numpy(dtype=float)
    low = data["Low"].to_numpy(dtype=float)
    close = data["Close"].to_numpy(dtype=float)
    latest = float(close[-1])
    best: PatternDetection | None = None

    for consolidation in (12, 16, 20, 25, 30, 36):
        if n < consolidation + 16:
            continue
        start = n - consolidation
        pole_start = max(0, start - max(14, consolidation))
        pole_move = close[start - 1] - close[pole_start]
        threshold = max(atr * 4.0, abs(close[pole_start]) * 0.055)
        if abs(pole_move) < threshold:
            continue
        x = np.arange(start, n, dtype=float)
        local_highs = [
            PatternPoint(pos, float(high[pos]))
            for pos in range(start + 2, n - 2)
            if high[pos] >= np.max(high[pos - 2 : pos + 3])
        ]
        local_lows = [
            PatternPoint(pos, float(low[pos]))
            for pos in range(start + 2, n - 2)
            if low[pos] <= np.min(low[pos - 2 : pos + 3])
        ]
        if len(local_highs) >= 2 and len(local_lows) >= 2:
            upper_slope, upper_intercept, _ = _regression(local_highs)
            lower_slope, lower_intercept, _ = _regression(local_lows)
        else:
            upper_slope, upper_intercept = np.polyfit(x, high[start:], 1)
            lower_slope, lower_intercept = np.polyfit(x, low[start:], 1)
        upper_start = upper_slope * start + upper_intercept
        upper_end = upper_slope * (n - 1) + upper_intercept
        lower_start = lower_slope * start + lower_intercept
        lower_end = lower_slope * (n - 1) + lower_intercept
        width_start = upper_start - lower_start
        width_end = upper_end - lower_end
        if width_start <= 0 or width_end <= 0:
            continue
        consolidation_range = float(np.nanmax(high[start:]) - np.nanmin(low[start:]))
        if consolidation_range > abs(pole_move) * 0.62:
            continue
        span_move_upper = upper_slope * consolidation
        span_move_lower = lower_slope * consolidation
        parallel = abs(span_move_upper - span_move_lower) <= max(atr * 1.2, consolidation_range * 0.30)
        converging = width_end < width_start * 0.72
        pole_up = pole_move > 0

        if converging and span_move_upper < 0 < span_move_lower:
            name = "Bull Pennant" if pole_up else "Bear Pennant"
        elif parallel and pole_up and span_move_upper <= atr * 0.5 and span_move_lower < atr * 0.5:
            name = "Bull Flag"
        elif parallel and not pole_up and span_move_upper > -atr * 0.5 and span_move_lower >= -atr * 0.5:
            name = "Bear Flag"
        else:
            continue

        bias = "bullish" if pole_up else "bearish"
        breakout = upper_end if pole_up else lower_end
        status, target = _status_and_target(bias, latest, breakout, abs(pole_move))
        tightness = 1 - consolidation_range / max(abs(pole_move), 1e-9)
        confidence = 63 + 20 * _clamp(tightness, 0, 1) + (5 if status == "Confirmed" else 0)
        candidate = PatternDetection(
            name, "continuation", bias, status, _clamp(confidence, 0, 96), pole_start, n - 1,
            (PatternPoint(pole_start, float(close[pole_start])), PatternPoint(start - 1, float(close[start - 1]))),
            (
                PatternLine(pole_start, start - 1, float(close[pole_start]), float(close[start - 1]), "pole"),
                PatternLine(start, n - 1, float(upper_start), float(upper_end), "resistance"),
                PatternLine(start, n - 1, float(lower_start), float(lower_end), "support"),
            ),
            float(breakout), target,
            "A strong directional pole is followed by a compact counter-trend consolidation.",
        )
        if best is None or candidate.confidence > best.confidence:
            best = candidate
    return [best] if best is not None else []


def _quadratic_fit(values: np.ndarray) -> tuple[float, float, float, float]:
    x = np.linspace(-1.0, 1.0, len(values))
    a, b, c = np.polyfit(x, values, 2)
    fitted = a * x * x + b * x + c
    total = float(np.sum((values - np.mean(values)) ** 2))
    residual = float(np.sum((values - fitted) ** 2))
    r2 = 1.0 - residual / total if total > 1e-12 else 0.0
    vertex = -b / (2 * a) if abs(a) > 1e-12 else 9.0
    return float(a), float(b), _clamp(r2, 0, 1), float(vertex)


def _rounded_patterns(data: pd.DataFrame, atr: float) -> list[PatternDetection]:
    n = len(data)
    close = data["Close"].to_numpy(dtype=float)
    latest = float(close[-1])
    best_round: PatternDetection | None = None
    best_cup: PatternDetection | None = None
    for window in (50, 65, 80, 100, 110, 130, 140, 160, 180):
        if n < window:
            continue
        start = n - window
        values = close[start:]
        a, _b, r2, vertex = _quadratic_fit(values)
        edge_level = float((values[0] + values[-1]) / 2)
        center = float(values[len(values) // 2])
        depth = abs(edge_level - center)
        if r2 < 0.68 or abs(vertex) > 0.42 or depth < max(atr * 3.2, abs(latest) * 0.045):
            continue
        bottom = a > 0
        bias = "bullish" if bottom else "bearish"
        breakout = max(values[0], values[-1]) if bottom else min(values[0], values[-1])
        status, target = _status_and_target(bias, latest, breakout, depth)
        name = "Rounding Bottom" if bottom else "Rounding Top"
        extremum_pos = start + int(np.argmin(values) if bottom else np.argmax(values))
        candidate = PatternDetection(
            name, "reversal", bias, status, _clamp(55 + 25 * r2 + (4 if status == "Confirmed" else 0), 0, 92),
            start, n - 1,
            (PatternPoint(start, float(values[0])), PatternPoint(extremum_pos, float(close[extremum_pos])), PatternPoint(n - 1, latest)),
            (PatternLine(start, n - 1, float(breakout), float(breakout), "breakout"),),
            float(breakout), target,
            "A smooth quadratic arc suggests a gradual transition between selling and accumulation." if bottom
            else "A smooth quadratic arc suggests a gradual transition between accumulation and distribution.",
        )
        if best_round is None or candidate.confidence > best_round.confidence:
            best_round = candidate

        cup_end_local = int(window * 0.78)
        if cup_end_local < 35 or window - cup_end_local < 8:
            continue
        cup = values[:cup_end_local]
        ca, _cb, cr2, cvertex = _quadratic_fit(cup)
        cup_bottom = ca > 0
        rim = float((cup[0] + cup[-1]) / 2)
        cup_extreme = float(np.min(cup) if cup_bottom else np.max(cup))
        cup_depth = abs(rim - cup_extreme)
        handle = values[cup_end_local:]
        handle_excursion = (rim - float(np.min(handle))) if cup_bottom else (float(np.max(handle)) - rim)
        rim_error = abs(float(cup[0]) - float(cup[-1]))
        if cr2 < 0.66 or abs(cvertex) > 0.48 or cup_depth < max(atr * 3, abs(latest) * 0.04):
            continue
        if rim_error > max(atr * 2, cup_depth * 0.35) or not (0 <= handle_excursion <= cup_depth * 0.48):
            continue
        inverse = not cup_bottom
        name = "Inverse Cup and Handle" if inverse else "Cup and Handle"
        bias = "bearish" if inverse else "bullish"
        breakout = min(cup[0], cup[-1]) if inverse else max(cup[0], cup[-1])
        status, target = _status_and_target(bias, latest, breakout, cup_depth)
        handle_start = start + cup_end_local
        candidate = PatternDetection(
            name, "continuation", bias, status, _clamp(60 + 25 * cr2 + (4 if status == "Confirmed" else 0), 0, 94),
            start, n - 1,
            (
                PatternPoint(start, float(cup[0])),
                PatternPoint(start + int(np.argmin(cup) if cup_bottom else np.argmax(cup)), cup_extreme),
                PatternPoint(handle_start - 1, float(cup[-1])),
                PatternPoint(n - 1, latest),
            ),
            (PatternLine(start, n - 1, float(breakout), float(breakout), "breakout"),),
            float(breakout), target,
            "A rounded base and shallow handle preserve a bullish continuation setup." if not inverse
            else "A rounded top and weak rebound preserve a bearish continuation setup.",
        )
        if best_cup is None or candidate.confidence > best_cup.confidence:
            best_cup = candidate
    return [item for item in (best_cup, best_round) if item is not None]


def _v_patterns(data: pd.DataFrame, atr: float) -> list[PatternDetection]:
    n = len(data)
    if n < 24:
        return []
    close = data["Close"].to_numpy(dtype=float)
    latest = float(close[-1])
    best: PatternDetection | None = None
    scan_start = max(6, n - min(100, int(n * 0.55)))
    for center in range(scan_start, n - 5):
        for arm in (5, 8, 12, 16, 20):
            left, right = center - arm, min(n - 1, center + arm)
            if left < 0 or right - center < max(4, arm // 2):
                continue
            is_bottom = close[center] <= np.min(close[left : right + 1])
            is_top = close[center] >= np.max(close[left : right + 1])
            if not (is_bottom or is_top):
                continue
            left_move = abs(close[center] - close[left])
            right_move = abs(close[right] - close[center])
            move = min(left_move, right_move)
            if move < max(atr * 3.5, abs(latest) * 0.045):
                continue
            symmetry = min(left_move, right_move) / max(left_move, right_move, 1e-9)
            if symmetry < 0.50:
                continue
            bottom = bool(is_bottom)
            name, bias = ("V-Bottom", "bullish") if bottom else ("V-Top", "bearish")
            breakout = close[left]
            status, target = _status_and_target(bias, latest, breakout, max(left_move, right_move))
            confidence = 60 + 24 * symmetry + (4 if status == "Confirmed" else 0)
            candidate = PatternDetection(
                name, "reversal", bias, status, _clamp(confidence, 0, 94), left, right,
                (PatternPoint(left, float(close[left])), PatternPoint(center, float(close[center])), PatternPoint(right, float(close[right]))),
                (), float(breakout), target,
                "A steep move into an extreme is followed by a comparably sharp reversal.",
            )
            if best is None or (candidate.end_pos, candidate.confidence) > (best.end_pos, best.confidence):
                best = candidate
    return [best] if best is not None else []


def _diamond_patterns(data: pd.DataFrame, atr: float) -> list[PatternDetection]:
    n = len(data)
    if n < 48:
        return []
    high = data["High"].to_numpy(dtype=float)
    low = data["Low"].to_numpy(dtype=float)
    close = data["Close"].to_numpy(dtype=float)
    latest = float(close[-1])
    best: PatternDetection | None = None
    for window in (48, 64, 80, 100):
        if n < window + 12:
            continue
        start = n - window
        segments = np.array_split(np.arange(start, n), 4)
        ranges = [float(np.max(high[s]) - np.min(low[s])) for s in segments]
        if not (ranges[1] > ranges[0] * 1.10 and ranges[3] < ranges[2] * 0.88):
            continue
        middle = start + window // 2
        before = _trend_change(close, start, lookback=min(40, start))
        top_pos = start + int(np.argmax(high[start:]))
        bottom_pos = start + int(np.argmin(low[start:]))
        if before > 0.035 and abs(top_pos - middle) <= window * 0.24:
            name, bias, extreme_pos, breakout = "Diamond Top", "bearish", top_pos, float(np.min(low[segments[-1]]))
        elif before < -0.035 and abs(bottom_pos - middle) <= window * 0.24:
            name, bias, extreme_pos, breakout = "Diamond Bottom", "bullish", bottom_pos, float(np.max(high[segments[-1]]))
        else:
            continue
        wide_pos = middle
        upper_wide, lower_wide = float(np.max(high[segments[1]])), float(np.min(low[segments[1]]))
        left_mid = float((high[start] + low[start]) / 2)
        right_mid = float((high[-1] + low[-1]) / 2)
        height = upper_wide - lower_wide
        if height < max(atr * 4, abs(latest) * 0.05):
            continue
        status, target = _status_and_target(bias, latest, breakout, height)
        candidate = PatternDetection(
            name, "reversal", bias, status, 67 + (5 if status == "Confirmed" else 0), start, n - 1,
            (PatternPoint(extreme_pos, float(high[extreme_pos] if bias == "bearish" else low[extreme_pos])),),
            (
                PatternLine(start, wide_pos, left_mid, upper_wide, "resistance"),
                PatternLine(start, wide_pos, left_mid, lower_wide, "support"),
                PatternLine(wide_pos, n - 1, upper_wide, right_mid, "resistance"),
                PatternLine(wide_pos, n - 1, lower_wide, right_mid, "support"),
            ),
            breakout, target,
            "Volatility broadens and then contracts around a major price extreme.",
        )
        if best is None or candidate.confidence > best.confidence:
            best = candidate
    return [best] if best is not None else []


def _overlap(first: PatternDetection, second: PatternDetection) -> float:
    intersection = max(0, min(first.end_pos, second.end_pos) - max(first.start_pos, second.start_pos))
    union = max(first.end_pos, second.end_pos) - min(first.start_pos, second.start_pos)
    return intersection / union if union > 0 else 1.0


def _confirmation_age(data: pd.DataFrame, pattern: PatternDetection) -> int | None:
    """Return bars since the first qualifying break after the pattern completed."""
    if pattern.status != "Confirmed" or pattern.breakout_level is None:
        return None

    closes = data["Close"].to_numpy(dtype=float)
    if not len(closes):
        return None
    start = max(0, min(len(closes) - 1, pattern.end_pos))
    breakout = float(pattern.breakout_level)
    buffer = max(abs(breakout) * 0.0015, 1e-9)
    post_pattern = closes[start:]
    if pattern.bias == "bullish":
        crossed = np.flatnonzero(post_pattern > breakout + buffer)
    elif pattern.bias == "bearish":
        crossed = np.flatnonzero(post_pattern < breakout - buffer)
    else:
        return None
    if not crossed.size:
        return None
    confirmation_pos = start + int(crossed[0])
    return len(closes) - 1 - confirmation_pos


def _is_current_pattern(data: pd.DataFrame, pattern: PatternDetection, atr: float) -> bool:
    """Reject stale or remote structures that no longer describe the active setup."""
    n = len(data)
    if n == 0:
        return False

    end_age = n - 1 - pattern.end_pos
    max_end_age = max(24, min(45, n // 5))
    if end_age > max_end_age:
        return False

    latest = float(data["Close"].iloc[-1])
    if pattern.status == "Confirmed":
        confirmation_age = _confirmation_age(data, pattern)
        max_confirmation_age = max(12, min(30, n // 8))
        return confirmation_age is not None and confirmation_age <= max_confirmation_age

    if pattern.breakout_level is not None and np.isfinite(pattern.breakout_level):
        distance = abs(latest - float(pattern.breakout_level))
        if distance > max(atr * 3.5, abs(float(pattern.breakout_level)) * 0.055):
            return False
    return True


def _rank_and_deduplicate(
    detections: list[PatternDetection],
    max_patterns: int,
    data: pd.DataFrame,
    atr: float,
) -> list[PatternDetection]:
    groups = {
        "Double Bottom": "base", "Triple Bottom": "base", "Inverse Head and Shoulders": "base",
        "Double Top": "top", "Triple Top": "top", "Head and Shoulders Top": "top",
        "V-Bottom": "base", "Diamond Bottom": "base",
        "V-Top": "top", "Diamond Top": "top",
        "Rounding Bottom": "curve-bottom", "Cup and Handle": "curve-bottom",
        "Rounding Top": "curve-top", "Inverse Cup and Handle": "curve-top",
        "Ascending Triangle": "envelope", "Descending Triangle": "envelope", "Symmetrical Triangle": "envelope",
        "Rectangle": "envelope", "Rising Wedge": "envelope", "Falling Wedge": "envelope",
        "Megaphone / Broadening Formation": "envelope",
        "Bull Flag": "pole", "Bear Flag": "pole", "Bull Pennant": "pole", "Bear Pennant": "pole",
    }
    current = [item for item in detections if _is_current_pattern(data, item, atr)]
    envelope = next((item for item in current if groups.get(item.name) == "envelope"), None)
    filtered: list[PatternDetection] = []
    for item in current:
        group = groups.get(item.name)
        envelope_conflict = (
            envelope is not None
            and item is not envelope
            and item.confidence <= envelope.confidence + 16
            and (
                (envelope.name == "Ascending Triangle" and group == "top")
                or (envelope.name == "Descending Triangle" and group == "base")
                or (envelope.name == "Rectangle" and group in {"top", "base"})
            )
        )
        if not envelope_conflict:
            filtered.append(item)
    ordered = sorted(
        (item for item in filtered if item.confidence >= 72),
        key=lambda item: (
            item.confidence
            + min(6.0, 6.0 * (item.end_pos + 1) / max(len(data), 1))
            + (6.0 if groups.get(item.name) == "envelope" else 0.0),
            item.status == "Confirmed",
            item.end_pos,
        ),
        reverse=True,
    )
    selected: list[PatternDetection] = []
    for candidate in ordered:
        opposing_window = max(16, min(30, len(data) // 10))
        duplicate = any(
            candidate.name == existing.name
            or (groups.get(candidate.name) == groups.get(existing.name) and groups.get(candidate.name) is not None)
            or (
                candidate.bias in {"bullish", "bearish"}
                and existing.bias in {"bullish", "bearish"}
                and candidate.bias != existing.bias
                and (
                    _overlap(candidate, existing) >= 0.20
                    or abs(candidate.end_pos - existing.end_pos) <= opposing_window
                )
            )
            or _overlap(candidate, existing) >= 0.72
            for existing in selected
        )
        if duplicate:
            continue
        selected.append(candidate)
        if len(selected) >= max_patterns:
            break
    return selected


def detect_chart_patterns(df: pd.DataFrame, max_patterns: int = 2) -> list[PatternDetection]:
    """Detect and rank the strongest current patterns in the supplied chart window."""
    data = _price_frame(df)
    if len(data) < 18 or max_patterns <= 0:
        return []
    atr = _atr_value(data)
    if not np.isfinite(atr) or atr <= 0:
        atr = max(abs(float(data["Close"].iloc[-1])) * 0.01, 1e-9)
    pivots = find_price_pivots(data)
    detections: list[PatternDetection] = []
    detections.extend(_reversal_patterns(data, pivots, atr))
    detections.extend(_envelope_patterns(data, pivots, atr))
    detections.extend(_flag_or_pennant(data, atr))
    detections.extend(_rounded_patterns(data, atr))
    detections.extend(_v_patterns(data, atr))
    detections.extend(_diamond_patterns(data, atr))
    return _rank_and_deduplicate(
        detections,
        max_patterns=max_patterns,
        data=data,
        atr=atr,
    )
