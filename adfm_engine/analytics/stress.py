from __future__ import annotations
from datetime import date,timedelta
from typing import Dict,List
import numpy as np
import pandas as pd
from adfm_engine.palette import PASTEL
TITLE = "Market Stress Composite"
SPX = "^GSPC"
IXIC = "^IXIC"
FOREIGN_EQUITIES: Dict[str, str] = {
    "^N225": "Nikkei 225",
    "^GDAXI": "DAX",
    "^FTSE": "FTSE 100",
    "^FCHI": "CAC 40",
    "^STOXX50E": "Euro Stoxx 50",
    "^HSI": "Hang Seng",
    "^AXJO": "ASX 200",
}
CARRY_FX: Dict[str, str] = {
    "AUDJPY=X": "AUD/JPY",
    "NZDJPY=X": "NZD/JPY",
}
HAVEN_FX: Dict[str, str] = {
    "JPY=X": "USD/JPY",
    "CHF=X": "USD/CHF",
}
FOREIGN_BONDS: Dict[str, str] = {
    "IGLT.L": "UK Gilts ETF",
    "IEGA.AS": "Euro Govt Bonds ETF",
    "2510.T": "Japan Govt Bonds ETF",
    "IGB.AX": "Australia Govt Bonds ETF",
}
ALL_TICKERS = sorted(
    set(
        [SPX, IXIC]
        + list(FOREIGN_EQUITIES)
        + list(CARRY_FX)
        + list(HAVEN_FX)
        + list(FOREIGN_BONDS)
    )
)
DEFAULT_Z_YEARS = 3
DEFAULT_SMOOTH_DAYS = 10
LEAD_HORIZON = 21
EARLY_STAGE_DD63 = -0.07
WATCH_RISK = 0.90
WATCH_DISLOCATION = 1.10
HEDGE_RISK = 1.25
HEDGE_DISLOCATION = 1.50
FRACTURE_LEVEL = 1.50
INDEX_COLOR = "#202124"
RISK_COLOR = PASTEL["rose"]
DISLOCATION_COLOR = PASTEL["lavender"]
WATCH_COLOR = PASTEL["amber"]
THRESHOLD_COLOR = "#B7B7B7"
def robust_z(s: pd.Series, window: int, min_periods: int = 126) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    minp = min(min_periods, max(40, window // 4))
    mean = s.rolling(window, min_periods=minp).mean()
    std = s.rolling(window, min_periods=minp).std(ddof=0)
    return ((s - mean) / std.replace(0, np.nan)).clip(-5, 5)

def pct_return(px: pd.Series, days: int) -> pd.Series:
    return px.pct_change(days, fill_method=None)

def safe_mean(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    return frame.mean(axis=1, skipna=True)

def available_cols(px: pd.DataFrame, universe: Dict[str, str]) -> List[str]:
    return [t for t in universe if t in px.columns and px[t].notna().sum() >= 80]

def latest_valid(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else np.nan

def fmt_score(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x:+.2f}"

def fmt_pct(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x * 100:+.2f}%"

def future_min(px: pd.Series, horizon: int) -> pd.Series:
    forward = pd.concat([px.shift(-i) for i in range(1, horizon + 1)], axis=1)
    return forward.min(axis=1, skipna=False)

def classify_market(direction_z: float, shock_z: float) -> str:
    direction = 0.0 if pd.isna(direction_z) else direction_z
    shock = 0.0 if pd.isna(shock_z) else shock_z
    if direction >= 1.50 or shock >= 2.00:
        return "High stress"
    if direction >= 0.90 or shock >= 1.20:
        return "Watch"
    return "Normal"

def regime_label(risk: float, dislocation: float) -> str:
    if pd.isna(risk) or pd.isna(dislocation):
        return "Insufficient data"
    if risk >= FRACTURE_LEVEL and dislocation >= FRACTURE_LEVEL:
        return "Global fracture"
    if risk >= HEDGE_RISK or dislocation >= HEDGE_DISLOCATION:
        return "Hedge pressure"
    if risk >= WATCH_RISK or dislocation >= WATCH_DISLOCATION:
        return "Watch"
    return "Neutral"

def action_label(risk: float, dislocation: float, us_dd63: float) -> str:
    if pd.isna(risk) or pd.isna(dislocation) or pd.isna(us_dd63):
        return "Insufficient data"
    if us_dd63 <= -0.10:
        return "Late: stress already in U.S. tape"
    if risk >= FRACTURE_LEVEL and dislocation >= FRACTURE_LEVEL:
        return "High alert: hedge"
    if (
        risk >= HEDGE_RISK
        or dislocation >= HEDGE_DISLOCATION
        or (risk >= WATCH_RISK and dislocation >= WATCH_DISLOCATION)
    ):
        return "Add protection"
    if risk >= WATCH_RISK or dislocation >= WATCH_DISLOCATION:
        return "Watch / prepare"
    return "No hedge signal"

def corr_with_future_drawdown(signal: pd.Series, target: pd.Series) -> float:
    fwd_dd = future_min(target, LEAD_HORIZON) / target - 1.0
    aligned = pd.concat([signal, fwd_dd], axis=1).dropna()
    if len(aligned) < 100:
        return np.nan
    return float(aligned.iloc[:, 0].corr(-aligned.iloc[:, 1]))

def choose_target(target_mode,px,risk_score,dislocation_score) -> tuple[str, str]:
    if target_mode == "S&P 500":
        return SPX, "S&P 500"
    if target_mode == "Nasdaq Composite":
        return (IXIC, "Nasdaq Composite") if IXIC in px.columns else (SPX, "S&P 500")

    combined = pd.concat(
        [risk_score.rename("Risk-Off"), dislocation_score.rename("Dislocation")],
        axis=1,
    ).mean(axis=1)

    candidates = [(SPX, "S&P 500")]
    if IXIC in px.columns:
        candidates.append((IXIC, "Nasdaq Composite"))

    best = candidates[0]
    best_metric = -np.inf
    for ticker, label in candidates:
        relationship = corr_with_future_drawdown(combined, px[ticker])
        move_21 = abs(latest_valid(pct_return(px[ticker], 21)))
        metric = 0.85 * (0.0 if pd.isna(relationship) else relationship) + 0.15 * (
            0.0 if pd.isna(move_21) else move_21
        )
        if metric > best_metric:
            best_metric = metric
            best = (ticker, label)
    return best

def compute_stress(px,z_window_years=3,smooth_days=10,target_mode="Auto"):
    warnings=[]
    calendar = px[SPX].dropna().index
    px = px.reindex(calendar).ffill(limit=2)
    eq_cols = available_cols(px, FOREIGN_EQUITIES)
    carry_cols = available_cols(px, CARRY_FX)
    haven_cols = available_cols(px, HAVEN_FX)
    bond_cols = available_cols(px, FOREIGN_BONDS)
    if len(eq_cols) < 3 or len(carry_cols) < 1:
        warnings.append('Some foreign-market series are unavailable today. Scores are reweighted across available inputs.')
    z_window = int(252 * z_window_years)
    eq_r21 = pd.DataFrame({t: pct_return(px[t], 21) for t in eq_cols})
    eq_r63 = pd.DataFrame({t: pct_return(px[t], 63) for t in eq_cols})
    eq_weak_21 = safe_mean(pd.DataFrame({t: -robust_z(eq_r21[t], z_window) for t in eq_cols}))
    eq_weak_63 = safe_mean(pd.DataFrame({t: -robust_z(eq_r63[t], z_window) for t in eq_cols}))
    breadth_neg21 = (eq_r21 < 0).mean(axis=1) if not eq_r21.empty else pd.Series(index=calendar, dtype=float)
    below_ma = pd.DataFrame({t: (px[t] < px[t].rolling(100, min_periods=60).mean()).astype(float) for t in eq_cols})
    breadth_ma = below_ma.mean(axis=1) if not below_ma.empty else pd.Series(index=calendar, dtype=float)
    breadth_z = robust_z(0.5 * breadth_neg21 + 0.5 * breadth_ma, z_window)
    spx_r21 = pct_return(px[SPX], 21)
    relative = pd.DataFrame({t: eq_r21[t] - spx_r21 for t in eq_cols})
    relative_weak = safe_mean(pd.DataFrame({t: -robust_z(relative[t], z_window) for t in eq_cols}))
    carry_5 = pd.DataFrame({t: -robust_z(pct_return(px[t], 5), z_window) for t in carry_cols})
    carry_21 = pd.DataFrame({t: -robust_z(pct_return(px[t], 21), z_window) for t in carry_cols})
    carry_stress = 0.35 * safe_mean(carry_5) + 0.65 * safe_mean(carry_21)
    haven_5 = pd.DataFrame({t: -robust_z(pct_return(px[t], 5), z_window) for t in haven_cols})
    haven_21 = pd.DataFrame({t: -robust_z(pct_return(px[t], 21), z_window) for t in haven_cols})
    haven_stress = 0.35 * safe_mean(haven_5) + 0.65 * safe_mean(haven_21)
    risk_components = pd.DataFrame({'Foreign equity weakness': 0.25 * eq_weak_21 + 0.1 * eq_weak_63, 'Foreign breadth': 0.2 * breadth_z, 'Foreign vs U.S.': 0.15 * relative_weak, 'Carry unwind': 0.2 * carry_stress, 'Haven FX': 0.1 * haven_stress})
    risk_raw = risk_components.sum(axis=1, min_count=2)
    risk_score = robust_z(risk_raw, z_window).ewm(span=smooth_days, adjust=False, min_periods=1).mean()
    eq_shock = safe_mean(pd.DataFrame({t: robust_z(pct_return(px[t], 5), z_window).abs() for t in eq_cols}))
    fx_all = carry_cols + haven_cols
    fx_shock = safe_mean(pd.DataFrame({t: robust_z(pct_return(px[t], 5), z_window).abs() for t in fx_all}))
    bond_shock = safe_mean(pd.DataFrame({t: robust_z(pct_return(px[t], 5), z_window).abs() for t in bond_cols}))
    eq_dispersion = eq_r21.std(axis=1, skipna=True) if not eq_r21.empty else pd.Series(index=calendar, dtype=float)
    dispersion_z = robust_z(eq_dispersion, z_window)
    dislocation_components = pd.DataFrame({'Foreign bond shock': 0.4 * bond_shock, 'FX shock': 0.25 * fx_shock, 'Foreign equity shock': 0.2 * eq_shock, 'Cross-country dispersion': 0.15 * dispersion_z})
    dislocation_raw = dislocation_components.sum(axis=1, min_count=2)
    dislocation_score = robust_z(dislocation_raw, z_window).ewm(span=smooth_days, adjust=False, min_periods=1).mean()
    target_ticker, target_label = choose_target(target_mode,px,risk_score,dislocation_score)
    target_px = px[target_ticker].dropna()
    risk_now = latest_valid(risk_score)
    dislocation_now = latest_valid(dislocation_score)
    regime = regime_label(risk_now, dislocation_now)
    us_high63 = target_px.rolling(63, min_periods=20).max()
    us_dd63 = target_px / us_high63 - 1.0
    us_dd63_now = latest_valid(us_dd63)
    action = action_label(risk_now, dislocation_now, us_dd63_now)
    watch_signal = (((risk_score >= WATCH_RISK) | (dislocation_score >= WATCH_DISLOCATION)) & (us_dd63 > EARLY_STAGE_DD63)).fillna(False)
    watch_onset = watch_signal & ~watch_signal.shift(1, fill_value=False)
    onset_dates = watch_onset[watch_onset].index
    signal_age = 'No active watch'
    if watch_signal.iloc[-1] and len(onset_dates):
        onset_pos = int(calendar.get_indexer([onset_dates[-1]])[0])
        current_pos = int(calendar.get_indexer([calendar[-1]])[0])
        signal_age = f'{current_pos - onset_pos} sessions'
    return {key:value for key,value in locals().items() if key not in ("key","value")}

def market_moves(px,eq_cols,carry_cols,haven_cols,bond_cols,z_window):
    rows = []

    for ticker in eq_cols:
        direction_z = latest_valid(-robust_z(pct_return(px[ticker], 21), z_window))
        shock_z = abs(latest_valid(robust_z(pct_return(px[ticker], 5), z_window)))
        rows.append(
            {
                "Bucket": "Foreign equities",
                "Market": FOREIGN_EQUITIES[ticker],
                "Ticker": ticker,
                "5D": latest_valid(pct_return(px[ticker], 5)),
                "21D": latest_valid(pct_return(px[ticker], 21)),
                "63D": latest_valid(pct_return(px[ticker], 63)),
                "Risk-Off Z": direction_z,
                "Shock Z": shock_z,
                "Status": classify_market(direction_z, shock_z),
            }
        )

    for ticker in carry_cols:
        direction_z = latest_valid(-robust_z(pct_return(px[ticker], 21), z_window))
        shock_z = abs(latest_valid(robust_z(pct_return(px[ticker], 5), z_window)))
        rows.append(
            {
                "Bucket": "Carry FX",
                "Market": CARRY_FX[ticker],
                "Ticker": ticker,
                "5D": latest_valid(pct_return(px[ticker], 5)),
                "21D": latest_valid(pct_return(px[ticker], 21)),
                "63D": latest_valid(pct_return(px[ticker], 63)),
                "Risk-Off Z": direction_z,
                "Shock Z": shock_z,
                "Status": classify_market(direction_z, shock_z),
            }
        )

    for ticker in haven_cols:
        direction_z = latest_valid(-robust_z(pct_return(px[ticker], 21), z_window))
        shock_z = abs(latest_valid(robust_z(pct_return(px[ticker], 5), z_window)))
        rows.append(
            {
                "Bucket": "Haven FX",
                "Market": HAVEN_FX[ticker],
                "Ticker": ticker,
                "5D": latest_valid(pct_return(px[ticker], 5)),
                "21D": latest_valid(pct_return(px[ticker], 21)),
                "63D": latest_valid(pct_return(px[ticker], 63)),
                "Risk-Off Z": direction_z,
                "Shock Z": shock_z,
                "Status": classify_market(direction_z, shock_z),
            }
        )

    for ticker in bond_cols:
        shock_z = abs(latest_valid(robust_z(pct_return(px[ticker], 5), z_window)))
        rows.append(
            {
                "Bucket": "Foreign bonds",
                "Market": FOREIGN_BONDS[ticker],
                "Ticker": ticker,
                "5D": latest_valid(pct_return(px[ticker], 5)),
                "21D": latest_valid(pct_return(px[ticker], 21)),
                "63D": latest_valid(pct_return(px[ticker], 63)),
                "Risk-Off Z": np.nan,
                "Shock Z": shock_z,
                "Status": classify_market(np.nan, shock_z),
            }
        )

    moves = pd.DataFrame(rows)
    if not moves.empty:
        status_rank = {"High stress": 2, "Watch": 1, "Normal": 0}
        moves["_rank"] = moves["Status"].map(status_rank).fillna(0)
        moves["_stress"] = moves[["Risk-Off Z", "Shock Z"]].max(axis=1, skipna=True)
        moves = moves.sort_values(["_rank", "_stress"], ascending=[False, False]).drop(
            columns=["_rank", "_stress"]
        )

    return moves
