"""Public API for the ADFM Sector Breadth and Rotation page."""

from adfm_core.sector_rotation_catalog import (
    COUNTRY_ROWS, LABEL_MODES, STATE_STREET_SECTOR_ETFS, STOCK_BASKETS, UNIVERSE_SCOPES,
    build_catalog, select_catalog,
)
from adfm_core.sector_rotation_data import (
    download_prices, drop_incomplete_us_session, fetch_state_street_holdings,
    parse_state_street_holdings_bytes, parse_state_street_holdings_frame,
    price_diagnostics, required_tickers,
)
from adfm_core.sector_rotation_analytics import (
    RotationWindow, adaptive_axis_range, attach_breadth, build_asset_levels,
    build_breadth_member_map, classify_quadrant, compute_breadth,
    compute_equal_weight_basket, compute_snapshot, confirmed_state_series,
    latest_state_metrics, movement_metrics, relative_series, trail_for_selected,
)

__all__ = [name for name in globals() if not name.startswith("_")]
