"""Shared ADFM application primitives.

Pages may adopt these modules incrementally.  The package intentionally has
no side effects at import time so legacy pages remain unchanged until they are
explicitly migrated.
"""

from .data_integrity import DataIntegrityPolicy, DataQualityReport
from .market_data import MarketDataConfig
from .ui import PageHeader

__all__ = ["DataIntegrityPolicy", "DataQualityReport", "MarketDataConfig", "PageHeader"]
