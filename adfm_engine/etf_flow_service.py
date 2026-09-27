"""Native transport for the full ETF flow-pressure coverage."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from adfm_engine import etf_flow_legacy_math as math
from adfm_engine.serialization import records

PERIOD_DAYS = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "12 Months": 365}


def load_etf_flow(*, period_label="1 Month", session_hour=None):
    now = datetime.now(ZoneInfo("America/New_York"))
    as_of = now.date()
    days = PERIOD_DAYS.get(period_label, max((as_of - as_of.replace(month=1, day=1)).days, 1))
    table = math.build_table(math.etf_tickers, period_label, days, as_of,
                             now.strftime("%Y-%m-%d-%H"))
    flow = f"{period_label} Flow Pressure Proxy"
    return {
        "as_of": now.strftime("%Y-%m-%d %H:%M %Z"),
        "period": period_label,
        "flow_column": flow,
        "return_column": f"{period_label} Return %",
        "coverage": len(math.etf_tickers),
        "rows": records(table),
    }
