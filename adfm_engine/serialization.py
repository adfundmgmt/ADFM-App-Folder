"""Stable JSON transport for pandas tables and Plotly figures."""
import json

import pandas as pd
from plotly.utils import PlotlyJSONEncoder


def records(frame: pd.DataFrame) -> list[dict]:
    # pandas maps NaN, NaT, and infinity to JSON null and preserves booleans.
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=15))


def figure_json(figure) -> dict:
    # Keep Plotly's typed-array encoding intact. The shipped JS matches Python's
    # Plotly distribution; no indicator or chart transformation runs in JS.
    return json.loads(json.dumps(figure.to_plotly_json(), cls=PlotlyJSONEncoder))
