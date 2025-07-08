import streamlit as st
import pandas as pd
import io, requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

@st.cache_data(show_spinner=False, ttl=6*60*60)
def load_series(series_id: str, start: str = "1990-01-01") -> pd.Series:
    def fetch():
        r = requests.get(FRED_CSV.format(sid=series_id), timeout=10)
        # Check if CSV and a valid 200 OK response
        if "text/csv" not in r.headers.get("Content-Type", "") or r.status_code != 200:
            st.error(
                f"FRED error: {series_id} - "
                f"Status {r.status_code}, content type: {r.headers.get('Content-Type','?')}\n"
                f"First 200 chars: {r.text[:200]!r}")
            return None
        return pd.read_csv(io.StringIO(r.text))
    try:
        df = fetch()
        if df is None:
            return pd.Series(dtype=float)
    except Exception as e:
        st.error(f"Error fetching {series_id} from FRED: {e}")
        return pd.Series(dtype=float)
    if "DATE" not in df or series_id not in df:
        st.error(f"Could not load {series_id} from FRED: column missing")
        return pd.Series(dtype=float)
    df["DATE"] = pd.to_datetime(df["DATE"])
    s = (df.set_index("DATE")[series_id]
           .replace({".": None})
           .astype(float)
           .asfreq("MS")
           .ffill())
    return s.loc[start:]
