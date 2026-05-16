import sys
import pandas as pd
from pathlib import Path

from scripts.helpers.preprocessing import preprocess_taxi_data

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_TAXI_DATA_PATH = _ROOT / "data" / "raw" / "chicago_taxi_trips_2024.csv"
_TS_COLS = ["trip_start_timestamp", "trip_end_timestamp"]


def load_taxi_data(preprocessed: bool = True) -> pd.DataFrame:
    df = _load_raw_taxi_data()
    if preprocessed:
        df = preprocess_taxi_data(df)
    return df


def _load_raw_taxi_data(path: Path = _TAXI_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in _TS_COLS:
        ts = pd.to_datetime(df[col], errors="coerce")
        try:
            df[col] = ts.dt.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="NaT")
        except Exception:
            # Fallback for environments without IANA timezone data (e.g. Windows without tzdata)
            df[col] = ts.dt.tz_localize("-06:00")
    return df
