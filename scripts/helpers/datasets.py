import sys
from typing import Literal

import pandas as pd
import geopandas as gpd
from pathlib import Path
import json

import osmnx as ox

from scripts.helpers.preprocessing import preprocess_taxi_data, preprocess_weather_data, merge_weather, _add_temporal_features

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_TAXI_DATA_PATH = _ROOT / "data" / "raw_parts" / "chicago_taxi_trips_2024.csv"
_TS_COLS = ["trip_start_timestamp", "trip_end_timestamp"]
_WEATHER_DATA_PATH = _ROOT / "data" / "raw" / "chicago_weather_hourly.csv"
_WEATHER_ZONES_PATH = _ROOT / "data" / "raw" / "weather_zones.json"

def load_taxi_data(
    preprocessed: bool = True,
    mode: Literal["trip", "demand"] = "trip",
) -> pd.DataFrame:
    """Load the raw Chicago taxi CSV, add temporal features, and optionally clean.

    Parameters
    ----------
    preprocessed:
        If True, apply ``preprocess_taxi_data`` according to ``mode``.
        If False, return raw data with temporal features only; ``mode`` is ignored.
    mode:
        ``'trip'``   – full cleaning pipeline; use for trip-level analyses
                       (fare, distance, speed, duration). Default.
        ``'demand'`` – minimal cleaning for demand-forecasting targets; keeps
                       every row with a valid pickup timestamp and location,
                       including trips with faulty trip metadata.
        Ignored when ``preprocessed=False``.
    """
    df = _load_raw_taxi_data()
    df = _add_temporal_features(df)
    if preprocessed:
        df = preprocess_taxi_data(df, mode=mode)
    return df

def load_weather_data(preprocessed: bool) -> pd.DataFrame:
    df = _load_raw_weather_data()
    if preprocessed:
        df = preprocess_weather_data(df)
    return df

def load_merged_data() -> pd.DataFrame:
    with open(_WEATHER_ZONES_PATH) as f:
        weather_zones = {int(k): v for k, v in json.load(f).items()}
    trips = load_taxi_data(preprocessed=True)
    weather = load_weather_data(preprocessed=True)
    return merge_weather(trips, weather, weather_zones)

def load_poi_data() -> gpd.GeoDataFrame:
    """Fetch Chicago POIs from OpenStreetMap and return a categorized GeoDataFrame.

    Columns: geometry (Point), poi_type (str), category (str).
    Rows not matching any category are dropped.
    """
    tag_cols = {"amenity": True, "shop": True, "tourism": True, "leisure": True}
    parts = []
    for tag_col, tag_val in tag_cols.items():
        gdf = ox.features_from_place("Chicago, Illinois", tags={tag_col: tag_val})
        cols = [c for c in [tag_col, "geometry"] if c in gdf.columns]
        slim = gdf[cols].copy()
        slim["poi_type"] = slim[tag_col]
        parts.append(slim.drop(columns=[tag_col]))

    pois = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")

    pois["geometry"] = pois["geometry"].apply(
        lambda g: g.centroid if g.geom_type != "Point" else g
    )
    pois = pois[pois["geometry"].geom_type == "Point"].reset_index(drop=True)

    _POI_CATEGORY_MAP = {
        "taxi": "transport", "bus_station": "transport", "bus_stop": "transport",
        "train_station": "transport", "subway_entrance": "transport",
        "ferry_terminal": "transport", "airport": "transport",
        "restaurant": "food_nightlife", "bar": "food_nightlife", "cafe": "food_nightlife",
        "fast_food": "food_nightlife", "pub": "food_nightlife", "nightclub": "food_nightlife",
        "food_court": "food_nightlife", "biergarten": "food_nightlife",
        "hospital": "healthcare", "clinic": "healthcare", "pharmacy": "healthcare",
        "doctors": "healthcare",
        "university": "education", "school": "education", "college": "education",
        "library": "education",
        "theatre": "entertainment", "cinema": "entertainment", "stadium": "entertainment",
        "museum": "entertainment", "arts_centre": "entertainment", "casino": "entertainment",
        "attraction": "entertainment", "theme_park": "entertainment",
        "hotel": "accommodation", "hostel": "accommodation", "motel": "accommodation",
        "guest_house": "accommodation",
        "mall": "shopping", "supermarket": "shopping", "department_store": "shopping",
        "convenience": "shopping", "marketplace": "shopping",
        "office": "office",
    }
    pois["category"] = pois["poi_type"].map(_POI_CATEGORY_MAP)
    pois = pois.dropna(subset=["category"]).reset_index(drop=True)
    return pois


def _load_raw_taxi_data(path: Path = _TAXI_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["pickup_census_tract", "dropoff_census_tract"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64").astype("string")
    for col in _TS_COLS:
        ts = pd.to_datetime(df[col], errors="coerce")
        try:
            df[col] = ts.dt.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="NaT")
        except Exception:
            # Fallback for environments without IANA timezone data (e.g. Windows without tzdata)
            df[col] = ts.dt.tz_localize("-06:00")
    return df


def _load_raw_weather_data(path: Path = _WEATHER_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["time"]:
        ts = pd.to_datetime(df[col], errors="coerce")
        try:
            df[col] = ts.dt.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="NaT")
        except Exception:
            # Fallback for environments without IANA timezone data (e.g. Windows without tzdata)
            df[col] = ts.dt.tz_localize("-06:00")
    return df