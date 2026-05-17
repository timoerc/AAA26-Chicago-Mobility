import sys
import pandas as pd
import geopandas as gpd
from pathlib import Path

import osmnx as ox

from scripts.helpers.preprocessing import preprocess_taxi_data

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_TAXI_DATA_PATH = _ROOT / "data" / "raw" / "chicago_taxi_trips_2024.csv"
_TS_COLS = ["trip_start_timestamp", "trip_end_timestamp"]


def load_taxi_data(preprocessed: bool) -> pd.DataFrame:
    df = _load_raw_taxi_data()
    if preprocessed:
        df = preprocess_taxi_data(df)
    return df


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
    for col in _TS_COLS:
        ts = pd.to_datetime(df[col], errors="coerce")
        try:
            df[col] = ts.dt.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="NaT")
        except Exception:
            # Fallback for environments without IANA timezone data (e.g. Windows without tzdata)
            df[col] = ts.dt.tz_localize("-06:00")
    return df
