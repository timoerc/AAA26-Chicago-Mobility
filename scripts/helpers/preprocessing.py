from typing import Literal

import pandas as pd
import geopandas as gpd
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import holidays

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CA_GEOJSON_PATH = ROOT_DIR / "data" / "raw" / "community_areas.geojson"

# --- Outlier / plausibility cutoffs (Rule 1: drop physically impossible records) ---
# Thresholds are derived and justified in notebooks/01_data_preparation.ipynb
# ("Outlier Diagnostics"). They target data errors, not genuine demand extremes.
SPEED_MIN_MPH     = 2.0        # below tariff-plausible avg speed -> stuck meter / under-recorded distance
SPEED_MAX_MPH     = 70.0       # above legal sustained speed (p99.9 = 54.5) -> GPS teleport / meter error
TRIP_SECONDS_MAX  = 3 * 3600   # 3 h -> meter left running (p99.9 = 7,624 s)
TRIP_MILES_MAX    = 100.0      # exceeds Chicago metro geography (p99.9 = 32.4 mi)
FARE_PER_MILE_MIN = 2.0        # Chicago $2.25/mi tariff floor (empirical p1 = 2.43)
FARE_PER_MILE_MAX = 100.0      # short-trip base-fare inflation is legit only below this (p99 = 117.5)


def preprocess_taxi_data(
    df: pd.DataFrame,
    ca_path: Path = CA_GEOJSON_PATH,
    mode: Literal["trip", "demand"] = "trip",
) -> pd.DataFrame:
    """Clean the raw taxi DataFrame.

    Two cleaning tiers are available via the ``mode`` parameter:

    ``mode='demand'``
        Minimal cleaning for building demand-forecasting targets (pickup counts
        per time bucket and spatial unit).  Only rows with a missing start
        timestamp are dropped, and community areas are imputed from centroid
        coordinates.  Every row with a valid pickup time and location is kept —
        including trips where other metadata (fare, distance, speed, taxi_id) is
        absent or implausible.  A real passenger hailing a cab is a real demand
        signal regardless of how the trip was recorded.

    ``mode='trip'``  *(default, backwards-compatible)*
        Full cleaning pipeline for trip-level analyses (distance, fare, speed,
        duration).  On top of the demand-mode steps, also drops: missing
        taxi_id, ghost trips (zero duration + same location), zero-mile GPS
        errors, and Rule-1 implausible outliers (fare, speed, distance, and
        fare-per-mile bounds — thresholds defined in this module and justified
        in ``notebooks/01a_taxi_data_preparation.ipynb``).
    """
    df = df.copy()

    # --- Shared by both modes ---
    # Drop DST transition artifacts (2024-11-03 fall-back hour → NaT timestamps)
    df = df.dropna(subset=["trip_start_timestamp", "trip_end_timestamp"])

    # Impute missing community areas from centroid coordinates via spatial join
    ca_gdf = (
        gpd.read_file(ca_path)[["area_numbe", "geometry"]]
        .rename(columns={"area_numbe": "area_number"})
    )
    ca_gdf["area_number"] = pd.to_numeric(ca_gdf["area_number"], errors="coerce")
    df = _fill_community_area(df, "pickup_centroid_latitude", "pickup_centroid_longitude", "pickup_community_area", ca_gdf)
    df = _fill_community_area(df, "dropoff_centroid_latitude", "dropoff_centroid_longitude", "dropoff_community_area", ca_gdf)

    if mode == "demand":
        return df.reset_index(drop=True)

    # --- Trip mode only: remove rows where trip metadata is invalid or implausible ---

    # Drop rows with missing taxi_id (can't link to a vehicle)
    df = df.dropna(subset=["taxi_id"])

    # Remove zero-movement trips (same location, zero duration) — no mobility signal
    zero_trip_mask = (
        (df["trip_seconds"] == 0)
        & (df["trip_end_timestamp"] == df["trip_start_timestamp"])
        & (df["pickup_centroid_latitude"] == df["dropoff_centroid_latitude"])
        & (df["pickup_centroid_longitude"] == df["dropoff_centroid_longitude"])
    )
    df = df[~zero_trip_mask]

    # Remove zero-miles trips with positive duration (likely GPS errors)
    zero_miles_mask = (
        (df["trip_miles"] == 0)
        & (df["trip_seconds"] > 0)
    )
    df = df[~zero_miles_mask]

    # Rule 1: drop physically impossible records
    # Derived features used purely for filtering; inf (residual zero-duration /
    # zero-mile rows) is treated as out-of-range so .between() drops it.
    df["speed_mph"] = (
        df["trip_miles"] / (df["trip_seconds"] / 3600)
    ).replace([np.inf, -np.inf], np.nan)
    fare_per_mile = (
        df["fare"] / df["trip_miles"]
    ).replace([np.inf, -np.inf], np.nan)

    keep = (
        (df["fare"] > 0)
        & (df["trip_miles"] <= TRIP_MILES_MAX)
        & (df["trip_seconds"] <= TRIP_SECONDS_MAX)
        & df["speed_mph"].between(SPEED_MIN_MPH, SPEED_MAX_MPH)
        & fare_per_mile.between(FARE_PER_MILE_MIN, FARE_PER_MILE_MAX)
    )
    df = df[keep].drop(columns=["speed_mph"])

    return df.reset_index(drop=True)


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar features from the (tz-aware, local) trip start time.

    These feed the demand-forecasting models (trips per spatial unit and time
    bucket). Raw integer/categorical fields are kept; cyclical (sin/cos)
    encodings are intentionally left to the modeling step.
    """
    df = df.copy()
    ts = df["trip_start_timestamp"].dt

    df["date"]        = ts.date                                # calendar day
    df["hour"]        = ts.hour.astype("Int64")                                # 0–23
    df["day_of_week"] = ts.dayofweek.astype("Int64")                         # 0 = Monday … 6 = Sunday
    df["is_weekend"]  = ts.dayofweek.isin([5, 6])              # Sat/Sun flag
    df["week"]        = ts.isocalendar().week.astype("Int64")  # ISO week number
    df["month"]       = ts.month.astype("Int64")                               # 1–12

    # US (Illinois) public-holiday flag(demand differs strongly on holidays)
    years = range(int(ts.year.min()), int(ts.year.max()) + 1)
    us_il_holidays = holidays.US(subdiv="IL", years=years)
    df["is_holiday"] = df["date"].isin(set(us_il_holidays))

    return df


def _fill_community_area(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    ca_col: str,
    ca_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    null_mask = df[ca_col].isna()
    if not null_mask.any():
        return df
    pts = gpd.GeoDataFrame(
        index=df.index[null_mask],
        geometry=gpd.points_from_xy(df.loc[null_mask, lon_col], df.loc[null_mask, lat_col]),
        crs="EPSG:4326",
    )
    joined = pts.sjoin(ca_gdf, how="left", predicate="within")
    df.loc[null_mask, ca_col] = joined["area_number"].values
    return df

def preprocess_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["latitude", "longitude", "weather_code"])
    return df.reset_index(drop=True)


def merge_weather(
    trips: pd.DataFrame,
    weather: pd.DataFrame,
    weather_zones: dict,
) -> pd.DataFrame:
    """Assign each trip its nearest weather zone then join hourly weather.

    Parameters
    ----------
    trips:         preprocessed taxi DataFrame with pickup lat/lon columns
    weather:       cleaned weather DataFrame with columns [time, zone, ...]
    weather_zones: dict mapping zone name → (lat, lon), e.g. from k-means centers
    """
    zone_names = list(weather_zones.keys())
    zone_coords = np.array(list(weather_zones.values()))  # shape (n_zones, 2)
    pickup_coords = trips[["pickup_centroid_latitude", "pickup_centroid_longitude"]].values

    # For a small number of zones, argmin over per-zone distances is sufficient
    sq_dists = np.array([
        np.sum((pickup_coords - zone) ** 2, axis=1) for zone in zone_coords
    ])  # shape (n_zones, n_trips)
    idx = sq_dists.argmin(axis=0)
    trips = trips.copy()
    trips["weather_zone"] = np.array(zone_names)[idx]
    trips["weather_hour"] = trips["trip_start_timestamp"].dt.floor("h")

    merged = trips.merge(
        weather,
        left_on=["weather_zone", "weather_hour"],
        right_on=["zone", "time"],
        how="left",
    ).drop(columns=["time", "weather_zone", "weather_hour"])

    return merged


def evaluate_weather_zones(
    coords: np.ndarray,
    k_range: range = range(2, 10),
    silhouette_sample: int = 10_000,
) -> pd.DataFrame:
    """Compute elbow (inertia) and silhouette score for each k.

    Returns a DataFrame indexed by k with columns [inertia, silhouette].
    Pass this to a notebook for plotting — no visualisation happens here.
    """
    from sklearn.metrics import silhouette_score

    rows = []
    for k in k_range:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(coords)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(
                coords, labels, sample_size=silhouette_sample, random_state=42
            ),
        })
    return pd.DataFrame(rows).set_index("k")


def get_weather_zone_centers(coords: np.ndarray, n_clusters: int) -> dict:
    """Run k-means on trip pickup coordinates and return cluster centers.

    Parameters
    ----------
    coords:      array of shape (n_trips, 2) with [lat, lon] pickup coordinates
    n_clusters:  number of weather zones (chosen from evaluate_weather_zones)

    Returns
    -------
    dict mapping zone index (int) → (lat, lon), ready to pass into merge_weather()
    """
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(coords)
    return {
        i: (float(lat), float(lon))
        for i, (lat, lon) in enumerate(km.cluster_centers_)
    }
