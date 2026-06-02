import pandas as pd
import geopandas as gpd
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
import numpy as np

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


def preprocess_taxi_data(df: pd.DataFrame, ca_path: Path = CA_GEOJSON_PATH) -> pd.DataFrame:
    # --- Drop missing timestamps (DST transition artifacts on 2024-11-03) ---
    df = df.dropna(subset=["trip_start_timestamp", "trip_end_timestamp"])

    # --- Fill missing community areas via spatial join ---
    ca_gdf = (
        gpd.read_file(ca_path)[["area_numbe", "geometry"]]
        .rename(columns={"area_numbe": "area_number"})
    )
    ca_gdf["area_number"] = pd.to_numeric(ca_gdf["area_number"], errors="coerce")
    df = _fill_community_area(df, "pickup_centroid_latitude", "pickup_centroid_longitude", "pickup_community_area", ca_gdf)
    df = _fill_community_area(df, "dropoff_centroid_latitude", "dropoff_centroid_longitude", "dropoff_community_area", ca_gdf)

    # --- Drop rows with missing taxi_id ---
    df = df.dropna(subset=["taxi_id"])

    # --- Remove zero-movement trips (same location, zero duration) ---
    zero_trip_mask = (
        (df["trip_seconds"] == 0)
        & (df["trip_end_timestamp"] == df["trip_start_timestamp"])
        & (df["pickup_centroid_latitude"] == df["dropoff_centroid_latitude"])
        & (df["pickup_centroid_longitude"] == df["dropoff_centroid_longitude"])
    )
    df = df[~zero_trip_mask]
    
    # --- Remove zero-miles trips with positive duration (likely GPS errors) ---
    zero_miles_mask = (
        (df["trip_miles"] == 0)
        & (df["trip_seconds"] > 0)
    )
    df = df[~zero_miles_mask]

    # --- Drop implausible outliers (Rule 1) ---
    # Derived features used purely for filtering; inf (residual zero-duration /
    # zero-mile rows) is treated as out-of-range so .between() drops it.
    df = df.copy()
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
    df = df[keep]

    return df.reset_index(drop=True)


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

    # --- Drop redundant columns ---
    # precipitation = rain + snowfall and adds no information; rain and snowfall are
    # kept separately because they have different behavioral effects on taxi demand.
    # latitude/longitude are redundant with the zone identifier.
    drop_cols = [c for c in ["precipitation", "latitude", "longitude"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # --- Remove duplicate (time, zone) rows ---
    df = df.drop_duplicates(subset=["time", "zone"])

    # --- Encode weather_code as ordered categories (WMO standard bins) ---
    # Raw WMO codes are not ordinal (code 95 ≠ "95× worse" than code 1), so feeding
    # them as integers would imply a false ordering to SVM and neural network models.
    if "weather_code" in df.columns:
        df["weather_category"] = df["weather_code"].apply(_categorize_weather_code)
        df = df.drop(columns=["weather_code"])
        df = pd.get_dummies(df, columns=["weather_category"], drop_first=True)

    return df.reset_index(drop=True)


def _categorize_weather_code(code: int) -> str:
    if code <= 3:
        return "clear"
    elif code <= 48:
        return "fog"
    elif code <= 67:
        return "rain"
    elif code <= 77:
        return "snow"
    elif code <= 82:
        return "showers"
    elif code <= 86:
        return "snow_showers"
    else:
        return "storm"


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
