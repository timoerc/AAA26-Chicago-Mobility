import pandas as pd
import geopandas as gpd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CA_GEOJSON_PATH = ROOT_DIR / "data" / "raw" / "community_areas.geojson"


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
        df["trip_miles"] == 0
        & (df["trip_seconds"] > 0)
    )
    df = df[~zero_miles_mask]

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
