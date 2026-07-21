"""
Demo Dataset Generator
=======================
Generates a small, synthetic taxi-trip + weather dataset with the same
column schema as the real Chicago data, so the notebooks can run end-to-end
without the real ~5.3GB taxi CSV or ~6MB weather CSV present.

The numbers are not meaningful (random, not real trips) — this exists purely
so `scripts/helpers/datasets.py` has something to fall back to when the real
files aren't found (see `AAA_FORCE_DEMO_DATA` / the auto-fallback logic
there), letting the pipeline be exercised end-to-end on a fresh clone.

Deterministic (fixed seed) and reruns cleanly if regeneration is ever needed.

Usage:
  uv run python scripts/generate_demo_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
CA_GEOJSON_PATH = REPO_ROOT / "data" / "raw" / "community_areas.geojson"
WEATHER_ZONES_PATH = REPO_ROOT / "data" / "raw" / "weather_zones.json"
OUTPUT_DIR = REPO_ROOT / "data" / "demo"
TAXI_OUTPUT = OUTPUT_DIR / "chicago_taxi_trips_2024_demo.csv"
WEATHER_OUTPUT = OUTPUT_DIR / "chicago_weather_hourly_demo.csv"

SEED = 42
N_TRIPS = 2_000
N_AREAS = 20          # subset of the 77 community areas used for demo trips
N_TAXIS = 120
START = pd.Timestamp("2024-03-04 00:00:00")
END = pd.Timestamp("2024-03-24 23:59:59")   # 3 full weeks

PAYMENT_TYPES = ["Credit Card", "Cash", "Mobile", "No Charge", "Dispute", "Unknown"]
PAYMENT_WEIGHTS = [0.45, 0.35, 0.12, 0.03, 0.03, 0.02]
COMPANIES = [
    "Demo Flash Cab", "Demo Sun Taxi", "Demo Globe Taxi",
    "Demo City Service", "Demo Taxi Co",
]

# Hour-of-day weights: light overnight lull, rush-hour + evening peaks.
HOUR_WEIGHTS = np.array([
    1, 1, 1, 1, 1, 2, 4, 6, 7, 5, 4, 4,
    5, 4, 4, 5, 6, 8, 7, 6, 5, 4, 3, 2,
], dtype=float)
HOUR_WEIGHTS /= HOUR_WEIGHTS.sum()


def sample_point_in_polygon(polygon, rng: np.random.Generator) -> Point:
    minx, miny, maxx, maxy = polygon.bounds
    while True:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        p = Point(x, y)
        if polygon.contains(p):
            return p


def random_timestamps(n: int, rng: np.random.Generator) -> pd.DatetimeIndex:
    total_days = (END.normalize() - START.normalize()).days + 1
    days = rng.integers(0, total_days, size=n)
    hours = rng.choice(24, size=n, p=HOUR_WEIGHTS)
    minutes = rng.integers(0, 60, size=n)
    seconds = rng.integers(0, 60, size=n)
    return pd.DatetimeIndex([
        START.normalize() + pd.Timedelta(days=int(d), hours=int(h), minutes=int(mi), seconds=int(s))
        for d, h, mi, s in zip(days, hours, minutes, seconds)
    ])


def generate_taxi_trips(areas: gpd.GeoDataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = N_TRIPS
    taxi_ids = [f"demo_taxi_{j:04d}" for j in range(N_TAXIS)]
    trip_taxi = rng.choice(taxi_ids, size=n)

    start_ts = random_timestamps(n, rng)

    area_idx_pickup = rng.integers(0, len(areas), size=n)
    area_idx_dropoff = rng.integers(0, len(areas), size=n)

    pickup_pts = [sample_point_in_polygon(areas.geometry.iloc[i], rng) for i in area_idx_pickup]
    dropoff_pts = [sample_point_in_polygon(areas.geometry.iloc[i], rng) for i in area_idx_dropoff]

    pickup_area = areas["area_number"].to_numpy()[area_idx_pickup]
    dropoff_area = areas["area_number"].to_numpy()[area_idx_dropoff]

    # Plausible urban cab speeds -> derive trip_miles from trip_seconds so
    # most rows satisfy the Rule-1 speed/fare-per-mile filters in preprocessing.py.
    trip_seconds = rng.integers(180, 2400, size=n)          # 3 min - 40 min
    speed_mph = rng.uniform(8, 35, size=n)
    trip_miles = np.round(speed_mph * trip_seconds / 3600, 2)
    fare_per_mile = rng.uniform(2.5, 5.0, size=n)
    fare = np.round(trip_miles * fare_per_mile, 2)
    tips = np.round(np.where(rng.random(n) < 0.4, fare * rng.uniform(0.1, 0.25, n), 0.0), 2)
    tolls = np.zeros(n)
    extras = np.round(np.where(rng.random(n) < 0.15, rng.uniform(0.5, 3.0, n), 0.0), 2)
    trip_total = np.round(fare + tips + tolls + extras, 2)

    end_ts = start_ts + pd.to_timedelta(trip_seconds, unit="s")

    df = pd.DataFrame({
        "trip_id": [f"demo_trip_{i:06d}" for i in range(n)],
        "taxi_id": trip_taxi,
        "trip_start_timestamp": start_ts,
        "trip_end_timestamp": end_ts,
        "trip_seconds": trip_seconds,
        "trip_miles": trip_miles,
        "pickup_census_tract": pd.NA,
        "dropoff_census_tract": pd.NA,
        "pickup_community_area": pickup_area,
        "dropoff_community_area": dropoff_area,
        "fare": fare,
        "tips": tips,
        "tolls": tolls,
        "extras": extras,
        "trip_total": trip_total,
        "payment_type": rng.choice(PAYMENT_TYPES, size=n, p=PAYMENT_WEIGHTS),
        "company": rng.choice(COMPANIES, size=n),
        "pickup_centroid_latitude": [p.y for p in pickup_pts],
        "pickup_centroid_longitude": [p.x for p in pickup_pts],
        "dropoff_centroid_latitude": [p.y for p in dropoff_pts],
        "dropoff_centroid_longitude": [p.x for p in dropoff_pts],
    })

    # --- Deliberately inject the messy edge cases preprocess_taxi_data() cleans up ---

    # Missing community area (exercises spatial-join imputation from centroid coords).
    missing_ca = rng.random(n) < 0.03
    df.loc[missing_ca, ["pickup_community_area", "dropoff_community_area"]] = np.nan

    # Zero-movement "ghost" trips (same start/end location, zero duration).
    ghost = rng.random(n) < 0.02
    df.loc[ghost, "trip_seconds"] = 0
    df.loc[ghost, "trip_end_timestamp"] = df.loc[ghost, "trip_start_timestamp"]
    df.loc[ghost, "dropoff_centroid_latitude"] = df.loc[ghost, "pickup_centroid_latitude"]
    df.loc[ghost, "dropoff_centroid_longitude"] = df.loc[ghost, "pickup_centroid_longitude"]

    # Zero-mile trips with positive duration (GPS-error rows).
    zero_miles = rng.random(n) < 0.02
    df.loc[zero_miles, "trip_miles"] = 0

    # Missing taxi_id (can't be linked to a vehicle).
    missing_taxi = rng.random(n) < 0.01
    df.loc[missing_taxi, "taxi_id"] = np.nan

    # Rule-1 outliers: implausible speed / fare-per-mile, dropped by preprocessing.
    outlier = rng.random(n) < 0.02
    df.loc[outlier, "trip_miles"] = rng.uniform(150, 300, size=outlier.sum())
    df.loc[outlier, "fare"] = rng.uniform(1, 5, size=outlier.sum())

    df["trip_start_timestamp"] = df["trip_start_timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.000")
    df["trip_end_timestamp"] = df["trip_end_timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.000")

    return df


def generate_weather(rng: np.random.Generator) -> pd.DataFrame:
    with open(WEATHER_ZONES_PATH) as f:
        zones = {int(k): v for k, v in json.load(f).items()}

    hours = pd.date_range(START.normalize(), END.normalize() + pd.Timedelta(hours=23), freq="h")
    rows = []
    for zone_id, (lat, lon) in zones.items():
        # Smooth seasonal + diurnal signal, cold March in Chicago, plus zone-level noise.
        day_frac = (hours - hours[0]) / pd.Timedelta(days=1)
        diurnal = -4 * np.cos(2 * np.pi * (day_frac % 1))
        base_temp = 4.0 + diurnal + rng.normal(0, 1.5, size=len(hours))
        apparent = base_temp - rng.uniform(1, 3, size=len(hours))
        precip = np.where(rng.random(len(hours)) < 0.1, rng.uniform(0, 2, len(hours)), 0.0)
        rain = np.where(base_temp > 0, precip, 0.0)
        snowfall = np.where(base_temp <= 0, precip * 0.7, 0.0)
        snow_depth = np.clip(np.cumsum(snowfall) * 0.02, 0, 5)
        windspeed = rng.uniform(5, 35, size=len(hours))
        windgusts = windspeed + rng.uniform(2, 15, size=len(hours))
        weather_code = rng.choice([0, 1, 2, 3, 61, 71], size=len(hours), p=[0.35, 0.25, 0.15, 0.15, 0.06, 0.04])
        cloud_cover = np.clip(rng.normal(55, 25, size=len(hours)), 0, 100)

        rows.append(pd.DataFrame({
            "time": hours.strftime("%Y-%m-%d %H:%M:%S"),
            "temperature_2m": np.round(base_temp, 1),
            "apparent_temperature": np.round(apparent, 1),
            "precipitation": np.round(precip, 2),
            "rain": np.round(rain, 2),
            "snowfall": np.round(snowfall, 2),
            "snow_depth": np.round(snow_depth, 2),
            "windspeed_10m": np.round(windspeed, 1),
            "windgusts_10m": np.round(windgusts, 1),
            "weather_code": weather_code,
            "cloud_cover": np.round(cloud_cover, 0).astype(int),
            "zone": zone_id,
            "latitude": lat,
            "longitude": lon,
        }))

    return pd.concat(rows, ignore_index=True)


def main():
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_areas = (
        gpd.read_file(CA_GEOJSON_PATH)[["area_numbe", "community", "geometry"]]
        .rename(columns={"area_numbe": "area_number"})
    )
    all_areas["area_number"] = pd.to_numeric(all_areas["area_number"], errors="coerce").astype(int)
    demo_areas = all_areas.sample(n=N_AREAS, random_state=SEED).reset_index(drop=True)

    taxi_df = generate_taxi_trips(demo_areas, rng)
    taxi_df.to_csv(TAXI_OUTPUT, index=False)
    print(f"Wrote {len(taxi_df):,} demo trips -> {TAXI_OUTPUT}")

    weather_df = generate_weather(rng)
    weather_df.to_csv(WEATHER_OUTPUT, index=False)
    print(f"Wrote {len(weather_df):,} demo weather rows -> {WEATHER_OUTPUT}")


if __name__ == "__main__":
    main()
