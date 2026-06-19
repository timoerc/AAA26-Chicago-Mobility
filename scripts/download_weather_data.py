"""
Chicago Hourly Weather Downloader (Open-Meteo)
===============================================
Downloads hourly weather data for Chicago from the Open-Meteo Historical
Archive API. No API key or registration required.

Coordinates are the k-means cluster centroids derived from taxi pickup
locations, written to data/raw/weather_zones.json by the zone-selection
step in notebooks/01b_weather_data_preparation.ipynb ("Determining
Weather Zone Locations via K-Means Clustering"). Run that step first —
this script reads the file it produces and will exit with a clear
message if it's missing.

The date range to fetch is derived automatically from the earliest and
latest trip timestamps in data/raw/chicago_taxi_trips_2024.csv.

API documentation:
  https://open-meteo.com/en/docs/historical-weather-api

Hourly variables downloaded:
  temperature_2m       — Air temperature at 2 m (°C)
  apparent_temperature — Feels-like temperature (°C)
  precipitation        — Total precipitation per hour (mm)
  rain                 — Rainfall component (mm)
  snowfall             — Snowfall (cm water-equivalent)
  snow_depth           — Snow depth on ground (m)
  windspeed_10m        — Wind speed at 10 m (km/h)
  windgusts_10m        — Wind gusts at 10 m (km/h)
  weather_code         — WMO weather condition code (0=clear … 99=thunderstorm)
  cloud_cover          — Total cloud cover (%)

Usage:
  uv run python scripts/download_weather_data.py

Requires:
  data/raw/weather_zones.json (written by the notebook's k-means step)
  data/raw/chicago_taxi_trips_2024.csv (for the date range)

Output:
  data/raw/chicago_weather_hourly.csv
  (one row per zone per hour; 'zone' column identifies the source)
"""

import json
import sys
import requests
import pandas as pd
from datetime import date
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARIABLES = [
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "windspeed_10m",
    "windgusts_10m",
    "weather_code",
    "cloud_cover",
]

REPO_ROOT          = Path(__file__).resolve().parents[1]
OUTPUT_DIR         = REPO_ROOT / "data" / "raw"
OUTPUT_FILE        = OUTPUT_DIR / "chicago_weather_hourly.csv"
WEATHER_ZONES_FILE = OUTPUT_DIR / "weather_zones.json"

# ── Helpers ────────────────────────────────────────────────────────────────────

def _fetch_zone(zone: str, lat: float, lon: float, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch hourly data for a single coordinate and return a tidy DataFrame."""
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date":   end_date.strftime("%Y-%m-%d"),
        "hourly":     ",".join(HOURLY_VARIABLES),
        "timezone":   "America/Chicago",
    }
    resp = requests.get(BASE_URL, params=params, timeout=120)
    resp.raise_for_status()

    hourly = resp.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df["zone"]      = zone
    df["latitude"]  = lat
    df["longitude"] = lon
    return df

# ── Inputs read from disk ───────────────────────────────────────────────────────

def _load_weather_zones() -> dict:
    """Load zone centroids written by the notebook's k-means step.

    Returns
    -------
    dict mapping zone index (int) → (lat, lon).
    """
    if not WEATHER_ZONES_FILE.exists():
        raise SystemExit(
            "\n"
            f"weather_zones.json not found at {WEATHER_ZONES_FILE}.\n"
            "Run the zone-selection step in "
            "notebooks/01b_weather_data_preparation.ipynb first (the cells under "
            "'Determining Weather Zone Locations via K-Means Clustering') — this "
            "writes weather_zones.json — then re-run this script.\n"
        )
    with open(WEATHER_ZONES_FILE) as f:
        return {int(k): tuple(v) for k, v in json.load(f).items()}


def _derive_date_range() -> tuple[date, date]:
    """Derive the (start, end) date range to fetch from the taxi trip data."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.helpers.datasets import load_taxi_data

    try:
        df = load_taxi_data(preprocessed=False)
    except FileNotFoundError:
        raise SystemExit(
            "\n"
            "data/raw/chicago_taxi_trips_2024.csv not found.\n"
            "Run `uv run python scripts/download_taxi_data.py` first, then re-run "
            "this script.\n"
        )
    start = df["trip_start_timestamp"].min().date()
    end = df["trip_end_timestamp"].max().date()
    return start, end

# ── Main download routine ──────────────────────────────────────────────────────

def download_weather_data(
    weather_zones: dict,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Download hourly weather for all zones and save to CSV.

    Reusable core fetch routine — call it directly if you already have a
    zones dict and date range in memory. For the standalone CLI entry
    point that derives both automatically, see download().

    Parameters
    ----------
    weather_zones:
        dict mapping zone name → (lat, lon), e.g. from get_weather_zone_centers().
    start_date:
        First date to fetch. Accepts datetime.date or pd.Timestamp (subclass of date).
    end_date:
        Last date to fetch. Accepts datetime.date or pd.Timestamp (subclass of date).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading hourly weather data from Open-Meteo …")
    print(f"   Period : {start_date} → {end_date}")
    print(f"   Zones  : {len(weather_zones)}\n")

    frames = []
    for zone, (lat, lon) in weather_zones.items():
        print(f"   Fetching {zone}  ({lat:.4f}, {lon:.4f}) …", end=" ", flush=True)
        df = _fetch_zone(zone, lat, lon, start_date, end_date)
        frames.append(df)
        print(f"{len(df):,} rows")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUTPUT_FILE, index=False)
    return combined


def download():
    weather_zones = _load_weather_zones()
    start_date, end_date = _derive_date_range()

    combined = download_weather_data(weather_zones, start_date, end_date)

    print(f"\n✅  Download complete.")
    print(f"   Output file  : {OUTPUT_FILE.resolve()}")
    print(f"   File size    : {OUTPUT_FILE.stat().st_size / 1_024:.1f} KB")
    print(f"   Rows written : {len(combined):,}")


if __name__ == "__main__":
    download()
