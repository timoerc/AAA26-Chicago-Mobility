"""
Chicago Hourly Weather Downloader (Open-Meteo)
===============================================
Downloads hourly weather data for Chicago from the Open-Meteo Historical
Archive API. No API key or registration required.

Coordinates are the k-means cluster centroids derived from taxi pickup
locations. Run the elbow / silhouette analysis in
notebooks/01_weather_data_preparation.ipynb first, then paste the
resulting zone dict into WEATHER_ZONES below.

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
  visibility           — Visibility (m)

Usage:
  python scripts/download_weather_data.py

Output:
  data/raw/chicago_weather_hourly.csv
  (one row per zone per hour; 'zone' column identifies the source)
"""

import requests
import pandas as pd
from datetime import date
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_URL   = "https://archive-api.open-meteo.com/v1/archive"

# Replace with the centroids from get_weather_zone_centers() in the notebook.
# Format: zone_name → (latitude, longitude)


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

REPO_ROOT   = Path(__file__).resolve().parents[1]
OUTPUT_FILE = REPO_ROOT / "data" / "raw" / "chicago_weather_hourly.csv"

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

# ── Main download routine ──────────────────────────────────────────────────────

def download_weather_data(
    weather_zones: dict,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Download hourly weather for all zones and save to CSV.

    Parameters
    ----------
    weather_zones:
        dict mapping zone name → (lat, lon), e.g. from get_weather_zone_centers().
    start_date:
        First date to fetch. Accepts datetime.date or pd.Timestamp (subclass of date).
    end_date:
        Last date to fetch. Accepts datetime.date or pd.Timestamp (subclass of date).
    """
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

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

    print(f"\nDone.")
    print(f"   Rows saved : {len(combined):,}  ({len(weather_zones)} zones × ~{len(combined) // len(weather_zones):,} hours)")
    print(f"   Output     : {OUTPUT_FILE.resolve()}")
    return combined


if __name__ == "__main__":
    raise SystemExit(
        "No WEATHER_ZONES defined. Call download_weather_data() from the notebook "
        "after running get_weather_zone_centers() — see notebooks/01_weather_data_preparation.ipynb."
    )
