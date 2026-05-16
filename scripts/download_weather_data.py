"""
Chicago Daily Weather Downloader
=================================
Downloads daily weather summaries for Chicago from the NCEI Access Data Service
(GHCN-Daily dataset) for the period 2024-01-01 – 2026-04-30.

Four stations are used to ensure full spatial coverage of all 77 Chicago
community areas across four geographic zones:

  Zone         Station ID    Name                              Coverage
  ──────────── ────────────  ────────────────────────────────  ─────────────────────────────────
  North/NW     USW00094846   Chicago O'Hare Intl Airport       Far North Side, Northwest Side,
                                                               O'Hare — primary reference station
  Southwest    USW00014819   Chicago Midway Airport            Southwest Side, West Side,
                                                               Garfield Ridge, Marquette Park
  Central      USC00111577   Chicago Loop / Downtown           Loop, Near North, Near West,
                                                               Lincoln Park — urban heat island
  South        USC00111549   Chicago South Shore               Hyde Park, Woodlawn, South Shore,
                                                               Calumet — southern lakefront zone

Dataset documentation:
  https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily

API reference:
  https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation

Data types downloaded (GHCN-D field names):
  TMAX  — Maximum temperature (°C, converted from tenths)
  TMIN  — Minimum temperature (°C, converted from tenths)
  TAVG  — Average temperature (°C, converted from tenths)
  PRCP  — Precipitation (mm, converted from tenths)
  SNOW  — Snowfall (mm)
  SNWD  — Snow depth (mm)
  AWND  — Average wind speed (m/s, converted from tenths)
  WDF2  — Direction of fastest 2-minute wind (degrees)
  WSF2  — Fastest 2-minute wind speed (m/s, converted from tenths)

Usage:
  python download_weather_data.py

Output:
  data/raw/chicago_weather_2024_2026.csv
  (one row per station per day; STATION column identifies the source)
"""

import os
import requests
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"

# Period of interest
START_DATE = "2024-01-01"
END_DATE   = "2026-04-30"

# GHCN-D station IDs — four stations for full coverage of all 77 community areas
STATIONS = [
    "USW00094846",  # [North/NW]   Chicago O'Hare Intl Airport   
    "USW00014819",  # [Southwest]  Chicago Midway Airport       
    "USC00111577",  # [Central]    Chicago Loop / Downtown       
    "USC00111549",  # [South]      Chicago South Shore           
]

# Human-readable labels (same order as STATIONS)
STATION_LABELS = {
    "USW00094846": "O'Hare Airport       (North/NW)",
    "USW00014819": "Midway Airport       (Southwest)",
    "USC00111577": "Chicago Loop/Central (Central)",
    "USC00111549": "Chicago South Shore  (South)",
}

# Variables to retrieve (comma-separated in the request)
DATA_TYPES = ",".join([
    "TMAX", "TMIN", "TAVG",
    "PRCP", "SNOW", "SNWD",
    "AWND", "WDF2", "WSF2",
])

# Output paths
REPO_ROOT   = Path(__file__).resolve().parents[1]
OUTPUT_DIR  = REPO_ROOT / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "chicago_weather_2024_2026.csv"

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_params() -> dict:
    """Assemble query parameters for the NCEI Access Data Service."""
    return {
        "dataset"   : "daily-summaries",
        "stations"  : ",".join(STATIONS),
        "startDate" : START_DATE,
        "endDate"   : END_DATE,
        "dataTypes" : DATA_TYPES,
        "format"    : "csv",
        "units"     : "metric",          # convert tenths-of-unit to standard SI
        "includeAttributes": "false",    # skip 
        "includeStationName": "true",
        "includeStationLocation": "true",
    }

# ── Main download routine ─────────────────────────────────────────────────────

def download():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    params = build_params()

    print("Downloading Chicago daily weather data from NCEI …")
    print(f"   Period  : {START_DATE} → {END_DATE}")
    print(f"   Stations ({len(STATIONS)}):")
    for sid in STATIONS:
        print(f"     • {sid}  {STATION_LABELS[sid]}")
    print(f"   URL     : {BASE_URL}\n")

    resp = requests.get(
        BASE_URL,
        params=params,
        timeout=120,   # large date ranges can take a moment
    )
    resp.raise_for_status()

    OUTPUT_FILE.write_bytes(resp.content)

    # Number of data rows (excluding header) for all stations combined
    lines = resp.text.strip().splitlines()
    row_count = max(0, len(lines) - 1)   # subtract header

    print(f"✅  Download complete.")
    print(f"   Output file : {OUTPUT_FILE.resolve()}")
    print(f"   File size   : {OUTPUT_FILE.stat().st_size / 1_024:.1f} KB")
    print(f"   Data rows   : {row_count:,}  (all stations combined; use STATION column to filter)")


if __name__ == "__main__":
    download()