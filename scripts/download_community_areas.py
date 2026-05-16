"""
Chicago Community Areas Boundary Downloader
============================================
Downloads the Chicago Community Areas boundary dataset from the City of Chicago
Data Portal via the Socrata SODA API (dataset ID: igwz-8jzy).

Dataset documentation:
  https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Community-Areas/igwz-8jzy/about_data

API reference:
  https://dev.socrata.com/foundry/data.cityofchicago.org/igwz-8jzy

Usage:
  python download_community_areas.py

Optional: Set an app token (recommended for higher rate limits):
  1. Register at https://data.cityofchicago.org/profile/app_tokens
  2. Set env var:  export SOCRATA_APP_TOKEN=your_token_here
     or paste it into APP_TOKEN below.
"""

import os
import requests
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

BASE_URL = "https://data.cityofchicago.org/resource/igwz-8jzy.geojson"

REPO_ROOT   = Path(__file__).resolve().parents[1]
OUTPUT_DIR  = REPO_ROOT / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "community_areas.geojson"

# 77 community areas — limit set well above that to be safe
LIMIT = 100

API_KEY_ID     = os.environ.get("CHICAGO_API_KEY_ID", "")
API_KEY_SECRET = os.environ.get("CHICAGO_API_KEY_SECRET", "")

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_auth() -> tuple | None:
    if API_KEY_ID and API_KEY_SECRET:
        return (API_KEY_ID, API_KEY_SECRET)
    return None

# ── Main download routine ─────────────────────────────────────────────────────

def download():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading Chicago community area boundaries …")
    resp = requests.get(
        BASE_URL,
        auth=build_auth(),
        params={"$limit": LIMIT},
        timeout=30,
    )
    resp.raise_for_status()

    OUTPUT_FILE.write_bytes(resp.content)

    print(f"\n✅  Download complete.")
    print(f"   Output file  : {OUTPUT_FILE.resolve()}")
    print(f"   File size    : {OUTPUT_FILE.stat().st_size / 1_024:.1f} KB")


if __name__ == "__main__":
    download()
