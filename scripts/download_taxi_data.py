"""
Chicago Taxi Trip Data Downloader
==================================
Downloads the Chicago Taxi Trips 2024+ dataset from the City of Chicago
Data Portal via the Socrata SODA API (dataset ID: ajtu-isnz).

Dataset documentation:
  https://data.cityofchicago.org/Transportation/Taxi-Trips-2024-/ajtu-isnz/about_data

API reference:
  https://dev.socrata.com/foundry/data.cityofchicago.org/ajtu-isnz

Usage:
  python download_chicago_taxi_data.py

Optional: Set an app token (recommended for higher rate limits):
  1. Register at https://data.cityofchicago.org/profile/app_tokens
  2. Set env var:  export SOCRATA_APP_TOKEN=your_token_here
     or paste it into APP_TOKEN below.
"""

import os
import time
import math
import requests
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Configuration ────────────────────────────────────────────────────────────

BASE_URL    = "https://data.cityofchicago.org/resource/ajtu-isnz.csv"

# Resolve paths relative to the repository root (works regardless of CWD / OS).
REPO_ROOT   = Path(__file__).resolve().parents[1]
OUTPUT_DIR  = REPO_ROOT / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "chicago_taxi_trips_2024.csv"

# Page size: Socrata's hard max per request is 50,000 rows.
PAGE_SIZE   = 25_000

# Set to None to download everything, or e.g. 500_000 to cap the download.
MAX_ROWS    = 15_000_000  # e.g. 500_000 for a smaller sample

# Optional API key — increases rate limits. From Developer Settings on the portal.
# Set both env vars, or paste the values directly here.
API_KEY_ID     = os.environ.get("CHICAGO_API_KEY_ID", "")
API_KEY_SECRET = os.environ.get("CHICAGO_API_KEY_SECRET", "")

# Only keep trips that have both census tracts (needed for spatial analysis).
WHERE_FILTER = (
    "pickup_census_tract IS NOT NULL "
    "AND dropoff_census_tract IS NOT NULL "
    "AND pickup_centroid_latitude IS NOT NULL "
    "AND pickup_centroid_longitude IS NOT NULL "
    "AND dropoff_centroid_latitude IS NOT NULL "
    "AND dropoff_centroid_longitude IS NOT NULL "
    "AND trip_start_timestamp IS NOT NULL "
    "AND trip_miles IS NOT NULL "
    "AND trip_seconds IS NOT NULL "
    "AND trip_total IS NOT NULL "
)

# Columns to keep (None = keep all). Reduces file size significantly.
# Full column list: see dataset documentation linked above.
COLUMNS = [
    "trip_id",
    "taxi_id",
    "trip_start_timestamp",
    "trip_end_timestamp",
    "trip_seconds",
    "trip_miles",
    "pickup_census_tract",
    "dropoff_census_tract",
    "pickup_community_area",
    "dropoff_community_area",
    "fare",
    "tips",
    "tolls",
    "extras",
    "trip_total",
    "payment_type",
    "company",
    "pickup_centroid_latitude",
    "pickup_centroid_longitude",
    "dropoff_centroid_latitude",
    "dropoff_centroid_longitude",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_headers(accept: str = "application/json") -> dict:
    return {"Accept": accept}


def build_auth() -> tuple | None:
    if API_KEY_ID and API_KEY_SECRET:
        return (API_KEY_ID, API_KEY_SECRET)
    return None


def fetch_row_count(where: str | None = None) -> int:
    """Query the API for the number of rows, optionally filtered by `where`."""
    url = "https://data.cityofchicago.org/resource/ajtu-isnz.json"
    params: dict = {"$select": "count(*)", "$limit": 1}
    if where:
        params["$where"] = where
    resp = requests.get(url, headers=build_headers(), auth=build_auth(), params=params, timeout=30)
    resp.raise_for_status()
    return int(resp.json()[0]["count"])


def fetch_page(offset: int, select_cols: list[str] | None) -> bytes:
    """Fetch one page of CSV data starting at `offset`."""
    params: dict = {
        "$limit":  PAGE_SIZE,
        "$offset": offset,
        "$order":  "trip_start_timestamp ASC",   # deterministic pagination
        "$where":  WHERE_FILTER,
    }
    if select_cols:
        params["$select"] = ", ".join(select_cols)

    for attempt in range(1, 7):                  # up to 6 retries
        try:
            resp = requests.get(
                BASE_URL,
                headers=build_headers(accept="text/csv"),
                auth=build_auth(),
                params=params,
                timeout=(10, 300),
            )
            resp.raise_for_status()
            return resp.content
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code < 500:
                raise                            # 4xx: don't retry, request is bad
            if attempt == 6:
                raise
            wait = 15 * (2 ** (attempt - 1))
            print(f"  ⚠  Request failed ({exc}). Retrying in {wait}s …")
            time.sleep(wait)
        except requests.RequestException as exc:
            if attempt == 6:
                raise
            wait = 15 * (2 ** (attempt - 1))    # 15, 30, 60, 120, 240 s
            print(f"  ⚠  Request failed ({exc}). Retrying in {wait}s …")
            time.sleep(wait)


# ── Main download routine ─────────────────────────────────────────────────────

def download():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching row counts …")
    try:
        total_rows    = fetch_row_count()
        filtered_rows = fetch_row_count(where=WHERE_FILTER)
        pct = 100 * filtered_rows / total_rows if total_rows else 0
        print(f"  Total rows in dataset : {total_rows:,}")
        print(f"  Rows matching filter  : {filtered_rows:,}  ({pct:.1f}%)")
        total_rows = filtered_rows
    except Exception as exc:
        print(f"  Could not retrieve row count ({exc}). Proceeding without progress estimate.")
        total_rows = None

    if MAX_ROWS and total_rows:
        rows_to_fetch = min(MAX_ROWS, total_rows)
    elif MAX_ROWS:
        rows_to_fetch = MAX_ROWS
    else:
        rows_to_fetch = total_rows

    if rows_to_fetch:
        n_pages = math.ceil(rows_to_fetch / PAGE_SIZE)
        print(f"  Will download up to {rows_to_fetch:,} rows in {n_pages} pages "
              f"({PAGE_SIZE:,} rows/page).")
    else:
        print("  Will download all rows (unknown total — stopping when empty page returned).")

    first_page    = True
    offset        = 0
    total_written = 0

    bar = (
        tqdm(total=n_pages if rows_to_fetch else None, unit="page", desc="Downloading")
        if HAS_TQDM else None
    )

    with open(OUTPUT_FILE, "wb") as out_f:
        page_num = 0
        while True:
            page_num += 1
            if bar is None:
                if rows_to_fetch:
                    print(f"  Page {page_num}/{n_pages}  (offset {offset:,}) …", end=" ", flush=True)
                else:
                    print(f"  Page {page_num}  (offset {offset:,}) …", end=" ", flush=True)

            content = fetch_page(offset, COLUMNS)

            # Split header from data rows
            lines = content.splitlines(keepends=True)
            if len(lines) <= 1:
                # Only header (or empty) — we've reached the end
                if bar is None:
                    print("done (empty page).")
                break

            header = lines[0]
            data_lines = lines[1:]

            if first_page:
                out_f.write(header)        # write header once
                first_page = False

            rows_this_page = len(data_lines)
            for line in data_lines:
                out_f.write(line)

            total_written += rows_this_page

            if bar is not None:
                bar.update(1)
                bar.set_postfix(rows=f"{total_written:,}", refresh=False)
            else:
                print(f"{rows_this_page:,} rows  (total so far: {total_written:,})")

            offset += PAGE_SIZE

            # Stop if we've hit the cap or got a partial page (last page)
            if MAX_ROWS and total_written >= MAX_ROWS:
                if bar is None:
                    print(f"\n  Reached MAX_ROWS cap ({MAX_ROWS:,}). Stopping.")
                break
            if rows_this_page < PAGE_SIZE:
                # Partial page → definitely the last one
                break

            # Polite pause between requests to avoid hammering the API
            time.sleep(0.5)

    if bar is not None:
        bar.close()

    print(f"\n✅  Download complete.")
    print(f"   Rows written : {total_written:,}")
    print(f"   Output file  : {OUTPUT_FILE.resolve()}")
    print(f"   File size    : {OUTPUT_FILE.stat().st_size / 1_048_576:.1f} MB")

if __name__ == "__main__":
    download()