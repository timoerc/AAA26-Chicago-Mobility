"""
Chicago Taxi Trip Data Downloader — Chunked / Resumable
=========================================================
Splits the target date range into N equal time windows and downloads each as
a separate part file. If a part file already exists it is skipped, so
interrupted runs can be resumed by simply re-running the script.

Usage:
  python download_taxi_data_chunked.py

Optional: Set an app token (recommended for higher rate limits):
  export CHICAGO_API_KEY_ID=your_id
  export CHICAGO_API_KEY_SECRET=your_secret
"""

import math
import os
import time
from datetime import datetime
from pathlib import Path

import requests

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL  = "https://data.cityofchicago.org/resource/ajtu-isnz.csv"
COUNT_URL = "https://data.cityofchicago.org/resource/ajtu-isnz.json"

REPO_ROOT  = Path(__file__).resolve().parents[1]
OUTPUT_FILE = REPO_ROOT / "data" / "raw_parts" / "chicago_taxi_trips_2024.csv"
PARTS_DIR   = REPO_ROOT / "data" / "raw_parts" / "parts"

# ── Download configuration ────────────────────────────────────────────────────

N_PARTS    = 12           # number of equal time chunks
START_DATE = "2024-01-01" # inclusive
END_DATE   = "2026-05-01" # exclusive
SKIP_CONCAT = False       # set True to keep only part files, skip final merge

PAGE_SIZE = 25_000

API_KEY_ID     = os.environ.get("CHICAGO_API_KEY_ID", "")
API_KEY_SECRET = os.environ.get("CHICAGO_API_KEY_SECRET", "")

BASE_WHERE = (
    "pickup_centroid_latitude IS NOT NULL "
    "AND pickup_centroid_longitude IS NOT NULL "
    "AND trip_start_timestamp IS NOT NULL "
)

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

def build_auth() -> tuple | None:
    if API_KEY_ID and API_KEY_SECRET:
        return (API_KEY_ID, API_KEY_SECRET)
    return None


def fetch_row_count(where: str) -> int:
    params: dict = {"$select": "count(*)", "$limit": 1, "$where": where}
    resp = requests.get(
        COUNT_URL,
        headers={"Accept": "application/json"},
        auth=build_auth(),
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    return int(resp.json()[0]["count"])


def fetch_page(offset: int, where: str) -> bytes:
    params: dict = {
        "$limit":  PAGE_SIZE,
        "$offset": offset,
        "$order":  "trip_start_timestamp ASC",
        "$where":  where,
        "$select": ", ".join(COLUMNS),
    }
    for attempt in range(1, 7):
        try:
            resp = requests.get(
                BASE_URL,
                headers={"Accept": "text/csv"},
                auth=build_auth(),
                params=params,
                timeout=(10, 300),
            )
            resp.raise_for_status()
            return resp.content
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code < 500:
                raise
            if attempt == 6:
                raise
            wait = 15 * (2 ** (attempt - 1))
            print(f"    ⚠  HTTP error ({exc}). Retrying in {wait}s …")
            time.sleep(wait)
        except requests.RequestException as exc:
            if attempt == 6:
                raise
            wait = 15 * (2 ** (attempt - 1))
            print(f"    ⚠  Request failed ({exc}). Retrying in {wait}s …")
            time.sleep(wait)


# ── Time-window splitting ─────────────────────────────────────────────────────

def make_time_windows(start: str, end: str, n: int) -> list[tuple[str, str]]:
    t0 = datetime.fromisoformat(start)
    t1 = datetime.fromisoformat(end)
    delta = (t1 - t0) / n
    windows = []
    for i in range(n):
        a = t0 + delta * i
        b = t0 + delta * (i + 1)
        windows.append((a.strftime("%Y-%m-%dT%H:%M:%S"), b.strftime("%Y-%m-%dT%H:%M:%S")))
    return windows


def part_where(t_start: str, t_end: str, is_last: bool) -> str:
    op = "<=" if is_last else "<"
    return (
        BASE_WHERE
        + f"AND trip_start_timestamp >= '{t_start}' "
        + f"AND trip_start_timestamp {op} '{t_end}'"
    )


# ── Single-part download ──────────────────────────────────────────────────────

def download_part(part_file: Path, where: str, label: str) -> int:
    """Download all pages for `where` into `part_file`. Returns rows written."""
    try:
        estimated = fetch_row_count(where)
        n_pages = math.ceil(estimated / PAGE_SIZE)
        print(f"    Estimated rows: {estimated:,}  ({n_pages} pages)")
    except Exception as exc:
        print(f"    Could not fetch row count ({exc}). Proceeding blind.")
        estimated = None
        n_pages = None

    bar = (
        tqdm(total=n_pages, unit="page", desc=f"  {label}", leave=True)
        if HAS_TQDM else None
    )

    first_page = True
    offset = 0
    total_written = 0
    tmp_file = part_file.with_suffix(".tmp")

    with open(tmp_file, "wb") as out_f:
        page_num = 0
        while True:
            page_num += 1
            if bar is None:
                suffix = f"/{n_pages}" if n_pages else ""
                print(f"    Page {page_num}{suffix}  (offset {offset:,}) …", end=" ", flush=True)

            content = fetch_page(offset, where)
            lines = content.splitlines(keepends=True)

            if len(lines) <= 1:
                if bar is None:
                    print("done (empty page).")
                break

            header = lines[0]
            data_lines = lines[1:]

            if first_page:
                out_f.write(header)
                first_page = False

            for line in data_lines:
                out_f.write(line)

            rows_this_page = len(data_lines)
            total_written += rows_this_page

            if bar is not None:
                bar.update(1)
                bar.set_postfix(rows=f"{total_written:,}", refresh=False)
            else:
                print(f"{rows_this_page:,} rows  (total: {total_written:,})")

            offset += PAGE_SIZE

            if rows_this_page < PAGE_SIZE:
                break

            time.sleep(0.5)

    if bar is not None:
        bar.close()

    tmp_file.rename(part_file)  # atomic: final filename only exists if download completed
    return total_written


# ── Concatenation ─────────────────────────────────────────────────────────────

def concatenate_parts(part_files: list[Path], output: Path) -> int:
    print(f"\nConcatenating {len(part_files)} parts → {output} …")
    total = 0
    with open(output, "wb") as out_f:
        for idx, pf in enumerate(part_files):
            with open(pf, "rb") as f:
                content = f.read()
            lines = content.splitlines(keepends=True)
            if not lines:
                continue
            if idx == 0:
                out_f.writelines(lines)
                total += len(lines) - 1   # minus header
            else:
                out_f.writelines(lines[1:])
                total += len(lines) - 1
    return total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    windows = make_time_windows(START_DATE, END_DATE, N_PARTS)
    pad = len(str(N_PARTS))

    print(f"Chunked download: {N_PARTS} parts  |  {START_DATE} → {END_DATE}")
    print(f"Parts dir : {PARTS_DIR}")
    print(f"Output    : {OUTPUT_FILE}\n")

    part_files: list[Path] = []
    grand_total = 0

    for i, (t_start, t_end) in enumerate(windows):
        part_num = str(i + 1).zfill(pad)
        total_num = str(N_PARTS).zfill(pad)
        part_file = PARTS_DIR / f"chicago_taxi_trips_part_{part_num}_of_{total_num}.csv"
        part_files.append(part_file)

        label = f"Part {part_num}/{total_num}  [{t_start} – {t_end}]"

        if part_file.exists() and part_file.stat().st_size > 0:
            print(f"  {label}: already exists, skipping.")
            continue

        print(f"  {label}")
        is_last = (i == N_PARTS - 1)
        where = part_where(t_start, t_end, is_last)

        try:
            rows = download_part(part_file, where, label)
            grand_total += rows
            size_mb = part_file.stat().st_size / 1_048_576
            print(f"  Done: {rows:,} rows, {size_mb:.1f} MB\n")
        except BaseException as exc:
            for f in (part_file, part_file.with_suffix(".tmp")):
                if f.exists():
                    f.unlink()
            if isinstance(exc, KeyboardInterrupt):
                print(f"\n  Interrupted. Partial file for {label} removed — safe to restart.")
                raise SystemExit(0)
            print(f"\n  ERROR on {label}: {exc}")
            raise SystemExit(1)

    if SKIP_CONCAT:
        print("\nSKIP_CONCAT=True. Parts are in:", PARTS_DIR)
        return

    total_rows = concatenate_parts(part_files, OUTPUT_FILE)
    size_mb = OUTPUT_FILE.stat().st_size / 1_048_576
    print(f"\n✅  Download complete.")
    print(f"   Rows in final file : {total_rows:,}")
    print(f"   Output file        : {OUTPUT_FILE.resolve()}")
    print(f"   File size          : {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
