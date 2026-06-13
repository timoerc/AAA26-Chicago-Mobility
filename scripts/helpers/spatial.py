import pandas as pd
import numpy as np
import h3


def _latlng_to_cell_safe(lat, lon, resolution):
    if pd.isna(lat) or pd.isna(lon):
        return pd.NA
    return h3.latlng_to_cell(lat, lon, resolution)


def add_h3_cells(df: pd.DataFrame, resolution: int) -> None:
    """Add pickup_h3 and dropoff_h3 columns in-place to df_demand."""
    # For all lat/lon pairs in pickup/dropoff, convert into h3 hexID the event occured in
    # pd.array(..., dtype='string') prevents pandas from mis-casting h3 HexID as complex128
    # {resolution} adds resolution number to column name, e.g. pickup_h3_r7, pickup_h3_r8, etc.
    # latlng_to_cell converts (lat, lon) to h3 hexID (the tuple lays in) at given resolution
    # Zip combines lat/long cols into pairs
    df[f'pickup_h3_r{resolution}'] = pd.array(
        [_latlng_to_cell_safe(lat, lon, resolution) for lat, lon in zip(df['pickup_centroid_latitude'], df['pickup_centroid_longitude'])],
        dtype='string'
    )
    df[f'dropoff_h3_r{resolution}'] = pd.array(
        [_latlng_to_cell_safe(lat, lon, resolution) for lat, lon in zip(df['dropoff_centroid_latitude'], df['dropoff_centroid_longitude'])],
        dtype='string'
    )


def _floor_local(s: pd.Series, freq: str) -> pd.Series:
    """Floor a datetime Series to ``freq`` in local wall-clock time.

    ``dt.floor`` on a tz-aware series operates on the underlying UTC instants,
    so for multi-hour ``freq`` it drifts out of step with a wall-clock
    ``pd.date_range`` whenever the UTC offset is not a multiple of ``freq``
    (e.g. Chicago's -5h DST offset vs a 3h/4h bucket). Dropping the tz first
    floors on the wall-clock the buckets are meant to represent, keeping the
    floored values and the bucket grid consistent across DST transitions.
    """
    if s.dt.tz is not None:
        s = s.dt.tz_localize(None)
    return s.dt.floor(freq)


def build_demand_panel(
    df: pd.DataFrame,
    resolution: int = 7,
    freq: str = "1h",
    spatial_role: str = "pickup",
    time_col: str = "trip_start_timestamp",
    hex_universe=None,
    drop_empty_hexes: bool = True,
) -> pd.DataFrame:
    """Aggregate trips into a complete (spatial unit x time bucket) demand panel.

    This is the forecasting target builder for Task 3: one row per
    (H3 hexagon, time bucket) with the number of pickups in that cell-slot as
    ``trip_count``. Both axes are parameterised so the same definition can drive
    a sensitivity sweep over spatial granularity and temporal aggregation:

    - ``resolution`` selects the H3 spatial unit (column ``{role}_h3_r{res}``
      must already exist; call :func:`add_h3_cells` first).
    - ``freq`` is a pandas offset alias (``'1h'``, ``'4h'``, ``'1D'``, ...) that
      drives *both* the bucketing of timestamps and the time grid used to fill
      zero-demand slots, so the two can never drift apart.

    Crucially, the returned panel is **complete**: every hexagon is paired with
    every time bucket in the observed range, and slots with no trips are filled
    with ``trip_count = 0``. A cell-hour with zero pickups is a real demand
    observation; omitting it (as a plain ``groupby().size()`` would) biases a
    forecaster toward over-prediction.

    Parameters
    ----------
    df : DataFrame
        Trip-level data with a tz-aware ``time_col`` and the H3 column for the
        requested ``resolution`` / ``spatial_role``.
    resolution : int
        H3 resolution defining the spatial unit.
    freq : str
        Pandas offset alias for the time-bucket width.
    spatial_role : {'pickup', 'dropoff'}
        ``'pickup'`` (origin) is demand; ``'dropoff'`` measures attraction.
    time_col : str
        Timestamp column to bucket. For demand this is the trip *start*.
    hex_universe : iterable of str, optional
        Hexagons to fill zeros over (e.g. the geographic-fill grid from
        ``build_h3_grid_from_trips``). Defaults to the cells observed in ``df``.
    drop_empty_hexes : bool
        If True, hexagons with zero trips across the whole period are excluded
        from the panel. This keeps boundary-fill cells (which never see a trip)
        from flooding the panel with all-zero rows, while preserving the
        informative within-active-hexagon zeros.

    Returns
    -------
    DataFrame
        Columns ``['h3_id', 'time_bucket', 'trip_count']``, sorted by
        ``['h3_id', 'time_bucket']`` and ready for feature joins / lagging.
    """
    if spatial_role not in {"pickup", "dropoff"}:
        raise ValueError("spatial_role must be 'pickup' or 'dropoff'")

    hex_col = f"{spatial_role}_h3_r{resolution}"
    if hex_col not in df.columns:
        raise KeyError(
            f"'{hex_col}' not found. Run add_h3_cells(df, {resolution}) first."
        )

    # Keep only rows with a valid cell and timestamp, then floor to the bucket
    # in local wall-clock time (see _floor_local: avoids DST grid drift).
    work = df[[hex_col, time_col]].dropna()
    work = work.assign(time_bucket=_floor_local(work[time_col], freq))

    counts = (
        work.groupby([hex_col, "time_bucket"], observed=True)
            .size()
            .rename("trip_count")
    )

    # Spatial axis: provided universe (e.g. boundary-filled grid) or observed cells.
    observed_hexes = counts.index.get_level_values(hex_col).unique()
    if hex_universe is None:
        hexes = observed_hexes
    else:
        hexes = pd.Index(pd.unique(pd.Series(list(hex_universe), dtype="string")))
        if drop_empty_hexes:
            # Restrict to cells that actually saw at least one trip.
            hexes = hexes.intersection(observed_hexes)
    hexes = hexes.dropna()

    # Time axis: contiguous grid at the same freq, inheriting tz from the data.
    buckets = pd.date_range(
        work["time_bucket"].min(), work["time_bucket"].max(), freq=freq
    )

    full_index = pd.MultiIndex.from_product(
        [hexes, buckets], names=[hex_col, "time_bucket"]
    )

    panel = (
        counts.reindex(full_index, fill_value=0)
              .reset_index()
              .rename(columns={hex_col: "h3_id"})
              .sort_values(["h3_id", "time_bucket"], ignore_index=True)
    )
    panel["trip_count"] = panel["trip_count"].astype("int64")
    return panel


# How each weather variable collapses when several hours fall into one bucket.
# Accumulations are summed, peak gusts take the max, everything else (state
# variables such as temperature or cloud cover) is averaged. Columns not listed
# here default to "mean".
_WEATHER_AGG_DEFAULTS = {
    "precipitation": "sum",
    "rain": "sum",
    "snowfall": "sum",
    "windgusts_10m": "max",
}


def add_weather_to_panel(
    panel: pd.DataFrame,
    weather: pd.DataFrame,
    weather_zones: dict,
    freq: str = "1h",
    weather_agg: dict | None = None,
    panel_hex_col: str = "h3_id",
    panel_time_col: str = "time_bucket",
    weather_time_col: str = "time",
    weather_zone_col: str = "zone",
) -> pd.DataFrame:
    """Join exogenous weather onto a demand panel, aggregated to the bucket width.

    Weather is joined *at the panel level* (not via the per-trip
    ``merge_weather``) so that every row receives weather — including
    zero-demand buckets, which have no trips to carry it but are exactly the
    rows where weather is most informative. Weather is exogenous: it exists for
    each ``(zone, hour)`` regardless of whether any trip occurred.

    The spatial link reuses the nearest-zone logic of ``merge_weather``, but
    applied to **hexagon centroids** instead of trip pickups: each panel
    hexagon is assigned the closest weather zone, then weather is merged on
    ``(zone, time_bucket)``.

    The weather table is hourly. When ``freq`` is coarser than ``'1h'`` the
    hourly records are aggregated to the bucket first, per-variable: see
    :data:`_WEATHER_AGG_DEFAULTS` (sums for accumulations, max for gusts, mean
    for state variables; unlisted columns default to ``"mean"``). Pass
    ``weather_agg`` to override. For ``freq='1h'`` the flooring is a no-op and
    the join is exact.

    Parameters
    ----------
    panel : DataFrame
        Output of :func:`build_demand_panel` (``[h3_id, time_bucket, ...]``).
    weather : DataFrame
        Cleaned hourly weather with a tz-aware ``weather_time_col``, a
        ``weather_zone_col``, and numeric weather columns.
    weather_zones : dict
        Mapping ``zone -> (lat, lon)`` (the k-means centers used by
        ``merge_weather``); keys must match the values in ``weather_zone_col``.
    freq : str
        Must match the ``freq`` used to build ``panel``.
    weather_agg : dict, optional
        Per-column aggregation overrides, e.g. ``{"snow_depth": "max"}``.

    Returns
    -------
    DataFrame
        ``panel`` with one column per weather variable appended; one row per
        original panel row (left join, so demand rows are never dropped).
    """
    # 1. Assign each hexagon to its nearest weather zone (argmin over centroids).
    zone_names = np.array(list(weather_zones.keys()))
    zone_coords = np.array(list(weather_zones.values()))            # (n_zones, 2)
    hex_ids = panel[panel_hex_col].unique()
    hex_coords = np.array([h3.cell_to_latlng(h) for h in hex_ids])  # (n_hex, 2)
    sq_dists = np.array([np.sum((hex_coords - z) ** 2, axis=1) for z in zone_coords])
    hex_to_zone = dict(zip(hex_ids, zone_names[sq_dists.argmin(axis=0)]))

    # 2. Aggregate hourly weather to the bucket width, per variable.
    w = weather.copy()
    # Floor in local wall-clock to match build_demand_panel's tz-naive buckets.
    w["_bucket"] = _floor_local(w[weather_time_col], freq)
    weather_cols = [c for c in w.columns
                    if c not in {weather_time_col, weather_zone_col, "_bucket"}]
    agg = {c: "mean" for c in weather_cols}
    agg.update({c: f for c, f in _WEATHER_AGG_DEFAULTS.items() if c in weather_cols})
    if weather_agg:
        agg.update({c: f for c, f in weather_agg.items() if c in weather_cols})

    w_agg = (
        w.groupby([weather_zone_col, "_bucket"], observed=True)
         .agg(agg)
         .reset_index()
         .rename(columns={"_bucket": panel_time_col})
    )

    # 3. Left-join weather onto every panel row via the hexagon's zone.
    out = panel.copy()
    out["_zone"] = out[panel_hex_col].map(hex_to_zone)
    out = out.merge(
        w_agg,
        left_on=["_zone", panel_time_col],
        right_on=[weather_zone_col, panel_time_col],
        how="left",
    ).drop(columns=["_zone", weather_zone_col])
    return out

def add_poi_to_panel(
    panel: pd.DataFrame,
    poi_features: pd.DataFrame,
    panel_hex_col: str = "h3_id",
) -> pd.DataFrame:
    """Left-join static per-hexagon POI counts onto a demand panel.
    POI counts are time-invariant, so they attach on the hexagon id alone and
    broadcast across every time bucket of that cell. Cells with no POIs receive
    ``0`` (a real absence, not missing data). ``poi_features`` must be aggregated
    at the **same H3 resolution** used to build ``panel`` — see
    :func:`scripts.helpers.datasets.load_poi_features`.
    Parameters
    ----------
    panel : DataFrame
        Demand panel with a ``panel_hex_col`` hexagon id.
    poi_features : DataFrame
        Per-hexagon counts: an ``h3_id`` column plus one ``n_poi_*`` column per
        category (output of :func:`load_poi_features`).
    Returns
    -------
    DataFrame
        ``panel`` with the ``n_poi_*`` columns appended (left join, every panel
        row preserved).
    """
    poi = poi_features.copy()
    # Align the join-key dtype (panel uses pandas 'string', POI ids are object).
    poi["h3_id"] = poi["h3_id"].astype(panel[panel_hex_col].dtype)
    poi_cols = [c for c in poi.columns if c != "h3_id"]
    out = panel.merge(poi, how="left", left_on=panel_hex_col, right_on="h3_id")
    if panel_hex_col != "h3_id" and "h3_id" in out.columns:
        out = out.drop(columns="h3_id")
    out[poi_cols] = out[poi_cols].fillna(0).astype("int64")
