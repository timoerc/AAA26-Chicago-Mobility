import pandas as pd
import numpy as np
import h3
import holidays
import json
from scripts.helpers.datasets import load_taxi_data, load_weather_data, _WEATHER_ZONES_PATH
from scripts.helpers.spatial import add_h3_cells, build_demand_panel, add_weather_to_panel








def prepare_modelling(
    resolution: int,
    freq: str,
) -> pd.DataFrame:
    """Build the model-ready demand panel: target, calendar/cyclic features, weather.

    Single entry point for Task 3 modelling. It turns the raw trip table into the
    ``(hexagon x time-bucket)`` panel that both the SVM and the neural network
    consume, then appends the calendar and exogenous features they share.

    All features are derived **on the panel** (from each bucket's start time),
    not on the trips: the modelling unit is the cell-slot, and
    :func:`build_demand_panel` aggregates trip-level columns away, so anything
    computed before aggregation would simply be dropped.

    Parameters
    ----------
    resolution : int
        H3 resolution defining the spatial unit (see :func:`add_h3_cells`).
    freq : str
        Pandas offset alias for the time-bucket width (``'1h'``, ``'3h'``,
        ``'1D'``, ...); drives both the panel grid and the weather aggregation.

    Returns
    -------
    DataFrame
        One row per ``(h3_id, time_bucket)`` with ``trip_count`` (the target),
        calendar + cyclic encodings, and per-bucket weather columns appended.
    """
    # 1. Load the preprocessed trip table.
    df = load_taxi_data(preprocessed=True)

    # 2. Assign each trip to its pickup H3 cell if not already present.
    if f"pickup_h3_r{resolution}" not in df.columns:
        add_h3_cells(df, resolution=resolution)

    # 3. Aggregate to the complete (hexagon x time-bucket) demand panel (zero-filled).
    panel = build_demand_panel(df, resolution=resolution, freq=freq, spatial_role="pickup")

    # 4. Derive calendar features from each bucket's start time (tz-aware, local Chicago time).
    ts = panel["time_bucket"].dt
    panel["day"]        = ts.day.astype("Int64")                                # calendar day
    panel["month"]       = ts.month.astype("Int64")              # 1–12
    panel["hour"]        = ts.hour.astype("Int64")               # 0–23
    panel["day_of_week"] = ts.dayofweek.astype("Int64")          # 0 = Monday … 6 = Sunday
    panel["week"]        = ts.isocalendar().week.astype("Int64") # ISO week number
    panel["is_weekend"]  = ts.dayofweek.isin([5, 6])             # Sat/Sun flag
    
    

    # 5. Flag US (Illinois) public holidays (demand deviates strongly on holidays).
    panel = add_holiday(panel)

    # 6. Add cyclic (sin/cos) encodings so the model sees calendar wrap-around.
    panel = create_cyclic_features(panel)
    panel.drop(columns=["day", "week", "month", "hour", "day_of_week"], inplace=True)

    # 7. Load the hourly weather table + k-means weather-zone centers (zone -> (lat, lon)).
    weather = load_weather_data(preprocessed=True)
    with open(_WEATHER_ZONES_PATH) as f:
        weather_zones = {int(k): v for k, v in json.load(f).items()}

    # 8. Join weather onto every panel row (zero-demand rows included), aggregated to freq.
    panel = add_weather_to_panel(panel, weather, weather_zones, freq=freq)

    return panel


def add_holiday(df: pd.DataFrame) -> pd.DataFrame:
    """Flag US (Illinois) public holidays from the frame's ``time_bucket`` column.

    Demand differs strongly on holidays such as Independence Day or Christmas.
    """
    df = df.copy()
    dates = df["time_bucket"].dt.date
    years = range(df["time_bucket"].min().year, df["time_bucket"].max().year + 1)
    us_il_holidays = holidays.US(subdiv="IL", years=years)
    df["is_holiday"] = dates.isin(set(us_il_holidays))
    return df

def create_cyclic_features(df):
    df = df.copy()

    # Hour of day
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of week
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Month of year
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    
    return df