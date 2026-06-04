import pandas as pd
import h3


def add_h3_cells(df: pd.DataFrame, resolution: int) -> None:
    """Add pickup_h3 and dropoff_h3 columns in-place to df_demand."""
    # For all lat/lon pairs in pickup/dropoff, convert into h3 hexID the event occured in
    # pd.array(..., dtype='string') prevents pandas from mis-casting h3 HexID as complex128
    # {resolution} adds resolution number to column name, e.g. pickup_h3_r7, pickup_h3_r8, etc.
    # latlng_to_cell converts (lat, lon) to h3 hexID (the tuple lays in) at given resolution
    # Zip combines lat/long cols into pairs
    df[f'pickup_h3_r{resolution}'] = pd.array(
        [h3.latlng_to_cell(lat, lon, resolution) for lat, lon in zip(df['pickup_centroid_latitude'], df['pickup_centroid_longitude'])],
        dtype='string'
    )
    df[f'dropoff_h3_r{resolution}'] = pd.array(
        [h3.latlng_to_cell(lat, lon, resolution) for lat, lon in zip(df['dropoff_centroid_latitude'], df['dropoff_centroid_longitude'])],
        dtype='string'
    )