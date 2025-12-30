"""
Geospatial utilities for facpy.
Handles coordinate transformations, local time calculation, and spatial filtering.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

# Predefined regions (min_lon, max_lon, min_lat, max_lat)
REGIONS = {
    "africa": (-20, 55, -40, 40),
    "europe": (-10, 40, 35, 75),
    "north_america": (-170, -50, 20, 75),
    "south_america": (-90, -30, -60, 15),
    "asia": (60, 150, 0, 75),
    "australia": (110, 160, -45, -10),
    "polar_north": (-180, 180, 60, 90),
    "polar_south": (-180, 180, -90, -60),
    "equatorial": (-180, 180, -20, 20),
}


def filter_region(
    df: pl.DataFrame,
    region: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    hemisphere: Optional[str] = None,
) -> pl.DataFrame:
    """
    Filter DataFrame by spatial region and/or hemisphere.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing 'latitude' and 'longitude' columns.
    region : str, optional
        Name of a predefined region (e.g., 'africa').
    bbox : tuple, optional
        Custom bounding box (min_lon, max_lon, min_lat, max_lat).
    hemisphere : {'north', 'south'}, optional
        Filter by hemisphere.

    Returns
    -------
    pl.DataFrame
        Filtered DataFrame.
    """
    if region and bbox:
         raise ValueError("Specify either 'region' or 'bbox', not both.")

    if region:
        if region.lower() not in REGIONS:
             raise ValueError(f"Unknown region '{region}'. Available: {list(REGIONS.keys())}")
        min_lon, max_lon, min_lat, max_lat = REGIONS[region.lower()]
    elif bbox:
        min_lon, max_lon, min_lat, max_lat = bbox
    else:
        min_lon, max_lon, min_lat, max_lat = None, None, None, None

    if min_lon is not None:
        # Handle longitude crossing 180?
        # Standard implementation for linear range:
        if min_lon <= max_lon:
             df = df.filter(
                 (pl.col("longitude") >= min_lon) &
                 (pl.col("longitude") <= max_lon)
             )
        else:
             # Crossing dateline (e.g. 170 to -170)
             df = df.filter(
                 (pl.col("longitude") >= min_lon) |
                 (pl.col("longitude") <= max_lon)
             )
             
        df = df.filter(
            (pl.col("latitude") >= min_lat) &
            (pl.col("latitude") <= max_lat)
        )

    if hemisphere:
        if hemisphere.lower().startswith("n"):
            df = df.filter(pl.col("latitude") >= 0)
        elif hemisphere.lower().startswith("s"):
            df = df.filter(pl.col("latitude") < 0)
        else:
             raise ValueError("Hemisphere must be 'north' or 'south'")

    return df


def add_local_time(df: pl.DataFrame, method: str = "slt") -> pl.DataFrame:
    """
    Add Local Time (LT) column to the DataFrame.
    
    Parameters
    ----------
    df : pl.DataFrame
        Must contain 'timestamp' and 'longitude'.
    method : str
        "slt" for Solar Local Time.
        "mlt" for Magnetic Local Time (not fully implemented without dependencies).

    Returns
    -------
    pl.DataFrame
        DataFrame with new 'local_time' column (float, 0-24).
    """
    if method == "slt":
        return _calculate_slt(df)
    elif method == "mlt":
        raise NotImplementedError("MLT calculation requires 'aacgmv2' or 'apexpy' which are not standard dependencies. Please use 'slt' or implement MLT locally.")
    else:
        raise ValueError(f"Unknown method {method}")


def _calculate_slt(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate Solar Local Time.
    SLT = UTC + Longitude / 15
    """
    # UTC hour from timestamp
    # timestamp is datetime.
    
    # We can use Polars expressions for speed.
    
    # Extract UTC hour (decimal)
    # hour + minute/60 + second/3600
    
    utc_hours = (
        pl.col("timestamp").dt.hour() + 
        pl.col("timestamp").dt.minute() / 60.0 + 
        pl.col("timestamp").dt.second() / 3600.0
    )
    
    # longitude offset
    lon_offset = pl.col("longitude") / 15.0
    
    slt = utc_hours + lon_offset
    
    # Wrap to 0-24
    slt = (slt % 24.0)
    
    return df.with_columns(slt.alias("local_time"))
