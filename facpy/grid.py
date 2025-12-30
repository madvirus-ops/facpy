"""
Gridding module for FAC data.
Aggregates point data into regular grids (Lat/Lon/LT).
"""

from typing import Optional, Tuple, Union, Literal

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

# valid statistics
STATISTICS = Literal["mean", "median", "std", "count", "sum"]

def grid_fac(
    df: pl.DataFrame,
    resolution: Tuple[float, float] = (2.0, 2.0),
    statistic: STATISTICS = "mean",
    local_time_bins: Optional[int] = None,
    hemisphere: Optional[str] = None,
) -> xr.Dataset:
    """
    Aggregate FAC data into a grid.

    Parameters
    ----------
    df : pl.DataFrame
        Input data. Must contain 'latitude', 'longitude', 'fac'.
        If local_time_bins is used, must contain 'local_time' or timestamps to compute it.
    resolution : tuple, default (2.0, 2.0)
        (lat_res, lon_res) in degrees.
    statistic : str, default 'mean'
        Aggregation statistic.
    local_time_bins : int, optional
        Number of Local Time bins (e.g. 24 for 1-hour bins).
        If provided, the output grid will have a 'local_time' dimension.
    hemisphere : str, optional
        Filter by hemisphere before gridding ('north' or 'south').

    Returns
    -------
    xr.Dataset
        Gridded data with coordinates (latitude, longitude, [local_time]).
    """
    
    # 1. Filter hemisphere if needed
    if hemisphere:
        if hemisphere.lower().startswith("n"):
            df = df.filter(pl.col("latitude") >= 0)
        elif hemisphere.lower().startswith("s"):
            df = df.filter(pl.col("latitude") < 0)

    # 2. Check/Add Local Time if needed
    if local_time_bins is not None:
        if "local_time" not in df.columns:
            # Lazy import to avoid circular dependency?
            # Or assume user prepared it?
            # The prompt says implementation of grid_fac should support it.
            # I can import add_local_time from .geo
            from facpy.geo import add_local_time
            df = add_local_time(df)

    # 3. Compute bin indices
    # We want centered bins usually? Or edge?
    # Let's use simple floor division for binning: val // res * res
    # This creates "lower edge" alignment.
    
    # Lat bins
    lat_res = resolution[0]
    lon_res = resolution[1]

    # Calculate bin centers or edges
    # Round to nearest bin?
    # Floor:
    df = df.with_columns([
        ((pl.col("latitude") / lat_res).floor() * lat_res).alias("lat_bin"),
        ((pl.col("longitude") / lon_res).floor() * lon_res).alias("lon_bin")
    ])
    
    group_cols = ["lat_bin", "lon_bin"]
    
    if local_time_bins is not None:
        # Bin LT
        # 0-24 range
        lt_res = 24.0 / local_time_bins
        df = df.with_columns(
            ((pl.col("local_time") / lt_res).floor() * lt_res).alias("lt_bin")
        )
        group_cols.append("lt_bin")

    # 4. Aggregation
    agg_expr = []
    if statistic == "mean":
        agg_expr.append(pl.col("fac").mean().alias("fac"))
    elif statistic == "median":
        agg_expr.append(pl.col("fac").median().alias("fac"))
    elif statistic == "std":
        agg_expr.append(pl.col("fac").std().alias("fac"))
    elif statistic == "count":
        agg_expr.append(pl.col("fac").count().alias("fac"))
    elif statistic == "sum":
        agg_expr.append(pl.col("fac").sum().alias("fac"))
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
        
    # Also count usually helpful
    agg_expr.append(pl.col("fac").count().alias("count"))

    grouped = df.group_by(group_cols).agg(agg_expr)
    
    # 5. Convert to Xarray
    # Polars to Pandas to Xarray usually easiest for multi-index reshaping
    # Or construct coords manually.
    
    # Collect to pandas
    # using to_dict to avoid pyarrow dependency issues in restricted envs
    # pdf = grouped.to_pandas() 
    pdf = pd.DataFrame(grouped.to_dict(as_series=False))
    
    if pdf.empty:
        # Return empty dataset with correct structure
        return xr.Dataset()

    index_cols = group_cols
    pdf = pdf.set_index(index_cols)
    
    ds = pdf.to_xarray()
    
    # Rename bins to proper coord names
    rename_map = {"lat_bin": "latitude", "lon_bin": "longitude"}
    if "lt_bin" in ds.coords:
        rename_map["lt_bin"] = "local_time"
        
    ds = ds.rename(rename_map)
    
    # Add attributes
    ds.attrs["description"] = f"FAC gridded data. Resolution: {resolution}"
    ds.attrs["statistic"] = statistic
    
    return ds
