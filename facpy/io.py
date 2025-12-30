"""
Input/Output module for facpy.
Handles loading Swarm FAC data from CDF or NetCDF files.
"""

import os
from pathlib import Path
from typing import List, Optional, Union, Literal

import cdflib
import netCDF4
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
import aacgmv2


def load_swarm_fac(
    file_path: Union[str, Path, List[Union[str, Path]]],
    satellite: Optional[Literal["A", "B", "C"]] = None,
    return_type: Literal["polars", "pandas"] = "polars",
    clean_columns: bool = True,
    add_magnetic: bool = True,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Load Swarm FAC data from one or multiple CDF/NetCDF files.

    Parameters
    ----------
    file_path : str, Path, or List
        Path to the file(s) or directory containing Swarm FAC data.
        If a directory is provided for a list of files, it will try to load all valid files.
    satellite : str, optional
        "A", "B", or "C". Used for validation if provided.
    return_type : {"polars", "pandas"}, default "polars"
        The desired return DataFrame type.
    clean_columns : bool, default True
        If True, drops invalid measurements (e.g. fill values) and standardizes column names.
    add_magnetic : bool, default True
        If True, computes and appends magnetic coordinates (mlat, mlon, mlt) using aacgmv2.

    Returns
    -------
    pl.DataFrame or pd.DataFrame
        The loaded data. Columns include 'timestamp', 'latitude', 'longitude', 'radius', 'fac',
        and optionally 'mlat', 'mlon', 'mlt'.
    """
    if isinstance(file_path, (str, Path)):
        if os.path.isdir(file_path):
            # TODO: Implement directory walking / globbing logic if needed
            # For now, treat as single file or error if not found
            raise ValueError(f"Directory loading not fully implemented yet. Please provide a list of files or a single file path. Path: {file_path}")
        files = [file_path]
    else:
        files = file_path

    dfs = []
    
    for f in files:
        f = Path(f)
        if not f.exists():
            raise FileNotFoundError(f"File not found: {f}")

        if f.suffix.lower() == ".cdf":
            df = _read_cdf(f)
        elif f.suffix.lower() == ".nc":
            df = _read_netcdf(f)
        else:
            # Fallback or error
            raise ValueError(f"Unsupported file extension: {f.suffix}")
        
        if df is not None:
             dfs.append(df)

    if not dfs:
        raise ValueError("No data loaded.")

    # Concatenate all loaded frames
    # Using Polars for efficient concatenation
    if len(dfs) > 1:
        final_df = pl.concat(dfs)
    else:
        final_df = dfs[0]

    if clean_columns:
        final_df = _clean_data(final_df)

    if add_magnetic:
        final_df = _add_magnetic_coords(final_df)

    if return_type == "pandas":
        return final_df.to_pandas()
    
    return final_df


def _read_cdf(file_path: Path) -> pl.DataFrame:
    """Read a CDF file into a Polars DataFrame using cdflib."""
    cdf = cdflib.CDF(str(file_path))
    
    # Extract relevant variables
    # Common Swarm FAC Level 2 product variables
    # timestamp, Latitude, Longitude, Radius, FAC, IRC, FAC_Error
    
    # Note: Variable names in CDF might depend on version.
    # Looking for standard names.
    
    try:
        # cdflib returns numpy arrays
        timestamp = cdf.varget("Timestamp")
        # Timestamp is usually CDF_EPOCH or CDF_TIME_TT2000
        # converting to datetime64[ns]
        
        info = cdf.varinq("Timestamp")
        if info['Data_Type_Description'] == 'CDF_TIME_TT2000':
             # cdflib < 1.0 had tt2000_to_datetime, 1.3+ uses to_datetime matching logic or class methods
             # We will use to_datetime which should handle it or unixtime
             timestamp = cdflib.cdfepoch.to_datetime(timestamp)
        else:
             timestamp = cdflib.cdfepoch.to_datetime(timestamp)

        lat = cdf.varget("Latitude")
        lon = cdf.varget("Longitude")
        radius = cdf.varget("Radius")
        fac = cdf.varget("FAC")
        
        # Some products might have IRC (Ionospheric Radial Current) or other fields
        # Ideally we check for existence.
        
        data = {
            "Timestamp": timestamp,
            "Latitude": lat,
            "Longitude": lon,
            "Radius": radius,
            "FAC": fac,
        }
        
    except Exception as e:
        # Fallback or specific error handling
        # Attempt to list variables to debug?
        raise ValueError(f"Error reading CDF file {file_path}: {e}")
    finally:
        cdf.close()

    return pl.DataFrame(data)


def _read_netcdf(file_path: Path) -> pl.DataFrame:
    """Read a NetCDF file into a Polars DataFrame using xarray/polars."""
    # xarray is very good at handling NetCDF
    ds = xr.open_dataset(file_path)
    
    # Select relevant variables to keep memory usage low if file is huge
    # Assuming standard Swarm names
    # Using a list of potential names
    
    keep_vars = ["Timestamp", "Latitude", "Longitude", "Radius", "FAC"]
    
    # Intersection of what's in the file and what we want
    available_vars = [v for v in keep_vars if v in ds]
    
    df = ds[available_vars].to_pandas() # xarray -> pandas
    ds.close()
    
    return pl.from_pandas(df)


def _clean_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean the FAC data:
    1. Handle fill values (often 9.9999e9 or similar in CDFs)
    2. Convert column names to snake_case if preferred, or keep standard.
    """
    
    # Swarm fill values can be large numbers.
    # We'll filter based on a reasonable physical range for FAC?
    # Or strict fill value check.
    
    # Let's standardize names to lowercase for consistency
    df = df.rename({
        "Timestamp": "timestamp",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Radius": "radius",
        "FAC": "fac"
    })
    
    # TODO: Handle fill values from metadata if possible.
    # For now, using hardcoded large value check common in space physics data
    # FAC is usually in uA/m^2, range is typically -100 to 100 or so locally,
    # certainly not > 1e9.
    
    df = df.filter(
        (pl.col("fac").abs() < 1e5) &  # Remove obvious fills
        (pl.col("latitude").abs() <= 90) &
        (pl.col("longitude").abs() <= 180)
    )
    
    # Ensure timestamp is sorted?
    df = df.sort("timestamp")
    
    return df


def _add_magnetic_coords(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute Altitude-Adjusted Corrected Geomagnetic (AACGM) coordinates.
    Appends 'mlat', 'mlon', and 'mlt' columns to the Polars DataFrame.
    """
    if df.height == 0:
        return df.with_columns([
            pl.lit(None).cast(pl.Float64).alias("mlat"),
            pl.lit(None).cast(pl.Float64).alias("mlon"),
            pl.lit(None).cast(pl.Float64).alias("mlt"),
        ])

    # AACGMv2 get_aacgm_coord takes (lat, lon, height, datetime)
    # Swarm 'radius' is in meters from Earth's center. 
    # AACGMv2 expects 'height' in km above surface (approx Re = 6371.2 km)
    
    # We'll use a conversion: height = (radius / 1000) - 6371.2
    # Note: Swarm Radius is usually around 6800 km.
    
    def compute_row(row):
        # row: (timestamp, lat, lon, radius)
        dt = row[0]
        lat = row[1]
        lon = row[2]
        h_km = (row[3] / 1000.0) - 6371.2
        
        try:
            mlat, mlon, mlt = aacgmv2.get_aacgm_coord(lat, lon, h_km, dt)
            return mlat, mlon, mlt
        except Exception:
            return None, None, None

    # For Polars, apply is often better if we have many rows, 
    # but aacgmv2 is a C-wrapper, so we might need to loop or use map_rows.
    # map_rows is generally preferred for row-wise python functions.
    
    coords = df.select(["timestamp", "latitude", "longitude", "radius"]).map_rows(compute_row)
    
    # coords is a DataFrame with 3 columns (column_0, column_1, column_2)
    new_cols = coords.rename({
        "column_0": "mlat",
        "column_1": "mlon",
        "column_2": "mlt"
    })
    
    return pl.concat([df, new_cols], how="horizontal")
