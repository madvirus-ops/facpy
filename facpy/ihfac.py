"""
Interhemispheric FAC (IHFAC) analysis.
Comparison of North and South hemisphere FAC structures.
"""

from typing import Optional, Union, Literal

import numpy as np
import polars as pl
import xarray as xr

from facpy.grid import grid_fac


def compare(
    north: Union[pl.DataFrame, xr.Dataset],
    south: Union[pl.DataFrame, xr.Dataset],
    method: Literal["difference", "ratio", "correlation"] = "difference",
    grid_resolution: tuple = (2.0, 2.0),
    align_coordinates: bool = True,
) -> xr.Dataset:
    """
    Compare North and South hemisphere FAC data.
    
    If inputs are DataFrames, they are gridded first using 'grid_resolution' 
    (and implicitly assuming simple lat/lon gridding, forcing alignment).

    Parameters
    ----------
    north : pl.DataFrame or xr.Dataset
        North hemisphere data.
    south : pl.DataFrame or xr.Dataset
        South hemisphere data.
    method : str, default 'difference'
        Comparison method:
        - 'difference': North - South (aligned)
        - 'ratio': North / South
        - 'correlation': Spatial correlation (returns scalar or map?) 
          Actually keeping it simple: (N-S).
    grid_resolution : tuple
        Resolution to grid inputs if they are DataFrames.
    align_coordinates : bool
        If True, flips South latitude to positive for comparison.
        
    Returns
    -------
    xr.Dataset
        Result of comparison.
    """
    
    # 1. Convert to xarray grids if needed
    if isinstance(north, pl.DataFrame):
        ds_n = grid_fac(north, resolution=grid_resolution)
    else:
        ds_n = north

    if isinstance(south, pl.DataFrame):
        ds_s = grid_fac(south, resolution=grid_resolution)
    else:
        ds_s = south
        
    # 2. Align coordinates
    # For comparison, we usually map South (-Lat) to North (+Lat) 
    # OR map both to MLat.
    # Assuming standard geographic or geomagnetic lat where South is negative.
    
    if align_coordinates:
        # Check if latitude is present
        if "latitude" in ds_s.coords:
            # Flip south latitude to match north range
            # ds_s coords: -90 to 0 -> 90 to 0
            # We want to align indices.
            # Easiest way: assign coords lat = -lat
            
            # Create new coord
            new_lat = -ds_s["latitude"]
            ds_s = ds_s.assign_coords(latitude=new_lat)
            
            # Now we need to interpolate or reindex to match North grid EXACTLY
            # xarray alignment handles this if values match.
            # But bin centers might differ slightly if not careful.
            # grid_fac uses floor division, so if range is -90..0 vs 0..90, 
            # bins: -88, -86... vs 0, 2, ...
            # -(-88) = 88. Matches.
            pass
            
    # 3. Perform comparison
    # xarray automatic alignment on coords (lat, lon, [local_time])
    
    # Intersection of valid data
    
    if method == "difference":
        # North - South
        # Note: FAC sign convention?
        # Usually FAC is defined defined by field B direction.
        # In North: Downward is parallel to B?
        # We need to handle sign. 
        # Usually we compare magnitude or signed value?
        # Assuming direct difference val_n - val_s (taking sign into account)
        # But if South FAC has opposite sign convention due to B direction, 
        # meaningful comparison might be sum?
        # Standard convention: Positive = Downward? Or Parallel?
        # Parallel to B:
        # North: B is Down. Positive = Down.
        # South: B is Up. Positive = Up (Away from Earth).
        # Physical FAC: Current into ionosphere.
        # North: Down.
        # South: Down.
        # If we use physical direction (Into/Away), they are consistent.
        # If we use Parallel/Anti-parallel, they flip.
        # Assuming Data provided is already corrected or consistent (e.g. Radial current is often used).
        # If it's Radial Current (IRC), Positive usually Radial Outward?
        # Let's assume user just wants mathematical difference of columns provided.
        
        result = ds_n - ds_s
        result.attrs["method"] = "North - South (aligned)"
        
    elif method == "ratio":
        result = ds_n / ds_s
        result.attrs["method"] = "North / South (aligned)"
        
    elif method == "correlation":
        # Compute correlation coefficient
        # This reduces dimension?
        # Maybe rolling correlation?
        raise NotImplementedError("Correlation method not fully implemented yet.")
        
    else:
        raise ValueError(f"Unknown method {method}")
        
    return result
