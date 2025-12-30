"""
Plotting module for facpy.
Generates maps and visualizations using Cartopy.
"""

from typing import Optional, Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def fac_map(
    grid: xr.Dataset,
    variable: str = "fac",
    projection: str = "platecarree",
    region: Optional[str] = None,
    title: Optional[str] = None,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    Plot a map of the FAC data.

    Parameters
    ----------
    grid : xr.Dataset
        Gridded data to plot.
    variable : str, default 'fac'
        Variable name in the dataset to plot.
    projection : str, default 'platecarree'
        Map projection. Options: 'platecarree', 'mercator', 'orthographic', 'robinson'.
    region : str, optional
        Focus region. E.g. 'africa'. Uses presets from geo.py.
    title : str, optional
        Plot title.
    cmap : str, default 'RdBu_r'
        Colormap. 'RdBu_r' is good for centered data (Red=Positive/Down, Blue=Negative/Up).
    vmin, vmax : float, optional
        Color scale limits. If None, symmetric max(abs) is used.
    show : bool, default True
        Whether to show the plot.
    save_path : str, optional
        Path to save the figure.
    """
    
    # 1. Setup Projection
    if projection == "platecarree":
        proj = ccrs.PlateCarree()
    elif projection == "mercator":
        proj = ccrs.Mercator()
    elif projection == "orthographic":
        # Centered roughly on Africa/Atlantic
        proj = ccrs.Orthographic(central_longitude=0, central_latitude=0)
    elif projection == "robinson":
         proj = ccrs.Robinson()
    else:
        raise ValueError(f"Unknown projection: {projection}")
        
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    
    # 2. Add features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)
    
    # 3. Handle region extent
    transform = ccrs.PlateCarree() # Data is assumed lat/lon
    
    if region:
        from facpy.geo import REGIONS
        if region.lower() in REGIONS:
             min_lon, max_lon, min_lat, max_lat = REGIONS[region.lower()]
             ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=transform)
             
    # 4. Prepare Data
    if variable not in grid:
         raise ValueError(f"Variable '{variable}' not found in grid. Available: {list(grid.data_vars)}")
         
    data_array = grid[variable]
    
    # Check if local_time dimension exists. If so, average or select?
    if "local_time" in data_array.dims:
        # TODO: Handle LT. Plot mean over LT? Or error?
        # User should reduce first usually.
        # But let's take mean to be safe/useful
        print("Warning: 'local_time' dimension found. Taking mean over 'local_time' for map.")
        data_array = data_array.mean(dim="local_time")
        
    # 5. Colormap Norm
    if vmin is None and vmax is None:
        # Symmetric
        mx = float(np.abs(data_array).max())
        vmin = -mx
        vmax = mx
        
    # 6. Plot using xarray's plot pcolormesh
    # Note: xarray plot uses matplotlib.
    # We pass ax.
    
    p = data_array.plot.pcolormesh(
        ax=ax, 
        transform=transform,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kwargs={"label": f"{variable} (uA/m^2 approx)"}
    )
    
    if title:
        ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        
    if show:
        plt.show()
    
    return ax
