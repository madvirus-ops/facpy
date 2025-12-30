
import pytest
import polars as pl
import numpy as np
import xarray as xr
from facpy.grid import grid_fac
from datetime import datetime

@pytest.fixture
def grid_data():
    # 4 points
    # 2 in same bin, 2 in diff bins
    
    # 2x2 grid
    # Bin 1: Lat [0, 2), Lon [0, 2)
    # P1: Lat 0.5, Lon 0.5, FAC 1.0
    # P2: Lat 1.5, Lon 1.5, FAC 3.0
    # Mean should be 2.0
    
    # Bin 2: Lat [2, 4), Lon [0, 2)
    # P3: Lat 2.5, Lon 0.5, FAC 5.0
    
    # P4: Lat 0.5, Lon 2.5, FAC 10.0 (Bin Lat [0, 2), Lon [2, 4))
    
    # Also need timestamp for LT test
    
    data = {
        "latitude": [0.5, 1.5, 2.5, 0.5],
        "longitude": [0.5, 1.5, 0.5, 2.5],
        "fac": [1.0, 3.0, 5.0, 10.0],
        "timestamp": [
            datetime(2021,1,1,12,0),
            datetime(2021,1,1,12,0),
            datetime(2021,1,1,12,0),
            datetime(2021,1,1,12,0)
        ]
    }
    return pl.DataFrame(data)

def test_grid_fac_basic(grid_data):
    ds = grid_fac(grid_data, resolution=(2.0, 2.0), statistic="mean")
    
    assert isinstance(ds, xr.Dataset)
    assert "fac" in ds.data_vars
    assert "count" in ds.data_vars
    
    # Check coords
    # Latitudes present: 0.0, 2.0
    # Longitudes present: 0.0, 2.0
    
    # Bin at lat=0, lon=0 -> P1 and P2
    # Mean (1+3)/2 = 2.0
    val_0_0 = ds["fac"].sel(latitude=0.0, longitude=0.0).item()
    assert abs(val_0_0 - 2.0) < 1e-6
    
    # Bin at lat=2, lon=0 -> P3 (5.0)
    val_2_0 = ds["fac"].sel(latitude=2.0, longitude=0.0).item()
    assert abs(val_2_0 - 5.0) < 1e-6
    
    # Bin at lat=0, lon=2 -> P4 (10.0)
    val_0_2 = ds["fac"].sel(latitude=0.0, longitude=2.0).item()
    assert abs(val_0_2 - 10.0) < 1e-6

def test_grid_fac_lt(grid_data):
    # Add LT binning
    # All points are approx LT 12 (Lon near 0)
    # Except if we change timestamps or reuse logic.
    # Lon 0, UTC 12 -> SLT 12
    # Lon 2.5, UTC 12 -> SLT 12.16
    
    ds = grid_fac(grid_data, resolution=(5.0, 5.0), local_time_bins=24)
    # 24 bins -> 1 hour resolution
    # All points fall in bin 12? (12.0 // 1 = 12.0)
    
    assert "local_time" in ds.coords
    
    # There should be data at local_time=12.0
    val = ds["fac"].sel(latitude=0.0, longitude=0.0, local_time=12.0)
    # Note: resolution 5x5 puts all in same spatial bin (0,0)
    # Mean of 1, 3, 5, 10 = 19/4 = 4.75
    assert abs(val.item() - 4.75) < 1e-6

def test_grid_fac_empty():
    df = pl.DataFrame({"latitude": [], "longitude": [], "fac": [], "timestamp": []}, schema={"latitude": pl.Float64, "longitude": pl.Float64, "fac": pl.Float64, "timestamp": pl.Datetime})
    ds = grid_fac(df)
    assert isinstance(ds, xr.Dataset)
    assert len(ds.data_vars) == 0 # or empty
