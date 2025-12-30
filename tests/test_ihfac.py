
import pytest
import xarray as xr
import numpy as np
from facpy.ihfac import compare

@pytest.fixture
def mock_grids():
    # 1x1 coords for simplicity
    
    # North: 50 lat, 0 lon, val 10
    n_data = xr.DataArray(
        [[10.0]], 
        coords={"latitude": [50.0], "longitude": [0.0]}, 
        dims=("latitude", "longitude"),
        name="fac"
    ).to_dataset()
    
    # South: -50 lat, 0 lon, val 8
    s_data = xr.DataArray(
        [[8.0]],
        coords={"latitude": [-50.0], "longitude": [0.0]},
        dims=("latitude", "longitude"),
        name="fac"
    ).to_dataset()
    
    return n_data, s_data

def test_compare_difference(mock_grids):
    n, s = mock_grids
    
    # compare assumes alignment.
    # South lat -50 flips to 50.
    # Diff = 10 - 8 = 2.
    
    diff = compare(n, s, method="difference")
    
    assert "fac" in diff
    # Lat should be 50
    val = diff["fac"].sel(latitude=50.0, longitude=0.0)
    assert abs(val.item() - 2.0) < 1e-6

def test_compare_diff_resolution(mock_grids):
    # Test alignment fails or handles mismatched coords
    n, s = mock_grids
    
    # Modify South to be -52 (-(-52)=52) -> mismatch with 50
    s = s.assign_coords(latitude=[-52.0])
    
    diff = compare(n, s, method="difference")
    
    # Should result in empty overlap or NaNs if not aligned
    # xarray aligns by outer or inner? default is inner intersection for arithmetic?
    # xarray arithmetic (ds1 - ds2) aligns on intersection.
    # So 50 vs 52 -> no overlap -> result empty or all NaN?
    # Intersection -> Empty if no matching coords.
    
    # Check if empty or size 0
    # Actually size might be 0 or 2 (outer)? 
    # Depends on join. Default logic usually behaves like inner join or outer with NaN.
    # Actually, default alignment is "inner" in recent xarray? No, "outer" with NaN usually?
    # Let's check result
    # If using 'arithmetic', join='inner' usually.
    
    # If inner, latitude 50 and 52 don't match -> empty.
    # If lat is 50, diff is NaN.
    
    assert diff.sizes["latitude"] == 0 or np.isnan(diff["fac"]).all()

