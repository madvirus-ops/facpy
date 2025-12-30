
import pytest
import xarray as xr
import numpy as np
from unittest.mock import patch
from facpy.plot import fac_map

@pytest.fixture
def mock_grid():
    # Create simple xarray grid
    data = np.random.randn(10, 20)
    ds = xr.Dataset(
        {"fac": (("latitude", "longitude"), data)},
        coords={
            "latitude": np.linspace(-90, 90, 10),
            "longitude": np.linspace(-180, 180, 20)
        }
    )
    return ds

def test_fac_map_basic(mock_grid):
    with patch("matplotlib.pyplot.show") as mock_show:
        ax = fac_map(mock_grid, show=True)
        assert ax is not None
        mock_show.assert_called_once()

def test_fac_map_region(mock_grid):
    with patch("matplotlib.pyplot.show"):
        ax = fac_map(mock_grid, region="africa")
        # Check extent was set? 
        # Hard to check directly on axis without transformation logic, 
        # but execution without error is good first step.
        assert ax is not None
