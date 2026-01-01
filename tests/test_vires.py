
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import polars as pl
import xarray as xr
import numpy as np
import datetime
from facpy.io import fetch_swarm_fac

@pytest.fixture
def mock_vires():
    with patch("facpy.io.SwarmRequest") as mock:
        yield mock

def test_fetch_swarm_fac(mock_vires):
    # Setup mock
    mock_instance = mock_vires.return_value
    mock_data = MagicMock()
    
    # Create a dummy xarray dataset to be returned by to_xarray()
    times = pd.to_datetime(["2021-01-01 00:00:00"])
    ds = xr.Dataset(
        data_vars={
            "FAC": (("Timestamp",), [0.5]),
            "Latitude": (("Timestamp",), [50.0]),
            "Longitude": (("Timestamp",), [10.0]),
            "Radius": (("Timestamp",), [6800.0]),
            "Kp": (("Timestamp",), [3.0]),
        },
        coords={"Timestamp": times}
    )
    
    mock_data.to_xarray.return_value = ds
    mock_instance.get_between.return_value = mock_data
    
    # Test fetch with auxiliaries
    df = fetch_swarm_fac(
        start_time="2021-01-01T00:00:00",
        end_time="2021-01-01T00:00:01",
        satellite="A",
        auxiliaries=["Kp"],
        add_magnetic=False # Skip magnetic coord calculation in mock test for speed
    )
    
    assert isinstance(df, pl.DataFrame)
    assert "fac" in df.columns
    assert "Kp" in df.columns
    assert df["Kp"][0] == 3.0
    assert df["fac"][0] == 0.5
    
    # Verify mock calls
    mock_instance.set_collection.assert_called_with("SW_OPER_FACATMS_2F")
    mock_instance.set_products.assert_called()
    mock_instance.get_between.assert_called()

def test_fetch_swarm_fac_pandas(mock_vires):
    mock_instance = mock_vires.return_value
    mock_data = MagicMock()
    
    ds = xr.Dataset(
        data_vars={
            "FAC": (("Timestamp",), [0.5]),
            "Latitude": (("Timestamp",), [50.0]),
            "Longitude": (("Timestamp",), [10.0]),
            "Radius": (("Timestamp",), [6800.0]),
        },
        coords={"Timestamp": pd.to_datetime(["2021-01-01"])}
    )
    
    mock_data.to_xarray.return_value = ds
    mock_instance.get_between.return_value = mock_data
    
    df = fetch_swarm_fac(
        start_time="2021-01-01",
        end_time="2021-01-01",
        return_type="pandas",
        add_magnetic=False
    )
    
    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert df["fac"][0] == 0.5
