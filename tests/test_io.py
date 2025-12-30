
import pytest
from unittest.mock import MagicMock, patch
import polars as pl
import pandas as pd
from pathlib import Path
import datetime
import numpy as np
from facpy.io import load_swarm_fac, _read_cdf, _read_netcdf

@pytest.fixture
def mock_cdf():
    with patch("facpy.io.cdflib.CDF") as mock:
        yield mock

@pytest.fixture
def mock_xarray():
    with patch("facpy.io.xr.open_dataset") as mock:
        yield mock

def test_load_swarm_fac_cdf(mock_cdf):
    # Setup mock CDF
    mock_instance = mock_cdf.return_value
    
    # Mock return values for varget
    # Timestamp, Latitude, Longitude, Radius, FAC
    
    # Using datetime objects for timestamp conversion mock
    # In reality cdflib returns numbers, but we mocked the call effectively
    
    times = [datetime.datetime(2021, 1, 1, 12, 0, 0)]
    
    def side_effect(var_name):
        if var_name == "Timestamp":
            return np.array([123456789]) # dummy epoch
        elif var_name == "Latitude":
            return np.array([50.0])
        elif var_name == "Longitude":
            return np.array([10.0])
        elif var_name == "Radius":
            return np.array([6800.0])
        elif var_name == "FAC":
            return np.array([0.5])
        return None

    mock_instance.varget.side_effect = side_effect
    
    # Mock varinq for timestamp type check
    mock_instance.varinq.return_value = {'Data_Type_Description': 'CDF_TIME_TT2000'}
    
    # Mock cdfepoch conversion
    with patch("facpy.io.cdflib.cdfepoch.to_datetime", return_value=np.array(times)) as mock_time:
         with patch("facpy.io.Path.exists", return_value=True):
            df = load_swarm_fac("dummy.cdf")
    
    assert isinstance(df, pl.DataFrame)
    assert df.height == 1
    assert "timestamp" in df.columns
    assert "fac" in df.columns
    assert df["fac"][0] == 0.5

def test_load_swarm_fac_netcdf(mock_xarray):
    # Setup mock ds
    mock_ds = MagicMock()
    
    # mock to_pandas
    data = {
        "Timestamp": [datetime.datetime(2021, 1, 1)],
        "Latitude": [50.0],
        "Longitude": [10.0],
        "Radius": [6800.0],
        "FAC": [0.5]
    }
    pandas_df = pd.DataFrame(data)
    
    # Constructing the mock chain: ds[vars].to_pandas() -> df
    # It's a bit tricky to mock exact item access on MagicMock easily for slice
    # So we'll mock internal _read_netcdf mostly or ensure the code path is hit
    
    # Simpler: mock read_netcdf directly if we trust the logic, 
    # but let's try to mock the library call.
    
    mock_xarray.return_value = mock_ds
    mock_ds.__getitem__.return_value.to_pandas.return_value = pandas_df
    mock_ds.__contains__.side_effect = lambda x: x in data # simulate containment
    
    with patch("facpy.io.Path.exists", return_value=True):
        df = load_swarm_fac("dummy.nc")
    assert isinstance(df, pl.DataFrame)
    assert df.height == 1
    
def test_clean_data():
    # Test valid and invalid data
    df = pl.DataFrame({
        "Timestamp": [1, 2],
        "Latitude": [50.0, 91.0], # 91 is invalid
        "Longitude": [10.0, 10.0],
        "Radius": [6800.0, 6800.0],
        "FAC": [0.5, 1e10] # 1e10 is fill value
    })
    
    # We need to expose _clean_data or test via load if public, 
    # but _clean_data is private.
    # However, load_swarm_fac calls it.
    
    # Let's import it directly for testing
    from facpy.io import _clean_data
    
    cleaned = _clean_data(df)
    
    assert cleaned.height == 1
    assert cleaned["latitude"][0] == 50.0
    # The second row should be filtered out because of Lat=91 and FAC=1e10
