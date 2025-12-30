
import pytest
import polars as pl
import datetime
from unittest.mock import patch, MagicMock
from facpy.io import _add_magnetic_coords, load_swarm_fac

def test_add_magnetic_coords_basic():
    # Create dummy data
    df = pl.DataFrame({
        "timestamp": [datetime.datetime(2021, 1, 1, 12, 0, 0)],
        "latitude": [60.0],
        "longitude": [10.0],
        "radius": [6800000.0], # 6800 km
        "fac": [0.1]
    })
    
    # Mock aacgmv2.get_aacgm_coord
    with patch("facpy.io.aacgmv2.get_aacgm_coord") as mock_get:
        mock_get.return_value = (65.0, 80.0, 13.5)
        
        result = _add_magnetic_coords(df)
        
        assert "mlat" in result.columns
        assert "mlon" in result.columns
        assert "mlt" in result.columns
        assert result["mlat"][0] == 65.0
        assert result["mlon"][0] == 80.0
        assert result["mlt"][0] == 13.5
        
        # Verify call arguments
        # h_km = (6800000 / 1000) - 6371.2 = 6800 - 6371.2 = 428.8
        mock_get.assert_called_once_with(60.0, 10.0, pytest.approx(428.8), datetime.datetime(2021, 1, 1, 12, 0, 0))

def test_add_magnetic_coords_empty():
    df = pl.DataFrame(schema={"timestamp": pl.Datetime, "latitude": pl.Float64, "longitude": pl.Float64, "radius": pl.Float64, "fac": pl.Float64})
    result = _add_magnetic_coords(df)
    assert result.height == 0
    assert "mlat" in result.columns

def test_add_magnetic_coords_exception():
    df = pl.DataFrame({
        "timestamp": [datetime.datetime(2021, 1, 1)],
        "latitude": [60.0],
        "longitude": [10.0],
        "radius": [6800000.0],
        "fac": [0.1]
    })
    
    with patch("facpy.io.aacgmv2.get_aacgm_coord", side_effect=Exception("error")):
        result = _add_magnetic_coords(df)
        assert result["mlat"][0] is None
