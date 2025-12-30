
import pytest
import polars as pl
from datetime import datetime
from unittest.mock import patch
from facpy.geo import filter_region, add_local_time, REGIONS

@pytest.fixture
def geo_df():
    # Helper to create a dataframe covering the globe
    # Lat: -90 to 90
    # Lon: -180 to 180
    
    # Points:
    # 1. Africa: Accrah (0, 5) approx
    # 2. USA: New York (-74, 40)
    # 3. Australia: Sydney (151, -33)
    # 4. Dateline: Fiji (180, -18) -> actually 178 or -178
    
    data = {
        "latitude": [5.0, 40.0, -33.0, -18.0],
        "longitude": [0.0, -74.0, 151.0, 179.0],
        "timestamp": [
            datetime(2021, 1, 1, 12, 0), # UTC 12:00 at Lon 0 -> SLT 12
            datetime(2021, 1, 1, 12, 0), # UTC 12:00 at Lon -74 -> SLT 12 + (-74/15 = -4.93) = 7.06
            datetime(2021, 1, 1, 0, 0),  # UTC 00:00 at Lon 151 -> SLT 0 + 10.06 = 10.06
            datetime(2021, 1, 1, 12, 0)
        ]
    }
    return pl.DataFrame(data)

def test_filter_region_preset(geo_df):
    # Africa: (-20, 55, -40, 40)
    # Should include point 1 (0, 5)
    
    african = filter_region(geo_df, region="africa")
    assert african.height == 1
    assert african["latitude"][0] == 5.0
    
def test_filter_region_hemisphere(geo_df):
    north = filter_region(geo_df, hemisphere="north")
    # Points 1 (5), 2 (40) are north (>=0)
    assert north.height == 2
    
    south = filter_region(geo_df, hemisphere="south")
    # Points 3 (-33), 4 (-18) are south (<0)
    assert south.height == 2

def test_add_local_time(geo_df):
    lt_df = add_local_time(geo_df, method="slt")
    assert "local_time" in lt_df.columns
    
    # Check Point 1: Lon 0, UTC 12 -> SLT 12
    # float comparison
    assert abs(lt_df["local_time"][0] - 12.0) < 0.01
    
    # Check Point 2: Lon -74, UTC 12 -> 12 + (-4.933) = 7.066
    slt_nyc = 12 + (-74/15)
    assert abs(lt_df["local_time"][1] - slt_nyc) < 0.01
    
    # Check Point 3: Lon 151, UTC 0 -> 0 + 10.06 = 10.06
    slt_syd = 151/15
    assert abs(lt_df["local_time"][2] - slt_syd) < 0.01

def test_filter_bbox_crossing(geo_df):
    # Test dateline crossing filter
    # Add a point at -179
    df2 = pl.concat([geo_df, pl.DataFrame({
        "latitude": [10.0], "longitude": [-179.0], "timestamp": [datetime(2021,1,1)]
    })])
    
    # Box crossing dateline: 170 to -170
    # min_lon=170, max_lon=-170
    
    # filter_region expects bbox=(min, max, ...)
    # If I pass (170, -170, -90, 90)
    
    res = filter_region(df2, bbox=(170, -170, -90, 90))
    
    # Should include 179 and -179
    # Should NOT include 0, -74, 151
    
    lons = res["longitude"].to_list()
    assert 179 in lons
    assert -179 in lons
    assert 0 not in lons

def test_add_local_time_mlt(geo_df):
    # mlt requires 'radius'
    df = geo_df.with_columns(pl.lit(6800000.0).alias("radius"))
    
    with patch("facpy.geo.aacgmv2.get_aacgm_coord") as mock_get:
        mock_get.return_value = (65.0, 80.0, 13.5)
        
        lt_df = add_local_time(df, method="mlt")
        
        assert "local_time" in lt_df.columns
        assert lt_df["local_time"][0] == 13.5
        
        # Verify call (Point 1: 5.0 lat, 0.0 lon, 6800km radius, 12:00 timestamp)
        # h_km = 6800 - 6371.2 = 428.8
        mock_get.assert_any_call(5.0, 0.0, pytest.approx(428.8), datetime(2021, 1, 1, 12, 0))

def test_add_local_time_mlt_missing_cols(geo_df):
    # missing 'radius'
    with pytest.raises(ValueError, match="MLT calculation requires 'radius' column."):
        add_local_time(geo_df, method="mlt")
