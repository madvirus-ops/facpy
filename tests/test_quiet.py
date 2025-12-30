
import pytest
import datetime
import polars as pl
from pathlib import Path
from facpy.quiet import quiet_days, _load_index_file

# Create a dummy index file fixture
@pytest.fixture
def index_file(tmp_path):
    # Create 10 days of data
    # Date, Kp
    # Kp sum for days:
    # 2021-01-01: Sum=10
    # 2021-01-02: Sum=50 (Storm)
    # 2021-01-03: Sum=5 (Quite)
    
    # We need 8 entries per day for Kp (3-hr)
    dates = []
    kps = []
    
    # Day 1: Moderate
    for i in range(8):
        dates.append(datetime.datetime(2021, 1, 1, i*3))
        kps.append(1.25) # Sum = 10
        
    # Day 2: Storm
    for i in range(8):
        dates.append(datetime.datetime(2021, 1, 2, i*3))
        kps.append(6.25) # Sum = 50
        
    # Day 3: Quiet
    for i in range(8):
        dates.append(datetime.datetime(2021, 1, 3, i*3))
        kps.append(0.625) # Sum = 5
        
    df = pl.DataFrame({
        "datetime": dates,
        "kp": kps
    })
    
    p = tmp_path / "kp_index.csv"
    df.write_csv(p)
    return p

def test_quiet_days_kp(index_file):
    start = datetime.date(2021, 1, 1)
    end = datetime.date(2021, 1, 3)
    
    # Get top 2 quietest
    # Should be Jan 3 (sum 5) then Jan 1 (sum 10)
    days = quiet_days(start, end, method="kp", top_n=2, index_file=index_file)
    
    assert len(days) == 2
    assert days[0] == datetime.date(2021, 1, 1) # sorted
    assert days[1] == datetime.date(2021, 1, 3)
    
    # Check strict threshold
    days_thresh = quiet_days(start, end, method="kp", threshold=8.0, index_file=index_file)
    # Only Jan 3 matches (sum 5 < 8)
    assert len(days_thresh) == 1
    assert days_thresh[0] == datetime.date(2021, 1, 3)

def test_load_index_loader(index_file):
    df = _load_index_file(index_file, "kp")
    assert "date" in df.columns
    assert "value" in df.columns
    assert df.height == 24
