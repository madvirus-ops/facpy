"""
Quiet time selection module.
Handles selection of quiet days based on geomagnetic indices (Kp, Dst).
"""

import datetime
from pathlib import Path
from typing import List, Optional, Union, Literal

import numpy as np
import pandas as pd
import polars as pl


def quiet_days(
    start_date: Union[str, datetime.date, datetime.datetime],
    end_date: Union[str, datetime.date, datetime.datetime],
    method: Literal["kp", "dst"] = "kp",
    top_n: int = 5,
    threshold: Optional[float] = None,
    index_file: Optional[Union[str, Path]] = None,
) -> List[datetime.date]:
    """
    Select the quietest days in a given range based on Kp or Dst index.

    Parameters
    ----------
    start_date : str or date
        Start of the period (inclusive).
    end_date : str or date
        End of the period (inclusive).
    method : {"kp", "dst"}, default "kp"
        The index to use.
        - "kp": Uses daily sum of Kp (or max Kp). Low values are quiet.
        - "dst": Uses daily minimum Dst. Values close to 0 (or positive) are quiet.
    top_n : int, default 5
        Number of quietest days to return per month? Or total?
        Currently implements total for the range if range is small, 
        or we can clarify. Let's return top_n days for the whole period.
    threshold : float, optional
        If provided, returns all days meeting the threshold (e.g. Kp_sum < 10).
        Overrides top_n if not None.
    index_file : str or Path, optional
        Path to a local file containing the index data.
        If None, attempts to download or use cached data (not fully implemented).

    Returns
    -------
    List[datetime.date]
        Sorted list of quiet dates.
    """
    
    # Normalize dates
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()
    if isinstance(start_date, datetime.datetime):
        start_date = start_date.date()
        
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).date()
    if isinstance(end_date, datetime.datetime):
        end_date = end_date.date()

    if index_file:
        df = _load_index_file(index_file, method)
    else:
        # TODO: Implement auto-download
        raise ValueError("Auto-download not implemented. Please provide 'index_file'.")

    # Filter by date range
    df = df.filter(
        (pl.col("date") >= start_date) & 
        (pl.col("date") <= end_date)
    )

    if df.height == 0:
        return []

    # Calculate daily statistic
    # Kp is usually 3-hour, Dst is 1-hour.
    # We need to aggregate to daily.
    
    # Assuming the loaded DF has "date" and "value" columns where "value" is the index
    # But often index data is high resolution.
    # Let's assume _load_index_file returns DAILY aggregates or we aggregate here?
    # Better to aggregate here if raw.
    
    # Check if df is daily
    # If duplicate dates exist, we aggregate
    
    daily_stats = df.group_by("date").agg([
        pl.col("value").sum().alias("sum"),
        pl.col("value").max().alias("max"),
        pl.col("value").min().alias("min"),
        pl.col("value").mean().alias("mean"),
        pl.col("value").count().alias("count") # valid points
    ])
    
    # Determine "quietness" metric
    if method == "kp":
        # Quiet = Low Sum Kp (or low Max Kp)
        # Using Sum Kp as primary metric
        metric = "sum"
        ascending = True
    elif method == "dst":
        # Quiet = Max Dst (closest to 0 or positive)?
        # Storms are negative Dst.
        # Quiet time is usually defined as Dst > -20 nT.
        # We want "Most quiet" -> Maximize Min(Dst) (least negative)?
        # Or Just Mean Dst close to 0?
        # Usually checking Minimum Dst of the day is sufficient to exclude storms.
        # So we want days where Min(Dst) is highest (e.g. -5 vs -100).
        metric = "min"
        ascending = False # We want higher values (closer to 0 or positive)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Remove incomplete days? (count < 8 for Kp, count < 24 for Dst)
    if method == "kp":
        daily_stats = daily_stats.filter(pl.col("count") >= 8)
    elif method == "dst":
        daily_stats = daily_stats.filter(pl.col("count") >= 24)

    # Sort
    daily_stats = daily_stats.sort(metric, descending=not ascending)

    if threshold is not None:
        if method == "kp":
            # threshold is max sum Kp allowed
            daily_stats = daily_stats.filter(pl.col(metric) <= threshold)
        elif method == "dst":
            # threshold is min Dst allowed (e.g. -20)
            daily_stats = daily_stats.filter(pl.col(metric) >= threshold)
    else:
        daily_stats = daily_stats.head(top_n)

    dates = daily_stats["date"].to_list()
    dates.sort()
    
    return dates


def _load_index_file(path: Union[str, Path], method: str) -> pl.DataFrame:
    """
    Load index data from a file.
    Supports simple CSV formats:
    - Date, Value
    - Or standard GFZ formats if we parse them (complex).
    
    For now, assumes a CSV with columns ['datetime', 'kp'] or ['datetime', 'dst'].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")

    # Read using polars with schema inference
    # Expecting 'datetime' column
    
    # If it's a standard format like WDC text, we need a parser.
    # Assuming prepared CSV for this stage.
    
    # Try reading as CSV
    try:
        df = pl.read_csv(path, try_parse_dates=True)
    except Exception:
        # Fallback to pandas for more robust parsing if needed?
        # Polars is stricter.
        df = pl.from_pandas(pd.read_csv(path, parse_dates=["datetime"]))
        
    df = df.rename({col: col.lower() for col in df.columns})
    
    if "datetime" not in df.columns:
        # Try to find a date column
        # Maybe 'date' or 'time'
        if "date" in df.columns:
            df = df.rename({"date": "datetime"})
    
    # Ensure datetime is date or datetime
    # We want to extract date for aggregation
    
    # Normalize columns
    if "datetime" in df.columns:
         df = df.with_columns(pl.col("datetime").dt.date().alias("date"))
    elif "date" in df.columns:
         # already date?
         pass
    else:
         raise ValueError("Could not find 'datetime' or 'date' column in index file.")
         
    target_col = method.lower()
    if target_col not in df.columns:
        # Check aliases
        if method == "kp" and "ap" in df.columns:
             # Maybe convert Ap to Kp? No, stick to Kp requirement.
             pass
        elif "value" in df.columns:
             df = df.rename({"value": target_col})
        else:
             raise ValueError(f"Could not find column '{target_col}' in index file. Columns: {df.columns}")

    df = df.rename({target_col: "value"})
    return df.select(["date", "value"]) # Keep only date and value (which is Kp or Dst)
