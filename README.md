# facpy: Field-Aligned Current Python Toolkit

[![PyPI version](https://badge.fury.io/py/facpy.svg)](https://pypi.org/project/facpy/)
[![Python Versions](https://img.shields.io/pypi/pyversions/facpy.svg)](https://pypi.org/project/facpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**facpy** is a research-grade Python package designed for standardized, fast, and reproducible analysis of Swarm satellite Field-Aligned Currents (FAC). It is optimized for quiet-time studies, regional analysis (e.g., Africa), and interhemispheric comparisons.

## 🚀 Key Features

*   **Data Loading**: Efficient loading of Swarm Level 2 FAC data (CDF and NetCDF formats) into Polars DataFrames.
*   **Quiet Time Selection**: Automatic selection of quietest days based on Kp (sum/max) or Dst (min) geomagnetic indices.
*   **Geospacial Tools**:
    *   Region-based filtering (presets for Africa, Europe, Polar Caps, etc.).
    *   **Magnetic Local Time (MLT)** and Solar Local Time (SLT) calculation.
    *   **Altitude-Adjusted Corrected Geomagnetic (AACGM)** coordinates (mlat, mlon).
    *   Hemisphere separation.
*   **Gridding**: Fast aggregation of point data into regular 2D (Lat/Lon) or 3D (Lat/Lon/LT) grids using vectorization.
*   **Interhemispheric Analysis (IHFAC)**: Tools to compare Northern and Southern hemisphere currents (Difference, Ratio) with automatic coordinate alignment.
*   **Visualization**: Publication-ready map generation using `cartopy`.

## 📦 Installation

**facpy** requires Python 3.9+.

```bash
# Install from PyPI
pip install facpy

# Or install from source
git clone https://github.com/madvirus-ops/facpy
cd facpy
pip install .

# Install with development dependencies
pip install ".[dev]"
```

## ⚡ Quick Start

Here is a complete workflow example demonstrating loading, filtering, gridding, and mapping.

```python
import facpy
from facpy import io, quiet, geo, grid, plot
import polars as pl

# 1. Load Data
# Supports single file or list of files (CDF/NetCDF)
df = io.load_swarm_fac("SW_OPER_FAC_A_20210101.cdf")

# 2. Select Quiet Days
# Get the 5 quietest days in Jan 2021 based on Kp index
quiet_dates = quiet.quiet_days(
    start_date="2021-01-01", 
    end_date="2021-01-31", 
    method="kp", 
    top_n=5,
    index_file="kp_index.txt" # Path to your index file
)

# Filter dataframe
df = df.filter(pl.col("timestamp").dt.date().is_in(quiet_dates))

# 3. Filter Region & Add Magnetic Local Time
# Focus on Africa and calculate Magnetic Local Time (MLT)
df_africa = geo.filter_region(df, region="africa")
df_africa = geo.add_local_time(df_africa, method="mlt")

# 4. Grid the Data
# Create a 2°x2° grid of Mean FAC values
ds_grid = grid.grid_fac(
    df_africa, 
    resolution=(2.0, 2.0), 
    statistic="mean"
)

# 5. Plot
# Generate a map using built-in Cartopy plotter
plot.fac_map(
    ds_grid, 
    title="Quiet Time Mean FAC - Africa", 
    projection="platecarree"
)
```

## 📚 Module Overview

### `facpy.io`
Handles file I/O.
*   `load_swarm_fac()`: Reads data, handles fill values, normalizes column names, and **automatically appends magnetic coordinates (mlat, mlon, mlt)** using `aacgmv2`.

### `facpy.quiet`
Geomagnetic activity selection.
*   `quiet_days()`: Returns dates of low activity defined by Kp or Dst.

### `facpy.geo`
Coordinate and spatial tools.
*   `filter_region()`: Spatial subsetting.
*   `add_local_time()`: Computes SLT or **MLT** (using `aacgmv2`) from satellite coordinates and timestamp.

### `facpy.grid`
Aggregation logic.
*   `grid_fac()`: Converts track data to `xarray.Dataset` grids. Supports multiple statistics (mean, median, std, count).

### `facpy.ihfac`
Interhemispheric analysis.
*   `compare()`: Aligns South hemisphere data to North coordinates and computes difference or ratio maps.

### `facpy.plot`
Visualization.
*   `fac_map()`: Wrapper around Cartopy for quick, consistent FAC maps.

## 🧪 Testing

Run the test suite to ensure everything is working correctly:

```bash
pytest tests/
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
