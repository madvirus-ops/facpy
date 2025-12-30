"""
Example workflow for using facpy.
Demonstrates loading, filtering, gridding, and plotting.
"""

import matplotlib.pyplot as plt
import polars as pl
import numpy as np
from datetime import datetime
from facpy import io, geo, grid, plot, quiet, ihfac

# 1. Create dummy data (Mocking load_swarm_fac)
print("1. Loading Data...")
# Simulate Swarm track data
dates = [datetime(2021, 1, 1, 12, 0, 0) for _ in range(100)]
lats = np.linspace(-40, 40, 100)
lons = np.zeros(100)
fac_vals = np.sin(lats * np.pi / 180) * 10  # fake signal

df = pl.DataFrame({
    "timestamp": dates,
    "latitude": lats,
    "longitude": lons,
    "fac": fac_vals
})

print(f"   Loaded {df.height} records.")

# 2. Select Quiet Days
# Mock index file would be needed, or we just trust the logic.
# dates = quiet.quiet_days(2021, method="kp", top_n=5)
# print(f"   Selected quiet dates: {dates}")
# For now, we skip filtering this dummy df by date as it only has 1 date.

# 3. Geographic filtering
print("2. Filtering for Africa...")
df_africa = geo.filter_region(df, region="africa")
print(f"   Records in Africa: {df_africa.height}")

# 4. Add Local Time
print("3. Calculating Local Time...")
df_africa = geo.add_local_time(df_africa)
print(f"   SLT range: {df_africa['local_time'].min():.2f} - {df_africa['local_time'].max():.2f}")

# 5. Grid
print("4. Gridding data...")
ds_grid = grid.grid_fac(
    df_africa, 
    resolution=(2.0, 2.0),
    statistic="mean"
)
print("   Grid created.")
print(ds_grid)

# 6. Plot
print("5. Plotting map...")
try:
    ax = plot.fac_map(
        ds_grid,
        title="Simulated FAC over Africa",
        show=False # Don't block
    )
    # plt.savefig("example_map.png")
    print("   Plot generated.")
except Exception as e:
    print(f"   Plotting failed (expected in headless env): {e}")

print("\nWorkflow completed successfully.")
