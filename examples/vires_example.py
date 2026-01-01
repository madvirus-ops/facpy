
import os
from facpy.io import fetch_swarm_fac, export_fac
import datetime

def main():
    # Note: Requires a VirES token to be configured.
    # See: https://viresclient.readthedocs.io/en/latest/installation.html#token-configuration
    
    start = datetime.datetime(2021, 1, 1, 0, 0, 0)
    end = datetime.datetime(2021, 1, 1, 1, 0, 0)
    
    print(f"Fetching Swarm-A FAC data from {start} to {end}...")
    
    try:
        # Fetch data with auxiliary variables Kp and Dst
        df = fetch_swarm_fac(
            start_time=start,
            end_time=end,
            satellite="A",
            auxiliaries=["Kp", "Dst"],
            return_type="polars"
        )
        
        print(f"Successfully fetched {len(df)} rows.")
        print(df.head())
        
        # Exporting data to different formats
        print("Exporting data to CSV and Parquet...")
        export_fac(df, "swarm_fac_data.csv", format="csv")
        export_fac(df, "swarm_fac_data.parquet", format="parquet")
        
        print("Done!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Make sure you have configured your VirES token.")

if __name__ == "__main__":
    main()
