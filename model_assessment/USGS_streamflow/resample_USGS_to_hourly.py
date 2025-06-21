###############################################################
# Author      : Peter La Follette [plafollette@lynker.com | April 2025]
# Some raw USGS streamflow data has a different temporal resolution than hourly. This esures that data in processed streamflow time series has an hourly resolution.
# note that this will not interpolate between missing data -- this code will preserve gaps in observed streamflow. 

import os
import pandas as pd
import sys
sys.path.append(os.path.join(os.getcwd(), "model_assessment", "configs"))
import path_config


# === Paths ===
input_dir = path_config.usgs_output_dir
output_dir = input_dir + "_resampled"

# === Directory Setup ===
os.makedirs(output_dir, exist_ok=True)

# === Resampling Loop ===
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            # Read and parse timestamps
            df = pd.read_csv(input_path, parse_dates=["value_time"])
            df.set_index("value_time", inplace=True)

            # Keep only rows on the exact hour
            df = df[df.index.minute == 0]

            if df.empty:
                print(f" Skipping {filename}: no hourly timestamps found")
                continue

            # Reindex to full hourly range from min to max
            full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="1H")
            df = df.reindex(full_range)

            # Save without interpolation
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'value_time'}, inplace=True)
            df.to_csv(output_path, index=False)
            print(f" Resampled: {filename} ")

        except Exception as e:
            print(f" Failed to process {filename}: {e}")

