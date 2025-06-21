###############################################################
# Author      : Peter La Follette [plafollette@lynker.com | June 2025]
# This script reads the .csv at basins_csv_path, determines which of the USGS gages in this .csv have data coverage >30% for both the calibration and validation periods specified in configs/time_config.yaml, 
# and then overwrites the original .csv with only the sites that had sufficient coverage. 

import os
import pandas as pd
from datetime import datetime
import yaml

# Base directory
base_dir = "/Users/peterlafollette/CIROH_project/NextGenSandboxHub/model_assessment"

# Paths
streamflow_dir = os.path.join(base_dir, "USGS_streamflow/successful_sites_resampled")
time_config_path = os.path.join(base_dir, "configs/time_config.yaml")
coverage_output_csv = os.path.join(base_dir, "USGS_streamflow/coverage_summary.csv")
basins_csv_path = "/Users/peterlafollette/CIROH_project/NextGenSandboxHub/out/basins_passed_custom.csv"

# Load time config
with open(time_config_path, 'r') as f:
    time_config = yaml.safe_load(f)

cal_start = pd.to_datetime(time_config["cal_start"])
cal_end = pd.to_datetime(time_config["cal_end"])
val_start = pd.to_datetime(time_config["val_start"])
val_end = pd.to_datetime(time_config["val_end"])

cal_hours = int((cal_end - cal_start).total_seconds() / 3600) + 1
val_hours = int((val_end - val_start).total_seconds() / 3600) + 1

# Initialize summary list
summary = []

# Process each CSV file
for filename in os.listdir(streamflow_dir):
    if not filename.endswith(".csv"):
        continue

    gage_id = filename.replace(".csv", "")
    file_path = os.path.join(streamflow_dir, filename)

    try:
        df = pd.read_csv(file_path, parse_dates=["value_time"])
    except Exception as e:
        print(f"Failed to read {filename}: {e}")
        continue

    df = df.set_index("value_time")

    cal_data = df.loc[cal_start:cal_end]
    val_data = df.loc[val_start:val_end]

    cal_coverage = len(cal_data) / cal_hours
    val_coverage = len(val_data) / val_hours

    summary.append({
        "gage_id": gage_id,
        "cal_coverage_percent": round(cal_coverage * 100, 2),
        "val_coverage_percent": round(val_coverage * 100, 2),
        "cal_ok": cal_coverage > 0.3,
        "val_ok": val_coverage > 0.3,
    })

# Create summary DataFrame
summary_df = pd.DataFrame(summary)
summary_df.to_csv(coverage_output_csv, index=False)
print(f"\nSaved coverage summary to: {coverage_output_csv}")

# Identify gages that passed both thresholds
passing_gages = set(summary_df[(summary_df["cal_ok"]) & (summary_df["val_ok"])]["gage_id"])

# Load and filter basins_passed_custom.csv
if os.path.exists(basins_csv_path):
    basins_df = pd.read_csv(basins_csv_path, dtype={"gage_id": str})
    original_count = len(basins_df)
    basins_df_filtered = basins_df[basins_df["gage_id"].isin(passing_gages)]
    filtered_count = len(basins_df_filtered)

    # Overwrite the original CSV
    basins_df_filtered.to_csv(basins_csv_path, index=False)
    print(f"\nUpdated {basins_csv_path}:")
    print(f"  - Original entries: {original_count}")
    print(f"  - Remaining after coverage filter: {filtered_count}")
else:
    print(f"\nERROR: {basins_csv_path} does not exist.")
