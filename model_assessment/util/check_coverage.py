import os
import pandas as pd
import sys
import yaml

# Add path to configs so we can import path_config
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "configs"))
from path_config import observed_q_root

# --- Load validation period from time_config.yaml ---
script_dir = os.path.dirname(os.path.abspath(__file__))
time_config_path = os.path.join(script_dir, "..", "configs", "time_config.yaml")

with open(time_config_path, "r") as f:
    time_config = yaml.safe_load(f)

val_start = pd.to_datetime(time_config["val_start"])
val_end = pd.to_datetime(time_config["val_end"])

# --- CONFIG ---
resampled_streamflow_dir = os.path.join(observed_q_root, "successful_sites_resampled")
thresh = 1.0  # Threshold in cubic meters per second
use_validation_period_only = True  # <<< Toggle this to True/False

# --- Initialize results ---
results = []

# --- Loop over streamflow CSVs ---
for filename in os.listdir(resampled_streamflow_dir):
    if not filename.endswith(".csv"):
        continue

    gage_id = filename[:-4]  # remove .csv
    print(f"Checking gage ID: {gage_id}")
    streamflow_file = os.path.join(resampled_streamflow_dir, filename)

    try:
        # FIX: Use the correct time column name
        time_col = "value_time"
        df = pd.read_csv(streamflow_file, parse_dates=[time_col])

        if "flow_m3_per_s" not in df.columns:
            print(f"'flow_m3_per_s' column not found in {filename}")
            continue

        if use_validation_period_only:
            if time_col not in df.columns:
                print(f"'{time_col}' column not found in {filename}")
                continue
            df = df[(df[time_col] >= val_start) & (df[time_col] <= val_end)]

        n_total = df["flow_m3_per_s"].notna().sum()
        n_above_thresh = (df["flow_m3_per_s"] > thresh).sum()
        coverage = 100 * n_above_thresh / n_total if n_total > 0 else 0

        results.append({
            "gage_id": gage_id,
            "n_values": n_total,
            "n_above_thresh": n_above_thresh,
            "coverage_percent": round(coverage, 2)
        })

    except Exception as e:
        print(f"Error processing {gage_id}: {e}")

# --- Save results ---
coverage_df = pd.DataFrame(results)
output_csv = os.path.join(script_dir, "streamflow_coverage_report.csv")
coverage_df.to_csv(output_csv, index=False)
print(f"Coverage report saved to: {output_csv}")















# ##### checks coverage for whole time series not just val period 
# import os
# import pandas as pd
# import sys

# # Add path to configs so we can import path_config
# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "configs"))
# from path_config import observed_q_root
# resampled_streamflow_dir = os.path.join(observed_q_root, "successful_sites_resampled")

# thresh = 1.0 ###the threshold, in cubic meters per second, above which we check for coverage 

# # --- Initialize results ---
# results = []

# # --- Loop over streamflow CSVs in successful_sites_resampled ---
# for filename in os.listdir(resampled_streamflow_dir):
#     if not filename.endswith(".csv"):
#         continue

#     gage_id = filename[:-4]  # remove .csv
#     print(f"Checking gage ID: {gage_id}")
#     streamflow_file = os.path.join(resampled_streamflow_dir, filename)

#     try:
#         df = pd.read_csv(streamflow_file)
#         if "flow_m3_per_s" not in df.columns:
#             print(f"'flow_m3_per_s' column not found in {filename}")
#             continue

#         # n_total = len(df) #would check for coverage over whole period instead, not just times for which data are present
#         # Count only valid (non-missing) values
#         n_total = df["flow_m3_per_s"].notna().sum()
#         n_above_thresh = (df["flow_m3_per_s"] > thresh).sum()
#         coverage = 100 * n_above_thresh / n_total if n_total > 0 else 0

#         results.append({
#             "gage_id": gage_id,
#             "n_values": n_total,
#             "n_above_thresh": n_above_thresh,
#             "coverage_percent": round(coverage, 2)
#         })

#     except Exception as e:
#         print(f"Error processing {gage_id}: {e}")

# # --- Save results ---
# coverage_df = pd.DataFrame(results)
# script_dir = os.path.dirname(os.path.abspath(__file__))
# output_csv = os.path.join(script_dir, "streamflow_coverage_report.csv")

# coverage_df.to_csv(output_csv, index=False)
# print(f"Coverage report saved to: {output_csv}")
