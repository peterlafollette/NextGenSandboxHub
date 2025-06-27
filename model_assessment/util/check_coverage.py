import os
import pandas as pd
import sys

# Add path to configs so we can import path_config
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "configs"))
from path_config import observed_q_root
resampled_streamflow_dir = os.path.join(observed_q_root, "successful_sites_resampled")

# --- Initialize results ---
results = []

# --- Loop over streamflow CSVs in successful_sites_resampled ---
for filename in os.listdir(resampled_streamflow_dir):
    if not filename.endswith(".csv"):
        continue

    gage_id = filename[:-4]  # remove .csv
    print(f"Checking gage ID: {gage_id}")
    streamflow_file = os.path.join(resampled_streamflow_dir, filename)

    try:
        df = pd.read_csv(streamflow_file)
        if "flow_m3_per_s" not in df.columns:
            print(f"'flow_m3_per_s' column not found in {filename}")
            continue

        # n_total = len(df) #would check for coverage over whole period instead, not just times for which data are present
        # Count only valid (non-missing) values
        n_total = df["flow_m3_per_s"].notna().sum()
        n_above_1cms = (df["flow_m3_per_s"] > 1.0).sum()
        coverage = 100 * n_above_1cms / n_total if n_total > 0 else 0

        results.append({
            "gage_id": gage_id,
            "n_values": n_total,
            "n_above_1cms": n_above_1cms,
            "coverage_percent": round(coverage, 2)
        })

    except Exception as e:
        print(f"Error processing {gage_id}: {e}")

# --- Save results ---
coverage_df = pd.DataFrame(results)
script_dir = os.path.dirname(os.path.abspath(__file__))
output_csv = os.path.join(script_dir, "streamflow_coverage_report.csv")

coverage_df.to_csv(output_csv, index=False)
print(f"Coverage report saved to: {output_csv}")
