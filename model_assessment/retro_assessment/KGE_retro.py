###############################################################
# Author      : Peter La Follette [plafollette@lynker.com | April 2025]
# This computes the KGE for the NWM version 3 retrospective simulations 

import os
import sys
import pandas as pd
import numpy as np
import yaml
from hydroeval import kge

# Add project root to path so we can import from model_assessment
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from model_assessment.configs import path_config as cfg

# === Load time config ===
with open("model_assessment/configs/time_config.yaml", "r") as f:
    time_cfg = yaml.safe_load(f)

val_start = pd.Timestamp(time_cfg["val_start"])
val_end   = pd.Timestamp(time_cfg["val_end"])

# === Derived paths ===
project_root = cfg.project_root
sim_dir = os.path.join(project_root, "retro_q_sims") ###update with directory with retrospective simulations, named based on COMID
###My approach to getting retrospective simulations was to download the net CDF files from here https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com/index.html#CONUS/ and then extract the data I needed.
###however, given that I wanted time series from 2010 to 2024, this ended up being a download for 8 TB of data. I recommend using the zarr format, the download size should be much smaller.
obs_dir = os.path.join(project_root, "model_assessment", "USGS_streamflow", "successful_sites_resampled")
map_path = os.path.join(
    project_root,
    "model_assessment", "retro_assessment", "get_COMIDS",
    "gage_comid_from_nhdplus_national.csv"
)

passed_path = cfg.gages_file

# Load COMID to gage mapping
mapping_df = pd.read_csv(map_path, dtype=str)

# if ONLY_USE_PASSED_GAGES:
passed_df = pd.read_csv(passed_path, dtype=str)
passed_gages = set(passed_df['gage_id'])
mapping_df = mapping_df[mapping_df['gage_id'].isin(passed_gages)]
# else:
# observed_gages = {
#     fname.replace(".csv", "") 
#     for fname in os.listdir(obs_dir) if fname.endswith(".csv")
# }
# mapping_df = mapping_df[mapping_df['gage_id'].isin(observed_gages)]

results = []

for _, row in mapping_df.iterrows():
    gage_id = row['gage_id']
    comid = row['COMID']

    sim_file = os.path.join(sim_dir, f"{comid}.csv")
    obs_file = os.path.join(obs_dir, f"{gage_id}.csv")

    if not os.path.exists(sim_file):
        print(f" Missing simulation file for COMID {comid}")
        continue
    if not os.path.exists(obs_file):
        print(f" Missing observed file for gage {gage_id}")
        continue

    kge_linear = np.nan
    kge_log = np.nan
    combined = pd.DataFrame()

    try:
        print(f"\n Processing gage {gage_id} | COMID {comid}")
        print(f"    Simulation file: {sim_file}")
        print(f"    Observation file: {obs_file}")

        # Read simulation data
        sim_df = pd.read_csv(sim_file)
        sim_df['time'] = pd.to_datetime(sim_df['time'], format="%Y%m%d%H%M")
        sim_df.set_index('time', inplace=True)
        sim_df = sim_df.rename(columns={'streamflow': 'sim_flow'})
        sim_df = sim_df.sort_index()

        # Read observed data
        obs_df = pd.read_csv(obs_file, parse_dates=['value_time'])
        obs_df.set_index('value_time', inplace=True)
        obs_df = obs_df.rename(columns={'flow_m3_per_s': 'obs_flow'})
        obs_hourly = obs_df['obs_flow'].to_frame()

        # Filter by validation period
        sim_df = sim_df[val_start:val_end]
        obs_hourly = obs_hourly[val_start:val_end]

        # Round timestamps to hour for consistent alignment
        sim_df.index = sim_df.index.round("h")
        obs_hourly.index = obs_hourly.index.round("h")

        print(f"    sim points after filtering: {len(sim_df)}")
        print(f"    obs points after filtering: {len(obs_hourly)}")
        overlap = sim_df.index.intersection(obs_hourly.index)
        print(f"    overlapping timestamps: {len(overlap)}")

        # Merge and drop NaNs
        combined = pd.merge(sim_df, obs_hourly, left_index=True, right_index=True, how="inner")
        combined = combined.dropna(subset=["sim_flow", "obs_flow"])

        if len(combined) == 0:
            print("     No valid data after dropping NaNs.")
        else:
            sim_vals = combined['sim_flow'].values
            obs_vals = combined['obs_flow'].values

            # Debug stats
            print("     Simulation flow summary:")
            print(sim_vals[:5], "...")
            print(f"       Mean: {np.mean(sim_vals):.3f}, Std: {np.std(sim_vals):.3f}, Min: {np.min(sim_vals):.3e}, Max: {np.max(sim_vals):.3e}")

            print("     Observation flow summary:")
            print(obs_vals[:5], "...")
            print(f"       Mean: {np.mean(obs_vals):.3f}, Std: {np.std(obs_vals):.3f}, Min: {np.min(obs_vals):.3e}, Max: {np.max(obs_vals):.3e}")

            # KGE calculation
            kge_linear = kge(sim_vals.reshape(-1, 1), obs_vals.reshape(-1, 1))[0][0]

            sim_vals_log = np.log10(np.clip(sim_vals, 1e-10, None))
            obs_vals_log = np.log10(np.clip(obs_vals, 1e-10, None))
            kge_log = kge(sim_vals_log.reshape(-1, 1), obs_vals_log.reshape(-1, 1))[0][0]

            print(f"     KGE: {kge_linear:.3f} | log-KGE: {kge_log:.3f} | merged points: {len(combined)}")

    except Exception as e:
        print(f" Exception for gage {gage_id} / COMID {comid}: {e}")

    results.append({
        "gage_id": gage_id,
        "COMID": comid,
        "KGE": kge_linear,
        "log_KGE": kge_log,
        "n_points": len(combined)
    })

# Output
results_df = pd.DataFrame(results)
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "kge_results.csv")
results_df.to_csv(output_path, index=False)
print("\nDone! Results written to", output_path)

