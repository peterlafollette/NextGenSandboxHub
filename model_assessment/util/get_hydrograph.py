###############################################################
# Author      : Peter La Follette [plafollette@lynker.com | Apr 2025]
# Updated     : Aug 2025 — allow explicit --troute_dir and --summary
#
# Extract the hydrograph for the target nexus (by gage) from a T-Route NetCDF.
# Priority for locating outputs:
#   1) If --troute_dir is provided, use it directly.
#   2) Else, if PARTICLE_ID or NGEN_PARTICLE_ID is set, use particle path:
#         {base_dir}/out/{gage_id}/particles/p{PID}/troute
#   3) Else legacy:
#         {base_dir}/out/{gage_id}/troute
###############################################################

import os
import sys
import argparse
import xarray as xr
import pandas as pd
from pathlib import Path
import traceback

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gage_id", type=str, required=True, help="USGS gage ID to extract")
    p.add_argument("--output", type=str, required=False, help="Path to save output CSV")
    p.add_argument("--base_dir", type=str, required=False,
                   default="/Users/peterlafollette/CIROH_project/NextGenSandboxHub",
                   help="Tile/model root (only used if --troute_dir not provided)")
    p.add_argument("--troute_dir", type=str, required=False,
                   help="Direct path to directory containing T-Route NetCDF(s)")
    p.add_argument("--summary", type=str, required=True,
                   help="Path to downstream_flowpath_summary.csv")
    return p.parse_args()

def resolve_troute_dir(base_dir, gage_id, explicit_troute=None):
    if explicit_troute:
        return explicit_troute

    pid = os.environ.get("NGEN_PARTICLE_ID") or os.environ.get("PARTICLE_ID")
    if pid:
        cand = os.path.join(base_dir, "out", gage_id, "particles", f"p{pid}", "troute")
        return cand
    else:
        return os.path.join(base_dir, "out", gage_id, "troute")

def choose_netcdf(troute_dir):
    if not os.path.isdir(troute_dir):
        raise FileNotFoundError(f"troute dir does not exist: {troute_dir}")
    files = [f for f in os.listdir(troute_dir) if f.endswith(".nc")]
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in troute dir: {troute_dir}")
    # Deterministic choice: sort and take first
    files.sort()
    if len(files) > 1:
        print(f" Multiple NetCDF files found; using first after sort: {files[0]}")
    return os.path.join(troute_dir, files[0])

def get_nexus_id(gage_id, summary_csv):
    df = pd.read_csv(summary_csv, dtype=str)
    if gage_id not in df["gage_id"].values:
        raise ValueError(f"Gage ID {gage_id} not found in summary file: {summary_csv}")

    row = df[df["gage_id"] == gage_id].iloc[0]
    if "most_downstream_nexus" in df.columns:
        nexus_str = row["most_downstream_nexus"]
    elif "nexus_before_it" in df.columns:
        nexus_str = row["nexus_before_it"]
    else:
        raise ValueError("Summary file missing 'most_downstream_nexus'/'nexus_before_it' columns")

    if not isinstance(nexus_str, str) or not nexus_str.startswith("nex-"):
        raise ValueError(f"Invalid nexus ID format for gage {gage_id}: {nexus_str}")
    return int(nexus_str.replace("nex-", ""))

def extract_nexus_flow(netcdf_path, nexus_id):
    ds = xr.open_dataset(netcdf_path)
    if "flow" not in ds or "feature_id" not in ds.coords:
        raise ValueError("NetCDF missing 'flow' variable or 'feature_id' coord")
    feature_ids = ds["feature_id"].values
    if nexus_id not in feature_ids:
        raise ValueError(f"Nexus ID {nexus_id} not in feature_id list")
    idx = int((feature_ids == nexus_id).nonzero()[0][0])
    flow = ds["flow"].isel(feature_id=idx).to_series()
    df = flow.reset_index().rename(columns={"time": "current_time", "flow": "flow"})
    return df

def main():
    args = parse_args()
    gage_id = args.gage_id

    # Resolve troute directory
    troute_dir = resolve_troute_dir(args.base_dir, gage_id, args.troute_dir)

    # Resolve summary CSV (required)
    summary_csv = args.summary
    if not os.path.isfile(summary_csv):
        print(f"ERROR: summary CSV not found at: {summary_csv}")
        sys.exit(1)

    # Output path
    if args.output:
        output_csv = args.output
    else:
        pid = os.environ.get("NGEN_PARTICLE_ID") or os.environ.get("PARTICLE_ID")
        out_dir = os.path.join(args.base_dir, "postproc")  # fallback
        os.makedirs(out_dir, exist_ok=True)
        output_csv = os.path.join(out_dir, f"{gage_id}_particle_{pid}.csv" if pid else f"{gage_id}.csv")

    try:
        netcdf_path = choose_netcdf(troute_dir)
        nexus_id = get_nexus_id(gage_id, summary_csv)
        flow_df = extract_nexus_flow(netcdf_path, nexus_id)
        flow_df.to_csv(output_csv, index=False)
        print(f" Saved nexus flow time series to: {output_csv}")
    except Exception as e:
        print(f" Error while processing {gage_id}: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()


