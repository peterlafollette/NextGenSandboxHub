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

def _parse_hf_numeric_id(value):
    value = str(value).strip()
    if "-" not in value:
        return int(value)
    return int(value.split("-", 1)[1])

def _parse_inflow_ids(row):
    wbs = str(row.get("wbs_into_that_nexus", "") or "").strip()
    if not wbs or wbs.startswith("["):
        return []
    ids = []
    for token in wbs.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            ids.append(_parse_hf_numeric_id(token))
        except ValueError:
            print(f" Skipping unparsable inflow WB id from summary: {token}")
    return ids

def get_target_ids(gage_id, summary_csv):
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
    return {
        "nexus_id": _parse_hf_numeric_id(nexus_str),
        "inflow_feature_ids": _parse_inflow_ids(row),
    }

def extract_nexus_flow(netcdf_path, target_ids):
    ds = xr.open_dataset(netcdf_path)
    if "flow" not in ds or "feature_id" not in ds.coords:
        raise ValueError("NetCDF missing 'flow' variable or 'feature_id' coord")
    feature_ids = ds["feature_id"].values
    nexus_id = target_ids["nexus_id"]
    if nexus_id not in feature_ids:
        inflow_ids = [fid for fid in target_ids.get("inflow_feature_ids", []) if fid in feature_ids]
        if not inflow_ids:
            raise ValueError(
                f"Nexus ID {nexus_id} not in feature_id list, and none of the "
                f"summary inflow IDs {target_ids.get('inflow_feature_ids', [])} were found"
            )
        print(
            f" Nexus ID {nexus_id} not in t-route feature IDs; "
            f"summing inflow WB features: {inflow_ids}"
        )
        flow = ds["flow"].sel(feature_id=inflow_ids).sum(dim="feature_id").to_series()
    else:
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
        target_ids = get_target_ids(gage_id, summary_csv)
        flow_df = extract_nexus_flow(netcdf_path, target_ids)
        flow_df.to_csv(output_csv, index=False)
        print(f" Saved nexus flow time series to: {output_csv}")
    except Exception as e:
        print(f" Error while processing {gage_id}: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
































###works but only if out is in the same directroy as NextGenSandboxHub because the path of downstream_flowpath_summary.csv is hardcoded
# ###############################################################
# # Author      : Peter La Follette [plafollette@lynker.com | April 2025]
# # extracts the hydrograph for the nexus we are interested in for objective function calculation
# ### Updated: extract only the flow time series for the correct nexus from a NetCDF file
# import os
# import sys
# import argparse
# import xarray as xr
# import pandas as pd
# from pathlib import Path
# import traceback

# # === PARSE ARGUMENTS ===
# parser = argparse.ArgumentParser()
# parser.add_argument("--gage_id", type=str, help="USGS gage ID to extract")
# parser.add_argument("--output", type=str, help="Optional path to save output CSV")
# parser.add_argument("--base_dir", type=str, default="/Users/peterlafollette/CIROH_project/NextGenSandboxHub", help="Root dir of the model (tile-specific)")

# args = parser.parse_args()

# # === CONFIGURATION ===
# gage_id = args.gage_id or os.environ.get("GAGE_ID") or "08103900"
# base_dir = args.base_dir
# troute_dir = os.path.join(base_dir, "out", gage_id, "troute")

# # UPDATED: summary CSV is now in model_assessment
# # summary_csv = os.path.join(base_dir, "model_assessment", "util", "downstream_flowpath_summary.csv")

# parent_dir = Path(base_dir).resolve().parent
# summary_csv = os.path.join(parent_dir, "NextGenSandboxHub", "model_assessment", "util", "downstream_flowpath_summary.csv")

# if not os.path.exists(summary_csv):
#     print(f"ERROR: Required CSV not found at:\n  {summary_csv}")
#     print("Please run get_penult_ids.py to generate it before continuing.")
#     sys.exit(1)

# # Handle output path
# if args.output:
#     output_csv = args.output
# else:
#     particle_id = os.environ.get("PARTICLE_ID")
#     output_dir = os.path.join(base_dir, "postproc")
#     os.makedirs(output_dir, exist_ok=True)
#     output_csv = os.path.join(output_dir, f"{gage_id}_particle_{particle_id}.csv" if particle_id else f"{gage_id}.csv")

# # === Step 1: Get the nexus feature ID for this gage ===
# def get_nexus_id(gage_id, summary_csv):
#     df = pd.read_csv(summary_csv, dtype=str)

#     if gage_id not in df["gage_id"].values:
#         raise ValueError(f" Gage ID {gage_id} not found in summary file.")

#     row = df[df["gage_id"] == gage_id].iloc[0]

#     # Determine which column to use
#     if "most_downstream_nexus" in df.columns:
#         nexus_str = row["most_downstream_nexus"]
#     elif "nexus_before_it" in df.columns:
#         nexus_str = row["nexus_before_it"]
#     else:
#         raise ValueError(f" Neither 'most_downstream_nexus' nor 'nexus_before_it' found in summary file.")

#     # Validate format
#     if not isinstance(nexus_str, str) or not nexus_str.startswith("nex-"):
#         raise ValueError(f" Invalid nexus ID format for gage {gage_id}: {nexus_str}")

#     return int(nexus_str.replace("nex-", ""))


# # === Step 2: Extract flow time series from NetCDF for specific nexus ===
# def extract_nexus_flow(netcdf_path, nexus_id):
#     ds = xr.open_dataset(netcdf_path)

#     if "flow" not in ds or "feature_id" not in ds.coords:
#         raise ValueError(" Expected variables 'flow' and coordinate 'feature_id' not found in NetCDF file.")

#     # Get index of the matching feature_id
#     feature_ids = ds["feature_id"].values
#     if nexus_id not in feature_ids:
#         raise ValueError(f" Nexus ID {nexus_id} not found in NetCDF file's feature_id list.")

#     idx = int((feature_ids == nexus_id).nonzero()[0][0])

#     flow = ds["flow"].isel(feature_id=idx).to_series()
#     df = flow.reset_index().rename(columns={"time": "current_time", "flow": "flow"})
#     return df

# # === MAIN EXECUTION ===
# try:
#     netcdf_files = [f for f in os.listdir(troute_dir) if f.endswith(".nc")]
#     if not netcdf_files:
#         raise FileNotFoundError(f"No NetCDF file found in {troute_dir}")
#     elif len(netcdf_files) > 1:
#         print(f" Multiple NetCDF files found in {troute_dir}, using the first one: {netcdf_files[0]}")

#     netcdf_path = os.path.join(troute_dir, netcdf_files[0])
#     nexus_id = get_nexus_id(gage_id, summary_csv)
#     flow_df = extract_nexus_flow(netcdf_path, nexus_id)
#     flow_df.to_csv(output_csv, index=False)
#     print(f" Saved nexus flow time series to: {output_csv}")

# except Exception as e:
#     print(f" Error while processing {gage_id}: {e}")
#     traceback.print_exc()
#     sys.exit(1)
