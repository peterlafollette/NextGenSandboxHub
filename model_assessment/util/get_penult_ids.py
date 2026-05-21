###############################################################
# Author      : Peter La Follette [plafollette@lynker.com | June 2025]
# Description : For each gage/catchment, find the most downstream nexus
#               using the 'network' layer of the geopackage.
#               This walks upstream to identify the last 'nex-*' 
#               and collects WBs that flow into it.
# # When running a NextGen formulation with a geopackage that has multiple divides and t-route, I beleive that it is best to use the most downstream nexus's output. This script identifies that for each catchment you want to model.
# # Further, I have found so far that the subsetting with hydrofabric version 2.1.1. seems to offer somewhat more accurate total catchment boundaries than 2.2's subsetting.

import geopandas as gpd
import pandas as pd
import os
import yaml
import argparse
import re
import sqlite3
from collections import defaultdict

# --- Paths ---
this_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(this_dir, "..", "..", "configs", "sandbox_config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

input_dir = os.path.expandvars(config["input_dir"])
output_csv_path = os.environ.get(
    "DOWNSTREAM_FLOWPATH_SUMMARY",
    os.path.join(this_dir, "downstream_flowpath_summary.csv")
)

parser = argparse.ArgumentParser(description="Build downstream nexus summary for HF geopackages.")
parser.add_argument("--gage-id", default=os.environ.get("NGEN_GAGE_ID") or os.environ.get("GAGE_ID"))
parser.add_argument("--input-dir", default=os.environ.get("CIROH_INPUT_DIR") or input_dir)
parser.add_argument(
    "--hf-gpkg",
    help="Optional full hydrofabric geopackage. When provided, use flowpath-attributes gage_nex_id.",
)
parser.add_argument(
    "--gage-file",
    default=os.environ.get("BASIN_CSV"),
    help="CSV containing gage_id values to process with --hf-gpkg.",
)
args = parser.parse_args()

results = []

input_dir = os.path.expandvars(args.input_dir)

def _read_gage_ids_from_csv(path):
    df = pd.read_csv(os.path.expandvars(path), dtype=str)
    col = "gage_id" if "gage_id" in df.columns else df.columns[0]
    return [
        str(g).strip().zfill(8)
        for g in df[col].dropna().tolist()
        if str(g).strip()
    ]

def _empty_row(gage_id, reason):
    return {
        "gage_id": gage_id,
        "most_downstream_wb": reason,
        "nexus_before_it": reason,
        "most_downstream_nexus": reason,
        "wbs_into_that_nexus": reason,
    }

def _split_gage_tokens(value):
    if value is None or pd.isna(value):
        return []
    return [
        token.strip()
        for token in re.split(r"[,;|\s]+", str(value))
        if token.strip()
    ]

def _wb_id(row):
    value = row.get("id")
    if value is not None and str(value).startswith("wb-"):
        return str(value)
    value = row.get("link")
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip()
    return value if value.startswith("wb-") else f"wb-{value}"

def _nexus_id(value):
    value = str(value).strip()
    if value.startswith("nex-"):
        return value
    if value.startswith("tnx-"):
        return f"nex-{value.split('-', 1)[1]}"
    return f"nex-{value}"

def _select_gage_row(rows, gage_id):
    exact = [
        row for row in rows
        if gage_id in _split_gage_tokens(row.get("gage"))
    ]
    if not exact:
        return None
    exact_with_nexus = [
        row for row in exact
        if row.get("gage_nex_id") is not None and str(row.get("gage_nex_id")).startswith("nex-")
    ]
    return (exact_with_nexus or exact)[0]

def _generate_from_full_hf(hf_gpkg, gage_ids):
    print(f"Reading flowpath-attributes from: {hf_gpkg}")
    con = sqlite3.connect(os.path.expandvars(hf_gpkg))
    con.row_factory = sqlite3.Row
    rows = [
        dict(row)
        for row in con.execute(
            'SELECT id, link, toid, gage, gage_nex_id FROM "flowpath-attributes"'
        )
    ]
    con.close()

    by_toid = defaultdict(list)
    gaged_rows = []
    for row in rows:
        toid = row.get("toid")
        if toid:
            by_toid[str(toid)].append(row)
        if row.get("gage"):
            gaged_rows.append(row)

    for gage_id in gage_ids:
        print(f"\nProcessing gage ID: {gage_id}")
        selected = _select_gage_row(gaged_rows, gage_id)
        if selected is None:
            print(" No flowpath-attributes row found for gage.")
            results.append(_empty_row(gage_id, "[None]"))
            continue

        nexus_before_it = _nexus_id(selected.get("gage_nex_id") or selected.get("toid"))
        most_downstream_wb = _wb_id(selected)
        wb_ids = [
            _wb_id(row)
            for row in by_toid.get(nexus_before_it, [])
            if _wb_id(row)
        ]
        if not wb_ids and most_downstream_wb:
            wb_ids = [most_downstream_wb]

        wb_ids_str = ",".join(dict.fromkeys(wb_ids))
        print(f" Most downstream WB:    {most_downstream_wb}")
        print(f" Nexus before it:       {nexus_before_it}")
        print(f" WBs into that nexus:   {wb_ids_str}")

        results.append({
            "gage_id": gage_id,
            "most_downstream_wb": most_downstream_wb,
            "nexus_before_it": nexus_before_it,
            "most_downstream_nexus": nexus_before_it,
            "wbs_into_that_nexus": wb_ids_str,
        })

if args.hf_gpkg:
    if args.gage_id:
        gage_ids = [str(args.gage_id).strip().zfill(8)]
    elif args.gage_file:
        gage_ids = _read_gage_ids_from_csv(args.gage_file)
    else:
        raise ValueError("--hf-gpkg requires either --gage-id or --gage-file")
    _generate_from_full_hf(args.hf_gpkg, gage_ids)

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"\nSaved results to: {output_csv_path}")
    raise SystemExit(0)

gage_ids = [args.gage_id] if args.gage_id else os.listdir(input_dir)

def _unique_network_rows(network_gdf):
    keep_cols = [
        c for c in [
            "id", "toid", "poi_id", "hl_uri", "hydroseq",
            "tot_drainage_areasqkm", "areasqkm"
        ]
        if c in network_gdf.columns
    ]
    return network_gdf[keep_cols].drop_duplicates(subset=["id", "toid"]).copy()

def _select_nexus_for_gage(network_gdf, gage_id):
    rows = _unique_network_rows(network_gdf)
    if "toid" not in rows.columns or "id" not in rows.columns:
        raise ValueError("network layer missing required id/toid columns")

    rows["id"] = rows["id"].astype(str)
    rows["toid"] = rows["toid"].astype(str)

    selected = None
    if "hl_uri" in rows.columns:
        hl = rows["hl_uri"].fillna("").astype(str)
        gage_rows = rows[hl.str.contains(str(gage_id), regex=False)]
        gage_rows = gage_rows[gage_rows["toid"].str.startswith("nex-")]
        if not gage_rows.empty:
            if "tot_drainage_areasqkm" in gage_rows.columns:
                scored = (
                    gage_rows.assign(_area=pd.to_numeric(gage_rows["tot_drainage_areasqkm"], errors="coerce"))
                    .groupby("toid", as_index=False)["_area"].max()
                    .sort_values("_area", ascending=False)
                )
                selected = scored.iloc[0]["toid"]
            else:
                selected = gage_rows.iloc[0]["toid"]

    if selected is None and "poi_id" in rows.columns:
        poi_rows = rows[
            rows["poi_id"].notna()
            & (rows["poi_id"].astype(str).str.strip() != "")
            & rows["toid"].str.startswith("nex-")
        ]
        if not poi_rows.empty:
            if "hydroseq" in poi_rows.columns:
                poi_rows = poi_rows.assign(_hydroseq=pd.to_numeric(poi_rows["hydroseq"], errors="coerce"))
                poi_rows = poi_rows.sort_values("_hydroseq", ascending=True, na_position="last")
            selected = poi_rows.iloc[0]["toid"]

    if selected is None:
        candidates = rows[rows["toid"].str.startswith("nex-")]
        if candidates.empty:
            return None, None, []
        if "hydroseq" in candidates.columns:
            candidates = candidates.assign(_hydroseq=pd.to_numeric(candidates["hydroseq"], errors="coerce"))
            candidates = candidates.sort_values("_hydroseq", ascending=True, na_position="last")
        selected = candidates.iloc[0]["toid"]

    inflowing = rows[rows["toid"] == selected].copy()
    if inflowing.empty:
        return None, selected, []

    if "tot_drainage_areasqkm" in inflowing.columns:
        inflowing = inflowing.assign(_area=pd.to_numeric(inflowing["tot_drainage_areasqkm"], errors="coerce"))
        most_downstream_wb = inflowing.sort_values("_area", ascending=False, na_position="last").iloc[0]["id"]
    else:
        most_downstream_wb = inflowing.iloc[0]["id"]
    wb_ids = inflowing["id"].astype(str).drop_duplicates().tolist()
    return most_downstream_wb, selected, wb_ids

for gage_id in gage_ids:
    gage_path = os.path.join(input_dir, gage_id)

    if not os.path.isdir(gage_path) or gage_id in ["dem", "failed_cats", ".DS_Store"]:
        continue

    print(f"\nProcessing gage ID: {gage_id}")
    gpkg_file = os.path.join(gage_path, "data", f"gage_{gage_id}.gpkg")

    if not os.path.exists(gpkg_file):
        print(f" GPKG not found for {gage_id}. Skipping.")
        continue

    try:
        network_gdf = gpd.read_file(gpkg_file, layer="network")
        most_downstream_wb, nexus_before_it, wb_ids = _select_nexus_for_gage(network_gdf, gage_id)

        if nexus_before_it is None:
            print(" No downstream nexus found.")
            results.append(_empty_row(gage_id, "[None]"))
            continue

        wb_ids_str = ",".join(wb_ids)
        print(f" Most downstream WB:    {most_downstream_wb}")
        print(f" Nexus before it:       {nexus_before_it}")
        print(f" WBs into that nexus:   {wb_ids_str}")

        results.append({
            "gage_id": gage_id,
            "most_downstream_wb": most_downstream_wb,
            "nexus_before_it": nexus_before_it,
            "most_downstream_nexus": nexus_before_it,
            "wbs_into_that_nexus": wb_ids_str,
        })

    except Exception as e:
        print(f" Error processing {gage_id}: {e}")
        results.append(_empty_row(gage_id, f"[Error: {str(e)}]"))

# --- Save output ---
df = pd.DataFrame(results)
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
df.to_csv(output_csv_path, index=False)
print(f"\nSaved results to: {output_csv_path}")















# ###############################################################
# # Author      : Peter La Follette [plafollette@lynker.com | April 2025]
# # When running a NextGen formulation with a geopackage that has multiple divides and t-route, I beleive that it is best to use the most downstream nexus's output. This script identifies that for each catchment you want to model.
# # Further, I have found so far that the subsetting with hydrofabric version 2.1.1. seems to offer somewhat more accurate total catchment boundaries than 2.2's subsetting.

# import geopandas as gpd
# import pandas as pd
# import os
# import yaml

# # --- Paths ---
# this_dir = os.path.dirname(os.path.abspath(__file__))
# config_path = os.path.join(this_dir, "..", "..", "configs", "sandbox_config.yaml")
# with open(config_path, "r") as f:
#     config = yaml.safe_load(f)
# input_dir = config["input_dir"]

# output_csv_path = os.path.join(this_dir, "downstream_flowpath_summary.csv")

# results = []


# for gage_id in os.listdir(input_dir):
#     gage_path = os.path.join(input_dir, gage_id)

#     if not os.path.isdir(gage_path) or gage_id in ["dem", "failed_cats", ".DS_Store"]:
#         continue

#     print(f"\n Processing gage ID: {gage_id}")

#     gpkg_file = os.path.join(gage_path, "data", f"gage_{gage_id}.gpkg")
#     if not os.path.exists(gpkg_file):
#         print(f" GPKG not found for {gage_id}. Skipping.")
#         continue

#     try:
#         # Load network layer
#         network_gdf = gpd.read_file(gpkg_file, layer="network")

#         # Step 1: Find the most downstream WB (its `toid` is not in the set of all `id`s)
#         all_ids = set(network_gdf["id"])
#         terminal_rows = network_gdf[~network_gdf["toid"].isin(all_ids)].drop_duplicates(subset="id")

#         if terminal_rows.empty:
#             print(" No terminal flowpath found.")
#             results.append({
#                 "gage_id": gage_id,
#                 "most_downstream_wb": "[None]",
#                 "nexus_before_it": "[None]",
#                 "wbs_into_that_nexus": "[None]"
#             })
#             continue

#         # Use hydroseq if available
#         if "hydroseq" in terminal_rows.columns:
#             most_downstream_row = terminal_rows.sort_values("hydroseq", ascending=False).iloc[0]
#         else:
#             most_downstream_row = terminal_rows.iloc[0]

#         most_downstream_wb = most_downstream_row["id"]

#         # Step 2: Find the nexus that leads into this flowpath
#         upstream_rows = network_gdf[network_gdf["toid"] == most_downstream_wb]
#         if upstream_rows.empty:
#             print(" Could not find nexus leading into downstream WB.")
#             results.append({
#                 "gage_id": gage_id,
#                 "most_downstream_wb": most_downstream_wb,
#                 "nexus_before_it": "[Not found]",
#                 "wbs_into_that_nexus": "[None]"
#             })
#             continue

#         # All upstream rows should have the same `toid`, which is the downstream WB
#         # Their `id`s are the nexuses
#         nexus_before_it = upstream_rows.iloc[0]["id"]  # this is the nexus flowing into the WB

#         # Step 3: Find all WBs that flow into that nexus
#         inflowing_wbs = network_gdf[network_gdf["toid"] == nexus_before_it]
#         wb_ids = inflowing_wbs["id"].drop_duplicates().tolist()
#         wb_ids_str = ",".join(wb_ids)

#         print(f"   Gage {gage_id}")
#         print(f"   Most downstream WB: {most_downstream_wb}")
#         print(f"   Nexus before it:    {nexus_before_it}")
#         print(f"   WBs into that nexus:{wb_ids_str}")

#         results.append({
#             "gage_id": gage_id,
#             "most_downstream_wb": most_downstream_wb,
#             "nexus_before_it": nexus_before_it,
#             "wbs_into_that_nexus": wb_ids_str
#         })

#     except Exception as e:
#         print(f" Error processing {gage_id}: {e}")
#         results.append({
#             "gage_id": gage_id,
#             "most_downstream_wb": "[Error]",
#             "nexus_before_it": "[Error]",
#             "wbs_into_that_nexus": f"[Error: {str(e)}]"
#         })

# # --- Save output ---
# df = pd.DataFrame(results)
# df.to_csv(output_csv_path, index=False)
# print(f"\n Saved results to: {output_csv_path}")
