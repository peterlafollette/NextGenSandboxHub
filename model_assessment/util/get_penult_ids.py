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

# --- Paths ---
this_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(this_dir, "..", "..", "configs", "sandbox_config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

input_dir = config["input_dir"]
output_csv_path = os.path.join(this_dir, "downstream_flowpath_summary.csv")

results = []

for gage_id in os.listdir(input_dir):
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

        # Create lookup: id -> toid
        id_to_toid = dict(zip(network_gdf["id"].astype(str), network_gdf["toid"].astype(str)))

        all_ids = set(id_to_toid.keys())
        toids = set(id_to_toid.values())
        terminal_ids = all_ids - toids

        if not terminal_ids:
            print(" No terminal flowpath found.")
            results.append({
                "gage_id": gage_id,
                "most_downstream_nexus": "[None]",
                "wbs_into_that_nexus": "[None]"
            })
            continue

        # Pick first terminal_id (if multiple, could be refined with hydroseq later)
        current_id = list(terminal_ids)[0]
        visited = set()
        last_nexus = None

        # Walk upstream
        while current_id in id_to_toid and current_id not in visited:
            visited.add(current_id)
            if current_id.startswith("nex-"):
                last_nexus = current_id
            current_id = id_to_toid.get(current_id, None)

        if last_nexus is None:
            print(" No nexus found in upstream chain.")
            results.append({
                "gage_id": gage_id,
                "most_downstream_nexus": "[None]",
                "wbs_into_that_nexus": "[None]"
            })
        else:
            # Find all WBs that flow into that nexus
            inflowing_wbs = network_gdf[network_gdf["toid"].astype(str) == last_nexus]
            wb_ids = inflowing_wbs["id"].astype(str).drop_duplicates().tolist()
            wb_ids_str = ",".join(wb_ids)

            print(f" Most downstream nexus: {last_nexus}")
            print(f" WBs into that nexus:   {wb_ids_str}")

            results.append({
                "gage_id": gage_id,
                "most_downstream_nexus": last_nexus,
                "wbs_into_that_nexus": wb_ids_str
            })

    except Exception as e:
        print(f" Error processing {gage_id}: {e}")
        results.append({
            "gage_id": gage_id,
            "most_downstream_nexus": f"[Error: {str(e)}]",
            "wbs_into_that_nexus": f"[Error: {str(e)}]"
        })

# --- Save output ---
df = pd.DataFrame(results)
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

