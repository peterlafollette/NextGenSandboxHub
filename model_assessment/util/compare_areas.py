###############################################################
# Author      : Peter La Follette [plafollette@lynker.com | April 2025]
# For some USGS gages, hydrofabric subsetting can yield a total catchment area that is significantly different than the reported area for the catchment according to the USGS.
# In the event that the mismatch is too large, perhaps consider using another catchment.
# Left in hardcoded paths.


import geopandas as gpd
import pandas as pd
import os

# --- Paths ---
### using hardcoded paths for now 
input_dir = "/Users/peterlafollette/CIROH_project/in"
# usgs_csv_path = "/Users/peterlafollette/NextGenSandboxHub/gages_arid.csv"
usgs_csv_path = "/Users/peterlafollette/Downloads/basinchar_and_report_sept_2011/spreadsheets-in-csv-format/conterm_bound_qa.txt" ### see https://www.sciencebase.gov/catalog/item/631405bbd34e36012efa304a ("GAGES-II: Geospatial Attributes of Gages for Evaluating Streamflow")
output_txt_path = "/Users/peterlafollette/CIROH_project/NextGenSandboxHub/downstream_wbs.txt"
output_csv_path = "/Users/peterlafollette/CIROH_project/NextGenSandboxHub/usable_catchments.csv"
successfully_downloaded_csv_path = "/Users/peterlafollette/CIROH_project/NextGenSandboxHub/model_assessment/USGS_streamflow/successful_sites"
usable_catchments = []

# --- Load USGS CSV ---
usgs_df = pd.read_csv(usgs_csv_path, dtype={"STAID": str})

# --- Set mismatch tolerance ---
threshold_percent = 30

# --- Prepare output file ---
with open(output_txt_path, "w") as out_file:
    out_file.write("GAGE_ID: MOST_DOWNSTREAM_FLOWPATH\n")
    out_file.write("--------------------------------\n")

# --- Loop over gage folders ---
count = 0
count_within_tol = 0
max_percent_diff = 0
mpd_gage = 0

for gage_id in os.listdir(input_dir):
    gage_path = os.path.join(input_dir, gage_id)

    if not os.path.isdir(gage_path) or gage_id in ["dem", "failed_cats"]:
        continue

    print(f"\n Checking gage ID: {gage_id}")

    # Try to get USGS area if available
    row = usgs_df[usgs_df["STAID"] == gage_id]
    has_usgs_area = not row.empty
    if has_usgs_area:
        usgs_area = float(row["DRAIN_SQKM"].values[0])
    else:
        print(f"  Gage ID {gage_id} not found in CSV. Will skip area comparison.")

    # Path to geopackage
    gpkg_file = os.path.join(gage_path, "data", f"gage_{gage_id}.gpkg")
    if not os.path.exists(gpkg_file):
        print(f"  GPKG file not found for gage {gage_id}. Skipping.")
        continue

    try:
        # --- Read divides layer and compute area ---
        divides_gdf = gpd.read_file(gpkg_file, layer="divides")
        if not divides_gdf.crs or not divides_gdf.crs.is_projected:
            divides_gdf = divides_gdf.to_crs(epsg=5070)
        gpkg_area = divides_gdf.geometry.area.sum() / 1e6

        if has_usgs_area:
            diff = abs(usgs_area - gpkg_area)
            pct_diff = (diff / usgs_area) * 100

            print(f"  USGS area:  {usgs_area:.2f} sq km")
            print(f"  GPKG area:  {gpkg_area:.2f} sq km")
            print(f"  Difference: {diff:.2f} sq km ({pct_diff:.2f}%)")

            count = count + 1

            if pct_diff > threshold_percent:
                print("   Area mismatch exceeds threshold.")
                if pct_diff>max_percent_diff:
                    max_percent_diff = pct_diff
                    mpd_gage = gage_id
            else:
                print("   Area within tolerance.")
                count_within_tol = count_within_tol + 1
                usable_catchments.append(gage_id)
        else:
            print(f"  GPKG area:  {gpkg_area:.2f} sq km")

        # --- Read network layer and find most downstream flowpath ---
        network_gdf = gpd.read_file(gpkg_file, layer="network")
        all_ids = set(network_gdf["id"])
        downstream_rows = network_gdf[~network_gdf["toid"].isin(all_ids)].drop_duplicates(subset="id")

        with open(output_txt_path, "a") as out_file:
            if downstream_rows.empty:
                print("   Could not identify a downstream flowpath.")
                out_file.write(f"{gage_id}: [No downstream flowpath found]\n")
            else:
                if "hydroseq" in downstream_rows.columns:
                    most_downstream = downstream_rows.sort_values("hydroseq", ascending=False).iloc[0]
                else:
                    most_downstream = downstream_rows.iloc[0]

                downstream_id = most_downstream['id']
                print("   Most downstream flowpath:")
                print(f"    ID: {downstream_id} -> toid: {most_downstream['toid']}")
                out_file.write(f"{gage_id}: {downstream_id}\n")

    except Exception as e:
        print(f" Error processing {gage_id}: {e}")
        with open(output_txt_path, "a") as out_file:
            out_file.write(f"{gage_id}: [Error: {str(e)}]\n")

# pd.DataFrame(usable_catchments, columns=["gage_id"]).to_csv(output_csv_path, index=False)

print("count:")
print(count)
print("count that passed:")
print(count_within_tol)
print("ratio:")
print(count_within_tol/count)
print("max percent diff:")
print(max_percent_diff)
print("at gage:")
print(mpd_gage)


print("usable_catchments df")
print(usable_catchments)

print("ids in successfully downloaded folder")
usable_catchments_final = []
for gage_id in os.listdir(successfully_downloaded_csv_path):
    if (str(gage_id)[0:-4] in usable_catchments):
        usable_catchments_final.append(str(gage_id)[0:-4])

print("usable_catchments_final")
print(usable_catchments_final)
print("len(usable_catchments_final)")
print(len(usable_catchments_final))

pd.DataFrame(usable_catchments_final, columns=["gage_id"]).to_csv(output_csv_path, index=False)











# import os
# import pandas as pd

# # Load usable gage IDs (excluding header)
# with open("usable_catchments.csv") as f:
#     next(f)  # skip header
#     usable_ids = set(line.strip() for line in f)

# # Make sure all gage IDs are treated as-is (not padded)
# usable_catchments_final = set(usable_ids)

# # Always preserve these folders
# always_keep = {"dem", "failed_cats"}

# # Directory to clean
# in_dir = "in"

# # Dry run toggle
# dry_run = False

# # Track stats
# to_delete = []

# for item in os.listdir(in_dir):
#     if item not in usable_catchments_final and item not in always_keep:
#         to_delete.append(item)
#         print(f"[Dry run] Would delete: {os.path.join(in_dir, item)}")

# print(f"\nTotal folders flagged for deletion: {len(to_delete)}")
# print(f"Expected to keep: {len(usable_catchments_final)} + {len(always_keep)} = {len(usable_catchments_final) + len(always_keep)}")
# print(f"Remaining should be: {len(os.listdir(in_dir)) - len(to_delete)} (if dry_run=False)")

# # Optional actual deletion
# if not dry_run:
#     for item in to_delete:
#         path = os.path.join(in_dir, item)
#         if os.path.isdir(path):
#             os.system(f"rm -rf '{path}'")




import os
import pandas as pd

# Paths
in_dir = input_dir
csv_path = output_csv_path

# 1. Get list of gage IDs from folder names
all_folders = os.listdir(in_dir)
folder_ids = sorted([
    name for name in all_folders
    if os.path.isdir(os.path.join(in_dir, name)) and name not in ['dem', 'failed_cats', '.DS_Store']
])

# 2. Load usable gage IDs from CSV
usable_df = pd.read_csv(csv_path, dtype=str)
usable_ids = sorted(usable_df['gage_id'].astype(str).tolist())

# 3. Compare
in_not_csv = sorted(set(folder_ids) - set(usable_ids))
csv_not_in = sorted(set(usable_ids) - set(folder_ids))

# 4. Print results
print(f"Total folders: {len(folder_ids)}")
print(f"Total usable gage IDs: {len(usable_ids)}")
print(f"Gage IDs in folder but not in CSV ({len(in_not_csv)}):\n{in_not_csv}")
print(f"Gage IDs in CSV but not in folder ({len(csv_not_in)}):\n{csv_not_in}")






