###############################################################
# Author      : Peter La Follette [plafollette@lynker.com | April 2025]
# Extracts and saves a mapping between USGS gage IDs and NHDPlus COMIDs

import os
import sys
import geopandas as gpd
import pandas as pd

# Add project root to path so we can import from model_assessment
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from model_assessment.configs import path_config as cfg

# Configuration
project_root = cfg.project_root

# Input shapefile path
### see https://www.epa.gov/waterdata/nhdplus-national-data 
shapefile_path = os.path.join(
    project_root,
    "model_assessment", "retro_assessment", "get_COMIDS",
    "NHDPlusNationalData_loc", "GageLoc.shp"
)

# Output directory (same as script location)
out_dir = os.path.join(
    project_root,
    "model_assessment", "retro_assessment", "get_COMIDS"
)
os.makedirs(out_dir, exist_ok=True)

mapping_csv = os.path.join(out_dir, "gage_comid_from_nhdplus_national.csv")
full_output_csv = os.path.join(out_dir, "gage_comid_full_nhdplus_national.csv")

# Load and clean shapefile
gdf = gpd.read_file(shapefile_path)

# Drop rows with invalid FLComID (0 or NaN)
gdf_cleaned = gdf[(gdf["FLComID"] != 0) & (~gdf["FLComID"].isna())]

# Save minimal gage_id to COMID mapping
df_mapping = gdf_cleaned[["SOURCE_FEA", "FLComID"]].copy()
df_mapping = df_mapping.rename(columns={"SOURCE_FEA": "gage_id", "FLComID": "COMID"})
df_mapping.to_csv(mapping_csv, index=False)

# Save full cleaned GeoDataFrame with geometry as WKT
gdf_cleaned.to_csv(full_output_csv, index=False)

# Confirm output
print("Saved gage-to-COMID mapping to", mapping_csv)
print("Saved full cleaned dataset (with geometry) to", full_output_csv)
