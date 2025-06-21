###############################################################
# Author      : Peter La Follette [plafollette@lynker.com | April 2025]
# downloads streamflow data from USGS gages

import pandas as pd
import requests
import os
from hydrotools.nwis_client.iv import IVDataService
from datetime import datetime
import sys
sys.path.append(os.path.join(os.getcwd(), "model_assessment", "configs"))
import path_config


cf_per_sec_to_m3_per_sec = 0.0283168
SECONDS_PER_HOUR = 3600

# Set time window
# start_date = '1995-10-01 00:00:00'
# end_date = '2024-09-30 23:00:00'
start_date = '2015-10-01 00:00:00'
end_date = '2015-10-02 00:00:00'
start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)

# Initialize site tracking lists
successful_sites = []
partial_data_sites = []
failed_sites = []

def load_gages_csv(gages_csv_path):
    return pd.read_csv(gages_csv_path, dtype={"gage_id": str})

def get_catchment_area_sqkm(site_id):
    url = f"https://waterservices.usgs.gov/nwis/site/?format=rdb&sites={site_id}&siteOutput=expanded"
    print(f"Fetching catchment area for site {site_id} ...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        lines = response.text.splitlines()

        # Skip comment lines and field definitions
        content_lines = [line for line in lines if line.strip() and not line.startswith("#") and not line.startswith("5s")]

        # Find headers and data lines
        header_line = None
        data_line = None
        for i, line in enumerate(content_lines):
            if line.startswith("agency_cd"):
                header_line = line
                data_line = content_lines[i + 1]
                break

        if header_line is None or data_line is None:
            print(f"   No valid metadata found for site {site_id}")
            return None

        headers = header_line.strip().split("\t")
        values = data_line.strip().split("\t")
        site_data = dict(zip(headers, values))

        if "drain_area_va" in site_data:
            val = site_data["drain_area_va"]
            if val not in ("", "na", "-999999"):
                area_mi2 = float(val)
                area_km2 = area_mi2 * 2.58999
                print(f"   Catchment area: {area_km2:.2f} km^2")
                return area_km2

        print(f"    No drainage area available in metadata for site {site_id}")
        return None

    except Exception as e:
        print(f"   Failed to fetch metadata for site {site_id}: {e}")
        return None

def download_usgs_streamflow(df_gages):
    for site in df_gages['gage_id']:
        print(f"\n Downloading data for site: {site}")

        # Step 1: Get catchment area
        area_km2 = get_catchment_area_sqkm(site)
        if area_km2 is None:
            print(f"  Skipping site {site} due to missing area.\n")
            failed_sites.append(site)
            continue
        area_m2 = area_km2 * 1_000_000  # km^2 to m^2

        # Step 2: Download streamflow
        print(f"Fetching streamflow data for site {site} ...")
        service = IVDataService(value_time_label="value_time")
        df_obs = service.get(
            sites=site,
            startDT=start_date,
            endDT=end_date
        )

        if df_obs.empty:
            print(f"    No streamflow data returned for site {site}.")
            failed_sites.append(site)
            continue
        print(f"   Retrieved {len(df_obs)} records of streamflow data.")

        if "value" not in df_obs.columns:
            print(f"   'value' column not found in streamflow data for site {site}. Skipping.")
            failed_sites.append(site)
            continue

        # Step 3: Convert and compute
        df_obs["flow_m3_per_s"] = df_obs["value"] * cf_per_sec_to_m3_per_sec
        df_obs["flow_cm_per_h"] = (df_obs["flow_m3_per_s"] * SECONDS_PER_HOUR / area_m2) * 100

        # Step 4: Check data range
        actual_start = pd.to_datetime(df_obs["value_time"].min())
        actual_end = pd.to_datetime(df_obs["value_time"].max())
        print(f"   Data coverage: {actual_start.date()} to {actual_end.date()}")

        if actual_start > start_dt or actual_end < end_dt:
            partial_data_sites.append(site)
        else:
            successful_sites.append(site)

        # Step 5: Save to CSV
        # output_filename = f"{site}.csv"
        os.makedirs(path_config.usgs_output_dir, exist_ok=True)
        # output_filename = os.path.join(path_config.observed_q_root, f"{site}.csv")
        output_filename = os.path.join(path_config.usgs_output_dir, f"{site}.csv")
        os.makedirs(path_config.usgs_output_dir, exist_ok=True)
        df_obs[["value_time", "flow_m3_per_s", "flow_cm_per_h"]].to_csv(output_filename, index=False)
        print(f"   Saved to {output_filename}")

def write_list_to_txt(filename, site_list):
    output_path = os.path.join(path_config.usgs_output_dir, filename)
    with open(output_path, "w") as f:
        for site in site_list:
            f.write(f"{site}\n")
    print(f" Wrote {len(site_list)} entries to {filename}")

# Run script
gages_list = load_gages_csv(path_config.gages_file)
print(" gages_list:")
print(gages_list)
download_usgs_streamflow(gages_list)

# Write site lists
write_list_to_txt("successful_sites.txt", successful_sites)
write_list_to_txt("partial_data_sites.txt", partial_data_sites)
write_list_to_txt("failed_sites.txt", failed_sites)

print("\n Done. Forcing exit.")
os._exit(0)
