############################################################################################
# Author  : Ahmad Jan Khattak
# Contact : ahmad.jan.khattak@noaa.gov
# Date    : July 16, 2024
############################################################################################
###############################################################
# edits from: Peter La Follette [plafollette@lynker.com | May 2025], to allow for tiled formulations

import os, sys
import subprocess
import yaml
import argparse
from pathlib import Path
import pandas as pd

path = Path(sys.argv[0]).resolve()
sandbox_dir = path.parent


from src.python import forcing, driver, runner


def CheckSandbox_VENV():
    VENV_SANDBOX = Path.home() / ".venv_sandbox_py3.11"

    # Check if the virtual environment exists
    if not VENV_SANDBOX.exists():
        print(f"Error: NextGen virtual environment {VENV_SANDBOX} not found under home directory...")
        sys.exit(1)


    # Check if the script is running inside that required environment
    VENV_ACTIVE = Path(sys.prefix)
    if VENV_ACTIVE.resolve() != VENV_SANDBOX.resolve():
        print(f"Warning: sandbox.py is not running in the expected Python virtual environment.")
        print(f"Expected: {VENV_SANDBOX}")
        print(f"Active:   {VENV_ACTIVE}")

        sys.exit(1)


formulations_supported = [
    "NOM,CFE",
    "PET,CFE",
    "NOM,LASAM",
    "PET,LASAM",
    "NOM,CFE,PET",
    "NOM,CFE,SMP,SFT",
    "NOM,LASAM,SMP,SFT",
    "NOM,TOPMODEL",
    "BASELINE,CFE",
    "BASELINE,LAS"
]

def Sandbox(sandbox_config, calib_config):
    
    if (args.subset):
        print ("Generating geopackages...")
        subset_basin = f"Rscript {sandbox_dir}/src/R/main.R {sandbox_config}"
        status = subprocess.call(subset_basin,shell=True)

        if (status):
            sys.exit("Failed during generating geopackge(s) step...")
        else:
            print ("DONE \u2713")

    if (args.forc):
        print ("Generating forcing data...")
        process_forcing = forcing.ForcingProcessor(sandbox_config)
        status          = process_forcing.download_forcing()

        if (status):
            sys.exit("Failed during generating geopackge(s) step...")
        else:
            print ("DONE \u2713")

    if (args.conf):
        print ("Generating config files...")
        _driver = driver.Driver(sandbox_config, formulations_supported)
        status  = _driver.run()

        if (status):
            sys.exit("Failed during generating config files step...")
        else:
            print ("DONE \u2713")

            # Disable Spotlight indexing in outputs/div and troute if they exist or might be used
            import platform
            from pathlib import Path
            import glob
            import yaml

            def disable_spotlight_indexing(path_str):
                path = Path(path_str)
                if platform.system() == "Darwin":
                    path.mkdir(parents=True, exist_ok=True)
                    (path / ".metadata_never_index").touch()

            # Parse output_dir from the sandbox config file
            with open(sandbox_config, "r") as f:
                sandbox_dict = yaml.safe_load(f)
            output_base = Path(sandbox_dict["output_dir"])

            # Create postproc directory at same level as output_dir
            postproc_dir = output_base.parent / "postproc"
            postproc_dir.mkdir(parents=True, exist_ok=True)
            disable_spotlight_indexing(postproc_dir)

            # Apply to all gage output dirs created
            gage_dirs = glob.glob(str(output_base / "*" / "outputs" / "div"))
            for div_path in gage_dirs:
                disable_spotlight_indexing(div_path)

            for gage_path in output_base.glob("*"):
                if not gage_path.is_dir():
                    continue  # skip files like .DS_Store
                div_path = gage_path / "outputs" / "div"
                troute_path = gage_path / "troute"

                disable_spotlight_indexing(div_path)

                if not troute_path.exists():
                    try:
                        troute_path.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        print(f"Warning: Could not create troute dir: {troute_path} - {e}")
                disable_spotlight_indexing(troute_path)


            # Create a filtered basins_passed_custom.csv file
            import pandas as pd

            passed_basins_csv = output_base / "basins_passed.csv"
            custom_csv = output_base / "basins_passed_custom.csv"
            gage_list_path = sandbox_dir / "basin_IDs" / "basin_IDs.csv"

            if passed_basins_csv.exists() and gage_list_path.exists():
                try:
                    # Read gage IDs from provided list
                    gage_ids_df = pd.read_csv(gage_list_path, dtype=str)
                    gage_ids = set(gage_ids_df['gage_id'].astype(str).str.strip())

                    # Read basins_passed.csv
                    basins_df = pd.read_csv(passed_basins_csv, dtype=str)

                    # Determine matching column name in basins_passed.csv
                    possible_id_cols = ["STAID", "gage_id", "gageID", "id"]
                    matching_col = next((col for col in possible_id_cols if col in basins_df.columns), None)

                    if matching_col is None:
                        print(f"Warning: No matching gage ID column found in {passed_basins_csv}")
                    else:
                        filtered_df = basins_df[basins_df[matching_col].astype(str).isin(gage_ids)]
                        filtered_df.to_csv(custom_csv, index=False)
                        print(f"Filtered basins_passed_custom.csv created at {custom_csv} with {len(filtered_df)} basins.")
                except Exception as e:
                    print(f"Warning: Failed to create filtered basins_passed_custom.csv - {e}")
            else:
                if not passed_basins_csv.exists():
                    print(f"Warning: basins_passed.csv not found at {passed_basins_csv}")
                if not gage_list_path.exists():
                    print(f"Warning: basin_IDs.csv not found at {gage_list_path}")


        
    if (args.run):
        print ("Calling Runner...")

        _runner = runner.Runner(sandbox_config, calib_config, gage_id=args.gage_id)
        status  = _runner.run()

        if (status):
            sys.exit("Failed during ngen-cal execution...")
        else:
            print ("DONE \u2713")
    
    print ("**********************************")
    
    

if __name__ == "__main__":
    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-subset", action='store_true',    help="Subset basin (generate .gpkg files)")
        parser.add_argument("-forc",   action='store_true',    help="Download forcing data")
        parser.add_argument("-conf",   action='store_true',    help="Generate config files")
        parser.add_argument("-run",    action='store_true',    help="Run NextGen simulations")
        parser.add_argument("-i",      dest="sandbox_infile", type=str, required=False,  help="sandbox config file")
        parser.add_argument("-j",      dest="calib_infile",    type=str, required=False,  help="caliberation config file")
        parser.add_argument("--gage_id", type=str, required=False, help="Run model only for this gage ID")
        args = parser.parse_args()
    except SystemExit:
        print("Formulations supported:\n" + "\n".join(formulations_supported))
        sys.exit(0)

    if (args.sandbox_infile):
        if (os.path.exists(args.sandbox_infile)):
            sandbox_config = Path(args.sandbox_infile).resolve()
        else:
            print ("sandbox config file DOES NOT EXIST, provided: ", args.sandbox_infile)
            sys.exit(0)
    else:
        sandbox_config = f"{sandbox_dir}/configs/sandbox_config.yaml"

    if (args.calib_infile):
        if (os.path.exists(args.calib_infile)):
            calib_config = Path(args.calib_infile).resolve()
        else:
            print ("caliberation config file DOES NOT EXIST, provided: ", args.calib_infile)
            sys.exit(0)
    else:
        calib_config = f"{sandbox_dir}/configs/calib_config.yaml"
    
    if (len(sys.argv) < 2):
        print ("No arguments are provide")
        sys.exit(0)

    # check if expected Python virtual env exists and activated
    CheckSandbox_VENV()

    Sandbox(sandbox_config, calib_config)
