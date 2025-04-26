############################################################################################
# Author  : Ahmad Jan Khattak
# Contact : ahmad.jan.khattak@noaa.gov
# Date    : July 16, 2024
############################################################################################

import os, sys
import subprocess
import yaml
import argparse
from pathlib import Path

path = Path(sys.argv[0]).resolve()
workflow_dir = path.parent


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
    "NON,LASAM",
    "PET,LASAM",
    "NOM,CFE,PET",
    "NOM,CFE,SMP,SFT",
    "NOM,LASAM,SMP,SFT",
    "NOM,TOPMODEL",
    "BASELINE,CFE",
    "BASELINE,LAS"
]

def Sandbox(workflow_config, calib_config):
    
    if (args.subset):
        print ("Generating geopackages...")
        subset_basin = f"Rscript {workflow_dir}/src/R/main.R {workflow_config}"
        status = subprocess.call(subset_basin,shell=True)

        if (status):
            sys.exit("Failed during generating geopackge(s) step...")
        else:
            print ("DONE \u2713")

    if (args.forc):
        print ("Generating forcing data...")
        process_forcing = forcing.ForcingProcessor(workflow_config)
        status          = process_forcing.download_forcing()

        if (status):
            sys.exit("Failed during generating geopackge(s) step...")
        else:
            print ("DONE \u2713")

    if (args.conf):
        print ("Generating config files...")
        _driver = driver.Driver(workflow_config, formulations_supported)
        status  = _driver.run()

        if (status):
            sys.exit("Failed during generating config files step...")
        else:
            print ("DONE \u2713")
        
    if (args.run):
        print ("Calling Runner...")

        _runner = runner.Runner(workflow_config, calib_config)
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
        parser.add_argument("-i",      dest="workflow_infile", type=str, required=False,  help="workflow config file")
        parser.add_argument("-j",      dest="calib_infile",    type=str, required=False,  help="caliberation config file")
        args = parser.parse_args()
    except SystemExit:
        print("Formulations supported:\n" + "\n".join(formulations_supported))
        sys.exit(0)

    if (args.workflow_infile):
        if (os.path.exists(args.workflow_infile)):
            workflow_config = Path(args.workflow_infile).resolve()
        else:
            print ("workflow config file DOES NOT EXIST, provided: ", args.workflow_infile)
            sys.exit(0)
    else:
        workflow_config = f"{workflow_dir}/configs/workflow_config.yaml"

    if (args.calib_infile):
        if (os.path.exists(args.calib_infile)):
            calib_config = Path(args.calib_infile).resolve()
        else:
            print ("caliberation config file DOES NOT EXIST, provided: ", args.calib_infile)
            sys.exit(0)
    else:
        calib_config = f"{workflow_dir}/configs/calib_config.yaml"
    
    if (len(sys.argv) < 2):
        print ("No arguments are provide")
        sys.exit(0)

    # check if expected Python virtual env exists and activated
    CheckSandbox_VENV()

    Sandbox(workflow_config, calib_config)
