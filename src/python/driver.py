############################################################################################
# Author  : Ahmad Jan Khattak
# Contact : ahmad.jan.khattak@noaa.gov
# Date    : October 11, 2023 
############################################################################################


import os
import sys
import subprocess
import pandas as pd
import glob
import shutil
import re
import geopandas as gpd
import csv
import yaml
import multiprocessing
from functools import partial
import time
import argparse
import json
from pathlib import Path

from src.python import helper
from src.python import generate

class Driver:
    def __init__(self, infile, formulations_supported):
        self.colors = helper.colors()

        self.sandbox_config = infile
        self.formulations_supported = formulations_supported
        self.load_config()
        
    def load_config(self):
        with open(self.sandbox_config, 'r') as file:
            d = yaml.safe_load(file)

        self.sandbox_dir = d["sandbox_dir"]
        self.input_dir    = d["input_dir"]
        self.output_dir   = Path(d["output_dir"])
        
        dformul = d['formulation']
        self.ngen_dir      = dformul["ngen_dir"]
        self.formulation   = dformul['models']
        self.clean         = self.process_clean_input_param(dformul.get('clean', "none"))
        self.verbosity     = dformul.get('verbosity', 0)
        self.basins_in_par = dformul.get('basins_in_par', 1)
        self.schema_type   = dformul.get('schema_type', "noaa-owp")

        self.setup_simulation = dformul.get('setup_simulation', True)
        self.rename_existing_simulation = dformul.get('rename_existing_simulation', "")

        # Forcing block
        dforcing = d['forcings']
        
        self.forcing_time   = dforcing["forcing_time"]
        self.forcing_format = dforcing.get('forcing_format', '.nc')
        forcing_start_yr    = pd.Timestamp(self.forcing_time['start_time']).year
        forcing_end_yr      = pd.Timestamp(self.forcing_time['end_time']).year + 1
        forcing_dir         = os.path.join(self.input_dir, "{*}", f'data/forcing/{forcing_start_yr}_to_{forcing_end_yr}')
        self.forcing_dir    = dforcing.get("forcing_dir", forcing_dir)
        self.forcing_format = dforcing.get('forcing_format', '.nc')

        self.is_netcdf_forcing = True
        if self.forcing_format == '.csv':
            self.is_netcdf_forcing = False

        # Simulation block
        dsim = d['simulation']
        self.ngen_cal_type = (dsim.get('task_type', 'control')).lower()

        if self.ngen_cal_type == 'calibration' or self.ngen_cal_type == 'calibvalid':
            if "calibration_time" not in dsim or not isinstance(dsim["calibration_time"], dict):
                raise ValueError("calibration_time is not provided or is not a valid dictionary.")
            
            self.simulation_time = dsim["calibration_time"]
            #self.calib_eval_time = dsim["calib_eval_time"]
        elif self.ngen_cal_type == 'validation':
            if "validation_time" not in dsim or not isinstance(dsim["validation_time"], dict):
                raise ValueError("validation_time is not provided or is not a valid dictionary.")

            self.simulation_time = dsim["validation_time"]
            #self.valid_eval_time = dsim["valid_eval_time"]
        elif self.ngen_cal_type == 'control':
            if "simulation_time" not in dsim or not isinstance(dsim["simulation_time"], dict):
                raise ValueError("simulation_time is not provided or is not a valid dictionary.")
            
            self.simulation_time = dsim["simulation_time"]
        else:
            raise ValueError("Not a valid task_type provided: valid options are [control, calibration, validation, calibvalid, restart]")
        
    def process_clean_input_param(self, clean):
        clean_lst = []
        if isinstance(clean, str):
            clean_lst = [clean]
        elif isinstance(clean, list):
            clean_lst.extend(clean)
        return clean_lst

    def get_forcing_files(self, gpkg_dirs, is_corrected=True):
        forcing_files = []

        if self.forcing_format == ".nc":
            if "{*}" in self.forcing_dir:
                for g in gpkg_dirs:
                    forcing_dir_local = self.forcing_dir
                    fdir = Path(forcing_dir_local.replace("{*}", Path(g).name))

                    if not fdir.exists() or not fdir.is_dir():
                        print(fdir)
                        raise ValueError("Forcing directory '{fdir}' does not exist.")
                    if is_corrected:
                        forcing_file = glob.glob(f"{fdir}/*_corrected.nc")[0]
                    else:
                        nc_file = glob.glob(f"{fdir}/*.nc")
                        forcing_file = [f for f in nc_file if not "_corrected" in f][0]

                    forcing_files.append(forcing_file)
            else:
                if not Path(self.forcing_dir).exists():
                    raise ValueError("Forcing directory '{self.forcing_dir}' does not exist.")

                if not Path(self.forcing_dir).is_dir():
                    forcing_file = self.forcing_dir
                else:
                    if is_corrected:
                        forcing_file = glob.glob(f"{self.forcing_dir}/*_corrected.nc")[0]
                    else:
                        nc_file = glob.glob(f"{fdir}/*.nc")
                        forcing_file = [f for f in nc_file if not "_corrected" in f][0]

                forcing_files.append(forcing_file)
        else:
            if "{*}" in self.forcing_dir:
                for g in gpkg_dirs:
                    forcing_dir_local = self.forcing_dir
                    fdir = Path(forcing_dir_local.replace("{*}", Path(g).name))

                    if not fdir.exists():
                        raise ValueError("Forcing directory '{fdir}' does not exist.")
                    if not fdir.is_dir():
                        raise ValueError("forcing format is .csv, so '{fdir}' should point to a directory and not file.")

                    forcing_files.append(fdir)

        return forcing_files

    def generate_catchment_files(self, dirs):
        i_dir = dirs[0]
        o_dir = dirs[1]
        f_dir = dirs[2]

        o_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(o_dir)

        basin_ids = []
        num_cats = []

        if self.verbosity >= 2:
            print("***********************************")
            print("cwd: ", os.getcwd())
            print("input_dir: ", i_dir)
            print("output_dir: ", o_dir)
            print("forcing_dir: ", f_dir)

        gpkg_name = Path(glob.glob(str(i_dir / "data" / "*.gpkg"))[0]).name
        gpkg_dir = Path(glob.glob(str(i_dir / "data" / "*.gpkg"))[0])
        gpkg_id = i_dir.name

        filled_dot = '●'

        if self.verbosity >= 1:
            print(filled_dot, gpkg_name, end="")

        gpkg_dir = os.path.join(i_dir, gpkg_dir)

        helper.create_clean_dirs(output_dir=o_dir,
                                 setup_simulation=self.setup_simulation,
                                 rename_existing_simulation=self.rename_existing_simulation,
                                 clean=self.clean)

        if not self.setup_simulation:
            return

        # Call generate files
        driver_ = generate.Generate(sandbox_dir = self.sandbox_dir,
                                    gpkg_file = gpkg_dir,
                                    forcing_dir = f_dir,
                                    ngen_dir = self.ngen_dir,
                                    sim_time = self.simulation_time,
                                    formulation = self.formulation,
                                    formulation_supported = self.formulations_supported,
                                    output_dir = o_dir,
                                    forcing_format = self.forcing_format,
                                    ngen_cal_type = self.ngen_cal_type,
                                    schema = self.schema_type)

        failed = False
        if not failed:
            basin_ids.append(gpkg_id)
            x = gpd.read_file(gpkg_dir, layer="divides")
            num_cats.append(len(x["divide_id"]))

        if self.verbosity >= 1:
            result = "Passed" if not failed else "Failed"
            print(self.colors.GREEN + "  %s " % result + self.colors.END)

        return basin_ids, num_cats
        #quit()
        
    def main(self, nproc=4):
        basins_passed = os.path.join(self.output_dir, "basins_passed.csv")

        if os.path.exists(basins_passed):
            os.remove(basins_passed)

        forcing_files = self.get_forcing_files(self.gpkg_dirs)

        basin_ids = []
        num_cats = []

        pool = multiprocessing.Pool(processes=nproc)
        tuple_list = list(zip(self.gpkg_dirs, self.output_dirs, forcing_files))
        results = pool.map(self.generate_catchment_files, tuple_list)
        results = [result for result in results if result is not None]

        for result in results:
            basin_ids.extend(result[0])
            num_cats.extend(result[1])

        with open(basins_passed, 'w', newline='') as file:
            dat = zip(basin_ids, num_cats)
            writer = csv.writer(file)
            writer.writerow(['gage_id', 'num_divides'])
            writer.writerows(dat)

        pool.close()
        pool.join()

        return len(num_cats)

    def run(self):
        start_time = time.time()

        if self.verbosity >= 2:
            print(self.simulation_time)

        if self.clean[0] == "all":
            check = input("\nDo you really want to delete all except \'data\' directory? you will lose all ngen output data: ")
            if check.lower() in ["y", "yes"]:
                print("Deleting all existing simulation data except \'data\' directory.")
            elif check.lower() in ["n", "no"]:
                sys.exit("Quiting...")

        assert os.path.exists(self.output_dir)
        assert os.path.exists(self.sandbox_dir)
        assert os.path.exists(self.ngen_dir)

        if not os.path.exists(os.path.join(self.sandbox_dir, "src/python")):
            sys.exit("check `sandbox_dir`, it should be the parent directory of `src/python` directory")

        all_dirs = glob.glob(os.path.join(self.input_dir, '*/'), recursive=True)
        self.gpkg_dirs = [Path(g) for g in all_dirs if os.path.exists(os.path.join(g, 'data')) and glob.glob(os.path.join(g, 'data', '*.gpkg'))]
        assert self.gpkg_dirs, f"No .gpkg files found in the data directory under {self.input_dir}."

        
        self.output_dirs = [self.output_dir / Path(g).name for g in self.gpkg_dirs]
        success_ncats = self.main(nproc=self.basins_in_par)
        
        end_time = time.time()
        total_time = end_time - start_time

        print("================== SUMMARY ===============================")
        print("| Total time         = %s [sec], %s [min]" % (round(total_time, 4), round(total_time / 60., 4)))
        print("| Total no of basins = %s " % len(self.gpkg_dirs))
        print("| Succeeded          = %s " % success_ncats)
        print("| Failed             = %s " % (len(self.gpkg_dirs) - success_ncats))
        print("==========================================================")
