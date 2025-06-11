############################################################################################
# Author  : Ahmad Jan
# Contact : ahmad.jan@noaa.gov
# Date    : September 28, 2023
# minor edits from Peter La Follette (plafollette@lynker.com) May 2025
############################################################################################

import os
import sys
import argparse
import re
import copy
import glob
import json
import subprocess
import pandas as pd
import geopandas as gpd
import numpy as np
import fiona
import yaml
import platform
import math
from pathlib import Path

try:
    from src.python import schema
except:
    import schema

os_name = platform.system()

class ConfigurationGenerator:
    def __init__(self, sandbox_dir, gpkg_file, forcing_dir, output_dir,
                 ngen_dir, formulation, simulation_time,
                 verbosity, ngen_cal_type, schema_type = None):
        self.sandbox_dir = sandbox_dir
        self.gpkg_file = gpkg_file
        self.forcing_dir = forcing_dir
        self.output_dir = output_dir
        self.ngen_dir = ngen_dir
        self.formulation = formulation
        self.simulation_time = simulation_time
        self.verbosity = verbosity
        #self.json_dir = json_dir
        #self.sim_output_dir = sim_output_dir
        self.ngen_cal_type = ngen_cal_type
        self.schema_type = schema_type

        if "CFE" in self.formulation:
            with open(os.path.join(self.sandbox_dir, "configs/basefiles", "custom.yaml"), 'r') as file:
                dcfe = yaml.safe_load(file)['models']['CFE']

            self.surface_runoff_scheme = dcfe['surface_runoff_scheme']

        self.soil_params_NWM_dir = os.path.join(self.ngen_dir,"extern/noah-owp-modular/noah-owp-modular/parameters")
                
        self.gdf, self.catids = self.read_gpkg_file()
        
        self.soil_class_NWM = self.get_soil_class_NWM()

        
    def get_soil_class_NWM(self):
        nom_soil_file = os.path.join(self.soil_params_NWM_dir, "SOILPARM.TBL")
        header = ['index', 'BB', 'DRYSMC', 'F11', 'MAXSMC', 'REFSMC', 'SATPSI', 'SATDK', 'SATDW', 'WLTSMC', 'QTZ', 'BVIC', 'AXAJ', 'BXAJ', 'XXAJ', 'BDVIC', 'BBVIC', 'GDVIC', 'ISLTYP']
        df = pd.read_table(nom_soil_file, delimiter=',', index_col=0, skiprows=3, nrows=19, names=header)
        return df

    def read_gpkg_file(self):
        try:
            gdf_soil = gpd.read_file(self.gpkg_file, layer='divide-attributes')
        except:
            try:
                gdf_soil = gpd.read_file(self.gpkg_file, layer='model-attributes')
            except:
                print("layer 'divide-attributes or model-attributes does not exist!'")
                sys.exit(1)

        gdf_soil.set_index("divide_id", inplace=True)
        gdf_div = gpd.read_file(self.gpkg_file, layer='divides')
        gdf_div = gdf_div.to_crs("EPSG:4326")

        layers = fiona.listlayers(self.gpkg_file)
        flowpath_layer = [layer for layer in layers if 'flowpath' in layer][0]

        if self.verbosity >= 3:
            print("Geopackage layers: ", layers)
            print("\n")

        params = schema.get_schema_model_attributes(gdf_soil)

        gdf_soil['soil_b'] = gdf_soil[params['soil_b']].fillna(16)
        gdf_soil['soil_dksat'] = gdf_soil[params['soil_dksat']].fillna(0.00000338)
        gdf_soil['soil_psisat'] = gdf_soil[params['soil_psisat']].fillna(0.355)
        gdf_soil['soil_smcmax'] = gdf_soil[params['soil_smcmax']].fillna(0.439)
        gdf_soil['soil_smcwlt'] = gdf_soil[params['soil_smcwlt']].fillna(0.066)
        gdf_soil['gw_Zmax'] = gdf_soil[params['gw_Zmax']].fillna(0.01)
        gdf_soil['gw_Coeff'] = gdf_soil[params['gw_Coeff']].fillna(1.8e-05)
        gdf_soil['gw_Expon'] = gdf_soil[params['gw_Expon']].fillna(6.0)
        gdf_soil['slope_1km'] = gdf_soil[params['slope_1km']].fillna(1.0)
        gdf_soil['ISLTYP'] = gdf_soil[params['ISLTYP']].fillna(1).astype(int)
        gdf_soil['IVGTYP'] = gdf_soil[params['IVGTYP']].fillna(1).astype(int)
        gdf_soil['gw_Zmax'] = gdf_soil['gw_Zmax'] / 1000.0
        gdf_soil['gw_Coeff'] = gdf_soil['gw_Coeff'] * 3600 / (7.337700 * 1000 * 1000)
        gdf_soil['elevation_mean'] = gdf_soil[params['elevation_mean']].fillna(4)
        gdf_soil['slope_mean'] = gdf_soil[params['slope_mean']].fillna(1.0)
        gdf_soil['aspect_mean'] = gdf_soil[params['aspect_mean']].fillna(1.0)

        if self.schema_type == 'dangermond':
            gdf_soil['elevation_mean'] = gdf_soil['elevation_mean'] / 100.0

        if 'refkdt' in gdf_soil:
            gdf_soil['soil_refkdt'] = gdf_soil[params['soil_refkdt']].fillna(3.0)
        else:
            gdf_soil['soil_refkdt'] = 3.0

        gdf = gpd.GeoDataFrame(data={'geometry': gdf_div['geometry'].values}, index=gdf_soil.index)
        gdf['soil_b'] = gdf_soil['soil_b'].copy()
        gdf['soil_satdk'] = gdf_soil['soil_dksat'].copy()
        gdf['soil_satpsi'] = gdf_soil['soil_psisat'].copy()
        gdf['soil_slop'] = gdf_soil['slope_1km'].copy()
        gdf['soil_smcmax'] = gdf_soil['soil_smcmax'].copy()
        gdf['soil_wltsmc'] = gdf_soil['soil_smcwlt'].copy()
        gdf['soil_refkdt'] = gdf_soil['soil_refkdt'].copy()
        gdf['max_gw_storage'] = gdf_soil['gw_Zmax'].copy()
        gdf['Cgw'] = gdf_soil['gw_Coeff'].copy()
        gdf['gw_expon'] = gdf_soil['gw_Expon'].copy()
        gdf['ISLTYP'] = gdf_soil['ISLTYP'].copy()
        gdf['IVGTYP'] = gdf_soil['IVGTYP'].copy()
        gdf['elevation_mean'] = gdf_soil['elevation_mean'].copy()
        gdf['slope_mean'] = gdf_soil['slope_mean'].copy()
        gdf['aspect_mean'] = gdf_soil['aspect_mean'].copy()

        mask = gdf['soil_b'].gt(0.0)
        min_value = gdf['soil_b'][mask].min()
        mask = gdf['soil_b'].le(0.0)
        gdf.loc[mask, 'soil_b'] = min_value

        mask = gdf['elevation_mean'].le(0.0)
        gdf.loc[mask, 'elevation_mean'] = 1.0

        if "topmodel" in self.formulation:
            gdf['twi'] = gdf_soil[params['twi']]
            gdf['width_dist'] = gdf_soil[params['width_dist']]


        gdf['giuh'] = gdf_soil[params['giuh']]
        gdf['N_nash_surface'] = gdf_soil[params['N_nash_surface']]
        gdf['K_nash_surface'] = gdf_soil[params['K_nash_surface']]
        
        df_cats = gpd.read_file(self.gpkg_file, layer='divides')
        catids = [int(re.findall('[0-9]+', s)[0]) for s in df_cats['divide_id']]

        return gdf, catids

    def write_nom_input_files(self):
        nom_dir = os.path.join(self.output_dir,"configs/noahowp")
        self.create_directory(nom_dir)
        
        # copy NOM params dir 
        str_sub ="cp -r "+ self.soil_params_NWM_dir + " %s"%nom_dir
        out=subprocess.call(str_sub,shell=True)
        
        nom_basefile = os.path.join(self.sandbox_dir, "configs/basefiles/config_noahowp.input")

        if not os.path.exists(nom_basefile):
            sys.exit(f"Sample NoahOWP config file does not exist, provided is {nom_basefile}")

        # Read infile line by line
        with open(nom_basefile, 'r') as infile:
            lines = infile.readlines()

        start_time = pd.Timestamp(self.simulation_time['start_time']).strftime("%Y%m%d%H%M")
        end_time = pd.Timestamp(self.simulation_time['end_time']).strftime("%Y%m%d%H%M")

        for catID in self.catids:
            cat_name = 'cat-' + str(catID)
            centroid_x = str(self.gdf['geometry'][cat_name].centroid.x)
            centroid_y = str(self.gdf['geometry'][cat_name].centroid.y)
            soil_type = str(self.gdf.loc[cat_name]['ISLTYP'])
            veg_type = str(self.gdf.loc[cat_name]['IVGTYP'])
            aspect = str(self.gdf.loc[cat_name]['aspect_mean'])
            slope  = self.gdf.loc[cat_name]['slope_mean']/100. # convert percent to ratio
            slope_deg = math.degrees(math.atan(slope)) # convert radian to degrees

            fname_nom = f'noahowp_config_{cat_name}.input'
            nom_file = os.path.join(nom_dir, fname_nom)

            with open(nom_file, 'w') as file:
                for line in lines:
                    if line.strip().startswith('startdate'):
                        file.write(f'  startdate      = \"{start_time}\"  \n')
                    elif line.strip().startswith('enddate'):
                        file.write(f'  enddate      = \"{end_time}\"  \n')
                    elif line.strip().startswith('forcing_filename'):
                        file.write(f'  forcing_filename   = \"{self.forcing_dir}\"  \n')
                    elif line.strip().startswith('output_filename'):
                        file.write(f'  output_filename   = \"output-{cat_name}.csv\"  \n')
                    elif line.strip().startswith('parameter_dir'):
                        file.write(f'  parameter_dir      = \"{os.path.join(nom_dir, "parameters")}\" \n')
                    elif line.strip().startswith('lat'):
                        file.write(f'  lat      = {centroid_y} \n')
                    elif line.strip().startswith('lon'):
                        file.write(f'  lon      = {centroid_x} \n')
                    elif line.strip().startswith('terrain_slope'):
                        file.write(f'  terrain_slope      = {slope_deg} \n')
                    elif line.strip().startswith('azimuth'):
                        file.write(f'  azimuth       = {aspect} \n')
                    elif line.strip().startswith('isltyp'):
                        file.write(f'  isltyp           = {soil_type} \n')
                    elif line.strip().startswith('vegtyp'):
                        file.write(f'  vegtyp        = {veg_type} \n')
                    else:
                        file.write(line)


    def write_cfe_input_files(self):

        cfe_dir = os.path.join(self.output_dir, "configs/cfe")
        self.create_directory(cfe_dir)

        cfe_basefile = os.path.join(self.sandbox_dir, "configs/basefiles/config_cfe.txt")

        if not os.path.exists(cfe_basefile):
            sys.exit(f"Sample CFE config file does not exist, provided is {cfe_basefile}")

        delimiter = ","
        
        # Read infile line by line
        with open(cfe_basefile, 'r') as infile:
            lines = infile.readlines()

        for catID in self.catids:
            cat_name = 'cat-' + str(catID)
            fname = cat_name + '*.txt'

            fname_cfe = f'cfe_config_{cat_name}.txt'
            cfe_file = os.path.join(cfe_dir, fname_cfe)

            with open(cfe_file, 'w') as file:
                for line in lines:
                    if line.startswith("#"):
                        continue
                    if line.strip().startswith('soil_params.b'):
                        if self.gdf['soil_b'][cat_name] == 1.0:
                            file.write(f'soil_params.b=1.1[]\n')
                        else:
                            file.write(f'soil_params.b={self.gdf["soil_b"][cat_name]}[]\n')
                    elif line.strip().startswith('soil_params.satdk'):
                        file.write(f'soil_params.satdk={self.gdf["soil_satdk"][cat_name]}[m s-1]\n')
                    elif line.strip().startswith('soil_params.satpsi'):
                        file.write(f'soil_params.satpsi={self.gdf["soil_satpsi"][cat_name]}[m]\n')
                    elif line.strip().startswith('soil_params.slop'):
                        file.write(f'soil_params.slop={self.gdf["soil_slop"][cat_name]}[m/m]\n')
                    elif line.strip().startswith('soil_params.smcmax'):
                        file.write(f'soil_params.smcmax={self.gdf["soil_smcmax"][cat_name]}[m/m]\n')
                    elif line.strip().startswith('soil_params.wltsmc'):
                        file.write(f'soil_params.wltsmc={self.gdf["soil_wltsmc"][cat_name]}[m/m]\n')
                    elif line.strip().startswith('soil_params.refkdt'):
                        file.write(f'refkdt={self.gdf["soil_refkdt"][cat_name]}\n')
                    elif line.strip().startswith('max_gw_storage'):
                        file.write(f'max_gw_storage={self.gdf["max_gw_storage"][cat_name]}[m]\n')
                    elif line.strip().startswith('Cgw'):
                        file.write(f'Cgw={self.gdf["Cgw"][cat_name]}[m h-1]\n')
                    elif line.strip().startswith('expon'):
                        file.write(f'expon={self.gdf["gw_expon"][cat_name]}[]\n')
                    #elif line.strip().startswith('soil_storage'):
                    #    file.write(f'soil_storage={self.gdf["soil_smcmax"][cat_name]}[m/m]\n')
                    elif line.strip().startswith('surface_runoff_scheme'):
                        surface_runoff_scheme = line.strip().split("=")[1]
                        file.write(line)

                        if surface_runoff_scheme == "GIUH" or surface_runoff_scheme == 1:
                            giuh_cat = json.loads(self.gdf['giuh'][cat_name])
                            giuh_cat = pd.DataFrame(giuh_cat, columns=['v', 'frequency'])
                            giuh_ordinates = ",".join(str(x) for x in np.array(giuh_cat["frequency"]))
                            file.write(f'giuh_ordinates={giuh_ordinates}\n')
                        elif surface_runoff_scheme == "NASH_CASCADE" or surface_runoff_scheme == 2:
                            file.write(f'N_nash_surface={int(self.gdf["N_nash_surface"][cat_name])}[]\n')
                            file.write(f'K_nash_surface={self.gdf["K_nash_surface"][cat_name]}[h-1]\n')
                            s = [str(0.0),] * int(self.gdf['N_nash_surface'][cat_name])
                            s = delimiter.join(s)
                            file.write(f'nash_storage_surface={s}[]\n')
                    elif line.strip().startswith('N_nash_surface') or line.strip().startswith('K_nash_surface') or \
                         line.strip().startswith('nash_storage_surface'):
                        continue
                    elif line.strip().startswith('surface_water_partitioning_scheme'):
                        surface_water_partitioning_scheme = line.strip().split("=")[1]
                        file.write(line)

                        if (surface_water_partitioning_scheme == "Xinanjiang"):
                            soil_id = self.gdf['ISLTYP'][cat_name]
                            file.write
                            file.write(f'a_Xinanjiang_inflection_point_parameter={self.soil_class_NWM["AXAJ"][soil_id]}\n')
                            file.write(f'b_Xinanjiang_shape_parameter={self.soil_class_NWM["BXAJ"][soil_id]}\n')
                            file.write(f'x_Xinanjiang_shape_parameter={self.soil_class_NWM["XXAJ"][soil_id]}\n')
                    elif line.strip().startswith('a_Xinanjiang_inflection_point_parameter') or \
                         line.strip().startswith('b_Xinanjiang_shape_parameter') or \
                         line.strip().startswith('x_Xinanjiang_shape_parameter'):
                        continue
                    elif line.strip().startswith('sft_coupled'):
                        sft_coupled = line.strip().split("=")[1]
                        if sft_coupled.lower() == "true":
                            ice_content_threshold = 0.3
                            file.write("sft_coupled=true")
                            file.write(f"ice_content_threshold={ice_content_threshold}")
                    else:
                        file.write(line)


    def write_topmodel_input_files(self):

        topmodel_dir = os.path.join(self.output_dir, "configs/topmodel")
        self.create_directory(topmodel_dir)
        
        for catID in self.catids:
            cat_name = 'cat-' + str(catID)
            fname = cat_name + '*.txt'

            topmod = [
                "0",
                f'{cat_name}',
                f"./forcing/{cat_name}.csv",
                f'./{topmodel_dir}/subcat_{cat_name}.dat',
                f'./{topmodel_dir}/params_{cat_name}.dat',
                f'./{topmodel_dir}/topmod_{cat_name}.out',
                f'./{topmodel_dir}/hyd_{cat_name}.out'
            ]

            fname_tm = f'topmod_{cat_name}.run'
            tm_file = os.path.join(topmodel_dir, fname_tm)
            with open(tm_file, "w") as f:
                f.writelines('\n'.join(topmod))

            params = [
                f'Extracted study basin: {cat_name}',
                "0.032  5.0  50.  3600.0  3600.0  0.05  0.0000328  0.002  0  1.0  0.02  0.1"
            ]

            fname_tm = f'params_{cat_name}.dat'
            tm_file = os.path.join(topmodel_dir, fname_tm)
            with open(tm_file, "w") as f:
                f.writelines('\n'.join(params))

            twi_cat = json.loads(self.gdf['twi'][cat_name])
            twi_cat = pd.DataFrame(twi_cat, columns=['v', 'frequency'])
            twi_cat = twi_cat.sort_values(by=['v'], ascending=False)

            width_f = json.loads(self.gdf['width_dist'][cat_name])
            df_width_f = pd.DataFrame(width_f, columns=['v', 'frequency'])
            v_cumm = np.cumsum(df_width_f['frequency'])

            nclasses_twi = len(twi_cat['frequency'].values)
            nclasses_width_function = len(df_width_f['frequency'].values)

            subcat = [
                "1 1 1",
                f'Extracted study basin: {cat_name}',
                f'{nclasses_twi} 1',
                'replace_with_twi',
                f'{nclasses_width_function}',
                'add_width_function',
                '$mapfile.dat'
            ]

            twi_str = ''
            for freq, value in zip(twi_cat['frequency'].values, twi_cat['v'].values):
                twi_str += f"{freq:.6f} {value:.6f}\n"

            subcat[3] = twi_str.strip()

            widthf_str = ''
            for freq, value in zip(v_cumm.values, df_width_f['v'].values):
                widthf_str += f"{freq:.6f} {value:.6f} "

            subcat[5] = widthf_str.strip()

            fname_tm = f'subcat_{cat_name}.dat'
            tm_file = os.path.join(topmodel_dir, fname_tm)
            with open(tm_file, "w") as f:
                f.writelines('\n'.join(subcat))

    def write_sft_input_files(self,
                              surface_water_partitioning_scheme):

        sft_dir = os.path.join(self.output_dir, "configs/sft")
        if not surface_water_partitioning_scheme in ["Schaake", "Xinanjiang"]:
            sys.exit("Runoff scheme should be: Schaake or Xinanjiang")

        ncells = 19
        soil_z = "0.1,0.15,0.18,0.23,0.29,0.36,0.44,0.55,0.69,0.86,1.07,1.34,1.66,2.07,2.58,3.22,4.01,5.0,6.0"
        delimiter = ','
        nsteps_yr = 365 * 24

        for catID in self.catids:
            cat_name = 'cat-' + str(catID)
            forcing_file = glob.glob(os.path.join(self.forcing_dir, cat_name + '*.csv'))[0]
            df_forcing = pd.read_csv(self.forcing_file, delimiter=',', usecols=['T2D'], nrows=nsteps_yr, index_col=None)
            MAAT = [str(round(df_forcing['T2D'].mean(), 2)),] * ncells
            MAAT = delimiter.join(MAAT)
            soil_id = self.gdf['ISLTYP'][cat_name]

            sft_params = [
                'verbosity=none',
                'soil_moisture_bmi=1',
                'end_time=1.0[d]',
                'dt=1.0[h]',
                f'soil_params.smcmax={self.gdf["soil_smcmax"][cat_name]}[m/m]',
                f'soil_params.b={self.gdf["soil_b"][cat_name]}[]',
                f'soil_params.satpsi={self.gdf["soil_satpsi"][cat_name]}[m]',
                f'soil_params.quartz={self.soil_class_NWM["QTZ"][soil_id]}[]',
                f'ice_fraction_scheme={surface_water_partitioning_scheme}',
                f'soil_z={soil_z}[m]',
                f'soil_temperature={MAAT}[K]'
            ]

            fname_sft = f'sft_config_{cat_name}.txt'
            sft_file = os.path.join(sft_dir, fname_sft)
            with open(sft_file, "w") as f:
                f.writelines('\n'.join(sft_params))

    def write_smp_input_files(self, cfe_coupled, lasam_coupled=False):
        
        smp_dir = os.path.join(self.output_dir, "configs/smp")
        
        soil_z = "0.1,0.15,0.18,0.23,0.29,0.36,0.44,0.55,0.69,0.86,1.07,1.34,1.66,2.07,2.58,3.22,4.01,5.0,6.0"

        for catID in self.catids:
            cat_name = 'cat-' + str(catID)
            soil_id = self.gdf['ISLTYP'][cat_name]

            smp_params = [
                'verbosity=none',
                f'soil_params.smcmax={self.gdf["soil_smcmax"][cat_name]}[m/m]',
                f'soil_params.b={self.gdf["soil_b"][cat_name]}[]',
                f'soil_params.satpsi={self.gdf["soil_satpsi"][cat_name]}[m]',
                f'soil_z={soil_z}[m]',
                'soil_moisture_fraction_depth=1.0[m]'
            ]

            if cfe_coupled:
                smp_params += ['soil_storage_model=conceptual', 'soil_storage_depth=2.0']
            elif lasam_coupled:
                smp_params += ['soil_storage_model=layered', 'soil_moisture_profile_option=constant',
                               'soil_depth_layers=2.0', 'water_table_depth=10[m]']

            fname_smp = f'smp_config_{cat_name}.txt'
            smp_file = os.path.join(smp_dir, fname_smp)
            with open(smp_file, "w") as f:
                f.writelines('\n'.join(smp_params))

    def write_lasam_input_files(self,
                                sft_coupled=False):
        
        lasam_dir = os.path.join(self.output_dir,"configs/lasam")
        self.create_directory(lasam_dir)

        lasam_params_file = os.path.join(self.ngen_dir,"extern/LASAM/LASAM/data/vG_params_stat_nom_ordered.dat")
        str_sub ="cp -r "+ lasam_params_file + " %s"%lasam_dir
        out=subprocess.call(str_sub,shell=True)

        sft_calib = "False"
        soil_z = "10.0,15.0,18.0,23.0,29.0,36.0,44.0,55.0,69.0,86.0,107.0,134.0,166.0,207.0,258.0,322.0,401.0,500.0,600.0"
        
        lasam_params_base = [
            'verbosity=none',
            f'soil_params_file={lasam_params_file}',
            'layer_thickness=200.0[cm]',
            'initial_psi=2000.0[cm]',
            'timestep=3600[sec]',
            'endtime=1000000000.0[d]',
            'forcing_resolution=3600[sec]',
            'ponded_depth_max=0[cm]',
            'use_closed_form_G=false',
            'layer_soil_type=',
            'wilting_point_psi=15495.0[cm]',
            'field_capacity_psi=340.9[cm]',
            'adaptive_timestep=true',
            'giuh_ordinates='
        ]

        if sft_coupled:
            lasam_params_base.append('sft_coupled=true')
            lasam_params_base.append(f'soil_z={soil_z}[cm]')

        if (sft_coupled and (sft_calib in ["true", "True"])):
            lasam_params_base.append('calib_params=true')
            
        if self.ngen_cal_type in ['calibration', 'validation', 'restart']:
            lasam_params_base.append('calib_params=true')
        
        soil_type_loc = lasam_params_base.index("layer_soil_type=")
            
        giuh_loc_id = lasam_params_base.index("giuh_ordinates=")
            

        for catID in self.catids:
            cat_name = 'cat-' + str(catID)
            fname = cat_name + '*.txt'

            lasam_params = lasam_params_base.copy()
            lasam_params[soil_type_loc] += str(self.gdf['ISLTYP'][cat_name])

            giuh_cat = json.loads(self.gdf['giuh'][cat_name])
            giuh_cat = pd.DataFrame(giuh_cat, columns=['v', 'frequency'])
            giuh_ordinates = ",".join(str(x) for x in np.array(giuh_cat["frequency"]))

            any_nans = np.any(np.isnan(giuh_cat["frequency"]))
            if any_nans:
                giuh_ordinates = str(1.0)

            lasam_params[giuh_loc_id] += giuh_ordinates

            fname_lasam = f'lasam_config_{cat_name}.txt'
            lasam_file = os.path.join(lasam_dir, fname_lasam)
            with open(lasam_file, "w") as f:
                f.writelines('\n'.join(lasam_params))


    def write_pet_input_files(self):
        pet_dir = os.path.join(self.output_dir,"configs/pet")
        self.create_directory(pet_dir)
                
        df_cats = gpd.read_file(self.gpkg_file, layer='divides')
        df_cats = df_cats.to_crs("EPSG:4326")
        df_cats.set_index("divide_id", inplace=True)
        pet_method = 3

        for catID in self.catids:
            cat_name = 'cat-' + str(catID)
            centroid_x = str(df_cats['geometry'][cat_name].centroid.x)
            centroid_y = str(df_cats['geometry'][cat_name].centroid.y)
            elevation_mean = self.gdf['elevation_mean'][cat_name]

            pet_params = [
                'verbose=0',
                f'pet_method={pet_method}',
                'forcing_file=BMI',
                'run_unit_tests=0',
                'yes_aorc=1',
                'yes_wrf=0',
                'wind_speed_measurement_height_m=10.0',
                'humidity_measurement_height_m=2.0',
                'vegetation_height_m=16.0',
                'zero_plane_displacement_height_m=0.0003',
                'momentum_transfer_roughness_length=0.0',
                'heat_transfer_roughness_length_m=0.0',
                'surface_longwave_emissivity=1.0',
                'surface_shortwave_albedo=0.17',
                'cloud_base_height_known=FALSE',
                'time_step_size_s=3600',
                'num_timesteps=720',
                'shortwave_radiation_provided=1',
                f'latitude_degrees={centroid_y}',
                f'longitude_degrees={centroid_x}',
                f'site_elevation_m={elevation_mean}'
            ]

            fname_pet = f'pet_config_{cat_name}.txt'
            pet_file = os.path.join(pet_dir, fname_pet)
            with open(pet_file, "w") as f:
                f.writelines('\n'.join(pet_params))

    def write_troute_input_files(self):

        troute_basefile = os.path.join(self.sandbox_dir, "configs/basefiles/config_troute.yaml")
        troute_dir = os.path.join(self.output_dir,"configs")
        gpkg_name = os.path.basename(self.gpkg_file).split(".")[0]

        if not os.path.exists(troute_basefile):
            sys.exit(f"Sample routing yaml file does not exist, provided is {troute_basefile}")

        with open(troute_basefile, 'r') as file:
            d = yaml.safe_load(file)

        d['network_topology_parameters']['supernetwork_parameters']['geo_file_path'] = self.gpkg_file
        d['network_topology_parameters']['waterbody_parameters']['level_pool']['level_pool_waterbody_parameter_file_path'] = self.gpkg_file
        d['network_topology_parameters']['supernetwork_parameters']['title_string'] = gpkg_name

        dt = 300
        params = self.get_flowpath_attributes(gage_id=self.gpkg_file, full_schema=True)

        columns = {
            'key': params['key'],
            'downstream': params['downstream'],
            'mainstem': params['mainstem'],
            'dx': params['dx'],
            'n': params['n'],
            'ncc': params['ncc'],
            's0': params['s0'],
            'bw': params['bw'],
            'waterbody': params['waterbody'],
            'gages': params['gages'],
            'tw': params['tw'],
            'twcc': params['twcc'],
            'musk': params['musk'],
            'musx': params['musx'],
            'cs': params['cs'],
            'alt': params['alt']
        }

        d['network_topology_parameters']['supernetwork_parameters']['columns'] = columns

        start_time = pd.Timestamp(self.simulation_time['start_time'])
        end_time = pd.Timestamp(self.simulation_time['end_time'])
        diff_time = (end_time - start_time).total_seconds()

        d['compute_parameters']['restart_parameters']['start_datetime'] = start_time.strftime("%Y-%m-%d_%H:%M:%S")

        if self.ngen_cal_type in ['calibration', 'validation', 'calibvalid', 'restart']:
            d['compute_parameters']['forcing_parameters']['qlat_input_folder'] = "./"
        else:
            d['compute_parameters']['forcing_parameters']['qlat_input_folder'] = os.path.join(self.output_dir, "outputs/div")

        d['compute_parameters']['forcing_parameters']['qlat_file_pattern_filter'] = "nex-*"
        del d['compute_parameters']['forcing_parameters']['binary_nexus_file_folder']
        d['compute_parameters']['forcing_parameters']['nts'] = int(diff_time / dt)
        d['compute_parameters']['forcing_parameters']['max_loop_size'] = 10000000

        d['compute_parameters']['cpu_pool'] = 10

        if self.ngen_cal_type in ['calibration', 'validation', 'calibvalid', 'restart']:
            stream_output = {
                "stream_output": {
                    "stream_output_directory": "./",
                    'stream_output_time': -1,
                    'stream_output_type': '.nc',
                    'stream_output_internal_frequency': 60
                }
            }
        else:
            stream_output = {
                "stream_output": {
                    'stream_output_directory': os.path.join(self.output_dir, "outputs/troute"),
                    'stream_output_time': -1,
                    'stream_output_type': '.nc',
                    'stream_output_internal_frequency': 60
                }
            }

        d['output_parameters'] = stream_output

        with open(os.path.join(troute_dir, "troute_config.yaml"), 'w') as file:
            yaml.dump(d, file, default_flow_style=False, sort_keys=False)


    def get_flowpath_attributes(self,
                                full_schema=False,
                                gage_id=False):

        layers = fiona.listlayers(self.gpkg_file)
        flowpath_layer = [layer for layer in layers if 'flowpath' in layer and not 'flowpaths' in layer][0]
        gdf_fp_attr = gpd.read_file(self.gpkg_file, layer=flowpath_layer)
        params = schema.get_schema_flowpath_attributes(gdf_fp_attr, for_gage_id=gage_id)

        if full_schema:
            return params
        elif gage_id:
            gage_id = params['gages']
            waterbody_id = params['key']
            gdf_fp_cols = gdf_fp_attr[[waterbody_id, gage_id]]
            basin_gage = gdf_fp_cols[gdf_fp_cols[gage_id].notna()]
            basin_gage_id = basin_gage[waterbody_id].tolist()
            return basin_gage_id

    def write_forcing_input_files(self,
                                  forcing_basefile,
                                  forcing_time,
                                  forcing_format):

        if not os.path.exists(forcing_basefile):
            sys.exit(f"Sample forcing yaml file does not exist, provided is {forcing_basefile}")

        with open(forcing_basefile, 'r') as file:
            d = yaml.safe_load(file)

        time_sim = json.loads(forcing_time)
        start_yr = pd.Timestamp(time_sim['start_time']).year
        end_yr   = pd.Timestamp(time_sim['end_time']).year

        if start_yr > end_yr:
            sys.exit(f"end_time ({end_yr})is less than the start_time ({start_yr}")

        if start_yr <= end_yr:
            end_yr = end_yr + 1

        d['gpkg'] = self.gpkg_file
        d["years"] = [start_yr, end_yr]
        d["out_dir"] = os.path.join(os.path.dirname(self.gpkg_file), "forcing")

        out_dir = Path(d['out_dir']) / f'{start_yr}_to_{end_yr}'
        are_identical = out_dir.resolve() == Path(self.forcing_dir).resolve()

        if not are_identical:
            raise RuntimeError(f"Directory mismatch: out_dir={out_dir} is not the same as forcing_dir={self.forcing_dir}.")

        if not os.path.exists(d["out_dir"]):
            os.makedirs("data/forcing")

        if forcing_format == '.csv':
            d['netcdf'] = False

        with open(os.path.join(d["out_dir"], "config_forcing.yaml"), 'w') as file:
            yaml.dump(d, file, default_flow_style=False, sort_keys=False)

        return os.path.join(d["out_dir"], "config_forcing.yaml")

    def create_directory(self, dir_name):
        if os.path.exists(dir_name):
            str_sub = "rm -rf " + dir_name
            out = subprocess.call(str_sub, shell=True)
        os.mkdir(dir_name)

class ConfigurationCalib:
    def __init__(self, gpkg_file, output_dir, ngen_dir, realization_file_par, troute_output_file,
                 ngen_cal_basefile, ngen_cal_type,
                 restart_dir, simulation_time, evaluation_time, num_proc):
        
        self.gpkg_file = gpkg_file
        self.output_dir = output_dir
        self.ngen_dir = ngen_dir
        self.realization_file_par = realization_file_par
        self.simulation_time = simulation_time
        self.evaluation_time = evaluation_time
        #self.verbosity = verbosity
        self.ngen_cal_type = ngen_cal_type
        self.num_proc = num_proc
        self.ngen_cal_basefile = ngen_cal_basefile
        self.troute_output_file = troute_output_file
        self.restart_dir = restart_dir

    def get_flowpath_attributes(self):

        layers = fiona.listlayers(self.gpkg_file)
        flowpath_layer = [layer for layer in layers if 'flowpath' in layer and not 'flowpaths' in layer][0]
        gdf_fp_attr = gpd.read_file(self.gpkg_file, layer=flowpath_layer)
        params = schema.get_schema_flowpath_attributes(gdf_fp_attr, for_gage_id=True)


        gage_id = params['gages']
        waterbody_id = params['key']
        gdf_fp_cols = gdf_fp_attr[[waterbody_id, gage_id]]
        basin_gage = gdf_fp_cols[gdf_fp_cols[gage_id].notna()]
        basin_gage_id = basin_gage[waterbody_id].tolist()
        return basin_gage_id
        
    def write_calib_input_files(self):
        
        conf_dir = os.path.join(self.output_dir, "configs")
        realization = glob.glob(os.path.join(self.output_dir, "json/realization_*.json"))

        assert len(realization) == 1

        if not os.path.exists(self.ngen_cal_basefile):
            sys.exit(f"Sample calib yaml file does not exist, provided is {self.ngen_cal_basefile}")

        basin_sandbox_dir = os.path.dirname(os.path.dirname(self.ngen_cal_basefile))
        gpkg_name = os.path.basename(self.gpkg_file).split(".")[0]

        with open(self.ngen_cal_basefile, 'r') as file:
            d = yaml.safe_load(file)

        d['general']['workdir']   = self.output_dir.as_posix()
        d['model']['binary']      = os.path.join(self.ngen_dir, "cmake_build/ngen")
        d['model']['realization'] = realization[0]
        d['model']['hydrofabric'] = self.gpkg_file.as_posix()
        d['model']['routing_output'] = self.troute_output_file

        gage_id = self.get_flowpath_attributes()
        """
        if len(gage_id) == 1:
            d['model']['eval_feature'] = gage_id[0]
        else:
            print("more than one rl_gages exist in the geopackage, using max drainage area to filter...")
            div = gpd.read_file(self.gpkg_file, layer='divides')
            df = div[['divide_id', 'tot_drainage_areasqkm']]
            index = df['divide_id'].map(lambda x: 'wb-' + str(x.split("-")[1]))
            df.set_index(index, inplace=True)
            idmax = df['tot_drainage_areasqkm'].idxmax()
            d['model']['eval_feature'] = idmax
        """

        d['model']['eval_feature'] = gpkg_name.split("_")[1]

        if self.num_proc > 1:
            d['model']['parallel'] = self.num_proc
            d['model']['partitions'] = self.realization_file_par

        if os_name == "Darwin":
            d['model']['binary'] = f'PYTHONEXECUTABLE=$(which python) ' + os.path.join(self.ngen_dir, "cmake_build/ngen")
        else:
            d['model']['binary'] = os.path.join(self.ngen_dir, "cmake_build/ngen")

        if self.ngen_cal_type == 'calibration':
            val_params = {
                'evaluation_start': self.evaluation_time['start_time'],
                'evaluation_stop': self.evaluation_time['end_time']
            }
            d['model']['eval_params'].update(val_params)

        if self.ngen_cal_type == 'validation':
            val_troute_output = {
                'ngen_cal_troute_output': {
                    'validation_routing_output': self.troute_output_file
                }
            }

            try:
                d['model']['plugin_settings'].update(val_troute_output)
            except:
                d['model']['plugin_settings'] = val_troute_output

            val_params = {
                'sim_start': self.simulation_time['start_time'],
                'evaluation_start': self.evaluation_time['start_time'],
                'evaluation_stop': self.evaluation_time['end_time'],
                'objective': "kling_gupta"
            }
            d['model']['val_params'] = val_params

        if (self.ngen_cal_type == 'restart' or self.ngen_cal_type == "validation"):

            state_file = glob.glob(str(Path(self.output_dir) / "*_worker" / "*_parameter_df_state.parquet"))[0]
            df_parq = pd.read_parquet(state_file)
            df_params = pd.read_csv(Path(state_file).parent / "best_params.txt", header = None)

            best_itr = str(int(df_params.values[1]))

            best_params_set = df_parq[best_itr]
            calib_params = best_params_set.index.to_list()

            for block in d:
                if '_params' in block:
                    for par in d[block]:
                        if par['name'] in calib_params:
                            par['init'] = float(best_params_set[par['name']])

        config_fname = ""
        if self.ngen_cal_type == 'calibration':
            config_fname = "ngen-cal_calib_config.yaml"
        elif self.ngen_cal_type == 'validation':
            config_fname = "ngen-cal_valid_config.yaml"

        with open(os.path.join(conf_dir, config_fname), 'w') as file:
            yaml.dump(d, file, default_flow_style=False, sort_keys=False)
