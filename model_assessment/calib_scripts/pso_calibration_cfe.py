###############################################################
# Author      : Peter La Follette [plafollette@lynker.com | May 2025]
# Calibrates cfe+pet+t-route or cfe+t-route+NOM, if NOM is in the model formulation then some of its parameters will be calibrated
# currently just 1 tile is supported

import os
import subprocess
import pandas as pd
import numpy as np
import math
from datetime import datetime
import random
import multiprocessing
from hydroeval import kge
import sys
import traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from model_assessment.util.metrics import compute_metrics
from model_assessment.util.update_NOM import update_mptable
import yaml
import json

np.random.seed(42)
random.seed(42)

# === CONFIGURATION ===
n_particles = 2
n_iterations = 2
max_cores_for_gages = 2
metric_to_calibrate_on = "kge"  # "kge", "log_kge", or "event_kge"

from model_assessment.configs import path_config as cfg
with open("model_assessment/configs/time_config.yaml", "r") as f:
    time_cfg = yaml.safe_load(f)

cal_start = pd.Timestamp(time_cfg["cal_start"])
cal_end   = pd.Timestamp(time_cfg["cal_end"])
val_start = pd.Timestamp(time_cfg["val_start"])
val_end   = pd.Timestamp(time_cfg["val_end"])

project_root = cfg.project_root
logging_dir = cfg.logging_dir
sandbox_path = cfg.sandbox_path
observed_q_root = cfg.observed_q_root

os.makedirs(logging_dir, exist_ok=True)


def clear_terminal():
    os.system('clear')

### As I was attempting my first calibration runs with 10s of gages running in parallel and hundreds of iterations on MacOS, I had spotlight indexing on.
### It is actually the case that you can run out of disk space if spotlight indexing is on and it includes outputs from nextgen, because nextgen model outputs amount to a huge amount of data written per day
### To address this, all directories that will contain nextgen outputs at the divide scale, as well as the t-route outputs, will have a .metadata_never_index file created with them during the -conf step in NextGenSandboxHub.
### This should make spotlight indexing skip these files and avoid the issue where the available disk space goes to 0, but just to be sure, this function stops the calibration execution in the event that disk space gets too low 
### if you have the ram to spare, writing the out directory on a ramdisk is a good idea anyway
def check_for_stop_signal_or_low_disk():
    stop_file = os.path.join(project_root, "STOP_NOW.txt")
    if os.path.exists(stop_file):
        return True
    statvfs = os.statvfs(project_root)
    free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024 ** 3)
    return free_gb < 100


# === PARAMETER SETUP ===
param_bounds = [
    (0.0, 21.94),
    (math.log10(2.77e-10), math.log10(0.000726)),
    (0.0, 0.995),
    (0.20554, 0.6),
    (0.01, 0.2),
    (math.log10(1.6266e-06), math.log10(0.1)),
    (1.0, 8.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.1, 4.0),
    (0.0, 1.0),
    (0.0, 0.138),
    (0.1, 1.0),
    (0.001, 0.005)
]

param_names = [
    "b", "satdk", "satpsi", "maxsmc", "max_gw_storage", "Cgw", "expon", "Kn",
    "Klf", "refkdt", "slope", "wltsmc", "alpha_fc", "Kinf_nash_surface"
]

# === NOM PARAMETERS (only used if NOM present) ===
nom_param_names = ["MFSNO", "RSURF_SNOW", "HVT", "CWPVT", "VCMX25", "MP"]
nom_param_bounds = [
    (0.625, 5.0),
    (0.1, 100.0),
    (0.0, 20.0),
    (0.18, 5.0),
    (0.0, 80.0),
    (3.6, 12.6)
]


log_scale_params = {"Cgw": True, "satdk": True}

param_bounds_dict = dict(zip(param_names, param_bounds))

# === HELPERS ===
def transform_params(params):
    return [10**p if log_scale_params.get(name, False) else p for name, p in zip(param_names, params)]


def detect_and_read_nom_params(gage_id):
    nom_dir = os.path.join(cfg.model_roots[0], f"out/{gage_id}/configs/noahowp/parameters")
    mptable_path = os.path.join(nom_dir, "MPTABLE.TBL")
    if not os.path.exists(mptable_path):
        return False, [], ""

    try:
        with open(mptable_path) as f:
            lines = f.readlines()

        nom_params = {}
        for line in lines:
            if "=" not in line or line.strip().startswith("!"):
                continue
            key, val = line.split("=", 1)
            name = key.strip()
            if name in nom_param_names:
                val = val.split("!")[0].split(",")[0].strip()
                nom_params[name] = float(val)

        if set(nom_params.keys()) != set(nom_param_names):
            raise ValueError(f"NOM param mismatch in {mptable_path}")

        return True, [nom_params[n] for n in nom_param_names], mptable_path

    except Exception as e:
        print(f"Error parsing NOM params from {mptable_path}: {e}")
        return False, [], mptable_path

def extract_initial_nom_params(nom_file_path):
    """
    Extract initial NOM parameters from a MPTABLE.TBL file.
    Only the first value from each line is used (even if multiple are listed).
    """
    nom_param_names = ["MFSNO", "RSURF_SNOW", "HVT", "CWPVT", "VCMX25", "MP"]
    nom_params_dict = {}

    try:
        with open(nom_file_path) as f:
            lines = f.readlines()

        for line in lines:
            if '=' not in line or line.strip().startswith('!'):
                continue
            key, value = line.split('=', 1)
            param_name = key.strip()
            if param_name in nom_param_names:
                value_str = value.split('!')[0]
                values = [v.strip() for v in value_str.split(',') if v.strip()]
                nom_params_dict[param_name] = float(values[0])

        if set(nom_params_dict.keys()) != set(nom_param_names):
            raise ValueError(f"Found NOM parameters {list(nom_params_dict.keys())}, expected {nom_param_names}. Check formatting in {nom_file_path}")

        return [nom_params_dict[p] for p in nom_param_names]

    except Exception as e:
        raise RuntimeError(f"Failed to parse NOM parameters from {nom_file_path}: {e}")


def extract_initial_cfe_params(config_path):
    param_map = {
        "b": "soil_params.b",
        "satdk": "soil_params.satdk",
        "satpsi": "soil_params.satpsi",
        "maxsmc": "soil_params.smcmax",
        "max_gw_storage": "max_gw_storage",
        "Cgw": "Cgw",
        "expon": "expon",
        "Kn": "K_nash_subsurface",
        "Klf": "K_lf",
        "refkdt": "refkdt",
        "slope": "soil_params.slop",
        "wltsmc": "soil_params.wltsmc",
        "alpha_fc": "alpha_fc",
        "Kinf_nash_surface": "Kinf_nash_surface"  # NEW: add Kinf
    }

    values = {}
    with open(config_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if "=" in line:
            key, val = line.strip().split("=", 1)
            key = key.strip()
            val = val.split("[")[0].strip()  # Remove units if present
            for param, config_key in param_map.items():
                if key == config_key:
                    values[param] = float(val)

    # Build parameter list
    param_list = []
    for name in param_names:
        if name in values:
            val = values[name]
        else:
            val = 0.002  # Always default to 0.002 if missing (old behavior)

        if log_scale_params.get(name, False):
            param_list.append(math.log10(val))
        else:
            param_list.append(val)

    return param_list


def get_observed_q(observed_path):
    df = pd.read_csv(observed_path, parse_dates=['value_time'])
    df.set_index('value_time', inplace=True)
    return df['flow_m3_per_s']


def regenerate_cfe_config(config_dir, params):
    replacements = dict(zip(param_names, params))
    param_map = { # same as above
        "b": "soil_params.b", "satdk": "soil_params.satdk", "satpsi": "soil_params.satpsi", "maxsmc": "soil_params.smcmax",
        "max_gw_storage": "max_gw_storage", "Cgw": "Cgw", "expon": "expon", "Kn": "K_nash_subsurface",
        "Klf": "K_lf", "refkdt": "refkdt", "slope": "soil_params.slop", "wltsmc": "soil_params.wltsmc",
        "alpha_fc": "alpha_fc", "Kinf_nash_surface": "Kinf_nash_surface"
    }

    if os.path.isfile(config_dir):
        config_dir = os.path.dirname(config_dir)

    for fname in sorted(f for f in os.listdir(config_dir) if f.startswith("cfe_config_cat")):
        path = os.path.join(config_dir, fname)
        lines = []
        with open(path) as f:
            for line in f:
                key = line.split('=')[0].strip()
                for pname, ckey in param_map.items():
                    if key == ckey:
                        unit = line[line.find("["):] if "[" in line else ""
                        lines.append(f"{ckey}={replacements[pname]}{unit}\n")
                        break
                else:
                    lines.append(line)

        with open(path, 'w') as f:
            f.writelines(lines)


# === OBJECTIVE FUNCTION ===
def objective_function(args):
    import json
    params, particle_idx, gage_id, config_path, observed_path, postproc_base_path, include_nom, nom_file_path = args

    true_params = transform_params(params)
    if check_for_stop_signal_or_low_disk():
        print(f" Stop signal detected, skipping particle {particle_idx}")
        return 10.0, -10.0, {}, {}

    print(f"\nEvaluating particle {particle_idx} for gage {gage_id}")

    # === Delete old .nc routing files in troute dir ===
    troute_dir = os.path.join(cfg.model_roots[0], "out", gage_id, "troute")
    if os.path.isdir(troute_dir):
        for fname in os.listdir(troute_dir):
            if fname.endswith(".nc"):
                file_path = os.path.join(troute_dir, fname)
                try:
                    os.remove(file_path)
                    print(f"[DEBUG] Deleted old routing file: {file_path}")
                except Exception as e:
                    print(f"[WARN] Could not delete {file_path}: {e}")

    regenerate_cfe_config(config_path, true_params)

    if include_nom:
        nom_params = dict(zip(nom_param_names, params[-6:]))
        update_mptable(
            original_file=nom_file_path,
            output_file=nom_file_path,
            updated_params=nom_params,
            verbose=False
        )


    try:
        # === Update realization JSON to use spinup + calibration only ===
        json_dir = os.path.join(cfg.model_roots[0], "out", gage_id, "json")
        realization_path = [f for f in os.listdir(json_dir) if f.endswith(".json")][0]
        realization_path = os.path.join(json_dir, realization_path)

        with open(realization_path, "r") as f:
            realization_config = json.load(f)

        realization_config["time"]["start_time"] = time_cfg["spinup_start"]
        realization_config["time"]["end_time"]   = time_cfg["cal_end"]

        with open(realization_path, "w") as f:
            json.dump(realization_config, f, indent=4)

        # === Update troute_config.yaml ===
        troute_path = os.path.join(cfg.model_roots[0], "out", gage_id, "configs", "troute_config.yaml")
        with open(troute_path, "r") as f:
            troute_cfg = yaml.safe_load(f)

        spinup_start = pd.Timestamp(time_cfg["spinup_start"])
        nts = int((cal_end - spinup_start) / pd.Timedelta(seconds=300))
        troute_cfg["compute_parameters"]["restart_parameters"]["start_datetime"] = spinup_start.strftime("%Y-%m-%d_%H:%M:%S")
        troute_cfg["compute_parameters"]["forcing_parameters"]["nts"] = nts

        with open(troute_path, "w") as f:
            yaml.dump(troute_cfg, f)

        # === Run the model ===
        subprocess.call(["python", "sandbox.py", "-run", "--gage_id", gage_id], cwd=project_root)

        # === Extract output ===
        env = os.environ.copy()
        env["PARTICLE_ID"] = str(particle_idx)

        get_hydrograph_path = os.path.join(project_root, "model_assessment", "util", "get_hydrograph.py")
        tile_root = cfg.model_roots[0]
        output_path = os.path.join(tile_root, "postproc", f"{gage_id}_particle_{particle_idx}.csv")

        subprocess.call(
            ["python", get_hydrograph_path, "--gage_id", gage_id, "--output", output_path, "--base_dir", tile_root],
            cwd=os.path.join(tile_root, "postproc")
        )

        sim_df = pd.read_csv(output_path, parse_dates=['current_time']).set_index('current_time')['flow'].resample('1h').mean()
        # Check output length (i.e. check for crashes if simulation length is unexpected)
        window_start = pd.Timestamp(time_cfg["spinup_start"])
        window_end = cal_end  # uses calibration period by default

        expected_length = int((window_end - window_start) / pd.Timedelta(hours=1)) + 1
        actual_length = len(sim_df)

        allowed_tolerance = 1
        if abs(actual_length - expected_length) > allowed_tolerance:
            raise RuntimeError(
                f"Gage {gage_id} produced {actual_length} time steps but expected {expected_length}. "
                f"Allowed tolerance is +/-{allowed_tolerance}. Simulation likely incomplete."
            )

        obs_df = get_observed_q(observed_path)

        sim_cal, obs_cal = sim_df[cal_start:cal_end].dropna(), obs_df[cal_start:cal_end].dropna()
        sim_val, obs_val = sim_df[val_start:val_end].dropna(), obs_df[val_start:val_end].dropna()

        sim_cal, obs_cal = sim_cal.align(obs_cal, join='inner')
        sim_val, obs_val = sim_val.align(obs_val, join='inner')

        if len(sim_val) > 0 and len(obs_val) > 0:
            sim_val.iloc[-1] += 1e-8
            obs_val.iloc[-1] += 1e-8

        if len(sim_cal) > 0 and len(obs_cal) > 0:
            sim_cal.iloc[-1] += 1e-8
            obs_cal.iloc[-1] += 1e-8

        cal_metrics = compute_metrics(sim_cal, obs_cal, event_threshold=1e-2)
        val_metrics = compute_metrics(sim_val, obs_val, event_threshold=1e-2)

        return -cal_metrics[metric_to_calibrate_on], val_metrics, cal_metrics

    except Exception as e:
        print(f"Error during evaluation for {gage_id}: {e}")
        traceback.print_exc()

        # === LOG the failed particle ===
        failed_row = {
            "gage_id": gage_id,
            "particle_idx": particle_idx,
            "params": params.tolist() if hasattr(params, "tolist") else list(params),
            "error_message": str(e)
        }
        log_path = os.path.join(logging_dir, "incomplete_simulations.csv")
        df = pd.DataFrame([failed_row])
        if os.path.exists(log_path):
            df.to_csv(log_path, mode='a', header=False, index=False)
        else:
            df.to_csv(log_path, index=False)

        placeholder_metrics = {
            "kge": -10.0,
            "log_kge": -10.0,
            "volume_error_percent": -999.0,
            "peak_time_error_hours": -999.0,
            "peak_flow_error_percent": -999.0,
            "event_kge": -10.0,
            "event_hours": 0,
            "total_hours": 0
        }
        return 10.0, placeholder_metrics, placeholder_metrics



class Particle:
    def __init__(self, bounds, init_position=None):
        if init_position is not None:
            self.position = np.array(init_position)
        else:
            self.position = np.array([np.random.uniform(low, high) for low, high in bounds])

        self.velocity = np.array([0.1 * (high - low) * np.random.uniform(-1, 1) for low, high in bounds])
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')
        self.current_value = float('inf')
        self.stagnation_counter = 0
        self.best_calibration_metric = -9999.0  # or np.nan
        self.best_validation_metric = -9999.0  # or np.nan


    def reset(self, bounds):
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.array([0.1 * (high - low) * np.random.uniform(-1, 1) for low, high in bounds])
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')
        self.current_value = float('inf')
        self.stagnation_counter = 0

    def update_velocity(self, global_best_position, w, c1, c2):
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, bounds):
        self.position += self.velocity
        for i in range(len(self.position)):
            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]; self.velocity[i] *= -0.5
            elif self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]; self.velocity[i] *= -0.5


class PSO:
    def __init__(self, n_particles, bounds, n_iterations, gage_id, init_position, config_path, observed_path, postproc_base_path, metric_to_calibrate_on="kge", include_nom=False, nom_file_path="", param_names=None):
        self.particles = [Particle(bounds, init_position=init_position if i == 0 else None) for i in range(n_particles)]
        self.bounds = bounds
        self.n_iterations = n_iterations
        self.gage_id = gage_id
        self.config_path = config_path
        self.observed_path = observed_path
        self.postproc_base_path = postproc_base_path
        self.global_best_position = self.particles[0].position
        self.global_best_value = float('inf')
        self.metric_to_calibrate_on = metric_to_calibrate_on
        self.best_cal_metrics = {}
        self.best_val_metrics = {}
        self.include_nom = include_nom
        self.nom_file_path = nom_file_path
        self.param_names = param_names if param_names else param_names

    def optimize(self):
        start_time = datetime.now()
        log_rows = []
        log_path = os.path.join(logging_dir, f"{self.gage_id}.csv")
        stagnation_threshold = 10
        w_start, w_end = 0.9, 0.4

        for iteration in range(self.n_iterations):
            clear_terminal()
            print(f"\n--- Iteration {iteration + 1} for gage {self.gage_id} ---")
            w = w_start - (w_start - w_end) * (iteration / self.n_iterations)

            results = [
                objective_function(
                    (p.position, i, self.gage_id, self.config_path, self.observed_path, self.postproc_base_path, self.include_nom, self.nom_file_path)
                )
                for i, p in enumerate(self.particles)
            ]


            for idx, (objective_value, val_metrics, cal_metrics) in enumerate(results):
                particle = self.particles[idx]
                particle.current_value = objective_value

                try:
                    metric_calibration = cal_metrics[self.metric_to_calibrate_on]
                    metric_validation = val_metrics[self.metric_to_calibrate_on]
                except KeyError:
                    raise ValueError(f"Metric '{self.metric_to_calibrate_on}' not found in computed metrics.")

                if objective_value < (particle.best_value - 0*0.001):
                    particle.best_value = objective_value
                    particle.best_position = np.copy(particle.position)
                    particle.stagnation_counter = 0
                else:
                    particle.stagnation_counter += 1

                if objective_value < self.global_best_value:
                    self.global_best_value = objective_value
                    self.global_best_position = np.copy(particle.position)
                    self.best_calibration_metric = metric_calibration
                    self.best_validation_metric = metric_validation
                    self.best_cal_metrics = cal_metrics
                    self.best_val_metrics = val_metrics
                    particle.stagnation_counter = 0

                    # Save new best hydrograph to /postproc/{gage_id}_best.csv
                    try:
                        particle_csv = os.path.join(self.postproc_base_path, f"{self.gage_id}_particle_{idx}.csv")
                        best_csv = os.path.join(self.postproc_base_path, f"{self.gage_id}_best.csv")
                        df = pd.read_csv(particle_csv)
                        df.to_csv(best_csv, index=False)
                    except Exception as e:
                        print(f" Failed to save best hydrograph for {self.gage_id}: {e}")



                if particle.stagnation_counter >= stagnation_threshold:
                    print(f" Resetting particle {idx} after {stagnation_threshold} stagnant iterations.")
                    particle.reset(self.bounds)

                # Logging each particle
                param_dict = {name: val for name, val in zip(self.param_names, particle.position)}
                row = {
                    "iteration": iteration + 1,
                    "particle": idx,
                    **param_dict,
                    f"{self.metric_to_calibrate_on}_calibration": cal_metrics.get(self.metric_to_calibrate_on, np.nan),
                    f"{self.metric_to_calibrate_on}_validation": val_metrics.get(self.metric_to_calibrate_on, np.nan)
                }
                log_rows.append(row)


            pd.DataFrame(log_rows).to_csv(log_path, index=False)

            for p in self.particles:
                p.update_velocity(self.global_best_position, w=w, c1=1.5, c2=1.5)
                p.update_position(self.bounds)

            print(f"Global best objective so far: {-self.global_best_value:.4f}")

        # Save final summary row
        params = self.global_best_position
        param_dict = {name: val for name, val in zip(param_names, params)}
        metric_to_log_fields = {
            "kge": ("kge_calibration", "kge_validation"),
            "log_kge": ("log_kge_calibration", "log_kge_validation"),
            "event_kge": ("event_kge_calibration", "event_kge_validation")
        }
        calibration_field, validation_field = metric_to_log_fields[self.metric_to_calibrate_on]

        # === Final full-period run using best parameters ===
        print(f"\nRunning final full-period validation for {self.gage_id}...")

        # Write best params to config
        true_best_params = transform_params(self.global_best_position)
        regenerate_cfe_config(self.config_path, true_best_params)

        if self.include_nom:
            nom_vals = self.global_best_position[-6:]
            nom_param_dict = dict(zip(nom_param_names, nom_vals))
            update_mptable(
                original_file=self.nom_file_path,
                output_file=self.nom_file_path,
                updated_params=nom_param_dict,
                verbose=False
            )


        # Update realization to use full period
        json_dir = os.path.join(cfg.model_roots[0], "out", self.gage_id, "json")
        realization_path = [f for f in os.listdir(json_dir) if f.endswith(".json")][0]
        realization_path = os.path.join(json_dir, realization_path)

        with open(realization_path, "r") as f:
            realization_config = json.load(f)

        realization_config["time"]["start_time"] = time_cfg["spinup_start"]
        realization_config["time"]["end_time"]   = time_cfg["val_end"]

        with open(realization_path, "w") as f:
            json.dump(realization_config, f, indent=4)

        # Update troute_config.yaml again
        troute_path = os.path.join(cfg.model_roots[0], "out", self.gage_id, "configs", "troute_config.yaml")
        with open(troute_path, "r") as f:
            troute_cfg = yaml.safe_load(f)

        spinup_start = pd.Timestamp(time_cfg["spinup_start"])
        val_end = pd.Timestamp(time_cfg["val_end"])
        nts_full = int((val_end - spinup_start) / pd.Timedelta(seconds=300))
        troute_cfg["compute_parameters"]["restart_parameters"]["start_datetime"] = spinup_start.strftime("%Y-%m-%d_%H:%M:%S")
        troute_cfg["compute_parameters"]["forcing_parameters"]["nts"] = nts_full

        with open(troute_path, "w") as f:
            yaml.dump(troute_cfg, f)

        # Run model
        subprocess.call(["python", "sandbox.py", "-run", "--gage_id", self.gage_id], cwd=project_root)

        # Extract and overwrite final best hydrograph
        get_hydrograph_path = os.path.join(project_root, "model_assessment", "util", "get_hydrograph.py")
        output_path = os.path.join(self.postproc_base_path, f"{self.gage_id}_best.csv")
        subprocess.call(
            ["python", get_hydrograph_path, "--gage_id", self.gage_id, "--output", output_path, "--base_dir", cfg.model_roots[0]],
            cwd=self.postproc_base_path
        )
        print(f" Final best hydrograph saved to {output_path}")

        # === Recompute validation metrics over the full period ===
        try:
            sim_df = pd.read_csv(output_path, parse_dates=['current_time']).set_index('current_time')['flow'].resample('1h').mean()
            window_start = pd.Timestamp(time_cfg["spinup_start"])
            window_end = pd.Timestamp(time_cfg["val_end"])

            expected_length = int((window_end - window_start) / pd.Timedelta(hours=1)) + 1
            actual_length = len(sim_df)

            allowed_tolerance = 1
            if abs(actual_length - expected_length) > allowed_tolerance:
                raise RuntimeError(
                    f"Final validation for gage {self.gage_id} produced {actual_length} time steps but expected {expected_length}. "
                    f"Allowed tolerance is ±{allowed_tolerance}. Simulation likely incomplete."
                )

            obs_df = get_observed_q(self.observed_path)

            sim_val, obs_val = sim_df[val_start:val_end].dropna(), obs_df[val_start:val_end].dropna()
            sim_val, obs_val = sim_val.align(obs_val, join='inner')

            if len(sim_val) > 0 and len(obs_val) > 0:
                sim_val.iloc[-1] += 1e-8
                obs_val.iloc[-1] += 1e-8

            val_metrics_final = compute_metrics(sim_val, obs_val, event_threshold=1e-2)

            # Create final row after computing full-period validation metrics
            final_summary = {
                "iteration": "FINAL",
                "particle": "BEST",
                **{name: val for name, val in zip(self.param_names, self.global_best_position)},
                f"{self.metric_to_calibrate_on}_calibration": self.best_cal_metrics.get(self.metric_to_calibrate_on, np.nan),
                f"{self.metric_to_calibrate_on}_validation": val_metrics_final.get(self.metric_to_calibrate_on, np.nan)
            }

            # Overwrite previous log and append only the final row
            pd.DataFrame(log_rows + [final_summary]).to_csv(log_path, index=False)

            # print the metric used for calibration
            metric_key = self.metric_to_calibrate_on
            metric_value = val_metrics_final.get(metric_key, None)
            if metric_value is not None:
                print(f" Final validation {metric_key.upper()}: {metric_value:.4f}")
            else:
                print(f" Final validation metric '{metric_key}' not found.")

        except Exception as e:
            print(f"Warning: Could not compute final validation metrics: {e}")


        return self.global_best_position, self.global_best_value, self.best_validation_metric, datetime.now() - start_time


# === PER-GAGE WRAPPER ===
def calibrate_gage(gage_id):
    import os
    import numpy as np

    observed_q_root = cfg.observed_q_root
    model_roots = cfg.model_roots
    root = model_roots[0]  # single-tile for now

    config_dir = os.path.join(root, f"out/{gage_id}/configs/cfe")
    if not os.path.exists(config_dir):
        print(f"Missing config dir for gage {gage_id}: {config_dir}")
        return

    config_path = os.path.join(config_dir, sorted(f for f in os.listdir(config_dir) if f.startswith("cfe_config_cat"))[0])
    observed_path = os.path.join(observed_q_root, "successful_sites_resampled", f"{gage_id}.csv")
    postproc_base_path = os.path.join(root, "postproc")

    # === Extract initial CFE parameters
    init_vals = extract_initial_cfe_params(config_path)
    bounds = param_bounds.copy()
    names = param_names.copy()

    # === Check for NOM config
    nom_config_dir = os.path.join(root, f"out/{gage_id}/configs/noahowp")
    include_nom = os.path.isdir(nom_config_dir)

    if include_nom:
        nom_file_path = os.path.join(nom_config_dir, "parameters", "MPTABLE.TBL")
        nom_init_vals = extract_initial_nom_params(nom_file_path)
        init_vals += nom_init_vals
        bounds += nom_param_bounds
        names += nom_param_names

    if len(init_vals) != len(bounds):
        raise ValueError(f"init_vals length {len(init_vals)} does not match bounds length {len(bounds)}")

    # === Launch PSO
    pso = PSO(
        n_particles=n_particles,
        bounds=bounds,
        n_iterations=n_iterations,
        gage_id=gage_id,
        init_position=init_vals,
        config_path=config_path,
        observed_path=observed_path,
        postproc_base_path=postproc_base_path,
        metric_to_calibrate_on="kge",
        include_nom=include_nom,
        nom_file_path=nom_file_path if include_nom else None,
        param_names=names
    )


    best_params, best_obj_value, best_val_metric, runtime = pso.optimize()


# === MAIN EXECUTION ===
if __name__ == "__main__":
    start_time = datetime.now()

    gages_file = os.path.join(project_root, "basin_IDs/basin_IDs.csv")
    gage_list = pd.read_csv(gages_file, dtype={"gage_id": str})["gage_id"].tolist()

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=max_cores_for_gages) as pool:
        pool.map(calibrate_gage, gage_list)

    end_time = datetime.now()
    total_duration = end_time - start_time
    print(f"\n Total wall time: {total_duration}")









































# ##works but only supports 1 tile and no NOM calibration 
# ###############################################################
# # Author      : Peter La Follette [plafollette@lynker.com | May 2025]
# # Calibrates cfe+pet+t-route or cfe+t-route+NOM, if NOM is in the model formulation then some of its parameters will be calibrated
# # currently just 1 tile is supported

# import os
# import subprocess
# import pandas as pd
# import numpy as np
# import math
# from datetime import datetime
# import random
# import multiprocessing
# from hydroeval import kge
# import sys
# import traceback
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# from model_assessment.util.metrics import compute_metrics
# from model_assessment.util.update_NOM import update_mptable
# import yaml
# import json

# np.random.seed(42)
# random.seed(42)

# # === CONFIGURATION ===
# n_particles = 2
# n_iterations = 2
# max_cores_for_gages = 2
# metric_to_calibrate_on = "kge"  # "kge", "log_kge", or "event_kge"

# from model_assessment.configs import path_config as cfg
# with open("model_assessment/configs/time_config.yaml", "r") as f:
#     time_cfg = yaml.safe_load(f)

# cal_start = pd.Timestamp(time_cfg["cal_start"])
# cal_end   = pd.Timestamp(time_cfg["cal_end"])
# val_start = pd.Timestamp(time_cfg["val_start"])
# val_end   = pd.Timestamp(time_cfg["val_end"])

# project_root = cfg.project_root
# logging_dir = cfg.logging_dir
# sandbox_path = cfg.sandbox_path
# observed_q_root = cfg.observed_q_root

# os.makedirs(logging_dir, exist_ok=True)


# def clear_terminal():
#     os.system('clear')

# ### As I was attempting my first calibration runs with 10s of gages running in parallel and hundreds of iterations on MacOS, I had spotlight indexing on.
# ### It is actually the case that you can run out of disk space if spotlight indexing is on and it includes outputs from nextgen, because nextgen model outputs amount to a huge amount of data written per day
# ### To address this, all directories that will contain nextgen outputs at the divide scale, as well as the t-route outputs, will have a .metadata_never_index file created with them during the -conf step in NextGenSandboxHub.
# ### This should make spotlight indexing skip these files and avoid the issue where the available disk space goes to 0, but just to be sure, this function stops the calibration execution in the event that disk space gets too low 
# def check_for_stop_signal_or_low_disk():
#     stop_file = os.path.join(project_root, "STOP_NOW.txt")
#     if os.path.exists(stop_file):
#         return True
#     statvfs = os.statvfs(project_root)
#     free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024 ** 3)
#     return free_gb < 100


# # === PARAMETER SETUP ===
# param_bounds = [
#     (0.0, 21.94),
#     (math.log10(2.77e-10), math.log10(0.000726)),
#     (0.0, 0.995),
#     (0.20554, 0.6),
#     (0.01, 0.2),
#     (math.log10(1.6266e-06), math.log10(0.1)),
#     (1.0, 8.0),
#     (0.0, 1.0),
#     (0.0, 1.0),
#     (0.1, 4.0),
#     (0.0, 1.0),
#     (0.0, 0.138),
#     (0.1, 1.0),
#     (0.001, 0.005)
# ]

# param_names = [
#     "b", "satdk", "satpsi", "maxsmc", "max_gw_storage", "Cgw", "expon", "Kn",
#     "Klf", "refkdt", "slope", "wltsmc", "alpha_fc", "Kinf_nash_surface"
# ]

# log_scale_params = {"Cgw": True, "satdk": True}


# # === HELPERS ===
# def transform_params(params):
#     return [10**p if log_scale_params.get(name, False) else p for name, p in zip(param_names, params)]


# def extract_initial_cfe_params(config_path):
#     param_map = {
#         "b": "soil_params.b",
#         "satdk": "soil_params.satdk",
#         "satpsi": "soil_params.satpsi",
#         "maxsmc": "soil_params.smcmax",
#         "max_gw_storage": "max_gw_storage",
#         "Cgw": "Cgw",
#         "expon": "expon",
#         "Kn": "K_nash_subsurface",
#         "Klf": "K_lf",
#         "refkdt": "refkdt",
#         "slope": "soil_params.slop",
#         "wltsmc": "soil_params.wltsmc",
#         "alpha_fc": "alpha_fc",
#         "Kinf_nash_surface": "Kinf_nash_surface"  # NEW: add Kinf
#     }

#     values = {}
#     with open(config_path, 'r') as f:
#         lines = f.readlines()
#     for line in lines:
#         if "=" in line:
#             key, val = line.strip().split("=", 1)
#             key = key.strip()
#             val = val.split("[")[0].strip()  # Remove units if present
#             for param, config_key in param_map.items():
#                 if key == config_key:
#                     values[param] = float(val)

#     # Build parameter list
#     param_list = []
#     for name in param_names:
#         if name in values:
#             val = values[name]
#         else:
#             val = 0.002  # Always default to 0.002 if missing (old behavior)

#         if log_scale_params.get(name, False):
#             param_list.append(math.log10(val))
#         else:
#             param_list.append(val)

#     return param_list


# def get_observed_q(observed_path):
#     df = pd.read_csv(observed_path, parse_dates=['value_time'])
#     df.set_index('value_time', inplace=True)
#     return df['flow_m3_per_s']


# def regenerate_cfe_config(config_dir, params):
#     replacements = dict(zip(param_names, params))
#     param_map = { # same as above
#         "b": "soil_params.b", "satdk": "soil_params.satdk", "satpsi": "soil_params.satpsi", "maxsmc": "soil_params.smcmax",
#         "max_gw_storage": "max_gw_storage", "Cgw": "Cgw", "expon": "expon", "Kn": "K_nash_subsurface",
#         "Klf": "K_lf", "refkdt": "refkdt", "slope": "soil_params.slop", "wltsmc": "soil_params.wltsmc",
#         "alpha_fc": "alpha_fc", "Kinf_nash_surface": "Kinf_nash_surface"
#     }

#     if os.path.isfile(config_dir):
#         config_dir = os.path.dirname(config_dir)

#     for fname in sorted(f for f in os.listdir(config_dir) if f.startswith("cfe_config_cat")):
#         path = os.path.join(config_dir, fname)
#         lines = []
#         with open(path) as f:
#             for line in f:
#                 key = line.split('=')[0].strip()
#                 for pname, ckey in param_map.items():
#                     if key == ckey:
#                         unit = line[line.find("["):] if "[" in line else ""
#                         lines.append(f"{ckey}={replacements[pname]}{unit}\n")
#                         break
#                 else:
#                     lines.append(line)

#         with open(path, 'w') as f:
#             f.writelines(lines)


# # === OBJECTIVE FUNCTION ===
# def objective_function(args):
#     import json
#     params, particle_idx, gage_id, config_path, observed_path, postproc_base_path = args

#     true_params = transform_params(params)
#     if check_for_stop_signal_or_low_disk():
#         print(f" Stop signal detected, skipping particle {particle_idx}")
#         return 10.0, -10.0, {}, {}

#     print(f"\nEvaluating particle {particle_idx} for gage {gage_id}")

#     regenerate_cfe_config(config_path, true_params)

#     try:
#         # === Update realization JSON to use spinup + calibration only ===
#         json_dir = os.path.join(cfg.model_roots[0], "out", gage_id, "json")
#         realization_path = [f for f in os.listdir(json_dir) if f.endswith(".json")][0]
#         realization_path = os.path.join(json_dir, realization_path)

#         with open(realization_path, "r") as f:
#             realization_config = json.load(f)

#         realization_config["time"]["start_time"] = time_cfg["spinup_start"]
#         realization_config["time"]["end_time"]   = time_cfg["cal_end"]

#         with open(realization_path, "w") as f:
#             json.dump(realization_config, f, indent=4)

#         # === Update troute_config.yaml ===
#         troute_path = os.path.join(cfg.model_roots[0], "out", gage_id, "configs", "troute_config.yaml")
#         with open(troute_path, "r") as f:
#             troute_cfg = yaml.safe_load(f)

#         spinup_start = pd.Timestamp(time_cfg["spinup_start"])
#         nts = int((cal_end - spinup_start) / pd.Timedelta(seconds=300))
#         troute_cfg["compute_parameters"]["restart_parameters"]["start_datetime"] = spinup_start.strftime("%Y-%m-%d_%H:%M:%S")
#         troute_cfg["compute_parameters"]["forcing_parameters"]["nts"] = nts

#         with open(troute_path, "w") as f:
#             yaml.dump(troute_cfg, f)

#         # === Run the model ===
#         subprocess.call(["python", "sandbox.py", "-run", "--gage_id", gage_id], cwd=project_root)

#         # === Extract output ===
#         env = os.environ.copy()
#         env["PARTICLE_ID"] = str(particle_idx)

#         get_hydrograph_path = os.path.join(project_root, "model_assessment", "util", "get_hydrograph.py")
#         tile_root = cfg.model_roots[0]
#         output_path = os.path.join(tile_root, "postproc", f"{gage_id}_particle_{particle_idx}.csv")

#         subprocess.call(
#             ["python", get_hydrograph_path, "--gage_id", gage_id, "--output", output_path, "--base_dir", tile_root],
#             cwd=os.path.join(tile_root, "postproc")
#         )

#         sim_df = pd.read_csv(output_path, parse_dates=['current_time']).set_index('current_time')['flow'].resample('1h').mean()
#         obs_df = get_observed_q(observed_path)

#         sim_cal, obs_cal = sim_df[cal_start:cal_end].dropna(), obs_df[cal_start:cal_end].dropna()
#         sim_val, obs_val = sim_df[val_start:val_end].dropna(), obs_df[val_start:val_end].dropna()

#         sim_cal, obs_cal = sim_cal.align(obs_cal, join='inner')
#         sim_val, obs_val = sim_val.align(obs_val, join='inner')

#         if len(sim_val) > 0 and len(obs_val) > 0:
#             sim_val.iloc[-1] += 1e-8
#             obs_val.iloc[-1] += 1e-8

#         if len(sim_cal) > 0 and len(obs_cal) > 0:
#             sim_cal.iloc[-1] += 1e-8
#             obs_cal.iloc[-1] += 1e-8

#         cal_metrics = compute_metrics(sim_cal, obs_cal, event_threshold=1e-2)
#         val_metrics = compute_metrics(sim_val, obs_val, event_threshold=1e-2)

#         return -cal_metrics[metric_to_calibrate_on], val_metrics, cal_metrics

#     except Exception as e:
#         print(f"Error during evaluation: {e}")
#         traceback.print_exc()
#         placeholder_metrics = {
#             "kge": -10.0,
#             "log_kge": -10.0,
#             "volume_error_percent": -999.0,
#             "peak_time_error_hours": -999.0,
#             "peak_flow_error_percent": -999.0,
#             "event_kge": -10.0,
#             "event_hours": 0,
#             "total_hours": 0
#         }
#         return 10.0, placeholder_metrics, placeholder_metrics


# class Particle:
#     def __init__(self, bounds, init_position=None):
#         if init_position is not None:
#             self.position = np.array(init_position)
#         else:
#             self.position = np.array([np.random.uniform(low, high) for low, high in bounds])

#         self.velocity = np.array([0.1 * (high - low) * np.random.uniform(-1, 1) for low, high in bounds])
#         self.best_position = np.copy(self.position)
#         self.best_value = float('inf')
#         self.current_value = float('inf')
#         self.stagnation_counter = 0
#         self.best_calibration_metric = -9999.0  # or np.nan
#         self.best_validation_metric = -9999.0  # or np.nan


#     def reset(self, bounds):
#         self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
#         self.velocity = np.array([0.1 * (high - low) * np.random.uniform(-1, 1) for low, high in bounds])
#         self.best_position = np.copy(self.position)
#         self.best_value = float('inf')
#         self.current_value = float('inf')
#         self.stagnation_counter = 0

#     def update_velocity(self, global_best_position, w, c1, c2):
#         r1 = np.random.rand(len(self.position))
#         r2 = np.random.rand(len(self.position))
#         cognitive = c1 * r1 * (self.best_position - self.position)
#         social = c2 * r2 * (global_best_position - self.position)
#         self.velocity = w * self.velocity + cognitive + social

#     def update_position(self, bounds):
#         self.position += self.velocity
#         for i in range(len(self.position)):
#             if self.position[i] < bounds[i][0]:
#                 self.position[i] = bounds[i][0]; self.velocity[i] *= -0.5
#             elif self.position[i] > bounds[i][1]:
#                 self.position[i] = bounds[i][1]; self.velocity[i] *= -0.5


# class PSO:
#     def __init__(self, n_particles, bounds, n_iterations, gage_id, config_path, observed_path, postproc_base_path, metric_to_calibrate_on="kge"):
#         self.particles = [Particle(bounds, init_position=extract_initial_cfe_params(config_path) if i == 0 else None) for i in range(n_particles)]
#         self.bounds = bounds
#         self.n_iterations = n_iterations
#         self.gage_id = gage_id
#         self.config_path = config_path
#         self.observed_path = observed_path
#         self.postproc_base_path = postproc_base_path
#         self.global_best_position = self.particles[0].position
#         self.global_best_value = float('inf')
#         self.metric_to_calibrate_on = metric_to_calibrate_on
#         self.best_cal_metrics = {}
#         self.best_val_metrics = {}

#     def optimize(self):
#         start_time = datetime.now()
#         log_rows = []
#         log_path = os.path.join(logging_dir, f"{self.gage_id}.csv")
#         stagnation_threshold = 10
#         w_start, w_end = 0.9, 0.4

#         for iteration in range(self.n_iterations):
#             clear_terminal()
#             print(f"\n--- Iteration {iteration + 1} for gage {self.gage_id} ---")
#             w = w_start - (w_start - w_end) * (iteration / self.n_iterations)

#             results = [
#                 objective_function(
#                     (p.position, i, self.gage_id, self.config_path, self.observed_path, self.postproc_base_path)
#                 )
#                 for i, p in enumerate(self.particles)
#             ]


#             for idx, (objective_value, val_metrics, cal_metrics) in enumerate(results):
#                 particle = self.particles[idx]
#                 particle.current_value = objective_value

#                 try:
#                     metric_calibration = cal_metrics[self.metric_to_calibrate_on]
#                     metric_validation = val_metrics[self.metric_to_calibrate_on]
#                 except KeyError:
#                     raise ValueError(f"Metric '{self.metric_to_calibrate_on}' not found in computed metrics.")

#                 if objective_value < (particle.best_value - 0*0.001):
#                     particle.best_value = objective_value
#                     particle.best_position = np.copy(particle.position)
#                     particle.stagnation_counter = 0
#                 else:
#                     particle.stagnation_counter += 1

#                 if objective_value < self.global_best_value:
#                     self.global_best_value = objective_value
#                     self.global_best_position = np.copy(particle.position)
#                     self.best_calibration_metric = metric_calibration
#                     self.best_validation_metric = metric_validation
#                     self.best_cal_metrics = cal_metrics
#                     self.best_val_metrics = val_metrics
#                     particle.stagnation_counter = 0

#                     # Save new best hydrograph to /postproc/{gage_id}_best.csv
#                     try:
#                         particle_csv = os.path.join(self.postproc_base_path, f"{self.gage_id}_particle_{idx}.csv")
#                         best_csv = os.path.join(self.postproc_base_path, f"{self.gage_id}_best.csv")
#                         df = pd.read_csv(particle_csv)
#                         df.to_csv(best_csv, index=False)
#                     except Exception as e:
#                         print(f" Failed to save best hydrograph for {self.gage_id}: {e}")



#                 if particle.stagnation_counter >= stagnation_threshold:
#                     print(f" Resetting particle {idx} after {stagnation_threshold} stagnant iterations.")
#                     particle.reset(self.bounds)

#                 # Logging each particle
#                 param_dict = {name: val for name, val in zip(param_names, particle.position)}
#                 row = {
#                     "iteration": iteration + 1,
#                     "particle": idx,
#                     **param_dict,
#                     f"{self.metric_to_calibrate_on}_calibration": cal_metrics.get(self.metric_to_calibrate_on, np.nan),
#                     f"{self.metric_to_calibrate_on}_validation": val_metrics.get(self.metric_to_calibrate_on, np.nan)
#                 }
#                 log_rows.append(row)


#             pd.DataFrame(log_rows).to_csv(log_path, index=False)

#             for p in self.particles:
#                 p.update_velocity(self.global_best_position, w=w, c1=1.5, c2=1.5)
#                 p.update_position(self.bounds)

#             print(f"Global best objective so far: {-self.global_best_value:.4f}")

#         # Save final summary row
#         params = self.global_best_position
#         param_dict = {name: val for name, val in zip(param_names, params)}
#         metric_to_log_fields = {
#             "kge": ("kge_calibration", "kge_validation"),
#             "log_kge": ("log_kge_calibration", "log_kge_validation"),
#             "event_kge": ("event_kge_calibration", "event_kge_validation")
#         }
#         calibration_field, validation_field = metric_to_log_fields[self.metric_to_calibrate_on]

#         # === Final full-period run using best parameters ===
#         print(f"\nRunning final full-period validation for {self.gage_id}...")

#         # Write best params to config
#         true_best_params = transform_params(self.global_best_position)
#         regenerate_cfe_config(self.config_path, true_best_params)

#         # Update realization to use full period
#         json_dir = os.path.join(cfg.model_roots[0], "out", self.gage_id, "json")
#         realization_path = [f for f in os.listdir(json_dir) if f.endswith(".json")][0]
#         realization_path = os.path.join(json_dir, realization_path)

#         with open(realization_path, "r") as f:
#             realization_config = json.load(f)

#         realization_config["time"]["start_time"] = time_cfg["spinup_start"]
#         realization_config["time"]["end_time"]   = time_cfg["val_end"]

#         with open(realization_path, "w") as f:
#             json.dump(realization_config, f, indent=4)

#         # Update troute_config.yaml again
#         troute_path = os.path.join(cfg.model_roots[0], "out", self.gage_id, "configs", "troute_config.yaml")
#         with open(troute_path, "r") as f:
#             troute_cfg = yaml.safe_load(f)

#         spinup_start = pd.Timestamp(time_cfg["spinup_start"])
#         val_end = pd.Timestamp(time_cfg["val_end"])
#         nts_full = int((val_end - spinup_start) / pd.Timedelta(seconds=300))
#         troute_cfg["compute_parameters"]["restart_parameters"]["start_datetime"] = spinup_start.strftime("%Y-%m-%d_%H:%M:%S")
#         troute_cfg["compute_parameters"]["forcing_parameters"]["nts"] = nts_full

#         with open(troute_path, "w") as f:
#             yaml.dump(troute_cfg, f)

#         # Run model
#         subprocess.call(["python", "sandbox.py", "-run", "--gage_id", self.gage_id], cwd=project_root)

#         # Extract and overwrite final best hydrograph
#         get_hydrograph_path = os.path.join(project_root, "model_assessment", "util", "get_hydrograph.py")
#         output_path = os.path.join(self.postproc_base_path, f"{self.gage_id}_best.csv")
#         subprocess.call(
#             ["python", get_hydrograph_path, "--gage_id", self.gage_id, "--output", output_path, "--base_dir", cfg.model_roots[0]],
#             cwd=self.postproc_base_path
#         )
#         print(f" Final best hydrograph saved to {output_path}")

#         # === Recompute validation metrics over the full period ===
#         try:
#             sim_df = pd.read_csv(output_path, parse_dates=['current_time']).set_index('current_time')['flow'].resample('1h').mean()
#             obs_df = get_observed_q(self.observed_path)

#             sim_val, obs_val = sim_df[val_start:val_end].dropna(), obs_df[val_start:val_end].dropna()
#             sim_val, obs_val = sim_val.align(obs_val, join='inner')

#             if len(sim_val) > 0 and len(obs_val) > 0:
#                 sim_val.iloc[-1] += 1e-8
#                 obs_val.iloc[-1] += 1e-8

#             val_metrics_final = compute_metrics(sim_val, obs_val, event_threshold=1e-2)

#             # Create final row after computing full-period validation metrics
#             final_summary = {
#                 "iteration": "FINAL",
#                 "particle": "BEST",
#                 **{name: val for name, val in zip(param_names, self.global_best_position)},
#                 f"{self.metric_to_calibrate_on}_calibration": self.best_cal_metrics.get(self.metric_to_calibrate_on, np.nan),
#                 f"{self.metric_to_calibrate_on}_validation": val_metrics_final.get(self.metric_to_calibrate_on, np.nan)
#             }

#             # Overwrite previous log and append only the final row
#             pd.DataFrame(log_rows + [final_summary]).to_csv(log_path, index=False)

#             # print the metric used for calibration
#             metric_key = self.metric_to_calibrate_on
#             metric_value = val_metrics_final.get(metric_key, None)
#             if metric_value is not None:
#                 print(f" Final validation {metric_key.upper()}: {metric_value:.4f}")
#             else:
#                 print(f" Final validation metric '{metric_key}' not found.")

#         except Exception as e:
#             print(f"Warning: Could not compute final validation metrics: {e}")


#         return self.global_best_position, self.global_best_value, self.best_validation_metric, datetime.now() - start_time


# # === PER-GAGE WRAPPER ===
# def calibrate_gage(gage_id):
#     model_root = cfg.model_roots[0]
#     tile_root = model_root
#     config_dir = os.path.join(model_root, f"out/{gage_id}/configs/cfe")
#     config_path = os.path.join(config_dir, sorted(f for f in os.listdir(config_dir) if f.startswith("cfe_config_cat"))[0])
#     # observed_path = os.path.join(project_root, f"USGS_streamflow/successful_sites_resampled/{gage_id}.csv")
#     observed_path = os.path.join(observed_q_root, "successful_sites_resampled", f"{gage_id}.csv")
#     postproc_base_path = os.path.join(model_root, "postproc")

#     pso = PSO(
#         n_particles,
#         param_bounds,
#         n_iterations,
#         gage_id,
#         config_path,
#         observed_path,
#         postproc_base_path,
#         metric_to_calibrate_on="kge"
#     )
#     best_params, best_obj_value, best_val_metric, runtime = pso.optimize()


# # === MAIN EXECUTION ===
# if __name__ == "__main__":
#     start_time = datetime.now()

#     gages_file = os.path.join(project_root, "out/basins_passed_custom.csv")
#     gage_list = pd.read_csv(gages_file, dtype={"gage_id": str})["gage_id"].tolist()

#     ctx = multiprocessing.get_context("spawn")
#     with ctx.Pool(processes=max_cores_for_gages) as pool:
#         pool.map(calibrate_gage, gage_list)

#     end_time = datetime.now()
#     total_duration = end_time - start_time
#     print(f"\n Total wall time: {total_duration}")



