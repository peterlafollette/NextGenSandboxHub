# Refactored PSO calibration script for CFE

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
project_root = "/Users/peterlafollette/NextGenSandboxHub"
logging_dir = os.path.join(project_root, "logging")
os.makedirs(logging_dir, exist_ok=True)
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


def clear_terminal():
    os.system('clear')


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

log_scale_params = {"Cgw": True, "satdk": True}


# === HELPERS ===
def transform_params(params):
    return [10**p if log_scale_params.get(name, False) else p for name, p in zip(param_names, params)]


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
    params, particle_idx, gage_id, config_path, observed_path, postproc_base_path = args

    true_params = transform_params(params)
    if check_for_stop_signal_or_low_disk():
        print(f" Stop signal detected, skipping particle {particle_idx}")
        return 10.0, -10.0, {}, {}

    print(f"\nEvaluating particle {particle_idx} for gage {gage_id}")

    regenerate_cfe_config(config_path, true_params)

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
        print(f"Error during evaluation: {e}")
        traceback.print_exc()
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
    def __init__(self, n_particles, bounds, n_iterations, gage_id, config_path, observed_path, postproc_base_path, metric_to_calibrate_on="kge"):
        self.particles = [Particle(bounds, init_position=extract_initial_cfe_params(config_path) if i == 0 else None) for i in range(n_particles)]
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
                    (p.position, i, self.gage_id, self.config_path, self.observed_path, self.postproc_base_path)
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
                param_dict = {name: val for name, val in zip(param_names, particle.position)}
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

        # summary_row = {
        #     "iteration": "BEST",
        #     "particle": "BEST",
        #     calibration_field: self.best_calibration_metric,
        #     validation_field: self.best_validation_metric,
        #     "kge_calibration": self.best_cal_metrics.get("kge", np.nan),
        #     "kge_validation": self.best_val_metrics.get("kge", np.nan),
        #     "log_kge_calibration": self.best_cal_metrics.get("log_kge", np.nan),
        #     "log_kge_validation": self.best_val_metrics.get("log_kge", np.nan),
        #     "event_kge_calibration": self.best_cal_metrics.get("event_kge", np.nan),
        #     "event_kge_validation": self.best_val_metrics.get("event_kge", np.nan),
        #     **param_dict
        # }


        # === Final full-period run using best parameters ===
        print(f"\nRunning final full-period validation for {self.gage_id}...")

        # Write best params to config
        true_best_params = transform_params(self.global_best_position)
        regenerate_cfe_config(self.config_path, true_best_params)

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
                **{name: val for name, val in zip(param_names, self.global_best_position)},
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


            # update the log file
            # final_row = {
            #     "iteration": "FINAL_FULL",
            #     "particle": "BEST",
            #     **{name: val for name, val in zip(param_names, self.global_best_position)},
            #     "kge_validation": val_metrics_final.get("kge", np.nan),
            #     "log_kge_validation": val_metrics_final.get("log_kge", np.nan),
            #     "event_kge_validation": val_metrics_final.get("event_kge", np.nan),
            # }

            # pd.concat([
            #     pd.read_csv(log_path),
            #     pd.DataFrame([final_row])
            # ]).to_csv(log_path, index=False)

        except Exception as e:
            print(f"Warning: Could not compute final validation metrics: {e}")


        return self.global_best_position, self.global_best_value, self.best_validation_metric, datetime.now() - start_time


# === PER-GAGE WRAPPER ===
def calibrate_gage(gage_id):
    model_root = cfg.model_roots[0]
    tile_root = model_root
    config_dir = os.path.join(model_root, f"out/{gage_id}/configs/cfe")
    config_path = os.path.join(config_dir, sorted(f for f in os.listdir(config_dir) if f.startswith("cfe_config_cat"))[0])
    # observed_path = os.path.join(project_root, f"USGS_streamflow/successful_sites_resampled/{gage_id}.csv")
    observed_path = os.path.join(observed_q_root, "successful_sites_resampled", f"{gage_id}.csv")
    postproc_base_path = os.path.join(model_root, "postproc")

    pso = PSO(
        n_particles,
        param_bounds,
        n_iterations,
        gage_id,
        config_path,
        observed_path,
        postproc_base_path,
        metric_to_calibrate_on="kge"
    )
    best_params, best_obj_value, best_val_metric, runtime = pso.optimize()


# === MAIN EXECUTION ===
if __name__ == "__main__":
    start_time = datetime.now()

    gages_file = os.path.join(project_root, "out/basins_passed_custom.csv")
    gage_list = pd.read_csv(gages_file, dtype={"gage_id": str})["gage_id"].tolist()

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=max_cores_for_gages) as pool:
        pool.map(calibrate_gage, gage_list)

    end_time = datetime.now()
    total_duration = end_time - start_time
    print(f"\n Total wall time: {total_duration}")











































# # Refactored PSO calibration script for CFE, is slower because always runs validation period 

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

# np.random.seed(42)
# random.seed(42)

# # === CONFIGURATION ===
# n_particles = 1
# n_iterations = 1
# max_cores_for_gages = 1
# project_root = "/Users/peterlafollette/NextGenSandboxHub"
# logging_dir = os.path.join(project_root, "logging")
# os.makedirs(logging_dir, exist_ok=True)
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


# def clear_terminal():
#     os.system('clear')


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
#     params, particle_idx, gage_id, config_path, observed_path, postproc_base_path = args

#     true_params = transform_params(params)
#     if check_for_stop_signal_or_low_disk():
#         print(f" Stop signal detected, skipping particle {particle_idx}")
#         return 10.0, -10.0, {}, {}

#     print(f"\nEvaluating particle {particle_idx} for gage {gage_id}")

#     regenerate_cfe_config(config_path, true_params)

#     try:
#         subprocess.call(["python", "sandbox.py", "-run", "--gage_id", gage_id], cwd=project_root)
#         env = os.environ.copy()
#         env["PARTICLE_ID"] = str(particle_idx)

#         # subprocess.call(["python", "get_hydrograph.py", "--gage_id", gage_id], cwd=os.path.join(project_root, "postproc"), env=env)
#         get_hydrograph_path = os.path.join(project_root, "model_assessment", "util", "get_hydrograph.py")
#         tile_root = cfg.model_roots[0]
#         output_path = os.path.join(tile_root, "postproc", f"{gage_id}_particle_{particle_idx}.csv")
#         # output_path = postproc_base_path
#         postproc_dir = os.path.join(cfg.model_roots[0], "postproc")
#         # subprocess.call(
#         #     # ["python", get_hydrograph_path, "--gage_id", gage_id, "--output", output_path, "--base_dir", tile_root],
#         #     ["python", get_hydrograph_path, "--gage_id", gage_id, "--output", output_path, "--base_dir", postproc_dir],
#         #     # cwd=os.path.join(tile_root, "postproc")
#         #     cwd=os.path.join(postproc_dir)
#         # )
#         postproc_dir = os.path.join(tile_root, "postproc")
#         subprocess.call(
#             ["python", get_hydrograph_path, "--gage_id", gage_id, "--output", output_path, "--base_dir", tile_root],
#             cwd=postproc_dir
#         )


#         # postproc_dir = os.path.join(cfg.model_roots[0], "postproc")
#         # subprocess.call(["python", "get_hydrograph.py", "--gage_id", gage_id], cwd=postproc_dir, env=env)

#         sim_path = os.path.join(postproc_base_path, f"{gage_id}_particle_{particle_idx}.csv")
#         sim_df = pd.read_csv(sim_path, parse_dates=['current_time']).set_index('current_time')['flow'].resample('1h').mean()
#         obs_df = get_observed_q(observed_path)

#         cal_start = pd.Timestamp("2014-10-01"); cal_end = pd.Timestamp("2018-09-30")
#         val_start = pd.Timestamp("2018-10-01"); val_end = pd.Timestamp("2020-09-30")

#         sim_cal, obs_cal = sim_df[cal_start:cal_end].dropna(), obs_df[cal_start:cal_end].dropna()
#         sim_val, obs_val = sim_df[val_start:val_end].dropna(), obs_df[val_start:val_end].dropna()

#         sim_cal, obs_cal = sim_cal.align(obs_cal, join='inner')
#         sim_val, obs_val = sim_val.align(obs_val, join='inner')

#         if len(sim_val) > 0 and len(obs_val) > 0:
#             sim_val.iloc[-1] += 1e-8
#             obs_val.iloc[-1] += 1e-8

#         cal_metrics = compute_metrics(sim_cal, obs_cal, event_threshold=1e-2)
#         val_metrics = compute_metrics(sim_val, obs_val, event_threshold=1e-2)

#         metric_calibration = cal_metrics[metric_to_calibrate_on]
#         objective_value = -metric_calibration

#         return objective_value, val_metrics, cal_metrics

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
#             # clear_terminal()
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
#                     "kge_calibration": cal_metrics.get("kge", np.nan),
#                     "kge_validation": val_metrics.get("kge", np.nan),
#                     "log_kge_calibration": cal_metrics.get("log_kge", np.nan),
#                     "log_kge_validation": val_metrics.get("log_kge", np.nan),
#                     "event_kge_calibration": cal_metrics.get("event_kge", np.nan),
#                     "event_kge_validation": val_metrics.get("event_kge", np.nan)
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

#         summary_row = {
#             "iteration": "BEST",
#             "particle": "BEST",
#             calibration_field: self.best_calibration_metric,
#             validation_field: self.best_validation_metric,
#             "kge_calibration": self.best_cal_metrics.get("kge", np.nan),
#             "kge_validation": self.best_val_metrics.get("kge", np.nan),
#             "log_kge_calibration": self.best_cal_metrics.get("log_kge", np.nan),
#             "log_kge_validation": self.best_val_metrics.get("log_kge", np.nan),
#             "event_kge_calibration": self.best_cal_metrics.get("event_kge", np.nan),
#             "event_kge_validation": self.best_val_metrics.get("event_kge", np.nan),
#             **param_dict
#         }

#         pd.DataFrame(log_rows + [summary_row]).to_csv(log_path, index=False)
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
#         metric_to_calibrate_on="event_kge"
#     )
#     best_params, best_obj_value, best_val_metric, runtime = pso.optimize()
#     print(f" Calibration done for {gage_id}: Validation metric = {best_val_metric:.3f}")


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


