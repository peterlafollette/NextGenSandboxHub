# Updated DDS calibration for CFE with full PSO parity
import os
import pandas as pd
import numpy as np
import random
import json
import yaml
from datetime import datetime
import multiprocessing
import subprocess
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from model_assessment.configs import path_config as cfg
from model_assessment.util.metrics import compute_metrics
from model_assessment.util.update_NOM import update_mptable

from pso_calibration_cfe import (
    check_for_stop_signal_or_low_disk,
    extract_initial_cfe_params,
    regenerate_cfe_config,
    objective_function,
    param_bounds,
    param_names,
    log_scale_params,
)

with open("model_assessment/configs/time_config.yaml", "r") as f:
    time_cfg = yaml.safe_load(f)

cal_start = pd.Timestamp(time_cfg["cal_start"])
cal_end   = pd.Timestamp(time_cfg["cal_end"])
val_start = pd.Timestamp(time_cfg["val_start"])
val_end   = pd.Timestamp(time_cfg["val_end"])
spinup_start = pd.Timestamp(time_cfg["spinup_start"])

np.random.seed(42)
random.seed(42)

n_iterations = 2
max_cores_for_gages = 2
metric_to_calibrate_on = "kge"

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def reflect_bounds(x, low, high):
    if x < low:
        return low + (low - x)
    elif x > high:
        return high - (x - high)
    else:
        return x

class DDS:
    def __init__(self, bounds, n_iterations, gage_id, config_path, observed_path, postproc_base_path,
                 init_params, metric_to_calibrate_on="kge", sigma=0.2):
        self.bounds = bounds
        self.n_iterations = n_iterations
        self.gage_id = gage_id
        self.config_path = config_path
        self.observed_path = observed_path
        self.postproc_base_path = postproc_base_path
        self.metric = metric_to_calibrate_on
        self.sigma = sigma
        self.best_position = np.copy(init_params)
        self.best_value = float("inf")

    def optimize(self):
        log_path = os.path.join(cfg.logging_dir, f"{self.gage_id}.csv")
        log_rows = []
        num_params = len(self.bounds)
        start_time = datetime.now()

        # === Initial run (spinup + calibration only)
        print(f"\n--- Initial evaluation (iteration 0) for gage {self.gage_id} ---")
        obj_val, val_metrics, cal_metrics = objective_function(
            (self.best_position, 0, self.gage_id, self.config_path, self.observed_path, self.postproc_base_path)
        )
        self.best_value = obj_val
        best_val_metrics = val_metrics
        best_cal_metrics = cal_metrics

        row = {
            "iteration": 0,
            "particle": 0,
            **dict(zip(param_names, self.best_position)),
            f"{self.metric}_calibration": cal_metrics.get(self.metric, np.nan),
            f"{self.metric}_validation": val_metrics.get(self.metric, np.nan),
        }

        log_rows.append(row)

        # === DDS main loop
        for iteration in range(1, self.n_iterations + 1):
            # clear_terminal()
            print(f"\n--- Iteration {iteration} for gage {self.gage_id} ---")

            if check_for_stop_signal_or_low_disk():
                break

            p = 1 - np.log(iteration) / np.log(self.n_iterations)
            perturb_mask = np.random.rand(num_params) < p
            if not np.any(perturb_mask):
                perturb_mask[np.random.randint(0, num_params)] = True

            candidate = np.copy(self.best_position)
            for i in range(num_params):
                if perturb_mask[i]:
                    low, high = self.bounds[i]
                    perturb = np.random.normal(0, self.sigma) * (high - low)
                    candidate[i] += perturb
                    candidate[i] = reflect_bounds(reflect_bounds(candidate[i], low, high), low, high)
                    candidate[i] = np.clip(candidate[i], low, high)

            obj_val, val_metrics, cal_metrics = objective_function(
                (candidate, 0, self.gage_id, self.config_path, self.observed_path, self.postproc_base_path)
            )

            if obj_val < self.best_value:
                self.best_value = obj_val
                self.best_position = candidate
                best_val_metrics = val_metrics
                best_cal_metrics = cal_metrics

            row = {
                "iteration": iteration,
                "particle": 0,
                **dict(zip(param_names, candidate)),
                f"{self.metric}_calibration": cal_metrics.get(self.metric, np.nan),
                f"{self.metric}_validation": val_metrics.get(self.metric, np.nan),
            }
            log_rows.append(row)

        # === Final full-period run ===
        print(f"\nRunning final full-period validation for {self.gage_id}...")
        from pso_calibration_cfe import transform_params
        true_best_params = transform_params(self.best_position)
        regenerate_cfe_config(self.config_path, true_best_params)

        json_dir = os.path.join(cfg.model_roots[0], "out", self.gage_id, "json")
        realization_path = [f for f in os.listdir(json_dir) if f.endswith(".json")][0]
        realization_path = os.path.join(json_dir, realization_path)

        with open(realization_path, "r") as f:
            realization_config = json.load(f)
        realization_config["time"]["start_time"] = time_cfg["spinup_start"]
        realization_config["time"]["end_time"]   = time_cfg["val_end"]
        with open(realization_path, "w") as f:
            json.dump(realization_config, f, indent=4)

        troute_path = os.path.join(cfg.model_roots[0], "out", self.gage_id, "configs", "troute_config.yaml")
        with open(troute_path, "r") as f:
            troute_cfg = yaml.safe_load(f)
        nts_full = int((val_end - spinup_start) / pd.Timedelta(seconds=300))
        troute_cfg["compute_parameters"]["restart_parameters"]["start_datetime"] = spinup_start.strftime("%Y-%m-%d_%H:%M:%S")
        troute_cfg["compute_parameters"]["forcing_parameters"]["nts"] = nts_full
        with open(troute_path, "w") as f:
            yaml.dump(troute_cfg, f)

        subprocess.call(["python", "sandbox.py", "-run", "--gage_id", self.gage_id], cwd=cfg.project_root)

        get_hydrograph_path = os.path.join(cfg.project_root, "model_assessment", "util", "get_hydrograph.py")
        output_path = os.path.join(self.postproc_base_path, f"{self.gage_id}_best.csv")
        subprocess.call([
            "python", get_hydrograph_path, "--gage_id", self.gage_id, "--output", output_path, "--base_dir", cfg.model_roots[0]
        ], cwd=self.postproc_base_path)

        sim_df = pd.read_csv(output_path, parse_dates=['current_time']).set_index('current_time')['flow'].resample('1h').mean()
        obs_df = pd.read_csv(self.observed_path, parse_dates=['value_time']).set_index('value_time')['flow_m3_per_s']
        sim_val, obs_val = sim_df[val_start:val_end].dropna(), obs_df[val_start:val_end].dropna()
        sim_val, obs_val = sim_val.align(obs_val, join='inner')
        val_metrics_final = compute_metrics(sim_val, obs_val, event_threshold=1e-2)

        summary_row = {
            "iteration": "FINAL",
            "particle": "BEST",
            **dict(zip(param_names, self.best_position)),
            f"{self.metric}_calibration": best_cal_metrics.get(self.metric, np.nan),
            f"{self.metric}_validation": val_metrics_final.get(self.metric, np.nan),
        }
        log_rows.append(summary_row)
        pd.DataFrame(log_rows).to_csv(log_path, index=False)
        print(f"\n Finished DDS for gage {self.gage_id}. Best obj = {-self.best_value:.4f} | Time: {datetime.now() - start_time}")
        return self.best_position, self.best_value, datetime.now() - start_time

def calibrate_gage_dds(gage_id):
    model_root = cfg.model_roots[0]
    config_dir = os.path.join(model_root, f"out/{gage_id}/configs/cfe")
    config_path = os.path.join(config_dir, sorted(f for f in os.listdir(config_dir) if f.startswith("cfe_config_cat"))[0])
    observed_path = os.path.join(cfg.observed_q_root, "successful_sites_resampled", f"{gage_id}.csv")
    postproc_path = os.path.join(model_root, "postproc")

    init_params = extract_initial_cfe_params(config_path)
    dds = DDS(param_bounds, n_iterations, gage_id, config_path, observed_path, postproc_path, init_params)
    dds.optimize()

if __name__ == "__main__":
    gages_file = os.path.join(cfg.project_root, "out/basins_passed_custom.csv")
    gage_list = pd.read_csv(gages_file, dtype={"gage_id": str})["gage_id"].tolist()

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=max_cores_for_gages) as pool:
        pool.map(calibrate_gage_dds, gage_list)



































