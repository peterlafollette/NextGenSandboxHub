#!/usr/bin/env python3
"""
DDS calibration for CASAM using the same particle-local execution path as
pso_calibration_casam.py.
"""

import argparse
import os
import random
import sys
import traceback
import multiprocessing
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pso_calibration_casam as casam

np.random.seed(42)
random.seed(42)

n_iterations = 2
max_cores_for_gages = 1
metric_to_calibrate_on = casam.metric_to_calibrate_on


def reflect_bounds(x, low, high):
    if x < low:
        return low + (low - x)
    if x > high:
        return high - (x - high)
    return x


class DDS:
    def __init__(
        self,
        bounds,
        n_iterations,
        gage_id,
        init_params,
        specs_by_tile,
        tile_counts,
        param_names,
        learn_tile_weight=False,
        metric_to_calibrate_on=metric_to_calibrate_on,
        sigma=0.2,
    ):
        self.bounds = bounds
        self.n_iterations = n_iterations
        self.gage_id = gage_id
        self.best_position = np.copy(init_params)
        self.best_value = float("inf")
        self.specs_by_tile = specs_by_tile
        self.tile_counts = tile_counts
        self.param_names = param_names or [f"p{i}" for i in range(len(bounds))]
        self.learn_tile_weight = learn_tile_weight
        self.metric = metric_to_calibrate_on
        self.sigma = sigma

    def evaluate(self, params, iteration):
        weights = [1.0 / len(casam.model_roots)] * len(casam.model_roots)
        return casam._safe_objective((
            params,
            0,
            self.gage_id,
            casam.model_roots,
            casam.observed_q_root,
            self.specs_by_tile,
            self.tile_counts,
            self.learn_tile_weight,
            weights,
            iteration,
            self.param_names,
        ))

    def log_row(self, start_time, job_cores, iteration, params, status, err, cal_metrics, val_metrics):
        return {
            "iteration": iteration,
            "particle": 0,
            **{name: val for name, val in zip(self.param_names, params)},
            f"{self.metric}_calibration": cal_metrics.get(self.metric, np.nan),
            f"{self.metric}_validation": val_metrics.get(self.metric, np.nan),
            "kge_calibration": cal_metrics.get("kge", np.nan),
            "kge_validation": val_metrics.get("kge", np.nan),
            "mappe_calibration": cal_metrics.get("mappe", np.nan),
            "mappe_validation": val_metrics.get("mappe", np.nan),
            "status": status,
            "error": (err or "")[:240],
            **casam.wall_time_log_fields(start_time, job_cores, 1),
        }

    def optimize(self):
        start_time = datetime.now()
        job_cores = casam.runtime_job_cores(default=1)
        log_path = os.path.join(casam.gage_logging_dir(self.gage_id), f"{self.gage_id}.csv")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_rows = []
        best_cal_metrics = {}
        best_val_metrics = {}

        print(f"\n--- Initial DDS evaluation for gage {self.gage_id} ---")
        status, err, obj_val, val_metrics, cal_metrics = self.evaluate(self.best_position, iteration=0)
        if status == "OK":
            self.best_value = obj_val
            best_cal_metrics = cal_metrics
            best_val_metrics = val_metrics

        log_rows.append(self.log_row(
            start_time, job_cores, 0, self.best_position, status, err, cal_metrics, val_metrics
        ))
        pd.DataFrame(log_rows).to_csv(log_path, index=False)

        num_params = len(self.bounds)
        for iteration in range(1, self.n_iterations + 1):
            casam.check_for_stop_signal_or_low_disk()
            print(f"\n--- DDS Iteration {iteration} for gage {self.gage_id} ---")

            probability = 1.0 if self.n_iterations <= 1 else 1.0 - np.log(iteration) / np.log(self.n_iterations)
            perturb_mask = np.random.rand(num_params) < probability
            if not np.any(perturb_mask):
                perturb_mask[np.random.randint(0, num_params)] = True

            candidate = np.copy(self.best_position)
            for i in range(num_params):
                if perturb_mask[i]:
                    low, high = self.bounds[i]
                    perturb = np.random.normal(0, self.sigma) * (high - low)
                    candidate[i] = reflect_bounds(candidate[i] + perturb, low, high)
                    candidate[i] = max(low, min(high, candidate[i]))

            status, err, obj_val, val_metrics, cal_metrics = self.evaluate(candidate, iteration=iteration)
            if status == "OK" and obj_val < self.best_value:
                self.best_value = obj_val
                self.best_position = candidate
                best_cal_metrics = cal_metrics
                best_val_metrics = val_metrics

            log_rows.append(self.log_row(
                start_time, job_cores, iteration, candidate, status, err, cal_metrics, val_metrics
            ))
            pd.DataFrame(log_rows).to_csv(log_path, index=False)

        if not np.isfinite(self.best_value):
            raise RuntimeError(f"No successful DDS evaluations for gage {self.gage_id}; skipping final validation")

        print(f"\n[INFO] Running final CASAM validation for DDS best parameters: {self.gage_id}")
        final_runner = casam.PSO(
            n_particles=1,
            bounds=self.bounds,
            n_iterations=0,
            gage_id=self.gage_id,
            init_position=self.best_position,
            metric_to_calibrate_on=self.metric,
            param_names=self.param_names,
            specs_by_tile=self.specs_by_tile,
            tile_counts=self.tile_counts,
            learn_tile_weight=self.learn_tile_weight,
            stagnation_threshold=10,
        )
        final_runner.global_best_position = np.copy(self.best_position)
        final_runner.global_best_value = self.best_value
        final_runner.best_cal_metrics = best_cal_metrics
        final_runner.best_val_metrics = best_val_metrics
        final_runner.optimize()

        if os.path.isfile(log_path):
            df = pd.read_csv(log_path)
            if len(df) > 0:
                final_idx = df.index[-1]
                fields = casam.wall_time_log_fields(start_time, job_cores, 1, final=True)
                for key, value in fields.items():
                    df.loc[final_idx, key] = value
                for col in ("status", "error"):
                    if col not in df.columns:
                        df[col] = ""
                    df[col] = df[col].astype("object")
                df.loc[final_idx, "status"] = "OK"
                df.loc[final_idx, "error"] = ""
                df.to_csv(log_path, index=False)

        print(f"\nDDS complete for {self.gage_id}. Best = {-self.best_value:.4f}")
        return self.best_position, self.best_value, datetime.now() - start_time


def calibrate_gage_dds(gage_id):
    try:
        specs_by_tile, tile_counts, bounds, init_params, names = casam.flatten_specs_for_all_tiles(
            gage_id,
            casam.model_roots,
        )

        learn_tile_weight = (
            len(casam.model_roots) == 2
            and getattr(casam, "LEARN_TILE_WEIGHT_IF_2TILES", False)
        )
        if learn_tile_weight:
            init_params.append(0.8)
            bounds.append((0.0, 1.0))
            names.append("tile_weight")

        dds = DDS(
            bounds=bounds,
            n_iterations=n_iterations,
            gage_id=gage_id,
            init_params=init_params,
            specs_by_tile=specs_by_tile,
            tile_counts=tile_counts,
            param_names=names,
            learn_tile_weight=learn_tile_weight,
            metric_to_calibrate_on=metric_to_calibrate_on,
        )
        dds.optimize()
    except Exception as exc:
        print(f"Error calibrating {gage_id}: {exc}")
        traceback.print_exc()
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Run CASAM DDS calibration.")
    parser.add_argument("--gage-id", default=os.environ.get("NGEN_GAGE_ID") or os.environ.get("GAGE_ID"))
    parser.add_argument("--n-iterations", type=int, default=n_iterations)
    parser.add_argument("--max-gage-procs", type=int, default=max_cores_for_gages)
    parser.add_argument("--sandbox-config", default=os.environ.get("NGEN_SANDBOX_CONFIG"))
    parser.add_argument("--spinup-start", default=os.environ.get("NGEN_SPINUP_START"))
    parser.add_argument("--cal-start", default=os.environ.get("NGEN_CAL_START"))
    parser.add_argument("--cal-end", default=os.environ.get("NGEN_CAL_END"))
    parser.add_argument("--val-start", default=os.environ.get("NGEN_VAL_START"))
    parser.add_argument("--val-end", default=os.environ.get("NGEN_VAL_END"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    n_iterations = args.n_iterations
    max_cores_for_gages = args.max_gage_procs
    if args.sandbox_config:
        casam.HYDRO_SANDBOX_CONFIG = casam.resolve_sandbox_config(args.sandbox_config)
        os.environ["NGEN_SANDBOX_CONFIG"] = casam.HYDRO_SANDBOX_CONFIG
    casam.set_time_windows({
        "spinup_start": args.spinup_start,
        "cal_start": args.cal_start,
        "cal_end": args.cal_end,
        "val_start": args.val_start,
        "val_end": args.val_end,
    })

    start = datetime.now()
    if args.gage_id:
        gage_list = [str(args.gage_id).strip()]
    else:
        gage_list = pd.read_csv(casam.cfg.gages_file, dtype={"gage_id": str})["gage_id"].tolist()

    if len(gage_list) == 1:
        calibrate_gage_dds(gage_list[0])
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=max_cores_for_gages) as pool:
            pool.map(calibrate_gage_dds, gage_list)

    print(f"\n=== Total DDS wall time: {datetime.now() - start} ===")
