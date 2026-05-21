"""
Author: Peter La Follette [plafollette@lynker.com | July 2025]
Multi-tile PSO calibration for CFE+PET+T-Route, with optional NOM parameter support.
Now supports concurrent per-iteration particle evaluation with per-particle workspaces.
"""

import os
import argparse
import subprocess
import pandas as pd
import numpy as np
import math
import yaml
import json
import random
import sys
import traceback
import multiprocessing
from datetime import datetime
from hydroeval import kge
import shutil
from pathlib import Path
import glob

# Project imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from model_assessment.util.metrics import compute_metrics
from model_assessment.util.update_NOM import (
    canonical_nom_param,
    update_mptable,
    update_noahowp_model_params,
)
from model_assessment.configs import path_config as cfg
from multiprocessing.pool import ThreadPool


# ==== Repro ====
np.random.seed(42)
random.seed(42)

# === CONFIGURATION ===
n_particles = 15                   # <-- particles per gage
n_iterations = 50
max_cores_for_gages = 1           # one catchment per job by default
metric_to_calibrate_on = "kge"

# run ALL particles concurrently per iteration (inner pool size defaults to n_particles)
max_particle_procs = None         # set to an int to cap, or None to use n_particles

with open("model_assessment/configs/time_config.yaml", "r") as f:
    time_cfg = yaml.safe_load(f)

TIME_FIELDS = ("spinup_start", "cal_start", "cal_end", "val_start", "val_end")

def set_time_windows(overrides=None):
    global spinup_start, cal_start, cal_end, val_start, val_end

    if overrides:
        for key, value in overrides.items():
            if value:
                time_cfg[key] = value

    spinup_start = pd.Timestamp(time_cfg["spinup_start"])
    cal_start = pd.Timestamp(time_cfg["cal_start"])
    cal_end = pd.Timestamp(time_cfg["cal_end"])
    val_start = pd.Timestamp(time_cfg["val_start"])
    val_end = pd.Timestamp(time_cfg["val_end"])

set_time_windows()

sandbox_config_override = os.environ.get("NGEN_SANDBOX_CONFIG")

project_root = cfg.project_root
sandbox_path = cfg.sandbox_path
logging_dir = cfg.logging_dir
observed_q_root = cfg.observed_q_root

os.makedirs(logging_dir, exist_ok=True)

# === Parameter definitions ===
# Defaults follow the upstream CFE-S (Schaake) calibration bounds. satdk and
# Cgw are still optimized in log10-space locally; satdk's upstream lower bound
# is zero, so use a tiny positive floor to keep the transform well-defined.
UPSTREAM_CFES_PARAM_NAMES = [
    "b", "satdk", "satpsi", "maxsmc", "max_gw_storage", "Cgw", "expon", "Kn",
    "Klf", "refkdt", "slope", "wltsmc", "alpha_fc"
]

CFE_LOG_FLOOR = 1e-12

ALL_CFE_PARAM_NAMES = UPSTREAM_CFES_PARAM_NAMES + [
    "Kinf_nash_surface"
]

ALL_CFE_PARAM_BOUNDS = [
    (0.0, 15.0),
    (math.log10(CFE_LOG_FLOOR), math.log10(0.00726)),
    (0.0, 0.995),
    (0.16, 0.6),
    (0.01, 0.25),
    (math.log10(1.8e-06), math.log10(1.8e-3)),
    (1.0, 8.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.001, 8.0),
    (0.0, 1.0),
    (0.0, 0.138),
    (0.15, 0.33),
    (0.001, 0.005)
]

log_scale_params = {"Cgw": True, "satdk": True}

# User-editable list of CFE parameters included in the optimizer vector.
# Default follows upstream CFE-S. Kinf_nash_surface remains available if added
# here explicitly, but it is not part of the upstream CFE-S default set.
CALIBRATION_REQUEST = list(UPSTREAM_CFES_PARAM_NAMES)

def select_cfe_parameters(request):
    bounds_by_name = dict(zip(ALL_CFE_PARAM_NAMES, ALL_CFE_PARAM_BOUNDS))
    selected = [str(name) for name in request]
    unknown = [name for name in selected if name not in bounds_by_name]
    if unknown:
        raise ValueError(f"Unknown CFE calibration parameter(s): {unknown}")
    return selected, [bounds_by_name[name] for name in selected]

param_names, param_bounds = select_cfe_parameters(CALIBRATION_REQUEST)

nom_param_names = ["MFSNO", "SCAMAX", "RSURF_SNOW", "CWP", "VCMX25", "MP", "RSURF_EXP"]
nom_param_bounds = [
    (0.5, 4.0),
    (0.7, 1.0),
    (0.136, 100.0),
    (0.09, 0.36),
    (20.0, 120.0),
    (3.6, 12.6),
    (1.0, 6.0),
]
nom_param_initials = {
    "MFSNO": 2.5,
    "SCAMAX": 0.8,
    "RSURF_SNOW": 50.0,
    "CWP": 0.1,
    "VCMX25": 40.0,
    "MP": 9.0,
    "RSURF_EXP": 3.0,
}
nom_param_bounds_by_name = dict(zip(nom_param_names, nom_param_bounds))

# === Helpers ===
def clear_terminal():
    os.system("clear")

def check_for_stop_signal_or_low_disk(threshold_gb=0):
    stop_file = os.path.join(project_root, "STOP_NOW.txt")
    if os.path.exists(stop_file):
        sys.exit(1)
    stat = os.statvfs("/")
    free_gb = (stat.f_frsize * stat.f_bavail) / (1024 ** 3)
    if free_gb < threshold_gb:
        sys.exit(1)

def runtime_job_cores(default=1):
    explicit = os.environ.get("NGEN_JOB_CORES")
    if explicit:
        try:
            return max(1, int(explicit))
        except ValueError:
            pass

    slurm_tasks = os.environ.get("SLURM_NTASKS")
    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_tasks and slurm_cpus_per_task:
        try:
            return max(1, int(slurm_tasks) * int(slurm_cpus_per_task))
        except ValueError:
            pass

    for name in ("SLURM_CPUS_PER_TASK", "SLURM_NTASKS", "SLURM_NPROCS", "PBS_NP"):
        value = os.environ.get(name)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return max(1, int(default))

def wall_time_log_fields(start_time, job_cores, particle_pool_size, final=False):
    elapsed_seconds = max(0.0, (datetime.now() - start_time).total_seconds())
    core_hours = elapsed_seconds * job_cores / 3600.0
    fields = {
        "elapsed_wall_time_seconds": elapsed_seconds,
        "job_cores": int(job_cores),
        "particle_pool_size": int(particle_pool_size),
        "core_hours_to_row": core_hours,
    }
    if final:
        fields["total_wall_time_seconds"] = elapsed_seconds
        fields["total_core_hours"] = core_hours
    return fields

def sandbox_config_for_tile(tile_idx):
    if sandbox_config_override:
        return sandbox_config_override
    return os.path.join(cfg.project_root, "configs", f"sandbox_config_tile{tile_idx+1}.yaml")

def transform_params(params, names):
    return [10**p if log_scale_params.get(name, False) else p for name, p in zip(names, params)]

def resolve_div_dir(tile_root, gage_id, pid):
    """
    Return the directory that actually contains CSV divide outputs for this tile & particle.
    Prefer the per-particle directory; fall back to the legacy non-particle dir if needed.
    """
    candidates = [
        os.path.join(pwork(tile_root, gage_id, pid), "outputs", "div"),        # particle-aware
        os.path.join(tile_root, "out", gage_id, "outputs", "div"),             # legacy
    ]
    first_existing = None
    for d in candidates:
        if os.path.isdir(d):
            if first_existing is None:
                first_existing = d
            # has CSVs?
            if any(name.endswith(".csv") for name in os.listdir(d)):
                return d
    # If none have CSVs but a dir exists, return the first existing; else return the particle path
    return first_existing or candidates[0]

def log_incomplete(gage_id, particle_idx, stage, err_msg):
    """Append a short line to {logging_dir}/{gage}_errors.log when a particle or final run fails."""
    try:
        os.makedirs(logging_dir, exist_ok=True)
        with open(os.path.join(logging_dir, f"{gage_id}_errors.log"), "a") as f:
            f.write(f"{datetime.now().isoformat()} | pid={particle_idx} | stage={stage} | {err_msg}\n")
    except Exception:
        pass  # never let logging itself crash

def _safe_objective(args):
    """Wrapper around objective_function_tiled that never raises; returns (status, error, objective, val_metrics, cal_metrics)."""
    try:
        obj, val, cal = objective_function_tiled(args)
        return ("OK", "", obj, val, cal)
    except Exception as e:
        # args = (params, particle_idx, gage_id, ...)
        _, particle_idx, gage_id, *_ = args
        log_incomplete(gage_id, particle_idx, "objective", str(e))
        return ("FAIL", str(e), float("inf"),
                {metric_to_calibrate_on: np.nan},
                {metric_to_calibrate_on: np.nan})


def retarget_realization_paths(realization_path: str, work_root: str, base_out_dir: str, use_shared_pet: bool = True):
    """
    - If use_shared_pet=True, PET init_config stays under the *shared* gage dir:
        <base_out_dir>/configs/pet/<file>
      (no per-particle PET folder is created)
    - CFE still retargets into <work_root>/configs/cfe/<file>.
    - output_root is set to <work_root>/outputs/div.
    """
    # Load
    with open(realization_path, "r") as f:
        rz = json.load(f)

    # Ensure particle dirs we do need
    cfg_cfe_dir = os.path.join(work_root, "configs", "cfe")
    div_dir     = os.path.join(work_root, "outputs", "div")
    os.makedirs(cfg_cfe_dir, exist_ok=True)
    os.makedirs(div_dir, exist_ok=True)

    # Where PET configs live (shared, non-particle)
    shared_pet_dir = os.path.join(base_out_dir, "configs", "pet")

    # Walk modules and retarget
    try:
        forms = rz["global"]["formulations"]
    except Exception:
        forms = []

    for form in forms:
        params = form.get("params", {})
        modules = params.get("modules", [])
        for m in modules:
            p = m.get("params", {})
            init_cfg = p.get("init_config", "")
            if not init_cfg:
                continue

            model_type = (p.get("model_type_name") or "").upper()
            # Decide new path root
            if "PET" in model_type or "/configs/pet/" in init_cfg:
                if use_shared_pet:
                    # Leave PET in the shared (gage-level) path
                    fname = Path(init_cfg).name
                    new_path = os.path.join(shared_pet_dir, fname)
                else:
                    # fallback: per-particle PET (not recommended for inode)
                    cfg_pet_dir = os.path.join(work_root, "configs", "pet")
                    os.makedirs(cfg_pet_dir, exist_ok=True)
                    fname = Path(init_cfg).name
                    new_path = os.path.join(cfg_pet_dir, fname)
            elif "CFE" in model_type or "/configs/cfe/" in init_cfg:
                fname = Path(init_cfg).name
                new_path = os.path.join(cfg_cfe_dir, fname)
            else:
                continue

            # Use absolute paths to avoid CWD sensitivity
            p["init_config"] = os.path.abspath(new_path)

    # output_root: particle-local
    new_out_root = os.path.join(work_root, "outputs", "div")
    rz["output_root"] = new_out_root
    if "global" in rz:
        rz["global"]["output_root"] = new_out_root

    with open(realization_path, "w") as f:
        json.dump(rz, f, indent=4)

    return new_out_root


def extract_tile_params(full_params, tile_idx, n_tiles):
    total_len = len(full_params)
    if n_tiles == 2 and total_len % 2 == 1:
        param_slice = full_params[:-1]
        chunk = len(param_slice) // n_tiles
        return param_slice[tile_idx * chunk : (tile_idx + 1) * chunk]
    chunk = total_len // n_tiles
    return full_params[tile_idx * chunk : (tile_idx + 1) * chunk]

def get_observed_q(observed_path):
    df = pd.read_csv(observed_path, parse_dates=['value_time']).set_index('value_time')
    return df['flow_m3_per_s']

def pwork(root, gage_id, pid):
    """Particle workspace root created by sandbox.py -conf --concurrent-particles"""
    return os.path.join(root, "out", gage_id, "particles", f"p{pid}")

def extract_initial_params(config_path):
    """Extract CFE params from cfe_config_cat*.txt and NOM params if present."""
    cfe_params = extract_initial_cfe_params(config_path)

    # Detect NOM
    config_root = os.path.dirname(os.path.dirname(config_path))
    nom_dir = os.path.join(config_root, "noahowp")
    nom_params = []
    if os.path.isdir(nom_dir):
        mptable_path = os.path.join(nom_dir, "parameters", "MPTABLE.TBL")
        nom_params = extract_initial_nom_params(mptable_path)

    return cfe_params + nom_params

def extract_initial_nom_params(mptable_path):
    """
    Extract NOM params for tile that uses Noah-MP from MPTABLE.TBL.
    """
    param_order = list(nom_param_names)
    values = []

    if not os.path.isfile(mptable_path):
        raise FileNotFoundError(f"[ERROR] MPTABLE.TBL not found: {mptable_path}")

    with open(mptable_path, "r") as f:
        lines = f.readlines()

    for pname in param_order:
        found = False
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith(("!", "&", "/")):
                continue
            key = stripped.split("=", 1)[0].strip() if "=" in stripped else stripped.split()[0]
            if canonical_nom_param(key, [pname]) == pname:
                if "=" in stripped:
                    val_part = stripped.split("=", 1)[1]
                else:
                    val_part = stripped[len(pname):].strip()
                val_clean = val_part.split("!")[0].split(",")[0].strip()
                try:
                    value = float(val_clean)
                    low, high = nom_param_bounds_by_name[pname]
                    if value < low or value > high:
                        value = float(nom_param_initials[pname])
                    values.append(value)
                    found = True
                    break
                except ValueError:
                    raise ValueError(f"[ERROR] Could not parse value for {pname} in line: {line.strip()}")
        if not found:
            values.append(float(nom_param_initials[pname]))

    if len(values) != len(param_order):
        raise ValueError(f"[ERROR] Expected {len(param_order)} NOM params but got {len(values)} from {mptable_path}")

    return values


def extract_initial_cfe_params(config_path):
    """
    Extracts initial CFE parameters from config file.
    If a parameter is missing, falls back to default value 0.002.
    """
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
        "Kinf_nash_surface": "Kinf_nash_surface"
    }

    values = {}
    with open(config_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "=" in line:
            key, val = line.strip().split("=", 1)
            key = key.strip()
            val = val.split("[")[0].strip()
            for param, config_key in param_map.items():
                if key == config_key:
                    values[param] = float(val)

    param_order = [name for name in param_names if name in param_map]

    extracted = []
    for name in param_order:
        if name in values:
            v = values[name]
        else:
            print(f"[WARN] {name} not found in {config_path}. Using default 0.002. This is expected for the first iteration.")
            v = 0.002
        if log_scale_params.get(name, False):
            extracted.append(math.log10(v))
        else:
            extracted.append(v)

    return extracted


def regenerate_cfe_config(config_path, params, names):
    replacements = dict(zip(names, params))
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
        "Kinf_nash_surface": "Kinf_nash_surface"
    }

    # Determine directory holding cfe_config_cat* files
    if os.path.isfile(config_path):
        config_dir = os.path.dirname(config_path)
    else:
        config_dir = config_path

    for fname in sorted(f for f in os.listdir(config_dir) if f.startswith("cfe_config_cat")):
        pth = os.path.join(config_dir, fname)
        found_keys = set()
        updated_lines = []
        with open(pth) as f:
            for line in f:
                key = line.split("=")[0].strip()
                replaced = False
                for pname, ckey in param_map.items():
                    if key == ckey and pname in replacements:
                        unit = line[line.find("["):] if "[" in line else ""
                        updated_lines.append(f"{ckey}={replacements[pname]}{unit}\n")
                        found_keys.add(pname)
                        replaced = True
                        break
                if not replaced:
                    updated_lines.append(line)

        # Add any missing keys at the end (without units)
        for pname, ckey in param_map.items():
            if pname not in found_keys and pname in replacements:
                updated_lines.append(f"{ckey}={replacements[pname]}\n")

        with open(pth, "w") as f:
            f.writelines(updated_lines)


# === Tiled objective function (particle-aware paths) ===
def objective_function_tiled(args):
    """
    Runs tiled CFE+PET model with weighted divide outputs pre-routing,
    entirely within the per-particle workspaces to avoid collisions.
    """
    (
        params, particle_idx, gage_id,
        model_roots, observed_q_root,
        include_nom_flags, nom_file_paths,
        weights
    ) = args

    check_for_stop_signal_or_low_disk()
    n_tiles = len(model_roots)

    # Tile weight handling
    if n_tiles == 2:
        tile_weight = params[-1]
        weights = [tile_weight, 1.0 - tile_weight]
        params = params[:-1]
    elif weights is None:
        weights = [1.0 / n_tiles] * n_tiles

    # === STEP 1: Run hydrology for each tile in the particle's workspace ===
    for tile_idx, tile_root in enumerate(model_roots):
        print(f"[INFO] Running hydrology for gage {gage_id} | Particle {particle_idx} | Tile {tile_idx}")

        # --- Build per-tile param vector and transform (handle log-scale) ---
        tile_params = extract_tile_params(params, tile_idx, n_tiles)
        names = param_names.copy()
        if include_nom_flags[tile_idx]:
            names += nom_param_names
        true_params = transform_params(tile_params, names)

        # --- Particle workspace locations (these were scaffolded by sandbox.py -conf --concurrent-particles) ---
        work_root   = pwork(tile_root, gage_id, particle_idx)               # .../out/<gage>/particles/pX
        cfg_dir_cfe = os.path.join(work_root, "configs", "cfe")
        json_dir    = os.path.join(work_root, "json")

        os.makedirs(cfg_dir_cfe, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

        # --- Update mutable configs in the PARTICLE workspace ---
        regenerate_cfe_config(cfg_dir_cfe, true_params, names)

        nom_param_dict = {}
        if include_nom_flags[tile_idx]:
            nom_vals = tile_params[-len(nom_param_names):]
            nom_param_dict = dict(zip(nom_param_names, nom_vals))
            nom_tbl  = os.path.join(work_root, "configs", "noahowp", "parameters", "MPTABLE.TBL")
            update_mptable(
                original_file=nom_tbl,
                output_file=nom_tbl,
                updated_params=nom_param_dict,
                verbose=True
            )

        # --- Use the PARTICLE realization JSON and clamp time window to spinup->cal_end ---
        json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
        if not json_files:
            raise FileNotFoundError(f"No realization JSON found in {json_dir}")
        realization_path = os.path.join(json_dir, sorted(json_files)[0])

        with open(realization_path, "r") as f:
            realization = json.load(f)
        realization["time"]["start_time"] = time_cfg["spinup_start"]
        realization["time"]["end_time"]   = time_cfg["cal_end"]
        with open(realization_path, "w") as f:
            json.dump(realization, f, indent=4)

        # --- (Optional) clear old divide outputs in this particle workspace to avoid mixing between iterations ---
        div_dir = os.path.join(work_root, "outputs", "div")
        os.makedirs(div_dir, exist_ok=True)
        for item in list(os.listdir(div_dir)):
            if item.startswith("."):
                continue
            p = os.path.join(div_dir, item)
            if os.path.isfile(p) or os.path.islink(p):
                os.remove(p)
            elif os.path.isdir(p):
                shutil.rmtree(p)

        # --- Run hydrology ONLY, targeting this particle workspace via env flags (runner.py must honor NGEN_PARTICLE_ID) ---
        tile_sandbox_config = sandbox_config_for_tile(tile_idx)
        env = os.environ.copy()
        env["NGEN_CONCURRENT_PARTICLES"] = "1"
        env["NGEN_PARTICLE_ID"] = str(particle_idx)
        env["NGEN_REALIZATION_PATH"] = realization_path

        base_out_dir = os.path.join(tile_root, "out", gage_id)   # legacy gage-level out/<gage>
        retarget_realization_paths(
            realization_path=realization_path,
            work_root=work_root,
            base_out_dir=base_out_dir,
        )
        update_noahowp_model_params(realization_path, nom_param_dict)

        ret = subprocess.call(
            [sys.executable, sandbox_path, "-i", tile_sandbox_config, "-run", "--gage_id", gage_id],
            cwd=tile_root,
            env=env,
        )
        if ret != 0:
            raise RuntimeError(f"Hydrology run failed for gage {gage_id} | particle {particle_idx} | tile {tile_idx}")


    # === STEP 2: Weighted divide outputs within particle workspace (tile 0 as router) ===
    router_tile_root = model_roots[0]
    router_work = pwork(router_tile_root, gage_id, particle_idx)

    # Always use a particle-local div_weighted folder for qlat input (even for 1 tile)
    weighted_div_dir = os.path.join(router_work, "outputs", "div_weighted")
    if os.path.exists(weighted_div_dir):
        shutil.rmtree(weighted_div_dir)
    os.makedirs(weighted_div_dir, exist_ok=True)

    # Resolve per-tile source dirs (particle-aware, with legacy fallback)
    src_div_dirs = [resolve_div_dir(root, gage_id, particle_idx) for root in model_roots]

    # Gather candidate file list from the FIRST non-empty source dir
    files = []
    for d in src_div_dirs:
        if os.path.isdir(d):
            cand = [f for f in os.listdir(d) if (f.startswith("cat-") or f.startswith("nex-")) and f.endswith(".csv")]
            if cand:
                files = cand
                break

    if not files:
        # Diagnostics to help debug
        print("[ERROR] No divide CSVs found for routing.")
        for idx, d in enumerate(src_div_dirs):
            exists = os.path.isdir(d)
            count = len(os.listdir(d)) if exists else 0
            print(f"  tile{idx} dir: {d} | exists={exists} | nfiles={count}")
        # Return a bad objective so PSO can proceed, rather than crashing the whole run
        dummy = {metric_to_calibrate_on: -np.inf}
        return (1e12, dummy, dummy)

    # Single-tile: just copy into weighted_div_dir
    if n_tiles == 1:
        src = src_div_dirs[0]
        for fname in files:
            inpath = os.path.join(src, fname)
            outpath = os.path.join(weighted_div_dir, fname)
            shutil.copy2(inpath, outpath)
    else:
        # Multi-tile: weighted average into weighted_div_dir
        for fname in files:
            dfs = []
            df_ref = None
            for tile_idx, div_dir in enumerate(src_div_dirs):
                fpath = os.path.join(div_dir, fname)
                if os.path.exists(fpath):
                    if fname.startswith("nex-"):
                        df = pd.read_csv(fpath, header=None)
                        df.columns = ["Time Step", "Time", "q_out"]
                    else:
                        df = pd.read_csv(fpath)
                    if df_ref is None:
                        df_ref = df.copy()
                    dfs.append(df["q_out"] * weights[tile_idx])
            if dfs:
                combined = sum(dfs)
                out_df = df_ref.copy()
                out_df["q_out"] = combined
                out_df["Time"] = pd.to_datetime(out_df["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                # Preserve header style
                outpath = os.path.join(weighted_div_dir, fname)
                if fname.startswith("nex-"):
                    out_df.to_csv(outpath, index=False, header=False)
                else:
                    out_df.to_csv(outpath, index=False)

    # === STEP 3: Run routing once on weighted outputs (particle workspace) ===
    troute_path = os.path.join(router_work, "configs", "troute_config.yaml")
    with open(troute_path) as f:
        troute_cfg = yaml.safe_load(f)

    nts = int((cal_end - spinup_start) / pd.Timedelta(seconds=300))
    troute_cfg["compute_parameters"]["restart_parameters"]["start_datetime"] = spinup_start.strftime("%Y-%m-%d_%H:%M:%S")
    troute_cfg["compute_parameters"]["forcing_parameters"]["nts"] = nts
    troute_cfg["compute_parameters"]["forcing_parameters"]["qlat_input_folder"] = weighted_div_dir

    # redirect nested stream outputs into the particle's troute dir
    particle_troute_dir = os.path.join(router_work, "troute")
    os.makedirs(particle_troute_dir, exist_ok=True)

    op = troute_cfg.setdefault("output_parameters", {})
    so = op.setdefault("stream_output", {})
    so["stream_output_directory"] = particle_troute_dir

    # KEEP mask_output pointing to the original (read-only) file if it exists.
    mask_path = so.get("mask_output")
    if not (isinstance(mask_path, str) and os.path.isfile(mask_path)):
        # try original gage-level mask path
        orig_mask = os.path.join(router_tile_root, "out", gage_id, "configs", "mask_output.yaml")
        if os.path.isfile(orig_mask):
            so["mask_output"] = orig_mask
        else:
            # no valid mask file; let T-Route proceed without one
            so.pop("mask_output", None)

    yaml.safe_dump(troute_cfg, open(troute_path, "w"))

    # clean particle's troute dir of old files
    troute_dir = particle_troute_dir
    if os.path.isdir(troute_dir):
        for fname in os.listdir(troute_dir):
            if fname.endswith((".nc", ".csv", ".parquet")):
                os.remove(os.path.join(troute_dir, fname))

    env = os.environ.copy()
    # env["NGEN_REALIZATION_PATH"] = realization_path
    env["NGEN_CONCURRENT_PARTICLES"] = "1"
    env["NGEN_PARTICLE_ID"] = str(particle_idx)
    env["NGEN_REALIZATION_PATH"] = realization_path
    ret = subprocess.call([sys.executable, "-m", "nwm_routing", "-f", "-V4", troute_path], env=env)
    if ret != 0:
        raise RuntimeError(f"T-route failed for gage {gage_id} | particle {particle_idx} | exit code {ret}")


    # === STEP 4: Extract routed hydrograph (particle workspace) ===
    postproc_dir = os.path.join(router_work, "postproc")
    os.makedirs(postproc_dir, exist_ok=True)
    output_path = os.path.join(postproc_dir, f"{gage_id}_particle_{particle_idx}.csv")
    get_hydrograph_path = os.path.join(project_root, "model_assessment", "util", "get_hydrograph.py")
    summary_csv = os.environ.get(
        "DOWNSTREAM_FLOWPATH_SUMMARY",
        os.path.join(project_root, "model_assessment", "util", "downstream_flowpath_summary.csv"),
    )

    troute_output_dir = os.path.join(router_work, "troute")
    if not os.path.isdir(troute_output_dir):
        raise FileNotFoundError(f"Particle troute dir missing: {troute_output_dir}")

    env = os.environ.copy()
    env["NGEN_CONCURRENT_PARTICLES"] = "1"
    env["NGEN_PARTICLE_ID"] = str(particle_idx)
    env["NGEN_REALIZATION_PATH"] = realization_path

    ret = subprocess.call(
        [
            sys.executable, get_hydrograph_path,
            "--gage_id", gage_id,
            "--output", output_path,
            "--troute_dir", troute_output_dir,
            "--summary", summary_csv,
        ],
        cwd=postproc_dir,
        env=env,
    )
    if ret != 0:
        raise RuntimeError(f"Hydrograph extraction failed for gage {gage_id} | particle {particle_idx} | exit code {ret}")


    # === STEP 5: Compute metrics ===
    sim_df = pd.read_csv(output_path, parse_dates=["current_time"]).set_index("current_time")["flow"].resample("1h").mean()
    obs_df = get_observed_q(os.path.join(observed_q_root, "successful_sites_resampled", f"{gage_id}.csv"))

    sim_cal, obs_cal = sim_df[cal_start:cal_end].dropna(), obs_df[cal_start:cal_end].dropna()
    sim_val, obs_val = sim_df[val_start:val_end].dropna(), obs_df[val_start:val_end].dropna()
    sim_cal, obs_cal = sim_cal.align(obs_cal, join="inner")
    sim_val, obs_val = sim_val.align(obs_val, join="inner")

    if len(sim_cal) > 0: sim_cal.iloc[-1] += 1e-8
    if len(obs_cal) > 0: obs_cal.iloc[-1] += 1e-8

    cal_metrics = compute_metrics(sim_cal, obs_cal, event_threshold=1e-2)
    val_metrics = compute_metrics(sim_val, obs_val, event_threshold=1e-2)

    # === After computing cal_metrics / val_metrics and before 'return' ===
    try:
        # Remove weighted qlat inputs (we can always regenerate them)
        if os.path.isdir(weighted_div_dir):
            shutil.rmtree(weighted_div_dir)
    except Exception:
        pass

    # Per-tile: remove hydrology divide CSVs now instead of waiting for next iteration
    for tile_root in model_roots:
        try:
            work_root = pwork(tile_root, gage_id, particle_idx)
            div_dir = os.path.join(work_root, "outputs", "div")
            if os.path.isdir(div_dir):
                for item in os.listdir(div_dir):
                    p = os.path.join(div_dir, item)
                    if os.path.isfile(p) or os.path.islink(p):
                        os.remove(p)
                    elif os.path.isdir(p):
                        shutil.rmtree(p)
        except Exception:
            pass

    # Particle routing outputs: keep the single hydrograph CSV, drop bulky files
    try:
        if os.path.isdir(troute_output_dir):
            for fname in os.listdir(troute_output_dir):
                if fname.endswith((".nc", ".csv", ".parquet")) and not fname.endswith("_best.csv"):
                    os.remove(os.path.join(troute_output_dir, fname))
    except Exception:
        pass


    return -cal_metrics[metric_to_calibrate_on], val_metrics, cal_metrics


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
        self.best_calibration_metric = -9999.0
        self.best_validation_metric = -9999.0

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
    def __init__(self,
                 n_particles, bounds, n_iterations, gage_id,
                 init_position, config_path, observed_path, postproc_base_path,
                 metric_to_calibrate_on="kge",
                 include_nom=False,
                 nom_file_paths=None,
                 param_names=None):
        self.particles = [
            Particle(bounds, init_position=init_position if i == 0 else None)
            for i in range(n_particles)
        ]
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
        self.nom_file_paths = nom_file_paths or []
        self.param_names = param_names or [f"p{i}" for i in range(len(bounds))]
        self.model_roots = []

    def optimize(self):
        start_time = datetime.now()
        log_rows = []
        log_path = os.path.join(logging_dir, f"{self.gage_id}.csv")
        stagnation_threshold = 10
        w_start, w_end = 0.9, 0.4

        ###
        ctx = multiprocessing.get_context("spawn")
        pool_size = max_particle_procs or len(self.particles)
        job_cores = runtime_job_cores(default=pool_size)

        for iteration in range(self.n_iterations):
            # clear_terminal()
            print(f"\n--- Iteration {iteration + 1} for gage {self.gage_id} ---")
            w = w_start - (w_start - w_end) * (iteration / self.n_iterations)

            # Build args once, then run all particles concurrently
            args_list = []
            for i, p in enumerate(self.particles):
                print(f"[INFO] Running gage {self.gage_id} | iteration {iteration + 1} | particle {i}")
                args_list.append((
                    p.position, i, self.gage_id, cfg.model_roots,
                    observed_q_root,
                    [self.include_nom] * len(cfg.model_roots),
                    self.nom_file_paths,
                    [1.0 / len(cfg.model_roots)] * len(cfg.model_roots)
                ))

            # ThreadPool avoids “daemonic processes can’t have children”
            with ThreadPool(processes=pool_size) as pool:
                results = pool.map(_safe_objective, args_list)


            # === Reduction step ===
            for idx, r in enumerate(results):
                status, err, objective_value, val_metrics, cal_metrics = r
                particle = self.particles[idx]
                particle.current_value = objective_value

                metric_calibration = cal_metrics.get(self.metric_to_calibrate_on, np.nan)
                metric_validation = val_metrics.get(self.metric_to_calibrate_on, np.nan)

                # Update per-particle and global bests only on success
                if status == "OK" and objective_value < (particle.best_value - 0.001):
                    particle.best_value = objective_value
                    particle.best_position = np.copy(particle.position)
                    particle.stagnation_counter = 0
                else:
                    particle.stagnation_counter += 1

                if status == "OK" and objective_value < self.global_best_value:
                    self.global_best_value = objective_value
                    self.global_best_position = np.copy(particle.position)
                    self.best_calibration_metric = metric_calibration
                    self.best_validation_metric = metric_validation
                    self.best_cal_metrics = cal_metrics
                    self.best_val_metrics = val_metrics
                    particle.stagnation_counter = 0

                if status != "OK":
                    # also mirror in the .log file with iteration context
                    log_incomplete(self.gage_id, idx, f"iter_{iteration+1}", err)

                if particle.stagnation_counter >= stagnation_threshold:
                    print(f"Resetting particle {idx} after {stagnation_threshold} stagnant iterations.")
                    particle.reset(self.bounds)

                param_dict = {name: val for name, val in zip(self.param_names, particle.position)}
                row = {
                    "iteration": iteration + 1,
                    "particle": idx,
                    **param_dict,
                    f"{self.metric_to_calibrate_on}_calibration": metric_calibration,
                    f"{self.metric_to_calibrate_on}_validation": metric_validation,
                    "status": status,
                    "error": (err or "")[:240],
                    **wall_time_log_fields(start_time, job_cores, pool_size),
                }
                log_rows.append(row)

            # write/flush after each iteration so partial progress is saved
            pd.DataFrame(log_rows).to_csv(log_path, index=False)


            # Update swarm
            for p in self.particles:
                p.update_velocity(self.global_best_position, w=w, c1=1.5, c2=1.5)
                p.update_position(self.bounds)

            print(f"Global best objective so far: {-self.global_best_value:.4f}")

        # === Final full-period multi-tile run in BEST particle workspace ===
        print(f"\n[INFO] Running final weighted-routing validation for {self.gage_id}...")
        if not np.isfinite(self.global_best_value):
            raise RuntimeError(f"No successful particle evaluations for gage {self.gage_id}; skipping final validation")
        n_tiles = len(cfg.model_roots)
        weights = [1.0 / n_tiles] * n_tiles
        if n_tiles == 2:
            weights = [self.global_best_position[-1], 1.0 - self.global_best_position[-1]]

        best_pid = 0  # we’ll reuse pid=0 workspace for the final (or choose any fixed ID)
        env = os.environ.copy()
        env["NGEN_CONCURRENT_PARTICLES"] = "1"
        env["NGEN_PARTICLE_ID"] = str(best_pid)
        # env["NGEN_REALIZATION_PATH"] = realization_path


        # 1) Hydrology (full period) per tile  — run in BEST particle workspace
        for tile_idx, tile_root in enumerate(cfg.model_roots):
            # --- Build per-tile param vector and transform (handle log-scale) ---
            tile_params = extract_tile_params(self.global_best_position, tile_idx, n_tiles)
            names = param_names.copy()
            if self.include_nom:
                names += nom_param_names
            true_best_params = transform_params(tile_params, names)

            # --- Particle workspace paths (re-use best_pid workspace) ---
            work_root   = pwork(tile_root, self.gage_id, best_pid)      # .../out/<gage>/particles/p<best_pid>
            cfg_dir_cfe = os.path.join(work_root, "configs", "cfe")
            json_dir    = os.path.join(work_root, "json")
            div_dir     = os.path.join(work_root, "outputs", "div")

            os.makedirs(cfg_dir_cfe, exist_ok=True)
            os.makedirs(json_dir, exist_ok=True)
            os.makedirs(div_dir, exist_ok=True)

            # --- Update mutable configs in the PARTICLE workspace ---
            regenerate_cfe_config(cfg_dir_cfe, true_best_params, names)

            nom_param_dict = {}
            if self.include_nom:
                nom_vals = tile_params[-len(nom_param_names):]
                nom_param_dict = dict(zip(nom_param_names, nom_vals))
                nom_tbl  = os.path.join(work_root, "configs", "noahowp", "parameters", "MPTABLE.TBL")
                update_mptable(
                    original_file=nom_tbl,
                    output_file=nom_tbl,
                    updated_params=nom_param_dict,
                    verbose=True
                )

            # --- Use the PARTICLE realization JSON and set time window spinup->val_end ---
            json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
            if not json_files:
                raise FileNotFoundError(f"No realization JSON found in {json_dir}")
            realization_path = os.path.join(json_dir, sorted(json_files)[0])

            with open(realization_path, "r") as f:
                realization = json.load(f)
            realization["time"]["start_time"] = time_cfg["spinup_start"]
            realization["time"]["end_time"]   = time_cfg["val_end"]
            with open(realization_path, "w") as f:
                json.dump(realization, f, indent=4)

            # --- Clean particle's divide outputs to avoid mixing between cal and final runs ---
            for item in list(os.listdir(div_dir)):
                if item.startswith("."):
                    continue
                p = os.path.join(div_dir, item)
                if os.path.isfile(p) or os.path.islink(p):
                    os.remove(p)
                elif os.path.isdir(p):
                    shutil.rmtree(p)

            # --- Run hydrology ONLY, targeting this BEST particle workspace via env flags ---
            tile_sandbox_config = sandbox_config_for_tile(tile_idx)
            env = os.environ.copy()
            env["NGEN_REALIZATION_PATH"] = realization_path
            env["NGEN_CONCURRENT_PARTICLES"] = "1"
            env["NGEN_PARTICLE_ID"] = str(best_pid)

            base_out_dir = os.path.join(tile_root, "out", self.gage_id)
            retarget_realization_paths(
                realization_path=realization_path,
                work_root=work_root,   # p<best_pid> workspace
                base_out_dir=base_out_dir,
            )
            update_noahowp_model_params(realization_path, nom_param_dict)

            ret = subprocess.call(
                [sys.executable, sandbox_path, "-i", tile_sandbox_config, "-run", "--gage_id", self.gage_id],
                cwd=tile_root,
                env=env,
            )
            if ret != 0:
                raise RuntimeError(f"Final-period hydrology failed for gage {self.gage_id} | particle {best_pid} | tile {tile_idx}")


        # 2) Weighted averaging of divide outputs (BEST workspace)
        router_tile_root = cfg.model_roots[0]
        router_work = pwork(router_tile_root, self.gage_id, best_pid)

        # Always produce a particle-local div_weighted and point routing to it
        weighted_div_dir = os.path.join(router_work, "outputs", "div_weighted")
        if os.path.exists(weighted_div_dir):
            shutil.rmtree(weighted_div_dir)
        os.makedirs(weighted_div_dir, exist_ok=True)

        # Resolve per-tile source dirs (particle-aware, with legacy fallback)
        src_div_dirs = [resolve_div_dir(root, self.gage_id, best_pid) for root in cfg.model_roots]

        # Pick file list from the first non-empty source dir
        files = []
        for d in src_div_dirs:
            if os.path.isdir(d):
                cand = [f for f in os.listdir(d) if (f.startswith("cat-") or f.startswith("nex-")) and f.endswith(".csv")]
                if cand:
                    files = cand
                    break

        if not files:
            print("[ERROR] No divide CSVs found for final routing.")
            for idx, d in enumerate(src_div_dirs):
                exists = os.path.isdir(d)
                count = len(os.listdir(d)) if exists else 0
                print(f"  tile{idx} dir: {d} | exists={exists} | nfiles={count}")
            dummy = {metric_to_calibrate_on: -np.inf}
            return (1e12, dummy, dummy)

        if len(cfg.model_roots) == 1:
            # Single tile: just copy into weighted_div_dir
            src = src_div_dirs[0]
            for fname in files:
                shutil.copy2(os.path.join(src, fname), os.path.join(weighted_div_dir, fname))
        else:
            # Multi-tile: weighted average
            for fname in files:
                dfs = []
                df_ref = None
                for tile_idx, div_dir in enumerate(src_div_dirs):
                    fpath = os.path.join(div_dir, fname)
                    if os.path.exists(fpath):
                        if fname.startswith("nex-"):
                            df = pd.read_csv(fpath, header=None); df.columns = ["Time Step", "Time", "q_out"]
                        else:
                            df = pd.read_csv(fpath)
                        if df_ref is None:
                            df_ref = df.copy()
                        dfs.append(df["q_out"] * weights[tile_idx])
                if dfs:
                    out_df = df_ref.copy()
                    out_df["q_out"] = sum(dfs)
                    out_df["Time"] = pd.to_datetime(out_df["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                    outpath = os.path.join(weighted_div_dir, fname)
                    if fname.startswith("nex-"):
                        out_df.to_csv(outpath, index=False, header=False)
                    else:
                        out_df.to_csv(outpath, index=False)


        # 3) Routing (full period) in BEST workspace
        troute_path = os.path.join(router_work, "configs", "troute_config.yaml")
        with open(troute_path) as f:
            troute_cfg = yaml.safe_load(f)

        nts_full = int((val_end - spinup_start) / pd.Timedelta(seconds=300))
        troute_cfg["compute_parameters"]["restart_parameters"]["start_datetime"] = spinup_start.strftime("%Y-%m-%d_%H:%M:%S")
        troute_cfg["compute_parameters"]["forcing_parameters"]["nts"] = nts_full
        troute_cfg["compute_parameters"]["forcing_parameters"]["qlat_input_folder"] = weighted_div_dir

        # redirect nested stream outputs into the particle's troute dir
        particle_troute_dir = os.path.join(router_work, "troute")
        os.makedirs(particle_troute_dir, exist_ok=True)

        op = troute_cfg.setdefault("output_parameters", {})
        so = op.setdefault("stream_output", {})
        so["stream_output_directory"] = particle_troute_dir

        # KEEP mask_output from original config if valid; else fall back; else drop
        mask_path = so.get("mask_output")
        if not (isinstance(mask_path, str) and os.path.isfile(mask_path)):
            orig_mask = os.path.join(router_tile_root, "out", self.gage_id, "configs", "mask_output.yaml")
            if os.path.isfile(orig_mask):
                so["mask_output"] = orig_mask
            else:
                so.pop("mask_output", None)


        with open(troute_path, "w") as f:
            yaml.safe_dump(troute_cfg, f)

        # clean particle troute dir of old files to avoid any overwrite/append issues
        for fname in os.listdir(particle_troute_dir):
            if fname.endswith((".nc", ".csv", ".parquet")):
                os.remove(os.path.join(particle_troute_dir, fname))

        ret = subprocess.call([sys.executable, "-m", "nwm_routing", "-f", "-V4", troute_path], env=env)
        if ret != 0:
            raise RuntimeError(f"Final T-route failed for gage {self.gage_id} | exit code {ret}")

        # 4) Extract final routed hydrograph (BEST workspace)
        postproc_dir = os.path.join(router_work, "postproc")
        os.makedirs(postproc_dir, exist_ok=True)
        final_output_path = os.path.join(postproc_dir, f"{self.gage_id}_best.csv")
        get_hydrograph_path = os.path.join(project_root, "model_assessment", "util", "get_hydrograph.py")
        summary_csv = os.environ.get(
            "DOWNSTREAM_FLOWPATH_SUMMARY",
            os.path.join(project_root, "model_assessment", "util", "downstream_flowpath_summary.csv"),
        )

        troute_output_dir = os.path.join(router_work, "troute")
        if not os.path.isdir(troute_output_dir):
            raise FileNotFoundError(f"Final troute dir missing: {troute_output_dir}")

        ret = subprocess.call(
            [
                sys.executable, get_hydrograph_path,
                "--gage_id", self.gage_id,
                "--output", final_output_path,
                "--troute_dir", troute_output_dir,
                "--summary", summary_csv,
            ],
            cwd=postproc_dir,
            env=env,
        )
        if ret != 0:
            raise RuntimeError(f"Final hydrograph extraction failed for gage {self.gage_id} | exit code {ret}")


        # 5) Final metrics (validation window)
        obs_df = get_observed_q(self.observed_path)
        sim_df = pd.read_csv(final_output_path, parse_dates=["current_time"]).set_index("current_time")["flow"].resample("1h").mean()
        sim_val, obs_val = sim_df[val_start:val_end].dropna(), obs_df[val_start:val_end].dropna()
        sim_val, obs_val = sim_val.align(obs_val, join="inner")
        if len(sim_val) > 0: sim_val.iloc[-1] += 1e-8
        if len(obs_val) > 0: obs_val.iloc[-1] += 1e-8
        val_metrics_final = compute_metrics(sim_val, obs_val, event_threshold=1e-2)

        final_row = {
            "iteration": "FINAL",
            "particle": "BEST",
            **{name: val for name, val in zip(self.param_names, self.global_best_position)},
            f"{self.metric_to_calibrate_on}_calibration": self.best_cal_metrics.get(self.metric_to_calibrate_on, np.nan),
            f"{self.metric_to_calibrate_on}_validation": val_metrics_final.get(self.metric_to_calibrate_on, np.nan),
            "status": "OK",
            "error": "",
            **wall_time_log_fields(start_time, job_cores, pool_size, final=True),
        }
        pd.DataFrame(log_rows + [final_row]).to_csv(log_path, index=False)
        print(f" Final {self.metric_to_calibrate_on.upper()} = {val_metrics_final.get(self.metric_to_calibrate_on, np.nan):.4f}")

        return self.global_best_position, self.global_best_value, self.best_validation_metric, datetime.now() - start_time


# === Per-gage wrapper ===
def calibrate_gage(gage_id):
    model_roots = cfg.model_roots
    n_tiles = len(model_roots)
    all_init_params, all_bounds, include_nom_flags, nom_file_paths = [], [], [], []

    for tile_idx, root in enumerate(model_roots):
        # Locate the CFE config file in the particle 0 workspace (scaffolded at -conf)
        config_dir = os.path.join(pwork(root, gage_id, 0), "configs", "cfe")
        config_file = sorted(f for f in os.listdir(config_dir) if f.startswith("cfe_config_cat"))[0]
        config_path = os.path.join(config_dir, config_file)

        init = extract_initial_params(config_path)  # Combines CFE + NOM if present
        bounds = param_bounds.copy()

        include_nom = os.path.isdir(os.path.join(pwork(root, gage_id, 0), "configs", "noahowp"))
        if include_nom:
            nom_path = os.path.join(pwork(root, gage_id, 0), "configs", "noahowp", "parameters", "MPTABLE.TBL")
            bounds += nom_param_bounds
            nom_file_paths.append(nom_path)
        else:
            nom_file_paths.append("")

        all_init_params.extend(init)
        all_bounds.extend(bounds)
        include_nom_flags.append(include_nom)

    names = []
    for tile_idx in range(n_tiles):
        tile_suffix = f"_tile{tile_idx+1}"
        tile_names = [f"{name}{tile_suffix}" for name in param_names]
        if include_nom_flags[tile_idx]:
            tile_names += [f"{name}{tile_suffix}" for name in nom_param_names]
        names.extend(tile_names)

    if n_tiles == 2:
        all_init_params.append(0.7)
        all_bounds.append((0.0, 1.0))
        names.append("tile_weight")

    print(f"[DEBUG] include_nom_flags: {include_nom_flags}")
    print(f"[DEBUG] nom_file_paths: {nom_file_paths}")

    pso = PSO(
        n_particles=n_particles,
        bounds=all_bounds,
        n_iterations=n_iterations,
        gage_id=gage_id,
        init_position=all_init_params,
        config_path="",
        observed_path=os.path.join(observed_q_root, "successful_sites_resampled", f"{gage_id}.csv"),
        postproc_base_path=os.path.join(pwork(model_roots[0], gage_id, 0), "postproc"),
        metric_to_calibrate_on=metric_to_calibrate_on,
        include_nom=any(include_nom_flags),
        nom_file_paths=nom_file_paths,
        param_names=names
    )
    pso.model_roots = model_roots
    pso.optimize()


def parse_args():
    parser = argparse.ArgumentParser(description="Run CFE PSO calibration.")
    parser.add_argument("--gage-id", default=os.environ.get("NGEN_GAGE_ID") or os.environ.get("GAGE_ID"))
    parser.add_argument("--n-particles", type=int, default=n_particles)
    parser.add_argument("--n-iterations", type=int, default=n_iterations)
    parser.add_argument("--max-particle-procs", type=int, default=max_particle_procs)
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

    n_particles = args.n_particles
    n_iterations = args.n_iterations
    max_particle_procs = args.max_particle_procs
    max_cores_for_gages = args.max_gage_procs
    sandbox_config_override = args.sandbox_config
    set_time_windows({
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
        gage_list = pd.read_csv(cfg.gages_file, dtype={"gage_id": str})["gage_id"].tolist()

    if len(gage_list) == 1:
        calibrate_gage(gage_list[0])
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=max_cores_for_gages) as pool:
            pool.map(calibrate_gage, gage_list)
    print(f"Total wall time: {datetime.now() - start}")








































# ####works well but does not support concurrent particle eval
# """
# Author: Peter La Follette [plafollette@lynker.com | July 2025]
# Multi-tile PSO calibration for CFE+PET+T-Route, with optional NOM parameter support.
# """

# import os
# import subprocess
# import pandas as pd
# import numpy as np
# import math
# import yaml
# import json
# import random
# import sys
# import traceback
# import multiprocessing
# from datetime import datetime
# from hydroeval import kge

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# from model_assessment.util.metrics import compute_metrics
# from model_assessment.util.update_NOM import update_mptable
# from model_assessment.configs import path_config as cfg

# np.random.seed(42)
# random.seed(42)

# # === CONFIGURATION ===
# n_particles = 2
# n_iterations = 2
# max_cores_for_gages = 2
# metric_to_calibrate_on = "kge"

# with open("model_assessment/configs/time_config.yaml", "r") as f:
#     time_cfg = yaml.safe_load(f)

# spinup_start = pd.Timestamp(time_cfg["spinup_start"])
# cal_start    = pd.Timestamp(time_cfg["cal_start"])
# cal_end      = pd.Timestamp(time_cfg["cal_end"])
# val_start    = pd.Timestamp(time_cfg["val_start"])
# val_end      = pd.Timestamp(time_cfg["val_end"])

# project_root = cfg.project_root
# sandbox_path = cfg.sandbox_path
# logging_dir = cfg.logging_dir
# observed_q_root = cfg.observed_q_root

# os.makedirs(logging_dir, exist_ok=True)

# # === Parameter definitions ===
# param_names = [
#     "b", "satdk", "satpsi", "maxsmc", "max_gw_storage", "Cgw", "expon", "Kn",
#     "Klf", "refkdt", "slope", "wltsmc", "alpha_fc", "Kinf_nash_surface"
# ]

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

# log_scale_params = {"Cgw": True, "satdk": True}

# nom_param_names = ["MFSNO", "RSURF_SNOW", "HVT", "CWPVT", "VCMX25", "MP"]
# nom_param_bounds = [
#     (0.625, 5.0), (0.1, 100.0), (0.0, 20.0), (0.18, 5.0), (0.0, 80.0), (3.6, 12.6)
# ]

# # === Helpers ===
# def clear_terminal():
#     os.system("clear")

# def check_for_stop_signal_or_low_disk(threshold_gb=50):
#     stop_file = os.path.join(project_root, "STOP_NOW.txt")
#     if os.path.exists(stop_file):
#         sys.exit(1)
#     stat = os.statvfs("/")
#     free_gb = (stat.f_frsize * stat.f_bavail) / (1024 ** 3)
#     if free_gb < threshold_gb:
#         sys.exit(1)

# def transform_params(params, names):
#     return [10**p if log_scale_params.get(name, False) else p for name, p in zip(names, params)]

# def extract_tile_params(full_params, tile_idx, n_tiles):
#     total_len = len(full_params)
#     if n_tiles == 2 and total_len % 2 == 1:
#         param_slice = full_params[:-1]
#         chunk = len(param_slice) // n_tiles
#         return param_slice[tile_idx * chunk : (tile_idx + 1) * chunk]
#     chunk = total_len // n_tiles
#     return full_params[tile_idx * chunk : (tile_idx + 1) * chunk]

# def get_observed_q(observed_path):
#     df = pd.read_csv(observed_path, parse_dates=['value_time']).set_index('value_time')
#     return df['flow_m3_per_s']

# def extract_initial_params(config_path):
#     """Extract CFE params from cfe_config_cat*.txt and NOM params if present."""
#     cfe_params = extract_initial_cfe_params(config_path)

#     # Detect NOM
#     config_root = os.path.dirname(os.path.dirname(config_path))
#     nom_dir = os.path.join(config_root, "noahowp")
#     nom_params = []
#     if os.path.isdir(nom_dir):
#         mptable_path = os.path.join(nom_dir, "parameters", "MPTABLE.TBL")
#         nom_params = extract_initial_nom_params(mptable_path)

#     return cfe_params + nom_params

# def extract_initial_nom_params(mptable_path):
#     """
#     Extract NOM params for tile that uses Noah-MP from MPTABLE.TBL.

#     Args:
#         mptable_path (str): Path to the MPTABLE.TBL file.

#     Returns:
#         list of float: NOM parameter values in the correct order.
#     """
#     param_order = ["MFSNO", "RSURF_SNOW", "HVT", "CWPVT", "VCMX25", "MP"]
#     values = []

#     if not os.path.isfile(mptable_path):
#         raise FileNotFoundError(f"[ERROR] MPTABLE.TBL not found: {mptable_path}")

#     with open(mptable_path, "r") as f:
#         lines = f.readlines()

#     for pname in param_order:
#         found = False
#         for line in lines:
#             stripped = line.strip()
#             if not stripped or stripped.startswith(("!", "&", "/")):
#                 continue  # Skip comments, section markers, blank lines
#             if stripped.startswith(pname):
#                 # Split at '=', then handle inline comments and commas
#                 if "=" in stripped:
#                     val_part = stripped.split("=", 1)[1]
#                 else:
#                     # Fallback: space-delimited (legacy style)
#                     val_part = stripped[len(pname):].strip()

#                 val_clean = val_part.split("!")[0].split(",")[0].strip()  # remove comments and take first value

#                 try:
#                     values.append(float(val_clean))
#                     found = True
#                     break
#                 except ValueError:
#                     raise ValueError(f"[ERROR] Could not parse value for {pname} in line: {line.strip()}")

#         if not found:
#             raise ValueError(f"[ERROR] Missing parameter {pname} in {mptable_path}")

#     if len(values) != len(param_order):
#         raise ValueError(f"[ERROR] Expected {len(param_order)} NOM params but got {len(values)} from {mptable_path}")

#     return values


# def extract_initial_cfe_params(config_path):
#     """
#     Extracts initial CFE parameters from config file.
#     If a parameter is missing, falls back to default value 0.002.
#     """
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
#         "Kinf_nash_surface": "Kinf_nash_surface"
#     }

#     values = {}
#     with open(config_path, "r") as f:
#         lines = f.readlines()
#     for line in lines:
#         if "=" in line:
#             key, val = line.strip().split("=", 1)
#             key = key.strip()
#             val = val.split("[")[0].strip()
#             for param, config_key in param_map.items():
#                 if key == config_key:
#                     values[param] = float(val)

#     param_order = list(param_map.keys())

#     extracted = []
#     for name in param_order:
#         if name in values:
#             v = values[name]
#         else:
#             print(f"[WARN] {name} not found in {config_path}. Using default 0.002. This is expected for the first iteration.")
#             v = 0.002
#         if log_scale_params.get(name, False):
#             extracted.append(math.log10(v))
#         else:
#             extracted.append(v)

#     return extracted


# # def extract_initial_nom_params(mptable_path):
# #     """Extract NOM params for tile that uses Noah-MP"""
# #     values = []
# #     param_order = nom_param_names.copy()
# #     with open(mptable_path) as f:
# #         for line in f:
# #             for pname in param_order:
# #                 if line.strip().startswith(pname):
# #                     val = float(line.strip().split()[1])
# #                     values.append(val)
# #     if len(values) != len(param_order):
# #         raise ValueError(f"Expected {len(param_order)} NOM params but got {len(values)} from {mptable_path}")
# #     return values


# def regenerate_cfe_config(config_path, params, names):
#     replacements = dict(zip(names, params))
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
#         "Kinf_nash_surface": "Kinf_nash_surface"
#     }

#     found_keys = set()
#     updated_lines = []

#     if os.path.isfile(config_path):
#         config_dir = os.path.dirname(config_path)
#     else:
#         config_dir = config_path

#     for fname in sorted(f for f in os.listdir(config_dir) if f.startswith("cfe_config_cat")):
#         path = os.path.join(config_dir, fname)
#         lines = []
#         with open(path) as f:
#             for line in f:
#                 key = line.split("=")[0].strip()
#                 replaced = False
#                 for pname, ckey in param_map.items():
#                     if key == ckey:
#                         unit = line[line.find("["):] if "[" in line else ""
#                         lines.append(f"{ckey}={replacements[pname]}{unit}\n")
#                         found_keys.add(pname)
#                         replaced = True
#                         break
#                 if not replaced:
#                     lines.append(line)
#         # Add missing keys
#         for pname, ckey in param_map.items():
#             if pname not in found_keys:
#                 lines.append(f"{ckey}={replacements[pname]}\n")
#         with open(path, "w") as f:
#             f.writelines(lines)

# # === Tiled objective function ===
# import shutil

# def objective_function_tiled(args):
#     """
#     Runs tiled CFE+PET model with weighted divide outputs pre-routing.
#     Routing now operates on weighted divide-scale flows rather than per-tile routed results.
#     """
#     (
#         params, particle_idx, gage_id,
#         model_roots, observed_q_root,
#         include_nom_flags, nom_file_paths,
#         weights
#     ) = args

#     check_for_stop_signal_or_low_disk()
#     n_tiles = len(model_roots)

#     # === CLEAR PREVIOUS RESULTS ===
#     for tile_root in model_roots:
#         outputs_dir = os.path.join(tile_root, "out", gage_id, "outputs")
#         div_dir = os.path.join(outputs_dir, "div")
#         div_weighted_dir = os.path.join(outputs_dir, "div_weighted")

#         for target_dir in [div_dir, div_weighted_dir]:
#             if os.path.isdir(target_dir):
#                 for item in os.listdir(target_dir):
#                     if item.startswith("."):  # preserve hidden files like .metadata_never_index
#                         continue
#                     full_path = os.path.join(target_dir, item)
#                     if os.path.isfile(full_path) or os.path.islink(full_path):
#                         os.remove(full_path)
#                     elif os.path.isdir(full_path):
#                         shutil.rmtree(full_path)


#         # Clear routed NetCDF files
#         troute_dir = os.path.join(tile_root, "out", gage_id, "troute")
#         if os.path.isdir(troute_dir):
#             for f in os.listdir(troute_dir):
#                 if f.endswith(".nc"):
#                     os.remove(os.path.join(troute_dir, f))

#         # Clear postprocessing particle files
#         postproc_dir = os.path.join(tile_root, "postproc")
#         if os.path.isdir(postproc_dir):
#             for f in os.listdir(postproc_dir):
#                 if f.startswith(f"{gage_id}_particle_") and f.endswith(".csv"):
#                     os.remove(os.path.join(postproc_dir, f))

#     # Handle tile weights
#     if n_tiles == 2:
#         tile_weight = params[-1]
#         weights = [tile_weight, 1.0 - tile_weight]
#         params = params[:-1]
#     elif weights is None:
#         weights = [1.0 / n_tiles] * n_tiles

#     # === STEP 1: Run hydrology for each tile ===
#     for tile_idx, tile_root in enumerate(model_roots):
#         print(f"[INFO] Running hydrology for gage {gage_id} | Particle {particle_idx} | Tile {tile_idx}")
#         tile_params = extract_tile_params(params, tile_idx, n_tiles)
#         names = param_names.copy()
#         if include_nom_flags[tile_idx]:
#             names += nom_param_names
#         true_params = transform_params(tile_params, names)

#         # Update configs (CFE + NOM if present)
#         config_dir = os.path.join(tile_root, f"out/{gage_id}/configs/cfe")
#         regenerate_cfe_config(config_dir, true_params, names)
#         if include_nom_flags[tile_idx]:
#             nom_vals = tile_params[-6:]
#             update_mptable(
#                 original_file=nom_file_paths[tile_idx],
#                 output_file=nom_file_paths[tile_idx],
#                 updated_params=dict(zip(nom_param_names, nom_vals)),
#                 verbose=True
#             )

#         # Update realization JSON (spinup+cal only)
#         json_dir = os.path.join(tile_root, "out", gage_id, "json")
#         realization_file = [f for f in os.listdir(json_dir) if f.endswith(".json")][0]
#         realization_path = os.path.join(json_dir, realization_file)
#         with open(realization_path) as f:
#             realization = json.load(f)
#         realization["time"]["start_time"] = time_cfg["spinup_start"]
#         realization["time"]["end_time"] = time_cfg["cal_end"]
#         with open(realization_path, "w") as f:
#             json.dump(realization, f, indent=4)

#         # Run hydrology only (skip routing)
#         tile_sandbox_config = os.path.join(cfg.project_root, "configs", f"sandbox_config_tile{tile_idx+1}.yaml")
#         subprocess.call(
#             ["python", sandbox_path, "-i", tile_sandbox_config, "-run", "--gage_id", gage_id],
#             cwd=tile_root
#         )

#     # === STEP 2: Weighted divide outputs ===
#     if n_tiles == 1:
#         # Skip weighting to save on I/O: use tile 0's existing divide outputs directly
#         weighted_div_dir = os.path.join(model_roots[0], "out", gage_id, "outputs", "div")
#     else:
#         # Perform weighted averaging for multiple tiles
#         weighted_div_dir = os.path.join(model_roots[0], "out", gage_id, "outputs", "div_weighted")
#         if os.path.exists(weighted_div_dir):
#             shutil.rmtree(weighted_div_dir)  # Explicitly clear old weighted outputs
#         os.makedirs(weighted_div_dir, exist_ok=True)

#         div_dirs = [os.path.join(root, "out", gage_id, "outputs", "div") for root in model_roots]
#         files = [f for f in os.listdir(div_dirs[0]) if (f.startswith("cat-") or f.startswith("nex-")) and f.endswith(".csv")]

#         for fname in files:
#             dfs = []
#             for tile_idx, div_dir in enumerate(div_dirs):
#                 fpath = os.path.join(div_dir, fname)
#                 if os.path.exists(fpath):
#                     if fname.startswith("nex-"):
#                         df = pd.read_csv(fpath, header=None)  # Nex files: no header
#                         df.columns = ["Time Step", "Time", "q_out"]
#                     else:
#                         df = pd.read_csv(fpath)  # Cat files: already have a header
#                     dfs.append(df["q_out"] * weights[tile_idx])
#             if dfs:
#                 combined = sum(dfs)
#                 out_df = df.copy()
#                 out_df["q_out"] = combined
#                 out_df["Time"] = pd.to_datetime(out_df["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")

#                 # Save with/without header depending on file type
#                 if fname.startswith("nex-"):
#                     out_df.to_csv(os.path.join(weighted_div_dir, fname), index=False, header=False)
#                 else:
#                     out_df.to_csv(os.path.join(weighted_div_dir, fname), index=False)


#     # === STEP 3: Run routing once on weighted outputs ===
#     troute_path = os.path.join(model_roots[0], "out", gage_id, "configs", "troute_config.yaml")
#     with open(troute_path) as f:
#         troute_cfg = yaml.safe_load(f)

#     nts = int((cal_end - spinup_start) / pd.Timedelta(seconds=300))
#     troute_cfg["compute_parameters"]["restart_parameters"]["start_datetime"] = spinup_start.strftime("%Y-%m-%d_%H:%M:%S")
#     troute_cfg["compute_parameters"]["forcing_parameters"]["nts"] = nts

#     troute_cfg["compute_parameters"]["forcing_parameters"]["qlat_input_folder"] = weighted_div_dir

#     yaml.safe_dump(troute_cfg, open(troute_path, "w"))


#     troute_dir = os.path.join(model_roots[0], "out", gage_id, "troute")
#     if os.path.isdir(troute_dir):
#         for fname in os.listdir(troute_dir):
#             if fname.endswith(".nc"):
#                 os.remove(os.path.join(troute_dir, fname))

#     subprocess.call(["python3", "-m", "nwm_routing", "-f", "-V4", troute_path])

#     # === STEP 4: Extract routed hydrograph ===
#     postproc_dir = os.path.join(model_roots[0], "postproc")
#     output_path = os.path.join(postproc_dir, f"{gage_id}_particle_{particle_idx}.csv")
#     get_hydrograph_path = os.path.join(project_root, "model_assessment", "util", "get_hydrograph.py")
#     subprocess.call(
#         ["python", get_hydrograph_path, "--gage_id", gage_id, "--output", output_path, "--base_dir", model_roots[0]],
#         cwd=postproc_dir
#     )

#     # === STEP 5: Compute metrics ===
#     sim_df = pd.read_csv(output_path, parse_dates=["current_time"]).set_index("current_time")["flow"].resample("1h").mean()
#     obs_df = get_observed_q(os.path.join(observed_q_root, "successful_sites_resampled", f"{gage_id}.csv"))

#     sim_cal, obs_cal = sim_df[cal_start:cal_end].dropna(), obs_df[cal_start:cal_end].dropna()
#     sim_val, obs_val = sim_df[val_start:val_end].dropna(), obs_df[val_start:val_end].dropna()
#     sim_cal, obs_cal = sim_cal.align(obs_cal, join="inner")
#     sim_val, obs_val = sim_val.align(obs_val, join="inner")

#     if len(sim_cal) > 0: sim_cal.iloc[-1] += 1e-8
#     if len(obs_cal) > 0: obs_cal.iloc[-1] += 1e-8

#     cal_metrics = compute_metrics(sim_cal, obs_cal, event_threshold=1e-2)
#     val_metrics = compute_metrics(sim_val, obs_val, event_threshold=1e-2)

#     return -cal_metrics[metric_to_calibrate_on], val_metrics, cal_metrics



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
#     def __init__(self,
#                  n_particles, bounds, n_iterations, gage_id,
#                  init_position, config_path, observed_path, postproc_base_path,
#                  metric_to_calibrate_on="kge",
#                  include_nom=False,
#                  nom_file_paths=None,
#                  param_names=None):
#         self.particles = [
#             Particle(bounds, init_position=init_position if i == 0 else None)
#             for i in range(n_particles)
#         ]
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
#         self.include_nom = include_nom
#         self.nom_file_paths = nom_file_paths or []   # <-- now stored correctly
#         self.param_names = param_names if param_names else param_names

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

#             results = []
#             for i, p in enumerate(self.particles):
#                 print(f"[INFO] Running gage {self.gage_id} | iteration {iteration + 1} | particle {i}")
#                 result = objective_function_tiled(
#                     (
#                         p.position, i, self.gage_id, cfg.model_roots,
#                         observed_q_root,
#                         [self.include_nom] * len(cfg.model_roots),
#                         self.nom_file_paths,    # <-- FIXED: pass the correct per-tile paths
#                         [1.0 / len(cfg.model_roots)] * len(cfg.model_roots)
#                     )
#                 )

#                 results.append(result)

#             for idx, (objective_value, val_metrics, cal_metrics) in enumerate(results):
#                 particle = self.particles[idx]
#                 particle.current_value = objective_value

#                 metric_calibration = cal_metrics[self.metric_to_calibrate_on]
#                 metric_validation = val_metrics[self.metric_to_calibrate_on]

#                 if objective_value < (particle.best_value - 0.001):
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

#                 if particle.stagnation_counter >= stagnation_threshold:
#                     print(f"Resetting particle {idx} after {stagnation_threshold} stagnant iterations.")
#                     particle.reset(self.bounds)

#                 param_dict = {name: val for name, val in zip(self.param_names, particle.position)}
#                 row = {
#                     "iteration": iteration + 1,
#                     "particle": idx,
#                     **param_dict,
#                     f"{self.metric_to_calibrate_on}_calibration": metric_calibration,
#                     f"{self.metric_to_calibrate_on}_validation": metric_validation
#                 }
#                 log_rows.append(row)

#             pd.DataFrame(log_rows).to_csv(log_path, index=False)

#             for p in self.particles:
#                 p.update_velocity(self.global_best_position, w=w, c1=1.5, c2=1.5)
#                 p.update_position(self.bounds)

#             print(f"Global best objective so far: {-self.global_best_value:.4f}")

#         # === Final full-period multi-tile run ===
#         print(f"\n[INFO] Running final weighted-routing validation for {self.gage_id}...")

#         n_tiles = len(cfg.model_roots)
#         weights = [1.0 / n_tiles] * n_tiles
#         if n_tiles == 2:
#             weights = [self.global_best_position[-1], 1.0 - self.global_best_position[-1]]

#         # === STEP 1: Run hydrology per tile ===
#         for tile_idx, tile_root in enumerate(cfg.model_roots):
#             tile_params = extract_tile_params(self.global_best_position, tile_idx, n_tiles)
#             names = param_names.copy()
#             if self.include_nom:
#                 names += nom_param_names
#             true_best_params = transform_params(tile_params, names)

#             # Update configs
#             config_dir = os.path.join(tile_root, f"out/{self.gage_id}/configs/cfe")
#             regenerate_cfe_config(config_dir, true_best_params, names)

#             # Update NOM if present
#             if self.include_nom:
#                 nom_vals = tile_params[-6:]
#                 update_mptable(
#                     original_file=os.path.join(tile_root, f"out/{self.gage_id}/configs/noahowp/parameters/MPTABLE.TBL"),
#                     output_file=os.path.join(tile_root, f"out/{self.gage_id}/configs/noahowp/parameters/MPTABLE.TBL"),
#                     updated_params=dict(zip(nom_param_names, nom_vals)),
#                     verbose=True
#                 )

#             # Update realization JSON (full spinup+val)
#             json_dir = os.path.join(tile_root, "out", self.gage_id, "json")
#             realization_file = [f for f in os.listdir(json_dir) if f.endswith(".json")][0]
#             realization_path = os.path.join(json_dir, realization_file)
#             with open(realization_path, "r") as f:
#                 realization = json.load(f)
#             realization["time"]["start_time"] = time_cfg["spinup_start"]
#             realization["time"]["end_time"] = time_cfg["val_end"]
#             with open(realization_path, "w") as f:
#                 json.dump(realization, f, indent=4)

#             # Run hydrology only
#             tile_sandbox_config = os.path.join(cfg.project_root, "configs", f"sandbox_config_tile{tile_idx+1}.yaml")
#             subprocess.call(
#                 ["python", sandbox_path, "-i", tile_sandbox_config, "-run", "--gage_id", self.gage_id],
#                 cwd=tile_root
#             )

#         # === STEP 2: Weighted averaging of divide outputs ===
#         if n_tiles == 1:
#             # Skip weighting to save on I/O: use tile 0's existing divide outputs directly
#             weighted_div_dir = os.path.join(self.model_roots[0], "out", self.gage_id, "outputs", "div")
#         else:
#             # Perform weighted averaging for multiple tiles
#             weighted_div_dir = os.path.join(self.model_roots[0], "out", self.gage_id, "outputs", "div_weighted")
#             if os.path.exists(weighted_div_dir):
#                 shutil.rmtree(weighted_div_dir)  # Explicitly clear old weighted outputs
#             os.makedirs(weighted_div_dir, exist_ok=True)

#             div_dirs = [os.path.join(root, "out", self.gage_id, "outputs", "div") for root in self.model_roots]
#             files = [f for f in os.listdir(div_dirs[0]) if (f.startswith("cat-") or f.startswith("nex-")) and f.endswith(".csv")]

#             for fname in files:
#                 dfs = []
#                 for tile_idx, div_dir in enumerate(div_dirs):
#                     fpath = os.path.join(div_dir, fname)
#                     if os.path.exists(fpath):
#                         if fname.startswith("nex-"):
#                             df = pd.read_csv(fpath, header=None)  # No header
#                             df.columns = ["Time Step", "Time", "q_out"]
#                         else:
#                             df = pd.read_csv(fpath)  # Has header
#                         dfs.append(df["q_out"] * weights[tile_idx])
#                 if dfs:
#                     combined = sum(dfs)
#                     out_df = df.copy()
#                     out_df["q_out"] = combined
#                     out_df["Time"] = pd.to_datetime(out_df["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")

#                     if fname.startswith("nex-"):
#                         out_df.to_csv(os.path.join(weighted_div_dir, fname), index=False, header=False)
#                     else:
#                         out_df.to_csv(os.path.join(weighted_div_dir, fname), index=False)



#         # === STEP 3: Run routing once ===
#         troute_path = os.path.join(cfg.model_roots[0], "out", self.gage_id, "configs", "troute_config.yaml")
#         with open(troute_path) as f:
#             troute_cfg = yaml.safe_load(f)
#         nts_full = int((val_end - spinup_start) / pd.Timedelta(seconds=300))
#         troute_cfg["compute_parameters"]["restart_parameters"]["start_datetime"] = spinup_start.strftime("%Y-%m-%d_%H:%M:%S")
#         troute_cfg["compute_parameters"]["forcing_parameters"]["nts"] = nts_full
#         troute_cfg["compute_parameters"]["forcing_parameters"]["qlat_input_folder"] = weighted_div_dir
#         yaml.safe_dump(troute_cfg, open(troute_path, "w"))

#         # === CLEAR OLD ROUTING FILES ===
#         troute_dir = os.path.join(cfg.model_roots[0], "out", self.gage_id, "troute")
#         if os.path.isdir(troute_dir):
#             for fname in os.listdir(troute_dir):
#                 if fname.endswith(".nc"):
#                     os.remove(os.path.join(troute_dir, fname))

#         # === RUN ROUTING ===
#         subprocess.call(["python3", "-m", "nwm_routing", "-f", "-V4", troute_path])

#         # === STEP 4: Extract final routed hydrograph ===
#         postproc_dir = os.path.join(cfg.model_roots[0], "postproc")
#         final_output_path = os.path.join(postproc_dir, f"{self.gage_id}_best.csv")
#         get_hydrograph_path = os.path.join(project_root, "model_assessment", "util", "get_hydrograph.py")
#         subprocess.call(
#             ["python", get_hydrograph_path, "--gage_id", self.gage_id, "--output", final_output_path, "--base_dir", cfg.model_roots[0]],
#             cwd=postproc_dir
#         )
#         print(f" Final best hydrograph saved to {final_output_path}")

#         # === STEP 5: Compute final metrics ===
#         obs_df = get_observed_q(self.observed_path)
#         sim_df = pd.read_csv(final_output_path, parse_dates=["current_time"]).set_index("current_time")["flow"].resample("1h").mean()
#         sim_val, obs_val = sim_df[val_start:val_end].dropna(), obs_df[val_start:val_end].dropna()
#         sim_val, obs_val = sim_val.align(obs_val, join="inner")
#         if len(sim_val) > 0: sim_val.iloc[-1] += 1e-8
#         if len(obs_val) > 0: obs_val.iloc[-1] += 1e-8
#         val_metrics_final = compute_metrics(sim_val, obs_val, event_threshold=1e-2)

#         # === Append final row to log ===
#         final_row = {
#             "iteration": "FINAL",
#             "particle": "BEST",
#             **{name: val for name, val in zip(self.param_names, self.global_best_position)},
#             f"{self.metric_to_calibrate_on}_calibration": self.best_cal_metrics.get(self.metric_to_calibrate_on, np.nan),
#             f"{self.metric_to_calibrate_on}_validation": val_metrics_final.get(self.metric_to_calibrate_on, np.nan)
#         }
#         pd.DataFrame(log_rows + [final_row]).to_csv(log_path, index=False)
#         print(f" Final {self.metric_to_calibrate_on.upper()} = {val_metrics_final.get(self.metric_to_calibrate_on, np.nan):.4f}")

#         return self.global_best_position, self.global_best_value, self.best_validation_metric, datetime.now() - start_time



# # === Per-gage wrapper ===
# def calibrate_gage(gage_id):
#     model_roots = cfg.model_roots
#     n_tiles = len(model_roots)
#     all_init_params, all_bounds, include_nom_flags, nom_file_paths = [], [], [], []

#     for tile_idx, root in enumerate(model_roots):
#         # --- Locate the CFE config file ---
#         config_dir = os.path.join(root, f"out/{gage_id}/configs/cfe")
#         config_file = sorted(f for f in os.listdir(config_dir) if f.startswith("cfe_config_cat"))[0]
#         config_path = os.path.join(config_dir, config_file)

#         # === NEW: Unified param extraction ===
#         init = extract_initial_params(config_path)  # Combines CFE + NOM if present
#         bounds = param_bounds.copy()

#         # === Detect NOM presence and append bounds ===
#         include_nom = os.path.isdir(os.path.join(root, f"out/{gage_id}/configs/noahowp"))
#         if include_nom:
#             nom_path = os.path.join(root, f"out/{gage_id}/configs/noahowp/parameters/MPTABLE.TBL")
#             bounds += nom_param_bounds
#             nom_file_paths.append(nom_path)
#         else:
#             nom_file_paths.append("")

#         # === Accumulate params and bounds for this tile ===
#         all_init_params.extend(init)
#         all_bounds.extend(bounds)
#         include_nom_flags.append(include_nom)

#     # === Expand parameter names per tile (CFE + NOM) ===
#     names = []
#     for tile_idx in range(n_tiles):
#         tile_suffix = f"_tile{tile_idx+1}"
#         tile_names = [f"{name}{tile_suffix}" for name in param_names]
#         if include_nom_flags[tile_idx]:
#             tile_names += [f"{name}{tile_suffix}" for name in nom_param_names]
#         names.extend(tile_names)

#     # === Add tile weight for 2-tile setup ===
#     if n_tiles == 2:
#         all_init_params.append(0.7)            # Initial tile weight
#         all_bounds.append((0.0, 1.0))          # Weight bounds
#         names.append("tile_weight")            # For logging

#     weights = [1.0 / n_tiles] * n_tiles
#     print(f"[DEBUG] include_nom_flags: {include_nom_flags}")
#     print(f"[DEBUG] nom_file_paths: {nom_file_paths}")

#     # === Create and run PSO ===
#     pso = PSO(
#         n_particles=n_particles,
#         bounds=all_bounds,
#         n_iterations=n_iterations,
#         gage_id=gage_id,
#         init_position=all_init_params,
#         config_path="",  # no longer needed
#         observed_path=os.path.join(observed_q_root, "successful_sites_resampled", f"{gage_id}.csv"),
#         postproc_base_path=model_roots[0] + "/postproc",
#         metric_to_calibrate_on=metric_to_calibrate_on,
#         include_nom=any(include_nom_flags),
#         nom_file_paths=nom_file_paths,   # <-- FIXED
#         param_names=names
#     )
#     pso.model_roots = model_roots


#     pso.optimize()


# if __name__ == "__main__":
#     start = datetime.now()
#     gage_list = pd.read_csv(cfg.gages_file, dtype={"gage_id": str})["gage_id"].tolist()
#     ctx = multiprocessing.get_context("spawn")
#     with ctx.Pool(processes=max_cores_for_gages) as pool:
#         pool.map(calibrate_gage, gage_list)
#     print(f"Total wall time: {datetime.now() - start}")





















































# """
# Author: Peter La Follette [plafollette@lynker.com | July 2025]
# Multi-tile PSO calibration for CFE+PET+T-Route, with optional NOM parameter support.
# """

# import os
# import subprocess
# import pandas as pd
# import numpy as np
# import math
# import yaml
# import json
# import random
# import sys
# import traceback
# import multiprocessing
# from datetime import datetime
# from hydroeval import kge

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# from model_assessment.util.metrics import compute_metrics
# from model_assessment.util.update_NOM import update_mptable
# from model_assessment.configs import path_config as cfg

# np.random.seed(42)
# random.seed(42)

# # === CONFIGURATION ===
# n_particles = 15
# n_iterations = 50
# max_cores_for_gages = 2
# metric_to_calibrate_on = "kge"

# with open("model_assessment/configs/time_config.yaml", "r") as f:
#     time_cfg = yaml.safe_load(f)

# spinup_start = pd.Timestamp(time_cfg["spinup_start"])
# cal_start    = pd.Timestamp(time_cfg["cal_start"])
# cal_end      = pd.Timestamp(time_cfg["cal_end"])
# val_start    = pd.Timestamp(time_cfg["val_start"])
# val_end      = pd.Timestamp(time_cfg["val_end"])

# project_root = cfg.project_root
# sandbox_path = cfg.sandbox_path
# logging_dir = cfg.logging_dir
# observed_q_root = cfg.observed_q_root

# os.makedirs(logging_dir, exist_ok=True)

# # === Parameter definitions ===
# param_names = [
#     "b", "satdk", "satpsi", "maxsmc", "max_gw_storage", "Cgw", "expon", "Kn",
#     "Klf", "refkdt", "slope", "wltsmc", "alpha_fc", "Kinf_nash_surface"
# ]

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

# log_scale_params = {"Cgw": True, "satdk": True}

# nom_param_names = ["MFSNO", "RSURF_SNOW", "HVT", "CWPVT", "VCMX25", "MP"]
# nom_param_bounds = [
#     (0.625, 5.0), (0.1, 100.0), (0.0, 20.0), (0.18, 5.0), (0.0, 80.0), (3.6, 12.6)
# ]

# # === Helpers ===
# def clear_terminal():
#     os.system("clear")

# def check_for_stop_signal_or_low_disk(threshold_gb=50):
#     stop_file = os.path.join(project_root, "STOP_NOW.txt")
#     if os.path.exists(stop_file):
#         sys.exit(1)
#     stat = os.statvfs("/")
#     free_gb = (stat.f_frsize * stat.f_bavail) / (1024 ** 3)
#     if free_gb < threshold_gb:
#         sys.exit(1)

# def transform_params(params, names):
#     return [10**p if log_scale_params.get(name, False) else p for name, p in zip(names, params)]

# def extract_tile_params(full_params, tile_idx, n_tiles):
#     total_len = len(full_params)
#     if n_tiles == 2 and total_len % 2 == 1:
#         param_slice = full_params[:-1]
#         chunk = len(param_slice) // n_tiles
#         return param_slice[tile_idx * chunk : (tile_idx + 1) * chunk]
#     chunk = total_len // n_tiles
#     return full_params[tile_idx * chunk : (tile_idx + 1) * chunk]

# def get_observed_q(observed_path):
#     df = pd.read_csv(observed_path, parse_dates=['value_time']).set_index('value_time')
#     return df['flow_m3_per_s']

# def extract_initial_params(config_path):
#     """Extract CFE params from cfe_config_cat*.txt and NOM params if present."""
#     cfe_params = extract_initial_cfe_params(config_path)

#     # Detect NOM
#     config_root = os.path.dirname(os.path.dirname(config_path))
#     nom_dir = os.path.join(config_root, "noahowp")
#     nom_params = []
#     if os.path.isdir(nom_dir):
#         mptable_path = os.path.join(nom_dir, "parameters", "MPTABLE.TBL")
#         nom_params = extract_initial_nom_params(mptable_path)

#     return cfe_params + nom_params

# def extract_initial_nom_params(mptable_path):
#     """
#     Extract NOM params for tile that uses Noah-MP from MPTABLE.TBL.

#     Args:
#         mptable_path (str): Path to the MPTABLE.TBL file.

#     Returns:
#         list of float: NOM parameter values in the correct order.
#     """
#     param_order = ["MFSNO", "RSURF_SNOW", "HVT", "CWPVT", "VCMX25", "MP"]
#     values = []

#     if not os.path.isfile(mptable_path):
#         raise FileNotFoundError(f"[ERROR] MPTABLE.TBL not found: {mptable_path}")

#     with open(mptable_path, "r") as f:
#         lines = f.readlines()

#     for pname in param_order:
#         found = False
#         for line in lines:
#             stripped = line.strip()
#             if not stripped or stripped.startswith(("!", "&", "/")):
#                 continue  # Skip comments, section markers, blank lines
#             if stripped.startswith(pname):
#                 # Split at '=', then handle inline comments and commas
#                 if "=" in stripped:
#                     val_part = stripped.split("=", 1)[1]
#                 else:
#                     # Fallback: space-delimited (legacy style)
#                     val_part = stripped[len(pname):].strip()

#                 val_clean = val_part.split("!")[0].split(",")[0].strip()  # remove comments and take first value

#                 try:
#                     values.append(float(val_clean))
#                     found = True
#                     break
#                 except ValueError:
#                     raise ValueError(f"[ERROR] Could not parse value for {pname} in line: {line.strip()}")

#         if not found:
#             raise ValueError(f"[ERROR] Missing parameter {pname} in {mptable_path}")

#     if len(values) != len(param_order):
#         raise ValueError(f"[ERROR] Expected {len(param_order)} NOM params but got {len(values)} from {mptable_path}")

#     return values


# def extract_initial_cfe_params(config_path):
#     """
#     Extracts initial CFE parameters from config file.
#     If a parameter is missing, falls back to default value 0.002.
#     """
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
#         "Kinf_nash_surface": "Kinf_nash_surface"
#     }

#     values = {}
#     with open(config_path, "r") as f:
#         lines = f.readlines()
#     for line in lines:
#         if "=" in line:
#             key, val = line.strip().split("=", 1)
#             key = key.strip()
#             val = val.split("[")[0].strip()
#             for param, config_key in param_map.items():
#                 if key == config_key:
#                     values[param] = float(val)

#     param_order = list(param_map.keys())

#     extracted = []
#     for name in param_order:
#         if name in values:
#             v = values[name]
#         else:
#             print(f"[WARN] {name} not found in {config_path}. Using default 0.002. This is expected for the first iteration.")
#             v = 0.002
#         if log_scale_params.get(name, False):
#             extracted.append(math.log10(v))
#         else:
#             extracted.append(v)

#     return extracted


# # def extract_initial_nom_params(mptable_path):
# #     """Extract NOM params for tile that uses Noah-MP"""
# #     values = []
# #     param_order = nom_param_names.copy()
# #     with open(mptable_path) as f:
# #         for line in f:
# #             for pname in param_order:
# #                 if line.strip().startswith(pname):
# #                     val = float(line.strip().split()[1])
# #                     values.append(val)
# #     if len(values) != len(param_order):
# #         raise ValueError(f"Expected {len(param_order)} NOM params but got {len(values)} from {mptable_path}")
# #     return values


# def regenerate_cfe_config(config_path, params, names):
#     replacements = dict(zip(names, params))
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
#         "Kinf_nash_surface": "Kinf_nash_surface"
#     }

#     found_keys = set()
#     updated_lines = []

#     if os.path.isfile(config_path):
#         config_dir = os.path.dirname(config_path)
#     else:
#         config_dir = config_path

#     for fname in sorted(f for f in os.listdir(config_dir) if f.startswith("cfe_config_cat")):
#         path = os.path.join(config_dir, fname)
#         lines = []
#         with open(path) as f:
#             for line in f:
#                 key = line.split("=")[0].strip()
#                 replaced = False
#                 for pname, ckey in param_map.items():
#                     if key == ckey:
#                         unit = line[line.find("["):] if "[" in line else ""
#                         lines.append(f"{ckey}={replacements[pname]}{unit}\n")
#                         found_keys.add(pname)
#                         replaced = True
#                         break
#                 if not replaced:
#                     lines.append(line)
#         # Add missing keys
#         for pname, ckey in param_map.items():
#             if pname not in found_keys:
#                 lines.append(f"{ckey}={replacements[pname]}\n")
#         with open(path, "w") as f:
#             f.writelines(lines)

# # === Tiled objective function ===
# def objective_function_tiled(args):
#     """
#     Runs tiled CFE+PET+T-Route model for one PSO particle.
#     Args should include:
#       - params
#       - particle_idx
#       - gage_id
#       - model_roots
#       - observed_q_root
#       - include_nom_flags
#       - nom_file_paths
#       - weights (optional)
#     """
#     (
#         params, particle_idx, gage_id,
#         model_roots, observed_q_root,
#         include_nom_flags, nom_file_paths,
#         weights
#     ) = args

#     check_for_stop_signal_or_low_disk()

#     n_tiles = len(model_roots)

#     if n_tiles == 2:
#         tile_weight = params[-1]
#         weights = [tile_weight, 1.0 - tile_weight]
#         params = params[:-1]
#     elif weights is None:
#         weights = [1.0 / n_tiles] * n_tiles

#     sim_dfs = []

#     for tile_idx, tile_root in enumerate(model_roots):
#         print(f"[INFO] Running gage {gage_id} | Particle {particle_idx} | Tile {tile_idx}")

#         # === Extract parameters for this tile
#         tile_params = extract_tile_params(params, tile_idx, n_tiles)
#         names = param_names.copy()
#         if include_nom_flags[tile_idx]:
#             names += nom_param_names

#         true_params = transform_params(tile_params, names)

#         # === Update CFE config for this tile
#         config_dir = os.path.join(tile_root, f"out/{gage_id}/configs/cfe")
#         regenerate_cfe_config(config_dir, true_params, names)

#         if include_nom_flags[tile_idx]:
#             nom_vals = tile_params[-6:]
#             nom_param_dict = dict(zip(nom_param_names, nom_vals))
#             update_mptable(
#                 original_file=nom_file_paths[tile_idx],
#                 output_file=nom_file_paths[tile_idx],
#                 updated_params=nom_param_dict,
#                 verbose=True
#             )

#         # === Update realization JSON ===
#         json_dir = os.path.join(tile_root, "out", gage_id, "json")
#         realization_file = [f for f in os.listdir(json_dir) if f.endswith(".json")][0]
#         realization_path = os.path.join(json_dir, realization_file)

#         with open(realization_path) as f:
#             realization = json.load(f)
#         realization["time"]["start_time"] = time_cfg["spinup_start"]
#         realization["time"]["end_time"] = time_cfg["cal_end"]
#         with open(realization_path, "w") as f:
#             json.dump(realization, f, indent=4)

#         # === Update troute_config.yaml ===
#         troute_path = os.path.join(tile_root, "out", gage_id, "configs", "troute_config.yaml")
#         with open(troute_path) as f:
#             troute_cfg = yaml.safe_load(f)
#         nts = int((cal_end - spinup_start) / pd.Timedelta(seconds=300))
#         troute_cfg["compute_parameters"]["restart_parameters"]["start_datetime"] = spinup_start.strftime("%Y-%m-%d_%H:%M:%S")
#         troute_cfg["compute_parameters"]["forcing_parameters"]["nts"] = nts
#         with open(troute_path, "w") as f:
#             yaml.dump(troute_cfg, f)

#         # === Delete stale .nc files ===
#         troute_dir = os.path.join(tile_root, "out", gage_id, "troute")
#         if os.path.isdir(troute_dir):
#             for fname in os.listdir(troute_dir):
#                 if fname.endswith(".nc"):
#                     file_path = os.path.join(troute_dir, fname)
#                     try:
#                         os.remove(file_path)
#                         print(f"[DEBUG] Deleted old routing file: {file_path}")
#                     except Exception as e:
#                         print(f"[WARN] Could not delete {file_path}: {e}")

#         # === Use TILE-SPECIFIC sandbox config ===
#         tile_sandbox_config = os.path.join(cfg.project_root, "configs", f"sandbox_config_tile{tile_idx+1}.yaml")

#         subprocess.call(
#             ["python", sandbox_path, "-i", tile_sandbox_config, "-run", "--gage_id", gage_id],
#             cwd=tile_root
#         )

#         # === Delete old postproc file for this tile+particle ===
#         output_path = os.path.join(tile_root, "postproc", f"{gage_id}_particle_{particle_idx}.csv")
#         if os.path.exists(output_path):
#             try:
#                 os.remove(output_path)
#                 print(f"[DEBUG] Deleted old postproc file: {output_path}")
#             except Exception as e:
#                 print(f"[WARN] Could not delete {output_path}: {e}")

#         # === Post-process hydrograph for this tile ===
#         output_path = os.path.join(tile_root, "postproc", f"{gage_id}_particle_{particle_idx}.csv")
#         get_hydrograph_path = os.path.join(project_root, "model_assessment", "util", "get_hydrograph.py")
#         subprocess.call(
#             ["python", get_hydrograph_path, "--gage_id", gage_id, "--output", output_path, "--base_dir", tile_root],
#             cwd=os.path.join(tile_root, "postproc")
#         )

#         sim_df = pd.read_csv(output_path, parse_dates=['current_time']).set_index('current_time')['flow'].resample('1h').mean()
#         sim_dfs.append(sim_df)

#     # === Average the hydrographs across tiles ===
#     avg_sim = sum(w * s for w, s in zip(weights, sim_dfs))

#     # === Save averaged hydrograph ===
#     avg_df = avg_sim.reset_index().rename(columns={'index': 'current_time', 0: 'flow'})
#     output_path = os.path.join(model_roots[0], "postproc", f"{gage_id}_particle_{particle_idx}.csv")
#     avg_df.to_csv(output_path, index=False)

#     obs_df = get_observed_q(os.path.join(observed_q_root, "successful_sites_resampled", f"{gage_id}.csv"))

#     sim_cal, obs_cal = avg_sim[cal_start:cal_end].dropna(), obs_df[cal_start:cal_end].dropna()
#     sim_val, obs_val = avg_sim[val_start:val_end].dropna(), obs_df[val_start:val_end].dropna()
#     sim_cal, obs_cal = sim_cal.align(obs_cal, join='inner')
#     sim_val, obs_val = sim_val.align(obs_val, join='inner')

#     sim_cal.iloc[-1] += 1e-8
#     obs_cal.iloc[-1] += 1e-8

#     cal_metrics = compute_metrics(sim_cal, obs_cal, event_threshold=1e-2)
#     val_metrics = compute_metrics(sim_val, obs_val, event_threshold=1e-2)

#     return -cal_metrics[metric_to_calibrate_on], val_metrics, cal_metrics


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
#     def __init__(self, 
#                  n_particles, bounds, n_iterations, gage_id,
#                  init_position, config_path, observed_path, postproc_base_path,
#                  metric_to_calibrate_on="kge",
#                  include_nom=False, 
#                  nom_file_paths=None,
#                  param_names=None):
#         self.particles = [
#             Particle(bounds, init_position=init_position if i == 0 else None)
#             for i in range(n_particles)
#         ]
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
#         self.include_nom = include_nom
#         self.nom_file_paths = nom_file_paths or []   # <-- now stored correctly
#         self.param_names = param_names if param_names else param_names

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

#             results = []
#             for i, p in enumerate(self.particles):
#                 print(f"[INFO] Running gage {self.gage_id} | iteration {iteration + 1} | particle {i}")
#                 result = objective_function_tiled(
#                     (
#                         p.position, i, self.gage_id, cfg.model_roots,
#                         observed_q_root,
#                         [self.include_nom] * len(cfg.model_roots),
#                         self.nom_file_paths,    # <-- FIXED: pass the correct per-tile paths
#                         [1.0 / len(cfg.model_roots)] * len(cfg.model_roots)
#                     )
#                 )

#                 results.append(result)

#             for idx, (objective_value, val_metrics, cal_metrics) in enumerate(results):
#                 particle = self.particles[idx]
#                 particle.current_value = objective_value

#                 metric_calibration = cal_metrics[self.metric_to_calibrate_on]
#                 metric_validation = val_metrics[self.metric_to_calibrate_on]

#                 if objective_value < (particle.best_value - 0.001):
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

#                 if particle.stagnation_counter >= stagnation_threshold:
#                     print(f"Resetting particle {idx} after {stagnation_threshold} stagnant iterations.")
#                     particle.reset(self.bounds)

#                 param_dict = {name: val for name, val in zip(self.param_names, particle.position)}
#                 row = {
#                     "iteration": iteration + 1,
#                     "particle": idx,
#                     **param_dict,
#                     f"{self.metric_to_calibrate_on}_calibration": metric_calibration,
#                     f"{self.metric_to_calibrate_on}_validation": metric_validation
#                 }
#                 log_rows.append(row)

#             pd.DataFrame(log_rows).to_csv(log_path, index=False)

#             for p in self.particles:
#                 p.update_velocity(self.global_best_position, w=w, c1=1.5, c2=1.5)
#                 p.update_position(self.bounds)

#             print(f"Global best objective so far: {-self.global_best_value:.4f}")

#         # === Final full-period multi-tile run ===
#         print(f"\nRunning final full-period validation for {self.gage_id}...")

#         sim_dfs = []
#         n_tiles = len(cfg.model_roots)
#         weights = [1.0 / n_tiles] * n_tiles

#         for tile_idx, tile_root in enumerate(cfg.model_roots):
#             tile_params = extract_tile_params(self.global_best_position, tile_idx, n_tiles)
#             names = param_names.copy()
#             if self.include_nom:
#                 names += nom_param_names

#             true_best_params = transform_params(tile_params, names)

#             # === Update CFE config ===
#             config_dir = os.path.join(tile_root, f"out/{self.gage_id}/configs/cfe")
#             regenerate_cfe_config(config_dir, true_best_params, names)

#             # === Update NOM if present ===
#             if self.include_nom:
#                 nom_vals = tile_params[-6:]
#                 nom_param_dict = dict(zip(nom_param_names, nom_vals))
#                 update_mptable(
#                     original_file=os.path.join(tile_root, f"out/{self.gage_id}/configs/noahowp/parameters/MPTABLE.TBL"),
#                     output_file=os.path.join(tile_root, f"out/{self.gage_id}/configs/noahowp/parameters/MPTABLE.TBL"),
#                     updated_params=nom_param_dict,
#                     verbose=True
#                 )

#             # === Update realization JSON ===
#             json_dir = os.path.join(tile_root, "out", self.gage_id, "json")
#             realization_file = [f for f in os.listdir(json_dir) if f.endswith(".json")][0]
#             realization_path = os.path.join(json_dir, realization_file)

#             with open(realization_path, "r") as f:
#                 realization = json.load(f)
#             realization["time"]["start_time"] = time_cfg["spinup_start"]
#             realization["time"]["end_time"] = time_cfg["val_end"]
#             with open(realization_path, "w") as f:
#                 json.dump(realization, f, indent=4)

#             # === Update t-route config ===
#             troute_path = os.path.join(tile_root, "out", self.gage_id, "configs", "troute_config.yaml")
#             with open(troute_path, "r") as f:
#                 troute_cfg = yaml.safe_load(f)
#             nts_full = int((val_end - spinup_start) / pd.Timedelta(seconds=300))
#             troute_cfg["compute_parameters"]["restart_parameters"]["start_datetime"] = spinup_start.strftime("%Y-%m-%d_%H:%M:%S")
#             troute_cfg["compute_parameters"]["forcing_parameters"]["nts"] = nts_full
#             with open(troute_path, "w") as f:
#                 yaml.dump(troute_cfg, f)

#             # === DELETE old .nc routing files ===
#             troute_dir = os.path.join(tile_root, "out", self.gage_id, "troute")
#             if os.path.isdir(troute_dir):
#                 for fname in os.listdir(troute_dir):
#                     if fname.endswith(".nc"):
#                         file_path = os.path.join(troute_dir, fname)
#                         try:
#                             os.remove(file_path)
#                             print(f"[DEBUG] Deleted old routing file: {file_path}")
#                         except Exception as e:
#                             print(f"[WARN] Could not delete {file_path}: {e}")

#             # === TILE-SPECIFIC sandbox config ===
#             tile_sandbox_config = os.path.join(cfg.project_root, "configs", f"sandbox_config_tile{tile_idx+1}.yaml")

#             print(f"[INFO] Running gage {self.gage_id} | FINAL | Tile {tile_idx}")

#             subprocess.call(
#                 ["python", sandbox_path, "-i", tile_sandbox_config, "-run", "--gage_id", self.gage_id],
#                 cwd=tile_root
#             )

#             # === Post-process ===
#             get_hydrograph_path = os.path.join(project_root, "model_assessment", "util", "get_hydrograph.py")
#             output_path = os.path.join(tile_root, "postproc", f"{self.gage_id}_best_tile_{tile_idx}.csv")
#             subprocess.call(
#                 ["python", get_hydrograph_path, "--gage_id", self.gage_id, "--output", output_path, "--base_dir", tile_root],
#                 cwd=os.path.join(tile_root, "postproc")
#             )

#             sim_df = pd.read_csv(output_path, parse_dates=['current_time']).set_index('current_time')['flow'].resample('1h').mean()
#             sim_dfs.append(sim_df)

#         # === Combine final tiles ===
#         avg_sim = sum(w * s for w, s in zip(weights, sim_dfs))
#         final_df = avg_sim.reset_index().rename(columns={'index': 'current_time', 0: 'flow'})
#         final_output_path = os.path.join(cfg.model_roots[0], "postproc", f"{self.gage_id}_best.csv")
#         final_df.to_csv(final_output_path, index=False)
#         print(f" Final best hydrograph saved to {final_output_path}")

#         obs_df = get_observed_q(self.observed_path)
#         sim_val, obs_val = avg_sim[val_start:val_end].dropna(), obs_df[val_start:val_end].dropna()
#         sim_val, obs_val = sim_val.align(obs_val, join='inner')
#         if len(sim_val) > 0 and len(obs_val) > 0:
#             sim_val.iloc[-1] += 1e-8
#             obs_val.iloc[-1] += 1e-8
#         val_metrics_final = compute_metrics(sim_val, obs_val, event_threshold=1e-2)

#         final_row = {
#             "iteration": "FINAL",
#             "particle": "BEST",
#             **{name: val for name, val in zip(self.param_names, self.global_best_position)},
#             f"{self.metric_to_calibrate_on}_calibration": self.best_cal_metrics.get(self.metric_to_calibrate_on, np.nan),
#             f"{self.metric_to_calibrate_on}_validation": val_metrics_final.get(self.metric_to_calibrate_on, np.nan)
#         }
#         pd.DataFrame(log_rows + [final_row]).to_csv(log_path, index=False)
#         print(f" Final {self.metric_to_calibrate_on.upper()} = {val_metrics_final.get(self.metric_to_calibrate_on, np.nan):.4f}")

#         return self.global_best_position, self.global_best_value, self.best_validation_metric, datetime.now() - start_time


# # === Per-gage wrapper ===
# def calibrate_gage(gage_id):
#     model_roots = cfg.model_roots
#     n_tiles = len(model_roots)
#     all_init_params, all_bounds, include_nom_flags, nom_file_paths = [], [], [], []

#     for tile_idx, root in enumerate(model_roots):
#         # --- Locate the CFE config file ---
#         config_dir = os.path.join(root, f"out/{gage_id}/configs/cfe")
#         config_file = sorted(f for f in os.listdir(config_dir) if f.startswith("cfe_config_cat"))[0]
#         config_path = os.path.join(config_dir, config_file)

#         # === NEW: Unified param extraction ===
#         init = extract_initial_params(config_path)  # Combines CFE + NOM if present
#         bounds = param_bounds.copy()

#         # === Detect NOM presence and append bounds ===
#         include_nom = os.path.isdir(os.path.join(root, f"out/{gage_id}/configs/noahowp"))
#         if include_nom:
#             nom_path = os.path.join(root, f"out/{gage_id}/configs/noahowp/parameters/MPTABLE.TBL")
#             bounds += nom_param_bounds
#             nom_file_paths.append(nom_path)
#         else:
#             nom_file_paths.append("")

#         # === Accumulate params and bounds for this tile ===
#         all_init_params.extend(init)
#         all_bounds.extend(bounds)
#         include_nom_flags.append(include_nom)

#     # === Expand parameter names per tile (CFE + NOM) ===
#     names = []
#     for tile_idx in range(n_tiles):
#         tile_suffix = f"_tile{tile_idx+1}"
#         tile_names = [f"{name}{tile_suffix}" for name in param_names]
#         if include_nom_flags[tile_idx]:
#             tile_names += [f"{name}{tile_suffix}" for name in nom_param_names]
#         names.extend(tile_names)

#     # === Add tile weight for 2-tile setup ===
#     if n_tiles == 2:
#         all_init_params.append(0.7)            # Initial tile weight
#         all_bounds.append((0.0, 1.0))          # Weight bounds
#         names.append("tile_weight")            # For logging

#     weights = [1.0 / n_tiles] * n_tiles
#     print(f"[DEBUG] include_nom_flags: {include_nom_flags}")
#     print(f"[DEBUG] nom_file_paths: {nom_file_paths}")

#     # === Create and run PSO ===
#     pso = PSO(
#         n_particles=n_particles,
#         bounds=all_bounds,
#         n_iterations=n_iterations,
#         gage_id=gage_id,
#         init_position=all_init_params,
#         config_path="",  # no longer needed
#         observed_path=os.path.join(observed_q_root, "successful_sites_resampled", f"{gage_id}.csv"),
#         postproc_base_path=model_roots[0] + "/postproc",
#         metric_to_calibrate_on=metric_to_calibrate_on,
#         include_nom=any(include_nom_flags),
#         nom_file_paths=nom_file_paths,   # <-- FIXED
#         param_names=names
#     )


#     pso.optimize()


# if __name__ == "__main__":
#     start = datetime.now()
#     gage_list = pd.read_csv(cfg.gages_file, dtype={"gage_id": str})["gage_id"].tolist()
#     ctx = multiprocessing.get_context("spawn")
#     with ctx.Pool(processes=max_cores_for_gages) as pool:
#         pool.map(calibrate_gage, gage_list)
#     print(f"Total wall time: {datetime.now() - start}")























































# """
# Author: Peter La Follette [plafollette@lynker.com | July 2025]
# Multi-tile PSO calibration for CFE+PET+T-Route, with optional NOM parameter support.
# """

# import os
# import subprocess
# import pandas as pd
# import numpy as np
# import math
# import yaml
# import json
# import random
# import sys
# import traceback
# import multiprocessing
# from datetime import datetime
# from hydroeval import kge

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# from model_assessment.util.metrics import compute_metrics
# from model_assessment.util.update_NOM import update_mptable
# from model_assessment.configs import path_config as cfg

# np.random.seed(42)
# random.seed(42)

# # === CONFIGURATION ===
# n_particles = 2
# n_iterations = 2
# max_cores_for_gages = 2
# metric_to_calibrate_on = "kge"

# with open("model_assessment/configs/time_config.yaml", "r") as f:
#     time_cfg = yaml.safe_load(f)

# spinup_start = pd.Timestamp(time_cfg["spinup_start"])
# cal_start    = pd.Timestamp(time_cfg["cal_start"])
# cal_end      = pd.Timestamp(time_cfg["cal_end"])
# val_start    = pd.Timestamp(time_cfg["val_start"])
# val_end      = pd.Timestamp(time_cfg["val_end"])

# project_root = cfg.project_root
# sandbox_path = cfg.sandbox_path
# logging_dir = cfg.logging_dir
# observed_q_root = cfg.observed_q_root

# os.makedirs(logging_dir, exist_ok=True)

# # === Parameter definitions ===
# param_names = [
#     "b", "satdk", "satpsi", "maxsmc", "max_gw_storage", "Cgw", "expon", "Kn",
#     "Klf", "refkdt", "slope", "wltsmc", "alpha_fc", "Kinf_nash_surface"
# ]

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

# log_scale_params = {"Cgw": True, "satdk": True}

# nom_param_names = ["MFSNO", "RSURF_SNOW", "HVT", "CWPVT", "VCMX25", "MP"]
# nom_param_bounds = [
#     (0.625, 5.0), (0.1, 100.0), (0.0, 20.0), (0.18, 5.0), (0.0, 80.0), (3.6, 12.6)
# ]

# # === Helpers ===
# def clear_terminal():
#     os.system("clear")

# def check_for_stop_signal_or_low_disk(threshold_gb=50):
#     stop_file = os.path.join(project_root, "STOP_NOW.txt")
#     if os.path.exists(stop_file):
#         sys.exit(1)
#     stat = os.statvfs("/")
#     free_gb = (stat.f_frsize * stat.f_bavail) / (1024 ** 3)
#     if free_gb < threshold_gb:
#         sys.exit(1)

# def transform_params(params, names):
#     return [10**p if log_scale_params.get(name, False) else p for name, p in zip(names, params)]

# def extract_tile_params(full_params, tile_idx, n_tiles):
#     total_len = len(full_params)
#     if n_tiles == 2 and total_len % 2 == 1:
#         param_slice = full_params[:-1]
#         chunk = len(param_slice) // n_tiles
#         return param_slice[tile_idx * chunk : (tile_idx + 1) * chunk]
#     chunk = total_len // n_tiles
#     return full_params[tile_idx * chunk : (tile_idx + 1) * chunk]

# def get_observed_q(observed_path):
#     df = pd.read_csv(observed_path, parse_dates=['value_time']).set_index('value_time')
#     return df['flow_m3_per_s']

# def extract_initial_cfe_params(config_path):
#     """
#     Extracts initial CFE parameters from config file.
#     If a parameter is missing, falls back to default value 0.002.
#     """
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
#         "Kinf_nash_surface": "Kinf_nash_surface"
#     }

#     values = {}
#     with open(config_path, "r") as f:
#         lines = f.readlines()
#     for line in lines:
#         if "=" in line:
#             key, val = line.strip().split("=", 1)
#             key = key.strip()
#             val = val.split("[")[0].strip()
#             for param, config_key in param_map.items():
#                 if key == config_key:
#                     values[param] = float(val)

#     param_order = list(param_map.keys())

#     extracted = []
#     for name in param_order:
#         if name in values:
#             v = values[name]
#         else:
#             print(f"[WARN] {name} not found in {config_path}. Using default 0.002")
#             v = 0.002
#         if log_scale_params.get(name, False):
#             extracted.append(math.log10(v))
#         else:
#             extracted.append(v)

#     return extracted


# def extract_initial_nom_params(mptable_path):
#     """Extract NOM params for tile that uses Noah-MP"""
#     values = []
#     param_order = nom_param_names.copy()
#     with open(mptable_path) as f:
#         for line in f:
#             for pname in param_order:
#                 if line.strip().startswith(pname):
#                     val = float(line.strip().split()[1])
#                     values.append(val)
#     if len(values) != len(param_order):
#         raise ValueError(f"Expected {len(param_order)} NOM params but got {len(values)} from {mptable_path}")
#     return values


# def regenerate_cfe_config(config_path, params, names):
#     replacements = dict(zip(names, params))
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
#         "Kinf_nash_surface": "Kinf_nash_surface"
#     }

#     found_keys = set()
#     updated_lines = []

#     if os.path.isfile(config_path):
#         config_dir = os.path.dirname(config_path)
#     else:
#         config_dir = config_path

#     for fname in sorted(f for f in os.listdir(config_dir) if f.startswith("cfe_config_cat")):
#         path = os.path.join(config_dir, fname)
#         lines = []
#         with open(path) as f:
#             for line in f:
#                 key = line.split("=")[0].strip()
#                 replaced = False
#                 for pname, ckey in param_map.items():
#                     if key == ckey:
#                         unit = line[line.find("["):] if "[" in line else ""
#                         lines.append(f"{ckey}={replacements[pname]}{unit}\n")
#                         found_keys.add(pname)
#                         replaced = True
#                         break
#                 if not replaced:
#                     lines.append(line)
#         # Add missing keys
#         for pname, ckey in param_map.items():
#             if pname not in found_keys:
#                 lines.append(f"{ckey}={replacements[pname]}\n")
#         with open(path, "w") as f:
#             f.writelines(lines)

# # === Tiled objective function ===
# def objective_function_tiled(args):
#     """
#     Runs tiled CFE+PET+T-Route model for one PSO particle.
#     Args should include:
#       - params
#       - particle_idx
#       - gage_id
#       - model_roots
#       - observed_q_root
#       - include_nom_flags
#       - nom_file_paths
#       - weights (optional)
#     """
#     (
#         params, particle_idx, gage_id,
#         model_roots, observed_q_root,
#         include_nom_flags, nom_file_paths,
#         weights
#     ) = args

#     check_for_stop_signal_or_low_disk()

#     n_tiles = len(model_roots)

#     if n_tiles == 2:
#         tile_weight = params[-1]
#         weights = [tile_weight, 1.0 - tile_weight]
#         params = params[:-1]
#     elif weights is None:
#         weights = [1.0 / n_tiles] * n_tiles

#     sim_dfs = []

#     for tile_idx, tile_root in enumerate(model_roots):
#         print(f"[INFO] Running gage {gage_id} | Particle {particle_idx} | Tile {tile_idx}")

#         # === Extract parameters for this tile
#         tile_params = extract_tile_params(params, tile_idx, n_tiles)
#         names = param_names.copy()
#         if include_nom_flags[tile_idx]:
#             names += nom_param_names

#         true_params = transform_params(tile_params, names)

#         # === Update CFE config for this tile
#         config_dir = os.path.join(tile_root, f"out/{gage_id}/configs/cfe")
#         regenerate_cfe_config(config_dir, true_params, names)

#         if include_nom_flags[tile_idx]:
#             nom_vals = tile_params[-6:]
#             nom_param_dict = dict(zip(nom_param_names, nom_vals))
#             update_mptable(
#                 original_file=nom_file_paths[tile_idx],
#                 output_file=nom_file_paths[tile_idx],
#                 updated_params=nom_param_dict,
#                 verbose=False
#             )

#         # === Update realization JSON ===
#         json_dir = os.path.join(tile_root, "out", gage_id, "json")
#         realization_file = [f for f in os.listdir(json_dir) if f.endswith(".json")][0]
#         realization_path = os.path.join(json_dir, realization_file)

#         with open(realization_path) as f:
#             realization = json.load(f)
#         realization["time"]["start_time"] = time_cfg["spinup_start"]
#         realization["time"]["end_time"] = time_cfg["cal_end"]
#         with open(realization_path, "w") as f:
#             json.dump(realization, f, indent=4)

#         # === Update troute_config.yaml ===
#         troute_path = os.path.join(tile_root, "out", gage_id, "configs", "troute_config.yaml")
#         with open(troute_path) as f:
#             troute_cfg = yaml.safe_load(f)
#         nts = int((cal_end - spinup_start) / pd.Timedelta(seconds=300))
#         troute_cfg["compute_parameters"]["restart_parameters"]["start_datetime"] = spinup_start.strftime("%Y-%m-%d_%H:%M:%S")
#         troute_cfg["compute_parameters"]["forcing_parameters"]["nts"] = nts
#         with open(troute_path, "w") as f:
#             yaml.dump(troute_cfg, f)

#         # === Delete stale .nc files ===
#         troute_dir = os.path.join(tile_root, "out", gage_id, "troute")
#         if os.path.isdir(troute_dir):
#             for fname in os.listdir(troute_dir):
#                 if fname.endswith(".nc"):
#                     file_path = os.path.join(troute_dir, fname)
#                     try:
#                         os.remove(file_path)
#                         print(f"[DEBUG] Deleted old routing file: {file_path}")
#                     except Exception as e:
#                         print(f"[WARN] Could not delete {file_path}: {e}")

#         # === Use TILE-SPECIFIC sandbox config ===
#         tile_sandbox_config = os.path.join(cfg.project_root, "configs", f"sandbox_config_tile{tile_idx+1}.yaml")

#         subprocess.call(
#             ["python", sandbox_path, "-i", tile_sandbox_config, "-run", "--gage_id", gage_id],
#             cwd=tile_root
#         )

#         # === Delete old postproc file for this tile+particle ===
#         output_path = os.path.join(tile_root, "postproc", f"{gage_id}_particle_{particle_idx}.csv")
#         if os.path.exists(output_path):
#             try:
#                 os.remove(output_path)
#                 print(f"[DEBUG] Deleted old postproc file: {output_path}")
#             except Exception as e:
#                 print(f"[WARN] Could not delete {output_path}: {e}")

#         # === Post-process hydrograph for this tile ===
#         output_path = os.path.join(tile_root, "postproc", f"{gage_id}_particle_{particle_idx}.csv")
#         get_hydrograph_path = os.path.join(project_root, "model_assessment", "util", "get_hydrograph.py")
#         subprocess.call(
#             ["python", get_hydrograph_path, "--gage_id", gage_id, "--output", output_path, "--base_dir", tile_root],
#             cwd=os.path.join(tile_root, "postproc")
#         )

#         sim_df = pd.read_csv(output_path, parse_dates=['current_time']).set_index('current_time')['flow'].resample('1h').mean()
#         sim_dfs.append(sim_df)

#     # === Average the hydrographs across tiles ===
#     avg_sim = sum(w * s for w, s in zip(weights, sim_dfs))

#     # === Save averaged hydrograph ===
#     avg_df = avg_sim.reset_index().rename(columns={'index': 'current_time', 0: 'flow'})
#     output_path = os.path.join(model_roots[0], "postproc", f"{gage_id}_particle_{particle_idx}.csv")
#     avg_df.to_csv(output_path, index=False)

#     obs_df = get_observed_q(os.path.join(observed_q_root, "successful_sites_resampled", f"{gage_id}.csv"))

#     sim_cal, obs_cal = avg_sim[cal_start:cal_end].dropna(), obs_df[cal_start:cal_end].dropna()
#     sim_val, obs_val = avg_sim[val_start:val_end].dropna(), obs_df[val_start:val_end].dropna()
#     sim_cal, obs_cal = sim_cal.align(obs_cal, join='inner')
#     sim_val, obs_val = sim_val.align(obs_val, join='inner')

#     sim_cal.iloc[-1] += 1e-8
#     obs_cal.iloc[-1] += 1e-8

#     cal_metrics = compute_metrics(sim_cal, obs_cal, event_threshold=1e-2)
#     val_metrics = compute_metrics(sim_val, obs_val, event_threshold=1e-2)

#     return -cal_metrics[metric_to_calibrate_on], val_metrics, cal_metrics


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
#     def __init__(self, n_particles, bounds, n_iterations, gage_id,
#                  init_position, config_path, observed_path, postproc_base_path,
#                  metric_to_calibrate_on="kge",
#                  include_nom=False, nom_file_path="",
#                  param_names=None):
#         self.particles = [
#             Particle(bounds, init_position=init_position if i == 0 else None)
#             for i in range(n_particles)
#         ]
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
#         self.include_nom = include_nom
#         self.nom_file_path = nom_file_path
#         self.param_names = param_names if param_names else param_names

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

#             results = []
#             for i, p in enumerate(self.particles):
#                 print(f"[INFO] Running gage {self.gage_id} | iteration {iteration + 1} | particle {i}")
#                 result = objective_function_tiled(
#                     (
#                         p.position, i, self.gage_id, cfg.model_roots,
#                         observed_q_root,
#                         [self.include_nom] * len(cfg.model_roots),
#                         [self.nom_file_path] * len(cfg.model_roots),
#                         [1.0 / len(cfg.model_roots)] * len(cfg.model_roots)
#                     )
#                 )
#                 results.append(result)

#             for idx, (objective_value, val_metrics, cal_metrics) in enumerate(results):
#                 particle = self.particles[idx]
#                 particle.current_value = objective_value

#                 metric_calibration = cal_metrics[self.metric_to_calibrate_on]
#                 metric_validation = val_metrics[self.metric_to_calibrate_on]

#                 if objective_value < (particle.best_value - 0.001):
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

#                 if particle.stagnation_counter >= stagnation_threshold:
#                     print(f"Resetting particle {idx} after {stagnation_threshold} stagnant iterations.")
#                     particle.reset(self.bounds)

#                 param_dict = {name: val for name, val in zip(self.param_names, particle.position)}
#                 row = {
#                     "iteration": iteration + 1,
#                     "particle": idx,
#                     **param_dict,
#                     f"{self.metric_to_calibrate_on}_calibration": metric_calibration,
#                     f"{self.metric_to_calibrate_on}_validation": metric_validation
#                 }
#                 log_rows.append(row)

#             pd.DataFrame(log_rows).to_csv(log_path, index=False)

#             for p in self.particles:
#                 p.update_velocity(self.global_best_position, w=w, c1=1.5, c2=1.5)
#                 p.update_position(self.bounds)

#             print(f"Global best objective so far: {-self.global_best_value:.4f}")

#         # === Final full-period multi-tile run ===
#         print(f"\nRunning final full-period validation for {self.gage_id}...")

#         sim_dfs = []
#         n_tiles = len(cfg.model_roots)
#         weights = [1.0 / n_tiles] * n_tiles

#         for tile_idx, tile_root in enumerate(cfg.model_roots):
#             tile_params = extract_tile_params(self.global_best_position, tile_idx, n_tiles)
#             names = param_names.copy()
#             if self.include_nom:
#                 names += nom_param_names

#             true_best_params = transform_params(tile_params, names)

#             # === Update CFE config ===
#             config_dir = os.path.join(tile_root, f"out/{self.gage_id}/configs/cfe")
#             regenerate_cfe_config(config_dir, true_best_params, names)

#             # === Update NOM if present ===
#             if self.include_nom:
#                 nom_vals = tile_params[-6:]
#                 nom_param_dict = dict(zip(nom_param_names, nom_vals))
#                 update_mptable(
#                     original_file=os.path.join(tile_root, f"out/{self.gage_id}/configs/noahowp/parameters/MPTABLE.TBL"),
#                     output_file=os.path.join(tile_root, f"out/{self.gage_id}/configs/noahowp/parameters/MPTABLE.TBL"),
#                     updated_params=nom_param_dict,
#                     verbose=False
#                 )

#             # === Update realization JSON ===
#             json_dir = os.path.join(tile_root, "out", self.gage_id, "json")
#             realization_file = [f for f in os.listdir(json_dir) if f.endswith(".json")][0]
#             realization_path = os.path.join(json_dir, realization_file)

#             with open(realization_path, "r") as f:
#                 realization = json.load(f)
#             realization["time"]["start_time"] = time_cfg["spinup_start"]
#             realization["time"]["end_time"] = time_cfg["val_end"]
#             with open(realization_path, "w") as f:
#                 json.dump(realization, f, indent=4)

#             # === Update t-route config ===
#             troute_path = os.path.join(tile_root, "out", self.gage_id, "configs", "troute_config.yaml")
#             with open(troute_path, "r") as f:
#                 troute_cfg = yaml.safe_load(f)
#             nts_full = int((val_end - spinup_start) / pd.Timedelta(seconds=300))
#             troute_cfg["compute_parameters"]["restart_parameters"]["start_datetime"] = spinup_start.strftime("%Y-%m-%d_%H:%M:%S")
#             troute_cfg["compute_parameters"]["forcing_parameters"]["nts"] = nts_full
#             with open(troute_path, "w") as f:
#                 yaml.dump(troute_cfg, f)

#             # === DELETE old .nc routing files ===
#             troute_dir = os.path.join(tile_root, "out", self.gage_id, "troute")
#             if os.path.isdir(troute_dir):
#                 for fname in os.listdir(troute_dir):
#                     if fname.endswith(".nc"):
#                         file_path = os.path.join(troute_dir, fname)
#                         try:
#                             os.remove(file_path)
#                             print(f"[DEBUG] Deleted old routing file: {file_path}")
#                         except Exception as e:
#                             print(f"[WARN] Could not delete {file_path}: {e}")

#             # === TILE-SPECIFIC sandbox config ===
#             tile_sandbox_config = os.path.join(cfg.project_root, "configs", f"sandbox_config_tile{tile_idx+1}.yaml")

#             print(f"[INFO] Running gage {self.gage_id} | FINAL | Tile {tile_idx}")

#             subprocess.call(
#                 ["python", sandbox_path, "-i", tile_sandbox_config, "-run", "--gage_id", self.gage_id],
#                 cwd=tile_root
#             )

#             # === Post-process ===
#             get_hydrograph_path = os.path.join(project_root, "model_assessment", "util", "get_hydrograph.py")
#             output_path = os.path.join(tile_root, "postproc", f"{self.gage_id}_best_tile_{tile_idx}.csv")
#             subprocess.call(
#                 ["python", get_hydrograph_path, "--gage_id", self.gage_id, "--output", output_path, "--base_dir", tile_root],
#                 cwd=os.path.join(tile_root, "postproc")
#             )

#             sim_df = pd.read_csv(output_path, parse_dates=['current_time']).set_index('current_time')['flow'].resample('1h').mean()
#             sim_dfs.append(sim_df)

#         # === Combine final tiles ===
#         avg_sim = sum(w * s for w, s in zip(weights, sim_dfs))
#         final_df = avg_sim.reset_index().rename(columns={'index': 'current_time', 0: 'flow'})
#         final_output_path = os.path.join(cfg.model_roots[0], "postproc", f"{self.gage_id}_best.csv")
#         final_df.to_csv(final_output_path, index=False)
#         print(f" Final best hydrograph saved to {final_output_path}")

#         obs_df = get_observed_q(self.observed_path)
#         sim_val, obs_val = avg_sim[val_start:val_end].dropna(), obs_df[val_start:val_end].dropna()
#         sim_val, obs_val = sim_val.align(obs_val, join='inner')
#         if len(sim_val) > 0 and len(obs_val) > 0:
#             sim_val.iloc[-1] += 1e-8
#             obs_val.iloc[-1] += 1e-8
#         val_metrics_final = compute_metrics(sim_val, obs_val, event_threshold=1e-2)

#         final_row = {
#             "iteration": "FINAL",
#             "particle": "BEST",
#             **{name: val for name, val in zip(self.param_names, self.global_best_position)},
#             f"{self.metric_to_calibrate_on}_calibration": self.best_cal_metrics.get(self.metric_to_calibrate_on, np.nan),
#             f"{self.metric_to_calibrate_on}_validation": val_metrics_final.get(self.metric_to_calibrate_on, np.nan)
#         }
#         pd.DataFrame(log_rows + [final_row]).to_csv(log_path, index=False)
#         print(f" Final {self.metric_to_calibrate_on.upper()} = {val_metrics_final.get(self.metric_to_calibrate_on, np.nan):.4f}")

#         return self.global_best_position, self.global_best_value, self.best_validation_metric, datetime.now() - start_time


# # === Per-gage wrapper ===
# def calibrate_gage(gage_id):
#     model_roots = cfg.model_roots
#     n_tiles = len(model_roots)
#     all_init_params, all_bounds, include_nom_flags, nom_file_paths = [], [], [], []

#     for tile_idx, root in enumerate(model_roots):
#         config_dir = os.path.join(root, f"out/{gage_id}/configs/cfe")
#         config_file = sorted(f for f in os.listdir(config_dir) if f.startswith("cfe_config_cat"))[0]
#         config_path = os.path.join(config_dir, config_file)
#         init = extract_initial_cfe_params(config_path)
#         bounds = param_bounds.copy()
#         include_nom = os.path.isdir(os.path.join(root, f"out/{gage_id}/configs/noahowp"))
        
#         if include_nom:
#             nom_path = os.path.join(root, f"out/{gage_id}/configs/noahowp/parameters/MPTABLE.TBL")
#             init += extract_initial_nom_params(nom_path)
#             bounds += nom_param_bounds
#             nom_file_paths.append(nom_path)
#         else:
#             nom_file_paths.append("")
        
#         all_init_params.extend(init)
#         all_bounds.extend(bounds)
#         include_nom_flags.append(include_nom)

#     # === EXPAND PARAMETER NAMES ===
#     names = []
#     for tile_idx in range(n_tiles):
#         tile_suffix = f"_tile{tile_idx+1}"
#         tile_names = [f"{name}{tile_suffix}" for name in param_names]
#         if include_nom_flags[tile_idx]:
#             tile_names += [f"{name}{tile_suffix}" for name in nom_param_names]
#         names.extend(tile_names)

#     # Correct: only now add tile_weight
#     if n_tiles == 2:
#         all_init_params.append(0.7)          # Example initial weight
#         all_bounds.append((0.0, 1.0))        # Bounds for weight
#         names.append("tile_weight")          # So it appears in logging

#     weights = [1.0 / n_tiles] * n_tiles

#     # === Create and run PSO ===
#     pso = PSO(
#         n_particles=n_particles,
#         bounds=all_bounds,
#         n_iterations=n_iterations,
#         gage_id=gage_id,
#         init_position=all_init_params,
#         config_path="",  # not needed anymore
#         observed_path=os.path.join(observed_q_root, "successful_sites_resampled", f"{gage_id}.csv"),
#         postproc_base_path=model_roots[0] + "/postproc",
#         metric_to_calibrate_on=metric_to_calibrate_on,
#         include_nom=any(include_nom_flags),
#         nom_file_path="",  # not needed anymore
#         param_names=names  # Pass expanded param names!
#     )

#     pso.optimize()


# if __name__ == "__main__":
#     start = datetime.now()
#     gage_list = pd.read_csv(cfg.gages_file, dtype={"gage_id": str})["gage_id"].tolist()
#     ctx = multiprocessing.get_context("spawn")
#     with ctx.Pool(processes=max_cores_for_gages) as pool:
#         pool.map(calibrate_gage, gage_list)
#     print(f"Total wall time: {datetime.now() - start}")

























































#####no tiled support
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
# ### if you have the ram to spare, writing the out directory on a ramdisk is a good idea anyway
# def check_for_stop_signal_or_low_disk():
#     stop_file = os.path.join(project_root, "STOP_NOW.txt")
#     if os.path.exists(stop_file):
#         print("Detected STOP_NOW.txt. ")
#         sys.exit(1)
#     statvfs = os.statvfs(project_root)
#     free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024 ** 3)
#     if (free_gb < 50.0):
#         print(f"Free disk space below threshold: {free_gb:.2f} GB, stopping.")
#         sys.exit(1)
#     # return free_gb < 100
#     return False


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

# # === NOM PARAMETERS (only used if NOM present) ===
# nom_param_names = ["MFSNO", "RSURF_SNOW", "HVT", "CWPVT", "VCMX25", "MP"]
# nom_param_bounds = [
#     (0.625, 5.0),
#     (0.1, 100.0),
#     (0.0, 20.0),
#     (0.18, 5.0),
#     (0.0, 80.0),
#     (3.6, 12.6)
# ]


# log_scale_params = {"Cgw": True, "satdk": True}

# param_bounds_dict = dict(zip(param_names, param_bounds))

# # === HELPERS ===
# def transform_params(params):
#     return [10**p if log_scale_params.get(name, False) else p for name, p in zip(param_names, params)]


# def detect_and_read_nom_params(gage_id):
#     nom_dir = os.path.join(cfg.model_roots[0], f"out/{gage_id}/configs/noahowp/parameters")
#     mptable_path = os.path.join(nom_dir, "MPTABLE.TBL")
#     if not os.path.exists(mptable_path):
#         return False, [], ""

#     try:
#         with open(mptable_path) as f:
#             lines = f.readlines()

#         nom_params = {}
#         for line in lines:
#             if "=" not in line or line.strip().startswith("!"):
#                 continue
#             key, val = line.split("=", 1)
#             name = key.strip()
#             if name in nom_param_names:
#                 val = val.split("!")[0].split(",")[0].strip()
#                 nom_params[name] = float(val)

#         if set(nom_params.keys()) != set(nom_param_names):
#             raise ValueError(f"NOM param mismatch in {mptable_path}")

#         return True, [nom_params[n] for n in nom_param_names], mptable_path

#     except Exception as e:
#         print(f"Error parsing NOM params from {mptable_path}: {e}")
#         return False, [], mptable_path

# def extract_initial_nom_params(nom_file_path):
#     """
#     Extract initial NOM parameters from a MPTABLE.TBL file.
#     Only the first value from each line is used (even if multiple are listed).
#     """
#     nom_param_names = ["MFSNO", "RSURF_SNOW", "HVT", "CWPVT", "VCMX25", "MP"]
#     nom_params_dict = {}

#     try:
#         with open(nom_file_path) as f:
#             lines = f.readlines()

#         for line in lines:
#             if '=' not in line or line.strip().startswith('!'):
#                 continue
#             key, value = line.split('=', 1)
#             param_name = key.strip()
#             if param_name in nom_param_names:
#                 value_str = value.split('!')[0]
#                 values = [v.strip() for v in value_str.split(',') if v.strip()]
#                 nom_params_dict[param_name] = float(values[0])

#         if set(nom_params_dict.keys()) != set(nom_param_names):
#             raise ValueError(f"Found NOM parameters {list(nom_params_dict.keys())}, expected {nom_param_names}. Check formatting in {nom_file_path}")

#         return [nom_params_dict[p] for p in nom_param_names]

#     except Exception as e:
#         raise RuntimeError(f"Failed to parse NOM parameters from {nom_file_path}: {e}")


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
#     params, particle_idx, gage_id, config_path, observed_path, postproc_base_path, include_nom, nom_file_path = args

#     true_params = transform_params(params)
#     if check_for_stop_signal_or_low_disk():
#         print(f" Stop signal detected, skipping particle {particle_idx}")
#         return 10.0, -10.0, {}, {}

#     print(f"\nEvaluating particle {particle_idx} for gage {gage_id}")

#     # === Delete old .nc routing files in troute dir ===
#     troute_dir = os.path.join(cfg.model_roots[0], "out", gage_id, "troute")
#     if os.path.isdir(troute_dir):
#         for fname in os.listdir(troute_dir):
#             if fname.endswith(".nc"):
#                 file_path = os.path.join(troute_dir, fname)
#                 try:
#                     os.remove(file_path)
#                     print(f"[DEBUG] Deleted old routing file: {file_path}")
#                 except Exception as e:
#                     print(f"[WARN] Could not delete {file_path}: {e}")

#     regenerate_cfe_config(config_path, true_params)

#     if include_nom:
#         nom_params = dict(zip(nom_param_names, params[-6:]))
#         update_mptable(
#             original_file=nom_file_path,
#             output_file=nom_file_path,
#             updated_params=nom_params,
#             verbose=False
#         )

#     incomplete_run = False
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
#         # Check output length (i.e. check for crashes if simulation length is unexpected)
#         window_start = pd.Timestamp(time_cfg["spinup_start"])
#         window_end = cal_end  # uses calibration period by default

#         expected_length = int((window_end - window_start) / pd.Timedelta(hours=1)) + 1
#         actual_length = len(sim_df)

#         allowed_tolerance = 1
#         if abs(actual_length - expected_length) > allowed_tolerance:
#             incomplete_run=True

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

#         if (incomplete_run):
#             # === LOG the failed particle ===
#             failed_row = {
#                 "gage_id": gage_id,
#                 "particle_idx": particle_idx,
#                 "params": params.tolist() if hasattr(params, "tolist") else list(params),
#                 "error_message": f"Incomplete simulation: expected {expected_length}, got {actual_length}"
#             }
#             log_path = os.path.join(logging_dir, "incomplete_simulations.csv")
#             df = pd.DataFrame([failed_row])
#             if os.path.exists(log_path):
#                 df.to_csv(log_path, mode='a', header=False, index=False)
#             else:
#                 df.to_csv(log_path, index=False)

#         return -cal_metrics[metric_to_calibrate_on], val_metrics, cal_metrics

#     except Exception as e:
#         print(f"Error during evaluation for {gage_id}: {e}")
#         traceback.print_exc()

#         # === LOG the failed particle ===
#         failed_row = {
#             "gage_id": gage_id,
#             "particle_idx": particle_idx,
#             "params": params.tolist() if hasattr(params, "tolist") else list(params),
#             "error_message": str(e)
#         }
#         log_path = os.path.join(logging_dir, "incomplete_simulations.csv")
#         df = pd.DataFrame([failed_row])
#         if os.path.exists(log_path):
#             df.to_csv(log_path, mode='a', header=False, index=False)
#         else:
#             df.to_csv(log_path, index=False)

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
#     def __init__(self, n_particles, bounds, n_iterations, gage_id, init_position, config_path, observed_path, postproc_base_path, metric_to_calibrate_on="kge", include_nom=False, nom_file_path="", param_names=None):
#         self.particles = [Particle(bounds, init_position=init_position if i == 0 else None) for i in range(n_particles)]
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
#         self.include_nom = include_nom
#         self.nom_file_path = nom_file_path
#         self.param_names = param_names if param_names else param_names

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
#                     (p.position, i, self.gage_id, self.config_path, self.observed_path, self.postproc_base_path, self.include_nom, self.nom_file_path)
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
#                 param_dict = {name: val for name, val in zip(self.param_names, particle.position)}
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

#         if self.include_nom:
#             nom_vals = self.global_best_position[-6:]
#             nom_param_dict = dict(zip(nom_param_names, nom_vals))
#             update_mptable(
#                 original_file=self.nom_file_path,
#                 output_file=self.nom_file_path,
#                 updated_params=nom_param_dict,
#                 verbose=False
#             )


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
#             window_start = pd.Timestamp(time_cfg["spinup_start"])
#             window_end = pd.Timestamp(time_cfg["val_end"])

#             expected_length = int((window_end - window_start) / pd.Timedelta(hours=1)) + 1
#             actual_length = len(sim_df)

#             allowed_tolerance = 1
#             if abs(actual_length - expected_length) > allowed_tolerance:
#                 raise RuntimeError(
#                     f"Final validation for gage {self.gage_id} produced {actual_length} time steps but expected {expected_length}. "
#                     f"Allowed tolerance is ±{allowed_tolerance}. Simulation likely incomplete."
#                 )

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
#                 **{name: val for name, val in zip(self.param_names, self.global_best_position)},
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
#     import os
#     import numpy as np

#     observed_q_root = cfg.observed_q_root
#     model_roots = cfg.model_roots
#     root = model_roots[0]  # single-tile for now

#     config_dir = os.path.join(root, f"out/{gage_id}/configs/cfe")
#     if not os.path.exists(config_dir):
#         print(f"Missing config dir for gage {gage_id}: {config_dir}")
#         return

#     config_path = os.path.join(config_dir, sorted(f for f in os.listdir(config_dir) if f.startswith("cfe_config_cat"))[0])
#     observed_path = os.path.join(observed_q_root, "successful_sites_resampled", f"{gage_id}.csv")
#     postproc_base_path = os.path.join(root, "postproc")

#     # === Extract initial CFE parameters
#     init_vals = extract_initial_cfe_params(config_path)
#     bounds = param_bounds.copy()
#     names = param_names.copy()

#     # === Check for NOM config
#     nom_config_dir = os.path.join(root, f"out/{gage_id}/configs/noahowp")
#     include_nom = os.path.isdir(nom_config_dir)

#     if include_nom:
#         nom_file_path = os.path.join(nom_config_dir, "parameters", "MPTABLE.TBL")
#         nom_init_vals = extract_initial_nom_params(nom_file_path)
#         init_vals += nom_init_vals
#         bounds += nom_param_bounds
#         names += nom_param_names

#     if len(init_vals) != len(bounds):
#         raise ValueError(f"init_vals length {len(init_vals)} does not match bounds length {len(bounds)}")

#     # === Launch PSO
#     pso = PSO(
#         n_particles=n_particles,
#         bounds=bounds,
#         n_iterations=n_iterations,
#         gage_id=gage_id,
#         init_position=init_vals,
#         config_path=config_path,
#         observed_path=observed_path,
#         postproc_base_path=postproc_base_path,
#         metric_to_calibrate_on="kge",
#         include_nom=include_nom,
#         nom_file_path=nom_file_path if include_nom else None,
#         param_names=names
#     )


#     best_params, best_obj_value, best_val_metric, runtime = pso.optimize()


# # === MAIN EXECUTION ===
# if __name__ == "__main__":
#     start_time = datetime.now()

#     gages_file = os.path.join(project_root, "basin_IDs/basin_IDs.csv")
#     gage_list = pd.read_csv(gages_file, dtype={"gage_id": str})["gage_id"].tolist()

#     ctx = multiprocessing.get_context("spawn")
#     with ctx.Pool(processes=max_cores_for_gages) as pool:
#         pool.map(calibrate_gage, gage_list)

#     end_time = datetime.now()
#     total_duration = end_time - start_time
#     print(f"\n Total wall time: {total_duration}")









































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
