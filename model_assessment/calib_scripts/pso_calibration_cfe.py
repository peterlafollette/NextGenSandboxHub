"""
Author: Peter La Follette [plafollette@lynker.com | July 2025]
Multi-tile PSO calibration for CFE+PET+T-Route, with optional NOM parameter support.
Now supports concurrent per-iteration particle evaluation with per-particle workspaces.
"""

import os
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
from model_assessment.util.update_NOM import update_mptable
from model_assessment.configs import path_config as cfg
from multiprocessing.pool import ThreadPool


# ==== Repro ====
np.random.seed(42)
random.seed(42)

# === CONFIGURATION ===
n_particles = 2                   # <-- particles per gage
n_iterations = 2
max_cores_for_gages = 5           # <-- process multiple gages in parallel (outer pool)
metric_to_calibrate_on = "kge"

# run ALL particles concurrently per iteration (inner pool size defaults to n_particles)
max_particle_procs = None         # set to an int to cap, or None to use n_particles

with open("model_assessment/configs/time_config.yaml", "r") as f:
    time_cfg = yaml.safe_load(f)

spinup_start = pd.Timestamp(time_cfg["spinup_start"])
cal_start    = pd.Timestamp(time_cfg["cal_start"])
cal_end      = pd.Timestamp(time_cfg["cal_end"])
val_start    = pd.Timestamp(time_cfg["val_start"])
val_end      = pd.Timestamp(time_cfg["val_end"])

project_root = cfg.project_root
sandbox_path = cfg.sandbox_path
logging_dir = cfg.logging_dir
observed_q_root = cfg.observed_q_root

os.makedirs(logging_dir, exist_ok=True)

# === Parameter definitions ===
param_names = [
    "b", "satdk", "satpsi", "maxsmc", "max_gw_storage", "Cgw", "expon", "Kn",
    "Klf", "refkdt", "slope", "wltsmc", "alpha_fc", "Kinf_nash_surface"
]

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

log_scale_params = {"Cgw": True, "satdk": True}

nom_param_names = ["MFSNO", "RSURF_SNOW", "HVT", "CWPVT", "VCMX25", "MP"]
nom_param_bounds = [
    (0.625, 5.0), (0.1, 100.0), (0.0, 20.0), (0.18, 5.0), (0.0, 80.0), (3.6, 12.6)
]

# === Helpers ===
def clear_terminal():
    os.system("clear")

def check_for_stop_signal_or_low_disk(threshold_gb=50):
    stop_file = os.path.join(project_root, "STOP_NOW.txt")
    if os.path.exists(stop_file):
        sys.exit(1)
    stat = os.statvfs("/")
    free_gb = (stat.f_frsize * stat.f_bavail) / (1024 ** 3)
    if free_gb < threshold_gb:
        sys.exit(1)

def update_nom_namelist_paramdir(namelist_path: str, new_param_dir: str):
    """Update `parameter_dir = "..."` in a NOM Fortran namelist."""
    if not os.path.isfile(namelist_path):
        return
    lines = []
    with open(namelist_path, "r") as f:
        for line in f:
            if "parameter_dir" in line and "=" in line and not line.strip().startswith("!"):
                # keep quotes as in file
                quote = '"' if '"' in line else "'"
                prefix = line.split("=", 1)[0]
                # preserve trailing comment if present
                comment = ""
                if "!" in line:
                    comment = "  !" + line.split("!", 1)[1].strip()
                line = f'{prefix}= {quote}{new_param_dir}{quote}{comment}\n'
            lines.append(line)
    with open(namelist_path, "w") as f:
        f.writelines(lines)


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
    with open(realization_path, "r") as f:
        rz = json.load(f)

    cfg_cfe_dir = os.path.join(work_root, "configs", "cfe")
    div_dir     = os.path.join(work_root, "outputs", "div")
    os.makedirs(cfg_cfe_dir, exist_ok=True)
    os.makedirs(div_dir, exist_ok=True)

    shared_pet_dir = os.path.join(base_out_dir, "configs", "pet")

    # particle NOM dirs
    cfg_nom_dir      = os.path.join(work_root, "configs", "noahowp")
    cfg_nom_paramdir = os.path.join(cfg_nom_dir, "parameters")
    os.makedirs(cfg_nom_paramdir, exist_ok=True)

    # source NOM paramdir (gage-level)
    shared_nom_paramdir = os.path.join(base_out_dir, "configs", "noahowp", "parameters")

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
            fname = Path(init_cfg).name

            if "PET" in model_type or "/configs/pet/" in init_cfg:
                if use_shared_pet:
                    new_path = os.path.join(shared_pet_dir, fname)
                else:
                    cfg_pet_dir = os.path.join(work_root, "configs", "pet")
                    os.makedirs(cfg_pet_dir, exist_ok=True)
                    new_path = os.path.join(cfg_pet_dir, fname)
                p["init_config"] = os.path.abspath(new_path)

            elif "CFE" in model_type or "/configs/cfe/" in init_cfg:
                new_path = os.path.join(cfg_cfe_dir, fname)
                p["init_config"] = os.path.abspath(new_path)

            elif "NOM" in model_type or "/noahowp/" in init_cfg or "noah" in model_type:
                # Put the NOM namelist into the particle workspace
                cfg_nom_namelist = os.path.join(cfg_nom_dir, fname)
                try:
                    # copy the namelist from its current location (init_cfg may be abs/rel)
                    src = init_cfg if os.path.isabs(init_cfg) else os.path.join(base_out_dir, init_cfg)
                    if os.path.isfile(src):
                        shutil.copy2(src, cfg_nom_namelist)
                except Exception:
                    pass

                # Ensure GENPARM/ SOILPARM exist in particle param dir (symlink preferred)
                for tbl in ("GENPARM.TBL", "SOILPARM.TBL"):
                    src_tbl = os.path.join(shared_nom_paramdir, tbl)
                    dst_tbl = os.path.join(cfg_nom_paramdir, tbl)
                    if os.path.isfile(src_tbl) and not os.path.exists(dst_tbl):
                        try:
                            os.symlink(src_tbl, dst_tbl)
                        except OSError:
                            try:
                                shutil.copy2(src_tbl, dst_tbl)
                            except Exception:
                                pass

                # We assume MPTABLE.TBL already exists per particle (scaffolded). If not, seed from shared.
                mpt_dst = os.path.join(cfg_nom_paramdir, "MPTABLE.TBL")
                if not os.path.isfile(mpt_dst):
                    mpt_src = os.path.join(shared_nom_paramdir, "MPTABLE.TBL")
                    if os.path.isfile(mpt_src):
                        shutil.copy2(mpt_src, mpt_dst)

                # Update `parameter_dir` inside the particle NOM namelist
                update_nom_namelist_paramdir(cfg_nom_namelist, cfg_nom_paramdir)

                # Point realization to the particle-local NOM namelist
                p["init_config"] = os.path.abspath(cfg_nom_namelist)

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
    param_order = ["MFSNO", "RSURF_SNOW", "HVT", "CWPVT", "VCMX25", "MP"]
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
            if stripped.startswith(pname):
                if "=" in stripped:
                    val_part = stripped.split("=", 1)[1]
                else:
                    val_part = stripped[len(pname):].strip()
                val_clean = val_part.split("!")[0].split(",")[0].strip()
                try:
                    values.append(float(val_clean))
                    found = True
                    break
                except ValueError:
                    raise ValueError(f"[ERROR] Could not parse value for {pname} in line: {line.strip()}")
        if not found:
            raise ValueError(f"[ERROR] Missing parameter {pname} in {mptable_path}")

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

    param_order = list(param_map.keys())

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
                    if key == ckey:
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

        if include_nom_flags[tile_idx]:
            nom_vals = tile_params[-6:]
            nom_tbl  = os.path.join(work_root, "configs", "noahowp", "parameters", "MPTABLE.TBL")
            update_mptable(
                original_file=nom_tbl,
                output_file=nom_tbl,
                updated_params=dict(zip(nom_param_names, nom_vals)),
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
        tile_sandbox_config = os.path.join(cfg.project_root, "configs", f"sandbox_config_tile{tile_idx+1}.yaml")
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

        ret = subprocess.call(
            ["python", sandbox_path, "-i", tile_sandbox_config, "-run", "--gage_id", gage_id],
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
    subprocess.call(["python3", "-m", "nwm_routing", "-f", "-V4", troute_path], env=env)


    # === STEP 4: Extract routed hydrograph (particle workspace) ===
    postproc_dir = os.path.join(router_work, "postproc")
    os.makedirs(postproc_dir, exist_ok=True)
    output_path = os.path.join(postproc_dir, f"{gage_id}_particle_{particle_idx}.csv")
    get_hydrograph_path = os.path.join(project_root, "model_assessment", "util", "get_hydrograph.py")
    summary_csv = os.path.join(project_root, "model_assessment", "util", "downstream_flowpath_summary.csv")

    troute_output_dir = os.path.join(router_work, "troute")
    if not os.path.isdir(troute_output_dir):
        raise FileNotFoundError(f"Particle troute dir missing: {troute_output_dir}")

    env = os.environ.copy()
    env["NGEN_CONCURRENT_PARTICLES"] = "1"
    env["NGEN_PARTICLE_ID"] = str(particle_idx)
    env["NGEN_REALIZATION_PATH"] = realization_path

    subprocess.call(
        [
            "python", get_hydrograph_path,
            "--gage_id", gage_id,
            "--output", output_path,
            "--troute_dir", troute_output_dir,
            "--summary", summary_csv,
        ],
        cwd=postproc_dir,
        env=env,
    )


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
        self.param_names = param_names if param_names else param_names
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
            with ThreadPool(processes=len(self.particles)) as pool:
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

            if self.include_nom:
                nom_vals = tile_params[-6:]
                nom_tbl  = os.path.join(work_root, "configs", "noahowp", "parameters", "MPTABLE.TBL")
                update_mptable(
                    original_file=nom_tbl,
                    output_file=nom_tbl,
                    updated_params=dict(zip(nom_param_names, nom_vals)),
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
            tile_sandbox_config = os.path.join(cfg.project_root, "configs", f"sandbox_config_tile{tile_idx+1}.yaml")
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

            ret = subprocess.call(
                ["python", sandbox_path, "-i", tile_sandbox_config, "-run", "--gage_id", self.gage_id],
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

        subprocess.call(["python3", "-m", "nwm_routing", "-f", "-V4", troute_path], env=env)

        # 4) Extract final routed hydrograph (BEST workspace)
        postproc_dir = os.path.join(router_work, "postproc")
        os.makedirs(postproc_dir, exist_ok=True)
        final_output_path = os.path.join(postproc_dir, f"{self.gage_id}_best.csv")
        get_hydrograph_path = os.path.join(project_root, "model_assessment", "util", "get_hydrograph.py")
        summary_csv = os.path.join(project_root, "model_assessment", "util", "downstream_flowpath_summary.csv")

        troute_output_dir = os.path.join(router_work, "troute")
        if not os.path.isdir(troute_output_dir):
            raise FileNotFoundError(f"Final troute dir missing: {troute_output_dir}")

        subprocess.call(
            [
                "python", get_hydrograph_path,
                "--gage_id", self.gage_id,
                "--output", final_output_path,
                "--troute_dir", troute_output_dir,
                "--summary", summary_csv,
            ],
            cwd=postproc_dir,
            env=env,
        )


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
            f"{self.metric_to_calibrate_on}_validation": val_metrics_final.get(self.metric_to_calibrate_on, np.nan)
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


if __name__ == "__main__":
    start = datetime.now()
    gage_list = pd.read_csv(cfg.gages_file, dtype={"gage_id": str})["gage_id"].tolist()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=max_cores_for_gages) as pool:
        pool.map(calibrate_gage, gage_list)
    print(f"Total wall time: {datetime.now() - start}")








