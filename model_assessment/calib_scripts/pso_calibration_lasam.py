"""
Author: Peter La Follette [plafollette@lynker.com | Refactor: Aug 2025]
Concurrent multi-tile PSO calibration for LASAM(+PET+T-Route), with optional NOM support.

- Runs all particles concurrently each iteration using a ThreadPool (safe, no child-proc nesting).
- Uses per-particle workspaces scaffolded by `sandbox.py -conf --concurrent-particles`:
    out/<gage_id>/particles/p{pid}/
  with particle-local configs/json/div/troute/postproc to avoid collisions.
- Retargets realization paths into particle workspaces and updates LASAM/PET/NOM config paths.
- Weighted tiling: optional `tile_weight` parameter (for 2 tiles); pre-routing weighted qlat in `div_weighted`.
- Final full-period run (spinup→val_end) in BEST particle workspace writes `{gage}_best.csv`.
"""

import os
import sys
import json
import yaml
import math
import shutil
import random
import traceback
import subprocess
from pathlib import Path
from datetime import datetime
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd

# Repro
np.random.seed(42)
random.seed(42)

# Project imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from model_assessment.util.metrics import compute_metrics
from model_assessment.util.update_NOM import update_mptable
from model_assessment.configs import path_config as cfg

# === CONFIGURATION ===
n_particles = 2                   # particles per gage (adjust)
n_iterations = 2                 # iterations (adjust)
max_cores_for_gages = 5           # process multiple gages in parallel (outer Pool)
metric_to_calibrate_on = "kge"

# run ALL particles concurrently per iteration (inner pool size). None => n_particles
max_particle_procs = None

with open("model_assessment/configs/time_config.yaml", "r") as f:
    time_cfg = yaml.safe_load(f)

spinup_start = pd.Timestamp(time_cfg["spinup_start"])  # pd.Timestamp for math
cal_start    = pd.Timestamp(time_cfg["cal_start"])
cal_end      = pd.Timestamp(time_cfg["cal_end"])
val_start    = pd.Timestamp(time_cfg["val_start"])
val_end      = pd.Timestamp(time_cfg["val_end"])

project_root     = cfg.project_root
sandbox_path     = cfg.sandbox_path
logging_dir      = cfg.logging_dir
observed_q_root  = cfg.observed_q_root
model_roots      = cfg.model_roots

os.makedirs(logging_dir, exist_ok=True)

# === LASAM+NOM parameter structure ===
# Soil layers are inferred per tile from config `layer_soil_type=`.
# For each layer: (log_alpha, n, log_Ks)
# Then LASAM scalars: (log10_a, b, frac_to_GW, field_capacity_psi, spf_factor, theta_e_1)
# Optional NOM (per tile): [MFSNO, RSURF_SNOW, HVT, CWPVT, VCMX25, MP]

nom_param_names = ["MFSNO", "RSURF_SNOW", "HVT", "CWPVT", "VCMX25", "MP"]
nom_param_bounds = [
    (0.625, 5.0), (0.1, 100.0), (0.0, 20.0), (0.18, 5.0), (0.0, 80.0), (3.6, 12.6)
]

# === Helpers ===
def check_for_stop_signal_or_low_disk(threshold_gb: float = 10):
    stop_file = os.path.join(project_root, "STOP_NOW.txt")
    if os.path.exists(stop_file):
        print("Detected STOP_NOW.txt")
        sys.exit(1)
    st = os.statvfs("/")
    free_gb = (st.f_frsize * st.f_bavail) / (1024 ** 3)
    if free_gb < threshold_gb:
        print(f"Free disk space below threshold: {free_gb:.2f} GB")
        sys.exit(1)

def pwork(root: str, gage_id: str, pid: int) -> str:
    """Particle workspace root created by `sandbox.py -conf --concurrent-particles`."""
    return os.path.join(root, "out", gage_id, "particles", f"p{pid}")

def resolve_div_dir(tile_root: str, gage_id: str, pid: int) -> str:
    """Return the directory that actually contains CSV divide outputs for this tile & particle.
    Prefer particle dir; fall back to legacy non-particle dir if needed.
    """
    candidates = [
        os.path.join(pwork(tile_root, gage_id, pid), "outputs", "div"),
        os.path.join(tile_root, "out", gage_id, "outputs", "div"),
    ]
    first_existing = None
    for d in candidates:
        if os.path.isdir(d):
            if first_existing is None:
                first_existing = d
            if any(name.endswith(".csv") for name in os.listdir(d)):
                return d
    return first_existing or candidates[0]

def log_incomplete(
    gage_id: str,
    particle_idx: int,
    stage: str,
    err_msg: str,
    params=None,
    param_names=None,
    iteration=None,
):
    try:
        os.makedirs(logging_dir, exist_ok=True)

        # Append to human-readable text log (kept as you had)
        with open(os.path.join(logging_dir, f"{gage_id}_errors.log"), "a") as f:
            f.write(
                f"{datetime.now().isoformat()} | iter={iteration} | pid={particle_idx} "
                f"| stage={stage} | {err_msg}\n"
            )

        # Append to CSV with parameter set
        csv_path = os.path.join(logging_dir, f"{gage_id}_incomplete.csv")
        row = {
            "timestamp": datetime.now().isoformat(),
            "gage_id": gage_id,
            "iteration": iteration,
            "particle": particle_idx,
            "stage": stage,
            "status": "INCOMPLETE",
            "error": err_msg,
        }

        if params is not None:
            p_list = np.asarray(params).tolist()
            if param_names is None:
                param_names = [f"p{i}" for i in range(len(p_list))]
            for k, v in zip(param_names, p_list):
                row[k] = v

        if os.path.isfile(csv_path):
            prev = pd.read_csv(csv_path)
            # union columns
            for col in row.keys():
                if col not in prev.columns:
                    prev[col] = np.nan
            new = pd.DataFrame([row])
            # ensure same order as existing
            for col in prev.columns:
                if col not in new.columns:
                    new[col] = np.nan
            pd.concat([prev, new[prev.columns]], ignore_index=True).to_csv(csv_path, index=False)
        else:
            pd.DataFrame([row]).to_csv(csv_path, index=False)
    except Exception:
        # Never fail the main run because logging hiccuped
        pass


# --- Realization retargeting (LASAM+PET+optional NOM) ---
def update_nom_namelist_paramdir(namelist_path: str, new_param_dir: str):
    if not os.path.isfile(namelist_path):
        return
    out_lines = []
    with open(namelist_path, "r") as f:
        for line in f:
            if "parameter_dir" in line and "=" in line and not line.strip().startswith("!"):
                quote = '"' if '"' in line else "'"
                prefix = line.split("=", 1)[0]
                comment = ""
                if "!" in line:
                    comment = "  !" + line.split("!", 1)[1].strip()
                line = f"{prefix}= {quote}{new_param_dir}{quote}{comment}\n"
            out_lines.append(line)
    with open(namelist_path, "w") as f:
        f.writelines(out_lines)

def retarget_realization_paths(realization_path: str, work_root: str, base_out_dir: str):
    with open(realization_path, "r") as f:
        rz = json.load(f)

    # Create only the directories we always need
    configs_root = os.path.join(work_root, "configs")
    os.makedirs(configs_root, exist_ok=True)
    div_dir = os.path.join(work_root, "outputs", "div")
    os.makedirs(div_dir, exist_ok=True)

    # Lazily create per-model config dirs only when we actually see that model
    cfg_lasam_dir = os.path.join(configs_root, "lasam")
    cfg_pet_dir   = os.path.join(configs_root, "pet")
    cfg_nom_dir   = os.path.join(configs_root, "noahowp")
    cfg_nom_param = os.path.join(cfg_nom_dir, "parameters")

    forms = rz.get("global", {}).get("formulations", [])

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

            # Resolve source path (allow relative to base out dir)
            src = init_cfg if os.path.isabs(init_cfg) else os.path.join(base_out_dir, init_cfg)

            # LASAM / LGAR
            if any(alias in model_type for alias in ("LASAM", "LGAR")) or "/configs/lasam/" in init_cfg:
                os.makedirs(cfg_lasam_dir, exist_ok=True)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(cfg_lasam_dir, fname))
                p["init_config"] = os.path.abspath(os.path.join(cfg_lasam_dir, fname))

            # PET
            elif "PET" in model_type or "/configs/pet/" in init_cfg:
                os.makedirs(cfg_pet_dir, exist_ok=True)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(cfg_pet_dir, fname))
                p["init_config"] = os.path.abspath(os.path.join(cfg_pet_dir, fname))

            # NOM (Noah-MP) — only create noahowp/ if needed
            elif any(k in model_type for k in ("NOM", "NOAH")) or "/noahowp/" in init_cfg:
                os.makedirs(cfg_nom_param, exist_ok=True)
                cfg_nom_namelist = os.path.join(cfg_nom_dir, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, cfg_nom_namelist)

                # Link/copy parameter tables from the shared (gage-level) location
                shared_nom_paramdir = os.path.join(base_out_dir, "configs", "noahowp", "parameters")
                for tbl in ("GENPARM.TBL", "SOILPARM.TBL"):
                    src_tbl = os.path.join(shared_nom_paramdir, tbl)
                    dst_tbl = os.path.join(cfg_nom_param, tbl)
                    if os.path.isfile(src_tbl) and not os.path.exists(dst_tbl):
                        try:
                            os.symlink(src_tbl, dst_tbl)
                        except OSError:
                            shutil.copy2(src_tbl, dst_tbl)

                # Seed MPTABLE if present
                mpt_src = os.path.join(shared_nom_paramdir, "MPTABLE.TBL")
                mpt_dst = os.path.join(cfg_nom_param, "MPTABLE.TBL")
                if os.path.isfile(mpt_src) and not os.path.isfile(mpt_dst):
                    shutil.copy2(mpt_src, mpt_dst)

                update_nom_namelist_paramdir(cfg_nom_namelist, cfg_nom_param)
                p["init_config"] = os.path.abspath(cfg_nom_namelist)

    # Redirect output_root into the particle workspace
    rz["output_root"] = div_dir
    if "global" in rz:
        rz["global"]["output_root"] = div_dir

    with open(realization_path, "w") as f:
        json.dump(rz, f, indent=4)

    return div_dir


# --- LASAM param extraction & writing ---
def get_observed_q(observed_path: str) -> pd.Series:
    df = pd.read_csv(observed_path, parse_dates=["value_time"]).set_index("value_time")
    return df["flow_m3_per_s"]

def extract_initial_params(example_config_path: str):
    with open(example_config_path) as f:
        lines = f.readlines()

    soil_file_line = next(line for line in lines if line.strip().startswith("soil_params_file"))
    soil_file = soil_file_line.split("=", 1)[1].strip()
    soil_path = Path(soil_file)
    if not soil_path.is_absolute():
        soil_path = (Path(example_config_path).parent / soil_path).resolve()

    # Read once, from the resolved path
    with open(soil_path) as f:
        soil_lines = f.readlines()

    soil_types_line = next(line for line in lines if line.startswith("layer_soil_type="))
    soil_types = list(map(int, soil_types_line.strip().split("=")[1].split(",")))


    # LASAM scalars
    a = float(next(line.split("=")[1] for line in lines if line.startswith("a=")))
    b = float(next(line.split("=")[1] for line in lines if line.startswith("b=")))
    frac_to_GW = float(next(line.split("=")[1] for line in lines if line.startswith("frac_to_GW=")))
    field_capacity_psi = float(next(line.split("=")[1].split("[")[0] for line in lines if line.startswith("field_capacity_psi=")))
    spf_factor = float(next(line.split("=")[1] for line in lines if line.startswith("spf_factor=")))

    layer_params = []
    for soil_type in soil_types:
        tokens = soil_lines[soil_type].split()
        alpha = float(tokens[3]); n = float(tokens[4]); Ks = float(tokens[5])
        layer_params.extend([math.log10(alpha), n, math.log10(Ks)])

    first_soil_type = soil_types[0]
    first_layer_tokens = soil_lines[first_soil_type].split()
    theta_e_1 = float(first_layer_tokens[2])

    # NOM optional
    config_root = os.path.dirname(os.path.dirname(example_config_path))  # trims /lasam
    nom_dir = os.path.join(config_root, "noahowp")
    nom_params = []
    if os.path.isdir(nom_dir):
        mptable_path = os.path.join(nom_dir, "parameters", "MPTABLE.TBL")
        try:
            with open(mptable_path) as f:
                lines = f.readlines()
            nom_params_dict = {}
            for line in lines:
                if "=" not in line or line.strip().startswith("!"):
                    continue
                key, value = line.split("=", 1)
                param_name = key.strip()
                if param_name in nom_param_names:
                    value_str = value.split("!")[0]
                    values = [v.strip() for v in value_str.split(",") if v.strip()]
                    nom_params_dict[param_name] = float(values[0])
            if set(nom_params_dict.keys()) != set(nom_param_names):
                raise ValueError("Incomplete NOM params in MPTABLE.TBL")
            nom_params = [nom_params_dict[p] for p in nom_param_names]
        except Exception as e:
            print(f"[WARN] NOM parse failed: {e}")

    # Return with 'a' already converted to log10 for optimization
    return layer_params + [math.log10(a), b, frac_to_GW, field_capacity_psi, spf_factor, theta_e_1] + nom_params

def extract_tile_params(full_params, tile_idx: int, n_tiles: int):
    full_params = np.array(full_params)
    total_len = len(full_params)
    if n_tiles == 2 and total_len % 2 == 1:
        param_slice = full_params[:-1]
        chunk = len(param_slice) // n_tiles
        return param_slice[tile_idx * chunk : (tile_idx + 1) * chunk]
    chunk = total_len // n_tiles
    return full_params[tile_idx * chunk : (tile_idx + 1) * chunk]

def regenerate_lasam_configs_in_workspace(work_cfg_dir: str, params, include_nom: bool):
    cfg_files = sorted(f for f in os.listdir(work_cfg_dir) if f.startswith("lasam_config_cat"))
    if not cfg_files:
        raise FileNotFoundError(f"No LASAM configs found in {work_cfg_dir}")

    with open(os.path.join(work_cfg_dir, cfg_files[0])) as f:
        lines0 = f.readlines()
    soil_types_line = next(line for line in lines0 if line.strip().startswith("layer_soil_type"))
    soil_types = list(map(int, soil_types_line.strip().split("=", 1)[1].split(",")))
    num_layers = len(soil_types)

    offset = 3 * num_layers
    if include_nom:
        lasam_slice = params
        scalars = lasam_slice[offset:-6]
        nom_vals = lasam_slice[-6:]
    else:
        lasam_slice = params
        scalars = lasam_slice[offset:]
        nom_vals = []

    log10_a, b, frac_to_GW, field_capacity_psi, spf_factor, theta_e_1 = scalars
    a = 10 ** log10_a

    for cfg_name in cfg_files:
        cfg_path = os.path.join(work_cfg_dir, cfg_name)
        with open(cfg_path, "r") as f:
            lines = f.readlines()

        # Find and resolve the soil file path
        soil_file = None
        for line in lines:
            if line.strip().startswith("soil_params_file"):
                soil_file = line.split("=", 1)[1].strip()
                break
        if not soil_file:
            raise FileNotFoundError(f"soil_params_file missing for {cfg_path}")

        soil_src = Path(soil_file)
        if not soil_src.is_absolute():
            soil_src = (Path(cfg_path).parent / soil_src).resolve()
        if not soil_src.is_file():
            raise FileNotFoundError(f"soil_params_file not found: {soil_src}")

        # Copy once into particle config dir, preserving the original filename (no suffix)
        local_soil = os.path.join(work_cfg_dir, soil_src.name)
        if not os.path.isfile(local_soil):
            shutil.copy2(soil_src, local_soil)

        # Update scalar lines & retarget soil file to the local copy
        new_lines = []
        for line in lines:
            s = line.strip()
            if s.startswith("a="):
                new_lines.append(f"a={a}\n")
            elif s.startswith("b="):
                new_lines.append(f"b={b}\n")
            elif s.startswith("frac_to_GW="):
                new_lines.append(f"frac_to_GW={frac_to_GW}\n")
            elif s.startswith("field_capacity_psi="):
                new_lines.append(f"field_capacity_psi={field_capacity_psi}[cm]\n")
            elif s.startswith("spf_factor="):
                new_lines.append(f"spf_factor={spf_factor}\n")
            elif s.startswith("soil_params_file"):
                new_lines.append(f"soil_params_file={os.path.abspath(local_soil)}\n")
            else:
                new_lines.append(line)
        with open(cfg_path, "w") as f:
            f.writelines(new_lines)

        # Edit the local soil file entries per layer
        with open(local_soil, "r") as f:
            soil_lines = f.readlines()
        for i, soil_type in enumerate(soil_types):
            start = i * 3
            log_alpha, n, log_Ks = lasam_slice[start:start + 3]
            alpha = 10 ** log_alpha
            Ks = 10 ** log_Ks
            toks = soil_lines[soil_type].split()
            if i == 0:
                toks[2] = str(theta_e_1)  # only top layer
            toks[3] = str(alpha)
            toks[4] = str(n)
            toks[5] = str(Ks)
            soil_lines[soil_type] = "\t".join(toks) + "\n"
        with open(local_soil, "w") as f:
            f.writelines(soil_lines)


# === Objective function (concurrent-safe; particle-aware) ===
def _safe_objective(args):
    try:
        obj, val_metrics, cal_metrics = objective_function_tiled(args)
        return ("OK", "", obj, val_metrics, cal_metrics)
    except Exception as e:
        (
            params, particle_idx, gage_id,
            model_roots, observed_q_root,
            include_nom_flags, weights,
            iteration, param_names
        ) = args
        log_incomplete(
            gage_id, particle_idx,
            stage="objective",
            err_msg=str(e),
            params=params,
            param_names=param_names,
            iteration=iteration,
        )
        dummy = {metric_to_calibrate_on: np.nan}
        return ("FAIL", str(e), float("inf"), dummy, dummy)


def objective_function_tiled(args):
    (
        params, particle_idx, gage_id,
        model_roots, observed_q_root,
        include_nom_flags, weights,
        *extra  # (iteration, param_names) comes in here; not needed inside
    ) = args

    check_for_stop_signal_or_low_disk()
    n_tiles = len(model_roots)

    # Tile weight handling
    if n_tiles == 2 and len(params) % 2 == 1:
        tile_weight = params[-1]
        weights = [tile_weight, 1.0 - tile_weight]
        params = params[:-1]
    elif weights is None:
        weights = [1.0 / n_tiles] * n_tiles

    # === STEP 1: Hydrology per tile in PARTICLE workspace ===
    for tile_idx, tile_root in enumerate(model_roots):
        tile_params = extract_tile_params(params, tile_idx, n_tiles)
        include_nom = include_nom_flags[tile_idx]

        work_root   = pwork(tile_root, gage_id, particle_idx)  # .../out/<gage>/particles/pX
        cfg_dir_lsm = os.path.join(work_root, "configs", "lasam")
        json_dir    = os.path.join(work_root, "json")
        os.makedirs(cfg_dir_lsm, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

        # Find particle-local realization.json
        json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
        if not json_files:
            raise FileNotFoundError(f"No realization JSON found in {json_dir}")
        realization_path = os.path.join(json_dir, sorted(json_files)[0])

        # Retarget realization paths into particle workspace (LASAM/PET/NOM)
        base_out_dir = os.path.join(tile_root, "out", gage_id)
        retarget_realization_paths(realization_path, work_root, base_out_dir)

        # Clamp time window to spinup→cal_end for calibration runs
        with open(realization_path, "r") as f:
            realization = json.load(f)
        realization["time"]["start_time"] = time_cfg["spinup_start"]
        realization["time"]["end_time"]   = time_cfg["cal_end"]
        with open(realization_path, "w") as f:
            json.dump(realization, f, indent=4)

        # Update LASAM configs in workspace
        regenerate_lasam_configs_in_workspace(cfg_dir_lsm, tile_params, include_nom)

        # Update NOM per tile if present (particle-local MPTABLE)
        if include_nom:
            nom_tbl = os.path.join(work_root, "configs", "noahowp", "parameters", "MPTABLE.TBL")
            nom_vals = tile_params[-6:]
            update_mptable(
                original_file=nom_tbl,
                output_file=nom_tbl,
                updated_params=dict(zip(nom_param_names, nom_vals)),
                verbose=False,
            )

        # Clear old divide outputs in particle workspace
        div_dir = os.path.join(work_root, "outputs", "div")
        os.makedirs(div_dir, exist_ok=True)
        for item in list(os.listdir(div_dir)):
            if item.startswith("."):  # keep hidden files
                continue
            p = os.path.join(div_dir, item)
            if os.path.isfile(p) or os.path.islink(p):
                os.remove(p)
            elif os.path.isdir(p):
                shutil.rmtree(p)

        # Run hydrology (divide scale) for this tile & particle
        tile_sandbox_config = os.path.join(cfg.project_root, "configs", f"sandbox_config_tile{tile_idx+1}.yaml")
        env = os.environ.copy()
        env["NGEN_CONCURRENT_PARTICLES"] = "1"
        env["NGEN_PARTICLE_ID"] = str(particle_idx)
        env["NGEN_REALIZATION_PATH"] = realization_path
        ret = subprocess.call(
            ["python", sandbox_path, "-i", tile_sandbox_config, "-run", "--gage_id", gage_id],
            cwd=tile_root,
            env=env,
        )
        if ret != 0:
            raise RuntimeError(f"Hydrology failed: gage {gage_id} | pid {particle_idx} | tile {tile_idx}")

    # === STEP 2: Weighted qlat build in router tile particle workspace ===
    router_tile_root = model_roots[0]
    router_work = pwork(router_tile_root, gage_id, particle_idx)

    weighted_div_dir = os.path.join(router_work, "outputs", "div_weighted")
    if os.path.exists(weighted_div_dir):
        shutil.rmtree(weighted_div_dir)
    os.makedirs(weighted_div_dir, exist_ok=True)

    src_div_dirs = [resolve_div_dir(root, gage_id, particle_idx) for root in model_roots]

    # Collect file list from first non-empty source
    files = []
    for d in src_div_dirs:
        if os.path.isdir(d):
            cand = [f for f in os.listdir(d) if (f.startswith("cat-") or f.startswith("nex-")) and f.endswith(".csv")]
            if cand:
                files = cand
                break
    if not files:
        details = []
        for idx, d in enumerate(src_div_dirs):
            exists = os.path.isdir(d)
            count = len(os.listdir(d)) if exists else 0
            details.append(f"tile{idx}: dir={d}, exists={exists}, nfiles={count}")
        raise RuntimeError("No divide CSVs for routing: " + "; ".join(details))


    if len(model_roots) == 1:
        src = src_div_dirs[0]
        for fname in files:
            shutil.copy2(os.path.join(src, fname), os.path.join(weighted_div_dir, fname))
    else:
        for fname in files:
            dfs = []
            df_ref = None
            for t_idx, div_dir in enumerate(src_div_dirs):
                fp = os.path.join(div_dir, fname)
                if os.path.exists(fp):
                    if fname.startswith("nex-"):
                        df = pd.read_csv(fp, header=None); df.columns = ["Time Step", "Time", "q_out"]
                    else:
                        df = pd.read_csv(fp)
                    if df_ref is None:
                        df_ref = df.copy()
                    dfs.append(df["q_out"] * weights[t_idx])
            if dfs:
                out_df = df_ref.copy()
                out_df["q_out"] = sum(dfs)
                out_df["Time"] = pd.to_datetime(out_df["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                outpath = os.path.join(weighted_div_dir, fname)
                if fname.startswith("nex-"):
                    out_df.to_csv(outpath, index=False, header=False)
                else:
                    out_df.to_csv(outpath, index=False)

    # === STEP 3: Routing once per particle (T-Route) ===
    troute_path = os.path.join(router_work, "configs", "troute_config.yaml")
    with open(troute_path) as f:
        troute_cfg = yaml.safe_load(f)

    nts = int((cal_end - spinup_start) / pd.Timedelta(seconds=300))
    troute_cfg["compute_parameters"]["restart_parameters"]["start_datetime"] = spinup_start.strftime("%Y-%m-%d_%H:%M:%S")
    troute_cfg["compute_parameters"]["forcing_parameters"]["nts"] = nts
    troute_cfg["compute_parameters"]["forcing_parameters"]["qlat_input_folder"] = weighted_div_dir

    particle_troute_dir = os.path.join(router_work, "troute")
    os.makedirs(particle_troute_dir, exist_ok=True)
    op = troute_cfg.setdefault("output_parameters", {})
    so = op.setdefault("stream_output", {})
    so["stream_output_directory"] = particle_troute_dir

    # mask_output handling: if invalid, try base; else drop
    mask_path = so.get("mask_output")
    if not (isinstance(mask_path, str) and os.path.isfile(mask_path)):
        orig_mask = os.path.join(router_tile_root, "out", gage_id, "configs", "mask_output.yaml")
        if os.path.isfile(orig_mask):
            so["mask_output"] = orig_mask
        else:
            so.pop("mask_output", None)

    with open(troute_path, "w") as f:
        yaml.safe_dump(troute_cfg, f)

    # Clean old particle T-Route outputs
    for fn in os.listdir(particle_troute_dir):
        if fn.endswith((".nc", ".csv", ".parquet")):
            os.remove(os.path.join(particle_troute_dir, fn))

    env = os.environ.copy()
    env["NGEN_CONCURRENT_PARTICLES"] = "1"
    env["NGEN_PARTICLE_ID"] = str(particle_idx)
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

    # === STEP 5: Metrics ===
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

    # Lightweight cleanup: remove weighted qlat and bulky routing artifacts
    try:
        if os.path.isdir(weighted_div_dir):
            shutil.rmtree(weighted_div_dir)
    except Exception:
        pass
    try:
        for fn in os.listdir(troute_output_dir):
            if fn.endswith((".nc", ".csv", ".parquet")) and not fn.endswith("_best.csv"):
                os.remove(os.path.join(troute_output_dir, fn))
    except Exception:
        pass

    return -cal_metrics[metric_to_calibrate_on], val_metrics, cal_metrics

# === PSO ===
class Particle:
    def __init__(self, bounds, init_position=None):
        self.position = np.array(init_position) if init_position is not None else np.array(
            [np.random.uniform(low, high) for low, high in bounds]
        )
        self.velocity = np.array([0.1 * (high - low) * np.random.uniform(-1, 1) for low, high in bounds])
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')
        self.current_value = float('inf')
        self.stagnation_counter = 0

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
    def __init__(self, n_particles, bounds, n_iterations, gage_id,
                 init_position, metric_to_calibrate_on="kge",
                 include_nom_flags=None, param_names=None):
        self.particles = [
            Particle(bounds, init_position=init_position if i == 0 else None)
            for i in range(n_particles)
        ]
        self.bounds = bounds
        self.n_iterations = n_iterations
        self.gage_id = gage_id
        self.global_best_position = self.particles[0].position
        self.global_best_value = float('inf')
        self.metric_to_calibrate_on = metric_to_calibrate_on
        self.include_nom_flags = include_nom_flags or []
        self.param_names = param_names or [f"p{i}" for i in range(len(bounds))]
        self.best_cal_metrics = {}
        self.best_val_metrics = {}

    def optimize(self):
        start_time = datetime.now()
        log_rows = []
        log_path = os.path.join(logging_dir, f"{self.gage_id}.csv")
        stagnation_threshold = 10
        w_start, w_end = 0.9, 0.4

        pool_size = max_particle_procs or len(self.particles)

        for iteration in range(self.n_iterations):
            print(f"\n--- Iteration {iteration + 1} for gage {self.gage_id} ---")
            w = w_start - (w_start - w_end) * (iteration / self.n_iterations)

            # Build argument list for all particles
            args_list = []
            for i, p in enumerate(self.particles):
                print(f"[INFO] gage {self.gage_id} | iter {iteration+1} | particle {i}")
                args_list.append((
                    p.position, i, self.gage_id,
                    model_roots, observed_q_root,
                    self.include_nom_flags,
                    [1.0 / len(model_roots)] * len(model_roots),
                    iteration + 1,          # NEW: for logging
                    self.param_names        # NEW: for column names
                ))


            with ThreadPool(processes=pool_size) as pool:
                results = pool.map(_safe_objective, args_list)

            # Reduction step
            for idx, r in enumerate(results):
                status, err, objective_value, val_metrics, cal_metrics = r
                particle = self.particles[idx]
                particle.current_value = objective_value

                metric_cal = cal_metrics.get(self.metric_to_calibrate_on, np.nan)
                metric_val = val_metrics.get(self.metric_to_calibrate_on, np.nan)

                if status == "OK" and objective_value < (particle.best_value - 1e-6):
                    particle.best_value = objective_value
                    particle.best_position = np.copy(particle.position)
                    particle.stagnation_counter = 0
                else:
                    particle.stagnation_counter += 1

                if status == "OK" and objective_value < self.global_best_value:
                    self.global_best_value = objective_value
                    self.global_best_position = np.copy(particle.position)
                    self.best_cal_metrics = cal_metrics
                    self.best_val_metrics = val_metrics
                    particle.stagnation_counter = 0

                # Log params and metrics per particle
                row = {
                    "iteration": iteration + 1,
                    "particle": idx,
                    **{name: val for name, val in zip(self.param_names, particle.position)},
                    f"{self.metric_to_calibrate_on}_calibration": metric_cal,
                    f"{self.metric_to_calibrate_on}_validation": metric_val,
                    "status": status,
                    "error": (err or "")[:240],
                }
                log_rows.append(row)

            # Persist log after each iteration
            pd.DataFrame(log_rows).to_csv(log_path, index=False)

            # Swarm update
            for p in self.particles:
                p.update_velocity(self.global_best_position, w=w, c1=1.5, c2=1.5)
                p.update_position(self.bounds)

            print(f"Global best objective so far: {-self.global_best_value:.4f}")

        # === Final full-period run in BEST particle workspace ===
        print(f"\n[INFO] Final full-period validation for gage {self.gage_id}...")
        n_tiles = len(model_roots)
        weights = [1.0 / n_tiles] * n_tiles
        if n_tiles == 2 and len(self.global_best_position) % 2 == 1:
            weights = [self.global_best_position[-1], 1.0 - self.global_best_position[-1]]

        best_pid = 0  # reuse pid=0 workspace for the final run

        # 1) Hydrology per tile with full window (spinup→val_end)
        for tile_idx, tile_root in enumerate(model_roots):
            tile_params = extract_tile_params(self.global_best_position, tile_idx, n_tiles)
            include_nom = self.include_nom_flags[tile_idx]

            work_root   = pwork(tile_root, self.gage_id, best_pid)
            cfg_dir_lsm = os.path.join(work_root, "configs", "lasam")
            json_dir    = os.path.join(work_root, "json")
            os.makedirs(cfg_dir_lsm, exist_ok=True)
            os.makedirs(json_dir, exist_ok=True)

            json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
            if not json_files:
                raise FileNotFoundError(f"No realization JSON found in {json_dir}")
            realization_path = os.path.join(json_dir, sorted(json_files)[0])

            base_out_dir = os.path.join(tile_root, "out", self.gage_id)
            retarget_realization_paths(realization_path, work_root, base_out_dir)

            with open(realization_path, "r") as f:
                realization = json.load(f)
            realization["time"]["start_time"] = time_cfg["spinup_start"]
            realization["time"]["end_time"]   = time_cfg["val_end"]
            with open(realization_path, "w") as f:
                json.dump(realization, f, indent=4)

            regenerate_lasam_configs_in_workspace(cfg_dir_lsm, tile_params, include_nom)

            if include_nom:
                nom_tbl = os.path.join(work_root, "configs", "noahowp", "parameters", "MPTABLE.TBL")
                nom_vals = tile_params[-6:]
                update_mptable(
                    original_file=nom_tbl,
                    output_file=nom_tbl,
                    updated_params=dict(zip(nom_param_names, nom_vals)),
                    verbose=False,
                )

            # Clean div outputs
            div_dir = os.path.join(work_root, "outputs", "div")
            os.makedirs(div_dir, exist_ok=True)
            for item in list(os.listdir(div_dir)):
                if item.startswith("."): continue
                pth = os.path.join(div_dir, item)
                if os.path.isfile(pth) or os.path.islink(pth): os.remove(pth)
                elif os.path.isdir(pth): shutil.rmtree(pth)

            tile_sandbox_config = os.path.join(cfg.project_root, "configs", f"sandbox_config_tile{tile_idx+1}.yaml")
            env = os.environ.copy()
            env["NGEN_CONCURRENT_PARTICLES"] = "1"
            env["NGEN_PARTICLE_ID"] = str(best_pid)
            env["NGEN_REALIZATION_PATH"] = realization_path
            ret = subprocess.call(
                ["python", sandbox_path, "-i", tile_sandbox_config, "-run", "--gage_id", self.gage_id],
                cwd=tile_root, env=env
            )
            if ret != 0:
                raise RuntimeError(f"Final hydrology failed: gage {self.gage_id} | tile {tile_idx}")

        # 2) Weighted qlat for final run
        router_tile_root = model_roots[0]
        router_work = pwork(router_tile_root, self.gage_id, best_pid)
        weighted_div_dir = os.path.join(router_work, "outputs", "div_weighted")
        if os.path.exists(weighted_div_dir): shutil.rmtree(weighted_div_dir)
        os.makedirs(weighted_div_dir, exist_ok=True)

        src_div_dirs = [resolve_div_dir(root, self.gage_id, best_pid) for root in model_roots]
        files = []
        for d in src_div_dirs:
            if os.path.isdir(d):
                cand = [f for f in os.listdir(d) if (f.startswith("cat-") or f.startswith("nex-")) and f.endswith(".csv")]
                if cand:
                    files = cand; break
        if not files:
            print("[ERROR] No divide CSVs found for final routing.")
            dummy = {metric_to_calibrate_on: -np.inf}
            return (1e12, dummy, dummy)

        if len(model_roots) == 1:
            src = src_div_dirs[0]
            for fname in files:
                shutil.copy2(os.path.join(src, fname), os.path.join(weighted_div_dir, fname))
        else:
            for fname in files:
                dfs = []; df_ref = None
                for t_idx, div_dir in enumerate(src_div_dirs):
                    fp = os.path.join(div_dir, fname)
                    if os.path.exists(fp):
                        if fname.startswith("nex-"):
                            df = pd.read_csv(fp, header=None); df.columns = ["Time Step", "Time", "q_out"]
                        else:
                            df = pd.read_csv(fp)
                        if df_ref is None: df_ref = df.copy()
                        dfs.append(df["q_out"] * weights[t_idx])
                if dfs:
                    out_df = df_ref.copy(); out_df["q_out"] = sum(dfs)
                    out_df["Time"] = pd.to_datetime(out_df["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                    outpath = os.path.join(weighted_div_dir, fname)
                    if fname.startswith("nex-"):
                        out_df.to_csv(outpath, index=False, header=False)
                    else:
                        out_df.to_csv(outpath, index=False)

        # 3) Final routing
        troute_path = os.path.join(router_work, "configs", "troute_config.yaml")
        with open(troute_path) as f:
            troute_cfg = yaml.safe_load(f)
        nts_full = int((val_end - spinup_start) / pd.Timedelta(seconds=300))
        troute_cfg["compute_parameters"]["restart_parameters"]["start_datetime"] = spinup_start.strftime("%Y-%m-%d_%H:%M:%S")
        troute_cfg["compute_parameters"]["forcing_parameters"]["nts"] = nts_full
        troute_cfg["compute_parameters"]["forcing_parameters"]["qlat_input_folder"] = weighted_div_dir
        particle_troute_dir = os.path.join(router_work, "troute")
        os.makedirs(particle_troute_dir, exist_ok=True)
        op = troute_cfg.setdefault("output_parameters", {})
        so = op.setdefault("stream_output", {})
        so["stream_output_directory"] = particle_troute_dir
        mask_path = so.get("mask_output")
        if not (isinstance(mask_path, str) and os.path.isfile(mask_path)):
            orig_mask = os.path.join(router_tile_root, "out", self.gage_id, "configs", "mask_output.yaml")
            if os.path.isfile(orig_mask): so["mask_output"] = orig_mask
            else: so.pop("mask_output", None)
        with open(troute_path, "w") as f:
            yaml.safe_dump(troute_cfg, f)
        for fn in os.listdir(particle_troute_dir):
            if fn.endswith((".nc", ".csv", ".parquet")):
                os.remove(os.path.join(particle_troute_dir, fn))
        env = os.environ.copy(); env["NGEN_CONCURRENT_PARTICLES"] = "1"; env["NGEN_PARTICLE_ID"] = str(best_pid)
        subprocess.call(["python3", "-m", "nwm_routing", "-f", "-V4", troute_path], env=env)

        # 4) Extract final hydrograph
        postproc_dir = os.path.join(router_work, "postproc"); os.makedirs(postproc_dir, exist_ok=True)
        final_output_path = os.path.join(postproc_dir, f"{self.gage_id}_best.csv")
        get_hydrograph_path = os.path.join(project_root, "model_assessment", "util", "get_hydrograph.py")
        summary_csv = os.path.join(project_root, "model_assessment", "util", "downstream_flowpath_summary.csv")
        subprocess.call([
            "python", get_hydrograph_path,
            "--gage_id", self.gage_id,
            "--output", final_output_path,
            "--troute_dir", os.path.join(router_work, "troute"),
            "--summary", summary_csv,
        ], cwd=postproc_dir, env=env)

        # 5) Final validation metrics
        obs_df = get_observed_q(os.path.join(observed_q_root, "successful_sites_resampled", f"{self.gage_id}.csv"))
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
        }
        existing = pd.DataFrame()
        if os.path.isfile(log_path):
            existing = pd.read_csv(log_path)
        pd.concat([existing, pd.DataFrame([final_row])]).to_csv(log_path, index=False)

        return self.global_best_position, self.global_best_value, self.best_val_metrics.get(self.metric_to_calibrate_on, np.nan), datetime.now() - start_time

# === Per-gage wrapper ===
def calibrate_gage(gage_id: str):
    n_tiles = len(model_roots)
    all_init_params, all_bounds, include_nom_flags = [], [], []
    tile_layer_counts = []

    for tile_idx, root in enumerate(model_roots):
        # Use particle 0 workspace as template source (scaffolded during -conf)
        cfg_dir = os.path.join(pwork(root, gage_id, 0), "configs", "lasam")
        cfg_file = sorted(f for f in os.listdir(cfg_dir) if f.startswith("lasam_config_cat"))[0]
        example_path = os.path.join(cfg_dir, cfg_file)

        tile_init = extract_initial_params(example_path)

        # Determine number of layers for bounds & naming
        with open(example_path) as f:
            lines = f.readlines()
        soil_types_line = next(line for line in lines if line.startswith("layer_soil_type="))
        num_layers = len(soil_types_line.strip().split("=")[1].split(","))
        tile_layer_counts.append(num_layers)

        # Bounds per tile
        tile_bounds = []
        for _ in range(num_layers):
            tile_bounds.extend([(-4, 0.0), (1.02, 3.0), (-4, 2)])  # log_alpha, n, log_Ks
        tile_bounds.extend([
            (-8, -1),         # log10_a
            (0.01, 5.0),      # b
            (1e-4, 1 - 1e-4), # frac_to_GW
            (10.0, 500.0),    # field_capacity_psi
            (0.1, 1.0),       # spf_factor
            (0.3, 0.6),       # theta_e_1
        ])

        # NOM detection
        include_nom = os.path.isdir(os.path.join(pwork(root, gage_id, 0), "configs", "noahowp"))
        if include_nom:
            tile_bounds.extend(nom_param_bounds)
        include_nom_flags.append(include_nom)

        all_init_params.extend(tile_init)
        all_bounds.extend(tile_bounds)

    # Build readable param names for logging
    names = []
    for tile_idx, num_layers in enumerate(tile_layer_counts):
        suffix = f"_tile{tile_idx+1}"
        for i in range(1, num_layers+1):
            names.extend([f"log_alpha_{i}{suffix}", f"n_{i}{suffix}", f"log_Ks_{i}{suffix}"])
        names.extend([f"log10_a{suffix}", f"b{suffix}", f"frac_to_GW{suffix}",
                      f"field_capacity_psi{suffix}", f"spf_factor{suffix}", f"theta_e_1{suffix}"])
        if include_nom_flags[tile_idx]:
            names.extend([f"{n}{suffix}" for n in nom_param_names])

    # Optional learned weight for 2 tiles
    if n_tiles == 2:
        all_init_params.append(0.8)  # tile0 weight
        all_bounds.append((0.0, 1.0))
        names.append("tile_weight")

    pso = PSO(
        n_particles=n_particles,
        bounds=all_bounds,
        n_iterations=n_iterations,
        gage_id=gage_id,
        init_position=all_init_params,
        metric_to_calibrate_on=metric_to_calibrate_on,
        include_nom_flags=include_nom_flags,
        param_names=names,
    )
    pso.optimize()

# === MAIN ===
if __name__ == "__main__":
    from multiprocessing import get_context

    start = datetime.now()
    gage_list = pd.read_csv(cfg.gages_file, dtype={"gage_id": str})["gage_id"].tolist()

    # Outer pool: parallelize across gages
    ctx = get_context("spawn")
    with ctx.Pool(processes=max_cores_for_gages) as pool:
        pool.map(calibrate_gage, gage_list)

    print(f"Total wall time: {datetime.now() - start}")





