#!/usr/bin/env python3
"""
Author: Peter La Follette [plafollette@lynker.com | Refactor: Jan 2026]
Single-tile PSO calibration for CASAM(+PET+T-Route), with optional NOM support.

Behavior preserved from Aug 2025 script:
- Runs all particles concurrently each iteration using a ThreadPool (safe, no child-proc nesting).
- Uses per-particle workspaces scaffolded by `sandbox.py -conf --concurrent-particles`:
    out/<gage_id>/particles/p{pid}/
  with particle-local configs/json/div/troute/postproc to avoid collisions.
- Retargets realization paths into particle workspaces and updates CASAM/PET/NOM config paths.
- Keeps one tile/model root per calibration job.
- Final full-period run (spinup->val_end) in BEST workspace writes '{gage}_best.csv'.
- STOP_NOW + low-disk abort.
- Incomplete-run logging to {gage}_errors.log and {gage}_incomplete.csv.
- Per-iteration CSV logging with parameters, calibration/validation metric, and errors.
- Stagnation resets (skip iteration leader).

Main change:
- The optimizer parameter vector is built from a user-editable CALIBRATION_REQUEST
  that can target specific layers and/or scalar/NOM parameters.

Default behavior restored:
- If NOM exists in a tile workspace and DEFAULT_INCLUDE_NOM_IF_PRESENT=True, the upstream NOM
  params are automatically appended.
- Layer params do not auto-expand; the user controls which layers to include.
"""

import os
import sys
import argparse
import json
import yaml
import math
import shutil
import random
import subprocess
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from multiprocessing.pool import ThreadPool
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Repro
np.random.seed(42)
random.seed(42)

# Project imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# from model_assessment.util.metrics import compute_metrics
from model_assessment.util.expanded_metrics import compute_metrics
from model_assessment.util.update_NOM import (
    canonical_nom_param,
    update_mptable,
    update_noahowp_model_params,
)
from model_assessment.configs import path_config as cfg

# =========================
# === CONFIGURATION =======
# =========================

n_particles = 15                 # particles per gage (adjust)
n_iterations = 50               # iterations (adjust)
max_cores_for_gages = 1        # one catchment per job by default
metric_to_calibrate_on = "kge"

# Run ALL particles concurrently per iteration (inner pool size). None => n_particles
max_particle_procs = None

# CASAM calibration is intentionally single-tile.
LEARN_TILE_WEIGHT_IF_2TILES = False

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


def ngen_time_string(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


set_time_windows()

project_root = cfg.project_root
sandbox_path = cfg.sandbox_path
logging_dir = cfg.logging_dir
observed_q_root = cfg.observed_q_root
model_roots = cfg.model_roots

HYDRO_MODEL_LABEL = "CASAM"
HYDRO_CONFIG_DIRNAME = "casam"
HYDRO_CONFIG_PREFIXES = ("casam_cfg_cat", "casam_config_cat")


def resolve_sandbox_config(path: str | None) -> str | None:
    if not path:
        return path
    return os.path.abspath(os.path.expandvars(path))


HYDRO_SANDBOX_CONFIG = os.environ.get(
    "NGEN_SANDBOX_CONFIG",
    os.path.join(project_root, "configs", "sandbox_config.yaml"),
)
HYDRO_SANDBOX_CONFIG = resolve_sandbox_config(HYDRO_SANDBOX_CONFIG)

def gage_logging_dir(gage_id: str) -> str:
    return cfg.gage_logging_dir(gage_id, model_roots[0], HYDRO_SANDBOX_CONFIG)


def gage_output_dir(root: str, gage_id: str) -> str:
    return cfg.gage_output_dir(gage_id, root, HYDRO_SANDBOX_CONFIG)

# =========================
# === NOM SETUP ===========
# =========================

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

# =========================
# === PARAMETER SELECTION ==
# =========================
# User-editable. Controls optimizer search space.
#
# Default request calibrates CASAM soil and scalar parameters:
# - soil: log_alpha for layer 1; log_Ks for layers 1 and 2
# - scalars: log10_a, b, frac_to_GW, spf_factor, and layer thickness for layers 1 and 2
# - Optional CASAM lateral-flow scalars can be uncommented; they are calibrated
#   in log10 space and applied model-wide.
#
# If NOM exists and DEFAULT_INCLUDE_NOM_IF_PRESENT=True, upstream NOM params are auto-appended.
#
# Per your note: the user specifies layers explicitly; we do not auto-expand by detected n_layers.

CALIBRATION_REQUEST = [
    # {"kind": "soil", "param": "log_alpha", "layers": [1, 2]},
    {"kind": "soil", "param": "log_alpha", "layers": [1]},
    # {"kind": "soil", "param": "n", "layers": [1, 2]},
    {"kind": "soil", "param": "log_Ks", "layers": [1, 2]},

    {"kind": "lasam", "param": "log10_a"},
    {"kind": "lasam", "param": "b"},
    {"kind": "lasam", "param": "frac_to_GW"},
    # {"kind": "lasam", "param": "log10_lateral_flow_psi_threshold"},
    # {"kind": "lasam", "param": "log10_lateral_flow_factor"},
    # {"kind": "lasam", "param": "field_capacity_psi"},
    {"kind": "lasam", "param": "spf_factor"},
    # {"kind": "lasam", "param": "theta_e_1"},
    {"kind": "lasam", "param": "layer_thickness", "layers": [1, 2]},
]

# If True and NOM exists in a tile workspace, NOM parameters are included by default
# (NOM-off => hydrology-only request; NOM-on => hydrology request plus upstream NOM knobs)
DEFAULT_INCLUDE_NOM_IF_PRESENT = True

# Default NOM list mirrors the upstream Noah-OWP calibration set when NOM exists.
DEFAULT_NOM_REQUEST = [
    {"kind": "nom", "param": "MFSNO"},
    {"kind": "nom", "param": "SCAMAX"},
    {"kind": "nom", "param": "RSURF_SNOW"},
    {"kind": "nom", "param": "CWP"},
    {"kind": "nom", "param": "VCMX25"},
    {"kind": "nom", "param": "MP"},
    {"kind": "nom", "param": "RSURF_EXP"},
]

# Bounds registry (extend freely as you add more knobs later)
BOUNDS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "soil": {
        "log_alpha": (-4.0, 0.0),
        "n": (1.02, 3.0),
        "log_Ks": (-4.0, 2.0),
        # Optional future:
        # "theta_e": (0.3, 0.6),
    },
    "lasam": {
        "log10_a": (-8.0, -1.0),
        "b": (0.01, 5.0),
        "frac_to_GW": (1e-4, 1.0 - 1e-4),
        "log10_lateral_flow_psi_threshold": (0.0, 4.0),
        "log10_lateral_flow_factor": (-3.0, 2.0),
        "field_capacity_psi": (10.0, 500.0),
        "spf_factor": (0.1, 1.0),
        "theta_e_1": (0.3, 0.6),
        "layer_thickness": (1.0, 1000.0),
    },
    "nom": dict(zip(nom_param_names, nom_param_bounds)),
}

# Per-layer bounds override for thickness (cm). If a layer isn't listed, fallback to BOUNDS["lasam"]["layer_thickness"].
LAYER_THICKNESS_BOUNDS = {
    1: (10.0, 100.0),     # top layer thinner
    2: (10.0, 400.0),  # bottom layer can be much thicker
}

# =========================
# === HELPERS ============
# =========================

def check_for_stop_signal_or_low_disk(threshold_gb: float = 0.0):
    stop_file = os.path.join(project_root, "STOP_NOW.txt")
    if os.path.exists(stop_file):
        print("Detected STOP_NOW.txt")
        sys.exit(1)
    st = os.statvfs("/")
    free_gb = (st.f_frsize * st.f_bavail) / (1024 ** 3)
    if free_gb < threshold_gb:
        print(f"Free disk space below threshold: {free_gb:.2f} GB")
        sys.exit(1)

def runtime_job_cores(default: int = 1) -> int:
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

def wall_time_log_fields(start_time: datetime, job_cores: int, particle_pool_size: int, final: bool = False) -> Dict[str, float]:
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

def runtime_cpu_pool(default: int = 1) -> int:
    for name in ("NGEN_TROUTE_CPU_POOL", "SLURM_CPUS_PER_TASK", "SLURM_NTASKS", "SLURM_NPROCS"):
        value = os.environ.get(name)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return max(1, int(default))

def pwork(root: str, gage_id: str, pid: int) -> str:
    """Particle workspace root created by `sandbox.py -conf --concurrent-particles`."""
    return os.path.join(gage_output_dir(root, gage_id), "particles", f"p{pid}")

def resolve_div_dir(tile_root: str, gage_id: str, pid: int) -> str:
    """Return the directory that actually contains CSV divide outputs for this tile & particle.
    Prefer particle dir; fall back to legacy non-particle dir if needed.
    """
    candidates = [
        os.path.join(pwork(tile_root, gage_id, pid), "outputs", "div"),
        os.path.join(gage_output_dir(tile_root, gage_id), "outputs", "div"),
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
        log_dir = gage_logging_dir(gage_id)

        with open(os.path.join(log_dir, f"{gage_id}_errors.log"), "a") as f:
            f.write(
                f"{datetime.now().isoformat()} | iter={iteration} | pid={particle_idx} "
                f"| stage={stage} | {err_msg}\n"
            )

        csv_path = os.path.join(log_dir, f"{gage_id}_incomplete.csv")
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
            for col in row.keys():
                if col not in prev.columns:
                    prev[col] = np.nan
            new = pd.DataFrame([row])
            for col in prev.columns:
                if col not in new.columns:
                    new[col] = np.nan
            pd.concat([prev, new[prev.columns]], ignore_index=True).to_csv(csv_path, index=False)
        else:
            pd.DataFrame([row]).to_csv(csv_path, index=False)
    except Exception:
        pass

# =========================
# === Realization retargeting (CASAM+PET+optional NOM)
# =========================

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

    configs_root = os.path.join(work_root, "configs")
    os.makedirs(configs_root, exist_ok=True)
    div_dir = os.path.join(work_root, "outputs", "div")
    os.makedirs(div_dir, exist_ok=True)

    cfg_model_dir = os.path.join(configs_root, HYDRO_CONFIG_DIRNAME)
    cfg_pet_dir = os.path.join(configs_root, "pet")
    cfg_nom_dir = os.path.join(configs_root, "noahowp")
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
            src = init_cfg if os.path.isabs(init_cfg) else os.path.join(base_out_dir, init_cfg)

            if any(alias in model_type for alias in ("CASAM", "LGAR")) or f"/configs/{HYDRO_CONFIG_DIRNAME}/" in init_cfg:
                os.makedirs(cfg_model_dir, exist_ok=True)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(cfg_model_dir, fname))
                p["init_config"] = os.path.abspath(os.path.join(cfg_model_dir, fname))

            elif "PET" in model_type or "/configs/pet/" in init_cfg:
                os.makedirs(cfg_pet_dir, exist_ok=True)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(cfg_pet_dir, fname))
                p["init_config"] = os.path.abspath(os.path.join(cfg_pet_dir, fname))

            elif any(k in model_type for k in ("NOM", "NOAH")) or "/noahowp/" in init_cfg:
                os.makedirs(cfg_nom_param, exist_ok=True)
                cfg_nom_namelist = os.path.join(cfg_nom_dir, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, cfg_nom_namelist)

                shared_nom_paramdir = os.path.join(base_out_dir, "configs", "noahowp", "parameters")
                for tbl in ("GENPARM.TBL", "SOILPARM.TBL"):
                    src_tbl = os.path.join(shared_nom_paramdir, tbl)
                    dst_tbl = os.path.join(cfg_nom_param, tbl)
                    if os.path.isfile(src_tbl) and not os.path.exists(dst_tbl):
                        try:
                            os.symlink(src_tbl, dst_tbl)
                        except OSError:
                            shutil.copy2(src_tbl, dst_tbl)

                mpt_src = os.path.join(shared_nom_paramdir, "MPTABLE.TBL")
                mpt_dst = os.path.join(cfg_nom_param, "MPTABLE.TBL")
                if os.path.isfile(mpt_src) and not os.path.isfile(mpt_dst):
                    shutil.copy2(mpt_src, mpt_dst)

                update_nom_namelist_paramdir(cfg_nom_namelist, cfg_nom_param)
                p["init_config"] = os.path.abspath(cfg_nom_namelist)

    rz["output_root"] = div_dir
    if "global" in rz:
        rz["global"]["output_root"] = div_dir

    with open(realization_path, "w") as f:
        json.dump(rz, f, indent=4)

    return div_dir

# =========================
# === Observations / metrics
# =========================

def get_observed_q(observed_path: str) -> pd.Series:
    df = pd.read_csv(observed_path, parse_dates=["value_time"]).set_index("value_time")
    return df["flow_m3_per_s"]

# =========================
# === PARAM SPEC SYSTEM ===
# =========================

@dataclass
class ParamSpec:
    name: str
    bounds: Tuple[float, float]
    init_value: float
    apply: Callable[["TileContext", float], None]

class TileContext:
    """Workspace context for a single tile workspace at (tile_root, gage_id, particle_id)."""

    def __init__(self, tile_root: str, gage_id: str, pid: int, work_root: str):
        self.tile_root = tile_root
        self.gage_id = gage_id
        self.pid = pid
        self.work_root = work_root

        self.lasam_cfg_dir = os.path.join(work_root, "configs", HYDRO_CONFIG_DIRNAME)
        self.lasam_cfg_files = sorted(
            f for f in os.listdir(self.lasam_cfg_dir)
            if any(f.startswith(prefix) for prefix in HYDRO_CONFIG_PREFIXES)
        )
        if not self.lasam_cfg_files:
            raise FileNotFoundError(f"No {HYDRO_MODEL_LABEL} configs found in {self.lasam_cfg_dir}")

        first_cfg = os.path.join(self.lasam_cfg_dir, self.lasam_cfg_files[0])
        with open(first_cfg, "r") as f:
            lines = f.readlines()

        soil_types_line = next(line for line in lines if line.strip().startswith("layer_soil_type="))
        self.soil_types: List[int] = list(map(int, soil_types_line.strip().split("=", 1)[1].split(",")))
        self.n_layers = len(self.soil_types)

        soil_file_line = next(line for line in lines if line.strip().startswith("soil_params_file"))
        soil_file = soil_file_line.split("=", 1)[1].strip()
        soil_path = Path(soil_file)
        if not soil_path.is_absolute():
            soil_path = (Path(first_cfg).parent / soil_path).resolve()
        self.src_soil_path = str(soil_path)
        self.local_soil_path = os.path.join(self.lasam_cfg_dir, Path(self.src_soil_path).name)

        self.nom_dir = os.path.join(work_root, "configs", "noahowp")
        self.include_nom = os.path.isdir(self.nom_dir)
        self.nom_mptable = os.path.join(self.nom_dir, "parameters", "MPTABLE.TBL")

        self._soil_lines_cache: Optional[List[str]] = None

    def ensure_local_soil(self):
        if not os.path.isfile(self.local_soil_path):
            if not os.path.isfile(self.src_soil_path):
                raise FileNotFoundError(f"soil_params_file not found: {self.src_soil_path}")
            shutil.copy2(self.src_soil_path, self.local_soil_path)

        for cfg_name in self.lasam_cfg_files:
            cfg_path = os.path.join(self.lasam_cfg_dir, cfg_name)
            with open(cfg_path, "r") as f:
                lines = f.readlines()
            out = []
            for line in lines:
                if line.strip().startswith("soil_params_file"):
                    out.append(f"soil_params_file={os.path.abspath(self.local_soil_path)}\n")
                else:
                    out.append(line)
            with open(cfg_path, "w") as f:
                f.writelines(out)

    def read_soil_lines(self) -> List[str]:
        if self._soil_lines_cache is not None:
            return self._soil_lines_cache
        if not os.path.isfile(self.local_soil_path):
            raise FileNotFoundError(f"Local soil file not found: {self.local_soil_path}")
        with open(self.local_soil_path, "r") as f:
            self._soil_lines_cache = f.readlines()
        return self._soil_lines_cache

    def write_soil_lines(self, lines: List[str]):
        with open(self.local_soil_path, "w") as f:
            f.writelines(lines)
        self._soil_lines_cache = lines

def read_layer_thickness_baseline(tile_ctx: TileContext) -> List[float]:
    """
    Reads 'layer_thickness=' from the first LASAM config in the workspace.
    Returns a list of floats (cm), e.g. [10.0, 190.0] or [190.0].
    """
    cfg_path = os.path.join(tile_ctx.lasam_cfg_dir, tile_ctx.lasam_cfg_files[0])
    with open(cfg_path, "r") as f:
        lines = f.readlines()

    line = next((ln for ln in lines if ln.strip().startswith("layer_thickness=")), None)
    if line is None:
        raise ValueError(f"Missing layer_thickness= in {cfg_path}")

    rhs = line.split("=", 1)[1].strip()
    rhs = rhs.split("[", 1)[0].strip()  # drop units like [cm]
    parts = [p.strip() for p in rhs.split(",") if p.strip()]
    return [float(p) for p in parts]

def apply_layer_thickness(tile_ctx: TileContext, layer_1based: int, value: float):
    """
    Updates the 'layer_thickness=' line in ALL LASAM configs in this workspace.
    Preserves any existing number of thickness entries; only replaces the selected layer index.
    Silently skips if layer index doesn't exist in the file.
    """
    for cfg_name in tile_ctx.lasam_cfg_files:
        cfg_path = os.path.join(tile_ctx.lasam_cfg_dir, cfg_name)
        with open(cfg_path, "r") as f:
            lines = f.readlines()

        out = []
        changed = False
        for line in lines:
            if line.strip().startswith("layer_thickness="):
                rhs = line.split("=", 1)[1].strip()

                # preserve unit suffix (e.g. "[cm]") if present
                unit = ""
                if "[" in rhs:
                    unit = "[" + rhs.split("[", 1)[1].strip()  # includes closing bracket
                vals_str = rhs.split("[", 1)[0].strip()

                parts = [p.strip() for p in vals_str.split(",") if p.strip()]
                vals = [float(p) for p in parts] if parts else []

                idx0 = layer_1based - 1
                if idx0 < 0 or idx0 >= len(vals):
                    # nothing to change (skip silently)
                    out.append(line)
                    continue

                vals[idx0] = float(value)
                joined = ",".join(f"{v:.6g}" for v in vals)
                newline = f"layer_thickness={joined}{unit if unit else '[cm]'}\n"
                out.append(newline)
                changed = True
            else:
                out.append(line)

        if changed:
            with open(cfg_path, "w") as f:
                f.writelines(out)

def read_lasam_scalar_baseline(tile_ctx: TileContext) -> Dict[str, float]:
    cfg_path = os.path.join(tile_ctx.lasam_cfg_dir, tile_ctx.lasam_cfg_files[0])
    with open(cfg_path, "r") as f:
        lines = f.readlines()

    def _get_float(prefix: str, default: Optional[float] = None) -> float:
        for line in lines:
            if line.strip().startswith(prefix):
                return float(line.split("=", 1)[1].strip().split("[")[0])
        if default is not None:
            return float(default)
        raise ValueError(f"Missing CASAM scalar config line: {prefix}")

    a = _get_float("a=")
    b = _get_float("b=")
    frac_to_GW = _get_float("frac_to_GW=")
    lateral_flow_psi_threshold = _get_float("lateral_flow_psi_threshold=", default=500.0)
    lateral_flow_factor = _get_float("lateral_flow_factor=", default=1.0)
    field_capacity_psi = _get_float("field_capacity_psi=")
    spf_factor = _get_float("spf_factor=")

    return {
        "log10_a": math.log10(a),
        "b": b,
        "frac_to_GW": frac_to_GW,
        "log10_lateral_flow_psi_threshold": math.log10(lateral_flow_psi_threshold),
        "log10_lateral_flow_factor": math.log10(lateral_flow_factor),
        "field_capacity_psi": field_capacity_psi,
        "spf_factor": spf_factor,
    }

def read_soil_layer_baseline(tile_ctx: TileContext, layer_1based: int) -> Dict[str, float]:
    if layer_1based < 1 or layer_1based > tile_ctx.n_layers:
        raise ValueError(f"Layer {layer_1based} out of range [1, {tile_ctx.n_layers}]")
    tile_ctx.ensure_local_soil()
    soil_lines = tile_ctx.read_soil_lines()
    soil_type = tile_ctx.soil_types[layer_1based - 1]
    toks = soil_lines[soil_type].split()
    theta_e = float(toks[2])
    alpha = float(toks[3])
    n = float(toks[4])
    Ks = float(toks[5])
    return {
        "theta_e": theta_e,
        "log_alpha": math.log10(alpha),
        "n": n,
        "log_Ks": math.log10(Ks),
    }

def read_nom_baseline(tile_ctx: TileContext) -> Dict[str, float]:
    if not tile_ctx.include_nom:
        return {}
    if not os.path.isfile(tile_ctx.nom_mptable):
        raise FileNotFoundError(f"NOM MPTABLE missing: {tile_ctx.nom_mptable}")

    with open(tile_ctx.nom_mptable, "r") as f:
        lines = f.readlines()

    vals: Dict[str, float] = {}
    for line in lines:
        if "=" not in line or line.strip().startswith("!"):
            continue
        key, value = line.split("=", 1)
        param = key.strip()
        requested = canonical_nom_param(param, nom_param_names)
        if requested is not None:
            value_str = value.split("!")[0]
            first = [v.strip() for v in value_str.split(",") if v.strip()][0]
            vals[requested] = float(first)
    for param, init in nom_param_initials.items():
        value = vals.get(param, float(init))
        low, high = nom_param_bounds_by_name[param]
        if value < low or value > high:
            value = float(init)
        vals[param] = value
    return vals

def apply_lasam_scalar(tile_ctx: TileContext, param: str, value: float):
    if param == "log10_a":
        a = 10 ** float(value)
        key = "a="
        out_line = f"a={a}\n"
    elif param == "b":
        key = "b="
        out_line = f"b={float(value)}\n"
    elif param == "frac_to_GW":
        key = "frac_to_GW="
        out_line = f"frac_to_GW={float(value)}\n"
    elif param == "log10_lateral_flow_psi_threshold":
        lateral_flow_psi_threshold = 10 ** float(value)
        key = "lateral_flow_psi_threshold="
        out_line = f"lateral_flow_psi_threshold={lateral_flow_psi_threshold}\n"
    elif param == "log10_lateral_flow_factor":
        lateral_flow_factor = 10 ** float(value)
        key = "lateral_flow_factor="
        out_line = f"lateral_flow_factor={lateral_flow_factor}\n"
    elif param == "field_capacity_psi":
        key = "field_capacity_psi="
        out_line = f"field_capacity_psi={float(value)}[cm]\n"
    elif param == "spf_factor":
        key = "spf_factor="
        out_line = f"spf_factor={float(value)}\n"
    else:
        raise ValueError(f"Unknown LASAM scalar param: {param}")

    for cfg_name in tile_ctx.lasam_cfg_files:
        cfg_path = os.path.join(tile_ctx.lasam_cfg_dir, cfg_name)
        with open(cfg_path, "r") as f:
            lines = f.readlines()
        out = []
        changed = False
        for line in lines:
            if line.strip().startswith(key):
                out.append(out_line)
                changed = True
            else:
                out.append(line)
        if not changed:
            out.append(out_line)
        with open(cfg_path, "w") as f:
            f.writelines(out)

def apply_soil_param(tile_ctx: TileContext, layer_1based: int, param: str, value: float):
    # Per your earlier script's behavior, skip silently if the layer is not present
    if layer_1based < 1 or layer_1based > tile_ctx.n_layers:
        return

    tile_ctx.ensure_local_soil()
    soil_lines = tile_ctx.read_soil_lines()
    soil_type = tile_ctx.soil_types[layer_1based - 1]
    toks = soil_lines[soil_type].split()

    if param == "log_alpha":
        alpha = 10 ** float(value)
        toks[3] = str(alpha)
    elif param == "n":
        toks[4] = str(float(value))
    elif param == "log_Ks":
        Ks = 10 ** float(value)
        toks[5] = str(Ks)
    elif param == "theta_e":
        toks[2] = str(float(value))
    else:
        raise ValueError(f"Unknown soil param: {param}")

    soil_lines[soil_type] = "\t".join(toks) + "\n"
    tile_ctx.write_soil_lines(soil_lines)

SOIL_TABLE_COLUMN_BY_PARAM = {
    "theta_r": 1,
    "theta_e": 2,
    "log_alpha": 3,
    "alpha": 3,
    "n": 4,
    "log_Ks": 5,
    "Ks": 5,
}

def calibrated_soil_columns_for_layer(specs: List[ParamSpec], layer_1based: int) -> set:
    calibrated = set()
    for spec in specs:
        name = spec.name
        if name == "theta_e_1":
            if layer_1based == 1:
                calibrated.add(SOIL_TABLE_COLUMN_BY_PARAM["theta_e"])
            continue
        if "_L" not in name:
            continue
        param, layer_str = name.rsplit("_L", 1)
        try:
            layer = int(layer_str)
        except ValueError:
            continue
        if layer != layer_1based:
            continue
        col = SOIL_TABLE_COLUMN_BY_PARAM.get(param)
        if col is not None:
            calibrated.add(col)
    return calibrated

def mirror_second_layer_uncalibrated_soil_params(tile_ctx: TileContext, specs: List[ParamSpec]):
    """
    CASAM uses two soil layers but a shared soil table. For the second layer,
    copy every non-calibrated soil-table parameter from the first layer after
    particle parameters have been written, so any calibrated L1 value is mirrored
    unless the corresponding L2 parameter is explicitly calibrated.
    """
    if tile_ctx.n_layers < 2:
        return

    top_soil_type = tile_ctx.soil_types[0]
    second_soil_type = tile_ctx.soil_types[1]
    if top_soil_type == second_soil_type:
        return

    tile_ctx.ensure_local_soil()
    soil_lines = tile_ctx.read_soil_lines()
    if top_soil_type >= len(soil_lines) or second_soil_type >= len(soil_lines):
        raise IndexError(
            f"CASAM soil type out of range: L1={top_soil_type}, L2={second_soil_type}, "
            f"soil table rows={len(soil_lines)}"
        )

    top_toks = soil_lines[top_soil_type].split()
    second_toks = soil_lines[second_soil_type].split()
    max_param_col = min(len(top_toks), len(second_toks)) - 1
    if max_param_col < 1:
        return

    calibrated_l2_cols = calibrated_soil_columns_for_layer(specs, layer_1based=2)
    for col in range(1, max_param_col + 1):
        if col in calibrated_l2_cols:
            continue
        second_toks[col] = top_toks[col]

    soil_lines[second_soil_type] = "\t".join(second_toks) + "\n"
    tile_ctx.write_soil_lines(soil_lines)

def apply_theta_e_1(tile_ctx: TileContext, value: float):
    apply_soil_param(tile_ctx, layer_1based=1, param="theta_e", value=float(value))

def apply_nom_param(tile_ctx: TileContext, param: str, value: float):
    if not tile_ctx.include_nom:
        return
    if param not in nom_param_names:
        raise ValueError(f"Unknown NOM param: {param}")
    if not os.path.isfile(tile_ctx.nom_mptable):
        raise FileNotFoundError(f"NOM MPTABLE missing: {tile_ctx.nom_mptable}")

    update_mptable(
        original_file=tile_ctx.nom_mptable,
        output_file=tile_ctx.nom_mptable,
        updated_params={param: float(value)},
        verbose=False,
    )

def build_specs_for_tile(tile_ctx: TileContext, tile_idx: int) -> List[ParamSpec]:
    specs: List[ParamSpec] = []
    lasam_base = read_lasam_scalar_baseline(tile_ctx)

    request_list = list(CALIBRATION_REQUEST)
    if DEFAULT_INCLUDE_NOM_IF_PRESENT and tile_ctx.include_nom:
        existing = {(r.get("kind", "").strip().lower(), r.get("param", "").strip()) for r in request_list}
        for r in DEFAULT_NOM_REQUEST:
            key = (r["kind"], r["param"])
            if key not in existing:
                request_list.append(r)

    nom_base = read_nom_baseline(tile_ctx) if tile_ctx.include_nom else {}

    for req in request_list:
        kind = req["kind"].strip().lower()
        param = req["param"].strip()

        if kind == "soil":
            layers = req.get("layers", [])
            if not isinstance(layers, list) or len(layers) == 0:
                raise ValueError(f"Soil request for {param} must include non-empty layers=[...]")
            if param not in BOUNDS["soil"]:
                raise ValueError(f"No bounds registered for soil param: {param}")

            for L in layers:
                L = int(L)
                if L < 1 or L > tile_ctx.n_layers:
                    # skip silently (mixed-layer tiles)
                    continue
                base = read_soil_layer_baseline(tile_ctx, L)
                if param not in base:
                    raise ValueError(f"Baseline missing soil param {param} for layer {L}")
                name = f"{param}_L{L}"
                bnd = BOUNDS["soil"][param]
                init = float(base[param])
                specs.append(
                    ParamSpec(
                        name=name,
                        bounds=bnd,
                        init_value=init,
                        apply=(lambda ctx, v, L=L, p=param: apply_soil_param(ctx, L, p, v))
                    )
                )

        elif kind == "lasam":
            if param not in BOUNDS["lasam"]:
                raise ValueError(f"No bounds registered for lasam param: {param}")

            # NEW: layer_thickness supports per-layer selection
            if param == "layer_thickness":
                layers = req.get("layers", [])
                if not isinstance(layers, list) or len(layers) == 0:
                    raise ValueError("lasam layer_thickness request must include non-empty layers=[...]")

                base_th = read_layer_thickness_baseline(tile_ctx)  # list of floats from file
                for L in layers:
                    L = int(L)
                    idx0 = L - 1
                    if idx0 < 0 or idx0 >= len(base_th):
                        # skip silently (user may ask for L2 but file is 1-layer)
                        continue

                    specs.append(
                        ParamSpec(
                            name=f"layer_thickness_L{L}",
                            bounds=LAYER_THICKNESS_BOUNDS.get(L, BOUNDS["lasam"][param]),
                            init_value=float(base_th[idx0]),
                            apply=(lambda ctx, v, L=L: apply_layer_thickness(ctx, L, v)),
                        )
                    )

                continue  # done handling thickness

            # existing theta_e_1 special-case
            if param == "theta_e_1":
                base1 = read_soil_layer_baseline(tile_ctx, 1)
                init = float(base1["theta_e"])
                specs.append(
                    ParamSpec(
                        name="theta_e_1",
                        bounds=BOUNDS["lasam"][param],
                        init_value=init,
                        apply=(lambda ctx, v: apply_theta_e_1(ctx, v))
                    )
                )
            else:
                if param not in lasam_base:
                    raise ValueError(f"Baseline missing lasam param: {param}")
                specs.append(
                    ParamSpec(
                        name=param,
                        bounds=BOUNDS["lasam"][param],
                        init_value=float(lasam_base[param]),
                        apply=(lambda ctx, v, p=param: apply_lasam_scalar(ctx, p, v))
                    )
                )


        elif kind == "nom":
            if param not in BOUNDS["nom"]:
                raise ValueError(f"No bounds registered for NOM param: {param}")
            if not tile_ctx.include_nom:
                continue
            if param not in nom_base:
                raise ValueError(f"NOM param {param} not found in MPTABLE: {tile_ctx.nom_mptable}")
            specs.append(
                ParamSpec(
                    name=param,
                    bounds=BOUNDS["nom"][param],
                    init_value=float(nom_base[param]),
                    apply=(lambda ctx, v, p=param: apply_nom_param(ctx, p, v))
                )
            )

        else:
            raise ValueError(f"Unknown calibration kind: {kind}")

    return specs

def flatten_specs_for_all_tiles(
    gage_id: str,
    model_roots_list: List[str],
) -> Tuple[List[List[ParamSpec]], List[int], List[Tuple[float, float]], List[float], List[str]]:
    specs_by_tile: List[List[ParamSpec]] = []
    tile_counts: List[int] = []
    bounds_all: List[Tuple[float, float]] = []
    init_all: List[float] = []
    names_all: List[str] = []

    for tile_idx, tile_root in enumerate(model_roots_list):
        work_root = pwork(tile_root, gage_id, 0)
        ctx0 = TileContext(tile_root, gage_id, 0, work_root)
        specs = build_specs_for_tile(ctx0, tile_idx)

        specs_by_tile.append(specs)
        tile_counts.append(len(specs))

        suffix = f"_tile{tile_idx+1}"
        for s in specs:
            bounds_all.append(s.bounds)
            init_all.append(s.init_value)
            names_all.append(s.name + suffix)

    return specs_by_tile, tile_counts, bounds_all, init_all, names_all

def apply_particle_params_for_tile(tile_ctx: TileContext, specs: List[ParamSpec], values: np.ndarray):
    needs_soil = any(
        s.name.startswith(("log_alpha_L", "n_L", "log_Ks_L", "theta_e_1")) or s.name.startswith("theta_e_L")
        for s in specs
    )
    if needs_soil:
        tile_ctx.ensure_local_soil()

    for spec, v in zip(specs, values):
        spec.apply(tile_ctx, float(v))

    mirror_second_layer_uncalibrated_soil_params(tile_ctx, specs)

def nom_updates_from_specs(specs: List[ParamSpec], values: np.ndarray) -> Dict[str, float]:
    return {
        spec.name: float(value)
        for spec, value in zip(specs, values)
        if spec.name in BOUNDS["nom"]
    }

# =========================
# === Objective function ===
# =========================

def _safe_objective(args):
    try:
        obj, val_metrics, cal_metrics = objective_function_tiled(args)
        return ("OK", "", obj, val_metrics, cal_metrics)
    except Exception as e:
        (
            params, particle_idx, gage_id,
            model_roots_list, observed_q_root_local,
            specs_by_tile, tile_counts, learn_tile_weight,
            weights_in,
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
        model_roots_list, observed_q_root_local,
        specs_by_tile, tile_counts, learn_tile_weight,
        weights,
        *extra  # (iteration, param_names)
    ) = args

    check_for_stop_signal_or_low_disk()
    n_tiles = len(model_roots_list)

    # Tile weight handling (if enabled and 2 tiles)
    if n_tiles == 2 and learn_tile_weight:
        tile_weight = float(params[-1])
        weights = [tile_weight, 1.0 - tile_weight]
        params = params[:-1]
    elif weights is None:
        weights = [1.0 / n_tiles] * n_tiles

    # === STEP 1: Hydrology per tile in PARTICLE workspace ===
    offset = 0
    for tile_idx, tile_root in enumerate(model_roots_list):
        n = tile_counts[tile_idx]
        tile_vals = np.array(params[offset:offset + n], dtype=float)
        offset += n

        work_root = pwork(tile_root, gage_id, particle_idx)
        cfg_dir_lsm = os.path.join(work_root, "configs", HYDRO_CONFIG_DIRNAME)
        json_dir = os.path.join(work_root, "json")
        os.makedirs(cfg_dir_lsm, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

        # Find particle-local realization.json
        json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
        if not json_files:
            raise FileNotFoundError(f"No realization JSON found in {json_dir}")
        realization_path = os.path.join(json_dir, sorted(json_files)[0])  # FIXED: os.patho -> os.path

        # Retarget realization paths into particle workspace (CASAM/PET/NOM)
        base_out_dir = gage_output_dir(tile_root, gage_id)
        retarget_realization_paths(realization_path, work_root, base_out_dir)

        # Clamp time window to spinup->cal_end for calibration runs
        with open(realization_path, "r") as f:
            realization = json.load(f)
        realization["time"]["start_time"] = ngen_time_string(spinup_start)
        realization["time"]["end_time"] = ngen_time_string(cal_end)
        with open(realization_path, "w") as f:
            json.dump(realization, f, indent=4)

        # Apply parameter set to particle-local CASAM/NOM files
        tile_ctx = TileContext(tile_root, gage_id, particle_idx, work_root)
        apply_particle_params_for_tile(tile_ctx, specs_by_tile[tile_idx], tile_vals)
        update_noahowp_model_params(
            realization_path,
            nom_updates_from_specs(specs_by_tile[tile_idx], tile_vals),
        )

        # Clear old divide outputs in particle workspace
        div_dir = os.path.join(work_root, "outputs", "div")
        os.makedirs(div_dir, exist_ok=True)
        for item in list(os.listdir(div_dir)):
            if item.startswith("."):
                continue
            pth = os.path.join(div_dir, item)
            if os.path.isfile(pth) or os.path.islink(pth):
                os.remove(pth)
            elif os.path.isdir(pth):
                shutil.rmtree(pth)

        # Run hydrology (divide scale) for this tile & particle
        tile_sandbox_config = HYDRO_SANDBOX_CONFIG
        env = os.environ.copy()
        env["NGEN_CONCURRENT_PARTICLES"] = "1"
        env["NGEN_PARTICLE_ID"] = str(particle_idx)
        env["NGEN_REALIZATION_PATH"] = realization_path

        ret = subprocess.call(
            [sys.executable, sandbox_path, "-i", tile_sandbox_config, "-run", "--gage_id", gage_id],
            cwd=tile_root,
            env=env,
        )
        if ret != 0:
            raise RuntimeError(f"Hydrology failed: gage {gage_id} | pid {particle_idx} | tile {tile_idx}")

    # === STEP 2: Weighted qlat build in router tile particle workspace ===
    router_tile_root = model_roots_list[0]
    router_work = pwork(router_tile_root, gage_id, particle_idx)

    weighted_div_dir = os.path.join(router_work, "outputs", "div_weighted")
    if os.path.exists(weighted_div_dir):
        shutil.rmtree(weighted_div_dir)
    os.makedirs(weighted_div_dir, exist_ok=True)

    src_div_dirs = [resolve_div_dir(root, gage_id, particle_idx) for root in model_roots_list]

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

    # Build weighted qlat (single tile -> copy)
    if len(model_roots_list) == 1:
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
                        df = pd.read_csv(fp, header=None)
                        df.columns = ["Time Step", "Time", "q_out"]
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
    troute_cfg["compute_parameters"]["cpu_pool"] = runtime_cpu_pool(default=1)
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
        orig_mask = os.path.join(gage_output_dir(router_tile_root, gage_id), "configs", "mask_output.yaml")
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
    subprocess.call([sys.executable, "-m", "nwm_routing", "-f", "-V4", troute_path], env=env)

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

    subprocess.call(
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

    # === STEP 5: Metrics ===
    sim_df = (
        pd.read_csv(output_path, parse_dates=["current_time"])
        .set_index("current_time")["flow"]
        .resample("1h")
        .mean()
    )
    obs_path = os.path.join(observed_q_root_local, "successful_sites_resampled", f"{gage_id}.csv")
    obs_df = get_observed_q(obs_path)

    sim_cal, obs_cal = sim_df[cal_start:cal_end].dropna(), obs_df[cal_start:cal_end].dropna()
    sim_val, obs_val = sim_df[val_start:val_end].dropna(), obs_df[val_start:val_end].dropna()
    sim_cal, obs_cal = sim_cal.align(obs_cal, join="inner")
    sim_val, obs_val = sim_val.align(obs_val, join="inner")

    if len(sim_cal) > 0:
        sim_cal.iloc[-1] += 1e-8
    if len(obs_cal) > 0:
        obs_cal.iloc[-1] += 1e-8

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

# =========================
# === PSO implementation ===
# =========================

class Particle:
    def __init__(self, bounds, init_position=None):
        self.position = np.array(init_position) if init_position is not None else np.array(
            [np.random.uniform(low, high) for low, high in bounds]
        )
        self.velocity = np.array([0.1 * (high - low) * np.random.uniform(-1, 1) for low, high in bounds])
        self.best_position = np.copy(self.position)
        self.best_value = float("inf")
        self.current_value = float("inf")
        self.stagnation_counter = 0

    def reset(self, bounds):
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.array([0.1 * (high - low) * np.random.uniform(-1, 1) for low, high in bounds])
        self.best_position = np.copy(self.position)
        self.best_value = float("inf")
        self.current_value = float("inf")
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
                self.position[i] = bounds[i][0]
                self.velocity[i] *= -0.5
            elif self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]
                self.velocity[i] *= -0.5

class PSO:
    def __init__(
        self,
        n_particles,
        bounds,
        n_iterations,
        gage_id,
        init_position,
        metric_to_calibrate_on="kge",
        param_names=None,
        specs_by_tile=None,
        tile_counts=None,
        learn_tile_weight=False,
        stagnation_threshold=10,
    ):
        self.particles = [
            Particle(bounds, init_position=init_position if i == 0 else None)
            for i in range(n_particles)
        ]
        self.bounds = bounds
        self.n_iterations = n_iterations
        self.gage_id = gage_id
        self.global_best_position = np.copy(self.particles[0].position)
        self.global_best_value = float("inf")
        self.metric_to_calibrate_on = metric_to_calibrate_on
        self.param_names = param_names or [f"p{i}" for i in range(len(bounds))]
        self.best_cal_metrics = {}
        self.best_val_metrics = {}
        self.stagnation_threshold = stagnation_threshold

        self.specs_by_tile = specs_by_tile or []
        self.tile_counts = tile_counts or []
        self.learn_tile_weight = learn_tile_weight

    def optimize(self):
        start_time = datetime.now()
        log_rows = []
        log_path = os.path.join(gage_logging_dir(self.gage_id), f"{self.gage_id}.csv")
        w_start, w_end = 0.9, 0.4
        pool_size = max_particle_procs or len(self.particles)
        job_cores = runtime_job_cores(default=pool_size)

        for iteration in range(self.n_iterations):
            print(f"\n--- Iteration {iteration + 1} for gage {self.gage_id} ---")
            w = w_start - (w_start - w_end) * (iteration / self.n_iterations)

            args_list = []
            for i, p in enumerate(self.particles):
                print(f"[INFO] gage {self.gage_id} | iter {iteration+1} | particle {i}")
                args_list.append((
                    p.position, i, self.gage_id,
                    model_roots, observed_q_root,
                    self.specs_by_tile, self.tile_counts, self.learn_tile_weight,
                    [1.0 / len(model_roots)] * len(model_roots),
                    iteration + 1,
                    self.param_names
                ))

            with ThreadPool(processes=pool_size) as pool:
                results = pool.map(_safe_objective, args_list)

            for idx, r in enumerate(results):
                status, err, objective_value, val_metrics, cal_metrics = r
                particle = self.particles[idx]
                particle.current_value = objective_value

                metric_cal = cal_metrics.get(self.metric_to_calibrate_on, np.nan)
                metric_val = val_metrics.get(self.metric_to_calibrate_on, np.nan)

                kge_cal = cal_metrics.get("kge", np.nan)
                kge_val = val_metrics.get("kge", np.nan)

                mappe_cal = cal_metrics.get("mappe", np.nan)
                mappe_val = val_metrics.get("mappe", np.nan)

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

                row = {
                    "iteration": iteration + 1,
                    "particle": idx,
                    **{name: val for name, val in zip(self.param_names, particle.position)},
                    f"{self.metric_to_calibrate_on}_calibration": metric_cal,
                    f"{self.metric_to_calibrate_on}_validation": metric_val,

                    # Added components for Pareto plotting
                    "kge_calibration": kge_cal,
                    "kge_validation": kge_val,
                    "mappe_calibration": mappe_cal,
                    "mappe_validation": mappe_val,

                    "status": status,
                    "error": (err or "")[:240],
                    **wall_time_log_fields(start_time, job_cores, pool_size),
                }

                log_rows.append(row)

            try:
                current_objs = [p.current_value for p in self.particles]
                best_idx_now = int(np.nanargmin(current_objs))
            except Exception:
                best_idx_now = None

            for i, p in enumerate(self.particles):
                if p.stagnation_counter >= self.stagnation_threshold and (best_idx_now is None or i != best_idx_now):
                    print(f"Resetting particle {i} after {self.stagnation_threshold} stagnant iterations.")
                    log_rows.append({
                        "iteration": iteration + 1,
                        "particle": i,
                        "status": "RESET",
                        "reason": f"stagnation >= {self.stagnation_threshold}",
                        **wall_time_log_fields(start_time, job_cores, pool_size),
                    })
                    p.reset(self.bounds)

            pd.DataFrame(log_rows).to_csv(log_path, index=False)

            for p in self.particles:
                p.update_velocity(self.global_best_position, w=w, c1=1.5, c2=1.5)
                p.update_position(self.bounds)

            print(f"Global best objective so far: {-self.global_best_value:.4f}")

        # === Final full-period run using BEST position, reusing pid=0 workspace (preserves original behavior) ===
        print(f"\n[INFO] Final full-period validation for gage {self.gage_id}...")
        best_pid = 0

        n_tiles = len(model_roots)
        weights = [1.0 / n_tiles] * n_tiles
        best_params = np.copy(self.global_best_position)

        if n_tiles == 2 and self.learn_tile_weight:
            wgt = float(best_params[-1])
            weights = [wgt, 1.0 - wgt]
            best_params = best_params[:-1]

        offset = 0
        for tile_idx, tile_root in enumerate(model_roots):
            n = self.tile_counts[tile_idx]
            tile_vals = np.array(best_params[offset:offset + n], dtype=float)
            offset += n

            work_root = pwork(tile_root, self.gage_id, best_pid)
            cfg_dir_lsm = os.path.join(work_root, "configs", HYDRO_CONFIG_DIRNAME)
            json_dir = os.path.join(work_root, "json")
            os.makedirs(cfg_dir_lsm, exist_ok=True)
            os.makedirs(json_dir, exist_ok=True)

            json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
            if not json_files:
                raise FileNotFoundError(f"No realization JSON found in {json_dir}")
            realization_path = os.path.join(json_dir, sorted(json_files)[0])

            base_out_dir = gage_output_dir(tile_root, self.gage_id)
            retarget_realization_paths(realization_path, work_root, base_out_dir)

            with open(realization_path, "r") as f:
                realization = json.load(f)
            realization["time"]["start_time"] = ngen_time_string(spinup_start)
            realization["time"]["end_time"] = ngen_time_string(val_end)
            with open(realization_path, "w") as f:
                json.dump(realization, f, indent=4)

            tile_ctx = TileContext(tile_root, self.gage_id, best_pid, work_root)
            apply_particle_params_for_tile(tile_ctx, self.specs_by_tile[tile_idx], tile_vals)
            update_noahowp_model_params(
                realization_path,
                nom_updates_from_specs(self.specs_by_tile[tile_idx], tile_vals),
            )

            div_dir = os.path.join(work_root, "outputs", "div")
            os.makedirs(div_dir, exist_ok=True)
            for item in list(os.listdir(div_dir)):
                if item.startswith("."):
                    continue
                pth = os.path.join(div_dir, item)
                if os.path.isfile(pth) or os.path.islink(pth):
                    os.remove(pth)
                elif os.path.isdir(pth):
                    shutil.rmtree(pth)

            tile_sandbox_config = HYDRO_SANDBOX_CONFIG
            env = os.environ.copy()
            env["NGEN_CONCURRENT_PARTICLES"] = "1"
            env["NGEN_PARTICLE_ID"] = str(best_pid)
            env["NGEN_REALIZATION_PATH"] = realization_path

            ret = subprocess.call(
                [sys.executable, sandbox_path, "-i", tile_sandbox_config, "-run", "--gage_id", self.gage_id],
                cwd=tile_root,
                env=env
            )
            if ret != 0:
                raise RuntimeError(f"Final hydrology failed: gage {self.gage_id} | tile {tile_idx}")

        router_tile_root = model_roots[0]
        router_work = pwork(router_tile_root, self.gage_id, best_pid)

        weighted_div_dir = os.path.join(router_work, "outputs", "div_weighted")
        if os.path.exists(weighted_div_dir):
            shutil.rmtree(weighted_div_dir)
        os.makedirs(weighted_div_dir, exist_ok=True)

        src_div_dirs = [resolve_div_dir(root, self.gage_id, best_pid) for root in model_roots]
        files = []
        for d in src_div_dirs:
            if os.path.isdir(d):
                cand = [f for f in os.listdir(d) if (f.startswith("cat-") or f.startswith("nex-")) and f.endswith(".csv")]
                if cand:
                    files = cand
                    break
        if not files:
            print("[ERROR] No divide CSVs found for final routing.")
            return

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
                            df = pd.read_csv(fp, header=None)
                            df.columns = ["Time Step", "Time", "q_out"]
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

        troute_path = os.path.join(router_work, "configs", "troute_config.yaml")
        with open(troute_path) as f:
            troute_cfg = yaml.safe_load(f)

        nts_full = int((val_end - spinup_start) / pd.Timedelta(seconds=300))
        troute_cfg["compute_parameters"]["cpu_pool"] = runtime_cpu_pool(default=1)
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
            orig_mask = os.path.join(gage_output_dir(router_tile_root, self.gage_id), "configs", "mask_output.yaml")
            if os.path.isfile(orig_mask):
                so["mask_output"] = orig_mask
            else:
                so.pop("mask_output", None)

        with open(troute_path, "w") as f:
            yaml.safe_dump(troute_cfg, f)

        for fn in os.listdir(particle_troute_dir):
            if fn.endswith((".nc", ".csv", ".parquet")):
                os.remove(os.path.join(particle_troute_dir, fn))

        env = os.environ.copy()
        env["NGEN_CONCURRENT_PARTICLES"] = "1"
        env["NGEN_PARTICLE_ID"] = str(best_pid)
        subprocess.call([sys.executable, "-m", "nwm_routing", "-f", "-V4", troute_path], env=env)

        postproc_dir = os.path.join(router_work, "postproc")
        os.makedirs(postproc_dir, exist_ok=True)
        final_output_path = os.path.join(postproc_dir, f"{self.gage_id}_best.csv")

        get_hydrograph_path = os.path.join(project_root, "model_assessment", "util", "get_hydrograph.py")
        summary_csv = os.environ.get(
            "DOWNSTREAM_FLOWPATH_SUMMARY",
            os.path.join(project_root, "model_assessment", "util", "downstream_flowpath_summary.csv"),
        )

        subprocess.call(
            [
                sys.executable, get_hydrograph_path,
                "--gage_id", self.gage_id,
                "--output", final_output_path,
                "--troute_dir", os.path.join(router_work, "troute"),
                "--summary", summary_csv,
            ],
            cwd=postproc_dir,
            env=env,
        )

        obs_path = os.path.join(observed_q_root, "successful_sites_resampled", f"{self.gage_id}.csv")
        obs_df = get_observed_q(obs_path)
        sim_df = pd.read_csv(final_output_path, parse_dates=["current_time"]).set_index("current_time")["flow"].resample("1h").mean()

        sim_val, obs_val = sim_df[val_start:val_end].dropna(), obs_df[val_start:val_end].dropna()
        sim_val, obs_val = sim_val.align(obs_val, join="inner")
        if len(sim_val) > 0:
            sim_val.iloc[-1] += 1e-8
        if len(obs_val) > 0:
            obs_val.iloc[-1] += 1e-8

        val_metrics_final = compute_metrics(sim_val, obs_val, event_threshold=1e-2)

        final_row = {
            "iteration": "FINAL",
            "particle": "BEST",
            **{name: val for name, val in zip(self.param_names, self.global_best_position)},
            f"{self.metric_to_calibrate_on}_calibration": self.best_cal_metrics.get(self.metric_to_calibrate_on, np.nan),
            f"{self.metric_to_calibrate_on}_validation": val_metrics_final.get(self.metric_to_calibrate_on, np.nan),
            "status": "OK",
            "error": "",

            # Added components
            "kge_calibration": self.best_cal_metrics.get("kge", np.nan),
            "kge_validation": val_metrics_final.get("kge", np.nan),
            "mappe_calibration": self.best_cal_metrics.get("mappe", np.nan),
            "mappe_validation": val_metrics_final.get("mappe", np.nan),
            **wall_time_log_fields(start_time, job_cores, pool_size, final=True),
        }

        existing = pd.DataFrame()
        if os.path.isfile(log_path):
            existing = pd.read_csv(log_path)
        pd.concat([existing, pd.DataFrame([final_row])]).to_csv(log_path, index=False)

        return

# =========================
# === Per-gage wrapper ===
# =========================

def calibrate_gage(gage_id: str):
    specs_by_tile, tile_counts, all_bounds, all_init_params, names = flatten_specs_for_all_tiles(gage_id, model_roots)

    learn_tile_weight = (len(model_roots) == 2 and LEARN_TILE_WEIGHT_IF_2TILES)
    if learn_tile_weight:
        all_init_params.append(0.8)
        all_bounds.append((0.0, 1.0))
        names.append("tile_weight")

    pso = PSO(
        n_particles=n_particles,
        bounds=all_bounds,
        n_iterations=n_iterations,
        gage_id=gage_id,
        init_position=all_init_params,
        metric_to_calibrate_on=metric_to_calibrate_on,
        param_names=names,
        specs_by_tile=specs_by_tile,
        tile_counts=tile_counts,
        learn_tile_weight=learn_tile_weight,
        stagnation_threshold=10,
    )
    pso.optimize()

# =========================
# === MAIN ===
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="Run single-tile CASAM PSO calibration.")
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
    from multiprocessing import get_context

    args = parse_args()

    n_particles = args.n_particles
    n_iterations = args.n_iterations
    max_particle_procs = args.max_particle_procs
    max_cores_for_gages = args.max_gage_procs
    if args.sandbox_config:
        HYDRO_SANDBOX_CONFIG = resolve_sandbox_config(args.sandbox_config)
        os.environ["NGEN_SANDBOX_CONFIG"] = HYDRO_SANDBOX_CONFIG
    set_time_windows({key: getattr(args, key) for key in TIME_FIELDS})

    if len(model_roots) != 1:
        raise RuntimeError(
            f"{HYDRO_MODEL_LABEL} calibration expects exactly one model root; "
            f"path_config.model_roots has {len(model_roots)} entries."
        )

    start = datetime.now()
    if args.gage_id:
        gage_list = [str(args.gage_id).strip()]
    else:
        gage_list = pd.read_csv(cfg.gages_file, dtype={"gage_id": str})["gage_id"].tolist()

    if len(gage_list) == 1:
        calibrate_gage(gage_list[0])
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=max_cores_for_gages) as pool:
            pool.map(calibrate_gage, gage_list)

    print(f"Total wall time: {datetime.now() - start}")
