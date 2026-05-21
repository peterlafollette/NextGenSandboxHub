#!/usr/bin/env python3
"""
Run two archived LASAM parameter sets once per gage using the particle workspaces
created by `sandbox.py -conf --concurrent-particles --num-particles 2`.

Behavior:
- p0 gets the row with best calibration KGE (max of kge_calibration)
- p1 gets the row with best calibration MAPPE (min of mappe_calibration)
- writes parameters into particle-local LASAM / NOM configs
- runs hydrology + routing once for each particle
- extracts and saves both hydrographs separately
- does NOT run a validation-only or final-best consolidation step
- does NOT delete the final routed outputs or hydrographs
- supports parallelism across gages via --n-workers

Typical use:
    python model_assessment/util/run_best_kge_and_mappe_lasam_parallel.py \
      --params-root /users/4/plafolle/params_for_ensemble/CIROH_project12 \
      --project-root /users/4/plafolle/CIROH_project12/NextGenSandboxHub \
      --n-workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

np.random.seed(42)
random.seed(42)

NON_PARAM_COLUMNS = {
    "iteration",
    "particle",
    "status",
    "error",
    "reason",
    "kge_calibration",
    "kge_validation",
    "mappe_calibration",
    "mappe_validation",
    "pareto_kge_mappe_calibration",
    "pareto_kge_mappe_validation",
}

SOIL_COL_RE = re.compile(r"^(?P<param>log_alpha|log_Ks|n|theta_e)_L(?P<layer>\d+)_tile(?P<tile>\d+)$")
THICKNESS_COL_RE = re.compile(r"^(?P<param>layer_thickness)_L(?P<layer>\d+)_tile(?P<tile>\d+)$")
SCALAR_COL_RE = re.compile(r"^(?P<param>log10_a|b|frac_to_GW|field_capacity_psi|spf_factor|theta_e_1|tile_weight)_tile(?P<tile>\d+)$")
NOM_COL_RE = re.compile(r"^(?P<param>MFSNO|RSURF_SNOW|HVT|CWPVT|VCMX25|MP)_tile(?P<tile>\d+)$")


def safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def list_gage_csvs(params_root: Path, requested_gages: Optional[Sequence[str]]) -> List[Path]:
    if requested_gages:
        out = []
        for g in requested_gages:
            fp = params_root / f"{g}.csv"
            if fp.is_file():
                out.append(fp)
            else:
                print(f"[WARN] Missing calibration CSV for gage {g}: {fp}")
        return out
    return sorted(fp for fp in params_root.glob("*.csv") if fp.is_file())


def choose_best_row(df: pd.DataFrame, score_column: str, direction: str) -> pd.Series:
    if score_column not in df.columns:
        raise ValueError(f"Missing required score column: {score_column}")

    work = df.copy()
    work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
    if "status" in work.columns:
        allowed = work["status"].astype(str).str.upper().isin(["OK", "FINAL", "BEST"])
        work = work[allowed | work["status"].isna()]
    work = work.dropna(subset=[score_column])
    if work.empty:
        raise ValueError(f"No valid rows found for {score_column}")

    if direction == "max":
        idx = work[score_column].idxmax()
    elif direction == "min":
        idx = work[score_column].idxmin()
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    return work.loc[idx]


@dataclass
class ProjectConfig:
    project_root: Path
    sandbox_path: str
    observed_q_root: str
    model_roots: List[str]
    time_cfg: Dict[str, str]
    logging_dir: str
    compute_metrics: Callable
    update_mptable: Callable


def load_project_config(project_root: Path) -> ProjectConfig:
    sys.path.insert(0, str(project_root))
    from model_assessment.configs import path_config as cfg  # type: ignore
    from model_assessment.util.expanded_metrics import compute_metrics  # type: ignore
    from model_assessment.util.update_NOM import update_mptable  # type: ignore

    time_cfg_path = project_root / "model_assessment" / "configs" / "time_config.yaml"
    with open(time_cfg_path, "r") as f:
        time_cfg = yaml.safe_load(f)

    model_roots = []
    for root in getattr(cfg, "model_roots", []):
        p = Path(root)
        if not p.is_absolute():
            p = (project_root / p).resolve()
        model_roots.append(str(p))
    if not model_roots:
        model_roots = [str(project_root.resolve())]

    sandbox_path = getattr(cfg, "sandbox_path", "sandbox.py")
    if not os.path.isabs(sandbox_path):
        sandbox_path = str((project_root / sandbox_path).resolve())

    logging_dir = getattr(cfg, "logging_dir", str(project_root / "logging"))
    if not os.path.isabs(logging_dir):
        logging_dir = str((project_root / logging_dir).resolve())
    os.makedirs(logging_dir, exist_ok=True)

    return ProjectConfig(
        project_root=project_root,
        sandbox_path=sandbox_path,
        observed_q_root=getattr(cfg, "observed_q_root"),
        model_roots=model_roots,
        time_cfg=time_cfg,
        logging_dir=logging_dir,
        compute_metrics=compute_metrics,
        update_mptable=update_mptable,
    )


def pwork(root: str, gage_id: str, pid: int) -> str:
    return os.path.join(root, "out", gage_id, "particles", f"p{pid}")


def resolve_div_dir(tile_root: str, gage_id: str, pid: int) -> str:
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


@dataclass
class ParamSpec:
    name: str
    apply: Callable[["TileContext", float], None]


class TileContext:
    def __init__(self, update_mptable: Callable, tile_root: str, gage_id: str, pid: int, work_root: str):
        self.update_mptable = update_mptable
        self.tile_root = tile_root
        self.gage_id = gage_id
        self.pid = pid
        self.work_root = work_root

        self.lasam_cfg_dir = os.path.join(work_root, "configs", "lasam")
        if not os.path.isdir(self.lasam_cfg_dir):
            raise FileNotFoundError(f"Missing LASAM config dir: {self.lasam_cfg_dir}")

        self.lasam_cfg_files = sorted(
            f for f in os.listdir(self.lasam_cfg_dir) if f.startswith("lasam_config_cat")
        )
        if not self.lasam_cfg_files:
            raise FileNotFoundError(f"No LASAM configs found in {self.lasam_cfg_dir}")

        first_cfg = os.path.join(self.lasam_cfg_dir, self.lasam_cfg_files[0])
        with open(first_cfg, "r") as f:
            lines = f.readlines()

        soil_types_line = next(line for line in lines if line.strip().startswith("layer_soil_type="))
        self.soil_types = list(map(int, soil_types_line.strip().split("=", 1)[1].split(",")))
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
        with open(self.local_soil_path, "r") as f:
            self._soil_lines_cache = f.readlines()
        return self._soil_lines_cache

    def write_soil_lines(self, lines: List[str]):
        with open(self.local_soil_path, "w") as f:
            f.writelines(lines)
        self._soil_lines_cache = lines


def update_key_in_text_file(fp: str, key: str, new_line: str) -> None:
    with open(fp, "r") as f:
        lines = f.readlines()
    out = []
    for line in lines:
        out.append(new_line if line.strip().startswith(key) else line)
    with open(fp, "w") as f:
        f.writelines(out)


def apply_soil_param(tile_ctx: TileContext, layer_1based: int, param: str, value: float):
    if layer_1based < 1 or layer_1based > tile_ctx.n_layers:
        return
    tile_ctx.ensure_local_soil()
    soil_lines = tile_ctx.read_soil_lines()
    soil_type = tile_ctx.soil_types[layer_1based - 1]
    toks = soil_lines[soil_type].split()
    if param == "log_alpha":
        toks[3] = str(10 ** float(value))
    elif param == "n":
        toks[4] = str(float(value))
    elif param == "log_Ks":
        toks[5] = str(10 ** float(value))
    elif param == "theta_e":
        toks[2] = str(float(value))
    else:
        raise ValueError(f"Unknown soil param: {param}")
    soil_lines[soil_type] = "\t".join(toks) + "\n"
    tile_ctx.write_soil_lines(soil_lines)


def apply_layer_thickness(tile_ctx: TileContext, layer_1based: int, value: float):
    for cfg_name in tile_ctx.lasam_cfg_files:
        cfg_path = os.path.join(tile_ctx.lasam_cfg_dir, cfg_name)
        with open(cfg_path, "r") as f:
            lines = f.readlines()
        out = []
        for line in lines:
            if line.strip().startswith("layer_thickness="):
                rhs = line.split("=", 1)[1].strip()
                unit = ""
                if "[" in rhs:
                    unit = "[" + rhs.split("[", 1)[1].strip()
                vals_str = rhs.split("[", 1)[0].strip()
                vals = [float(x.strip()) for x in vals_str.split(",") if x.strip()]
                idx = layer_1based - 1
                if 0 <= idx < len(vals):
                    vals[idx] = float(value)
                out.append(f"layer_thickness={','.join(f'{v:.6g}' for v in vals)}{unit or '[cm]'}\n")
            else:
                out.append(line)
        with open(cfg_path, "w") as f:
            f.writelines(out)


def apply_lasam_scalar(tile_ctx: TileContext, param: str, value: float):
    if param == "log10_a":
        key, line = "a=", f"a={10 ** float(value)}\n"
    elif param == "b":
        key, line = "b=", f"b={float(value)}\n"
    elif param == "frac_to_GW":
        key, line = "frac_to_GW=", f"frac_to_GW={float(value)}\n"
    elif param == "field_capacity_psi":
        key, line = "field_capacity_psi=", f"field_capacity_psi={float(value)}[cm]\n"
    elif param == "spf_factor":
        key, line = "spf_factor=", f"spf_factor={float(value)}\n"
    elif param == "theta_e_1":
        apply_soil_param(tile_ctx, 1, "theta_e", value)
        return
    elif param == "tile_weight":
        return
    else:
        raise ValueError(f"Unknown LASAM scalar param: {param}")

    for cfg_name in tile_ctx.lasam_cfg_files:
        update_key_in_text_file(os.path.join(tile_ctx.lasam_cfg_dir, cfg_name), key, line)


def apply_nom_param(tile_ctx: TileContext, param: str, value: float):
    if not tile_ctx.include_nom:
        return
    if not os.path.isfile(tile_ctx.nom_mptable):
        raise FileNotFoundError(f"NOM MPTABLE missing: {tile_ctx.nom_mptable}")
    tile_ctx.update_mptable(
        original_file=tile_ctx.nom_mptable,
        output_file=tile_ctx.nom_mptable,
        updated_params={param: float(value)},
        verbose=False,
    )


def parse_row_to_tile_values(row: pd.Series) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Optional[float]]]:
    tile_values: Dict[int, Dict[str, float]] = {}
    tile_weights: Dict[int, Optional[float]] = {}

    for col, raw in row.items():
        if col in NON_PARAM_COLUMNS:
            continue
        val = safe_float(raw)
        if val is None:
            continue

        m = SOIL_COL_RE.match(col)
        if m:
            tile = int(m.group("tile"))
            name = f"{m.group('param')}_L{int(m.group('layer'))}"
            tile_values.setdefault(tile, {})[name] = val
            continue

        m = THICKNESS_COL_RE.match(col)
        if m:
            tile = int(m.group("tile"))
            name = f"layer_thickness_L{int(m.group('layer'))}"
            tile_values.setdefault(tile, {})[name] = val
            continue

        m = SCALAR_COL_RE.match(col)
        if m:
            tile = int(m.group("tile"))
            param = m.group("param")
            if param == "tile_weight":
                tile_weights[tile] = val
            else:
                tile_values.setdefault(tile, {})[param] = val
            continue

        m = NOM_COL_RE.match(col)
        if m:
            tile = int(m.group("tile"))
            tile_values.setdefault(tile, {})[m.group("param")] = val
            continue

    return tile_values, tile_weights


def build_specs_for_row(tile_ctx: TileContext, tile_value_map: Dict[str, float]) -> List[ParamSpec]:
    specs: List[ParamSpec] = []
    for name in tile_value_map.keys():
        if name.startswith("layer_thickness_L"):
            layer = int(name.split("_L", 1)[1])
            specs.append(ParamSpec(name=name, apply=lambda ctx, v, layer=layer: apply_layer_thickness(ctx, layer, v)))
        elif re.match(r"^(log_alpha|log_Ks|n|theta_e)_L\d+$", name):
            param, layer_str = name.rsplit("_L", 1)
            layer = int(layer_str)
            specs.append(ParamSpec(name=name, apply=lambda ctx, v, layer=layer, param=param: apply_soil_param(ctx, layer, param, v)))
        elif name in {"log10_a", "b", "frac_to_GW", "field_capacity_psi", "spf_factor", "theta_e_1", "tile_weight"}:
            specs.append(ParamSpec(name=name, apply=lambda ctx, v, param=name: apply_lasam_scalar(ctx, param, v)))
        elif name in {"MFSNO", "RSURF_SNOW", "HVT", "CWPVT", "VCMX25", "MP"}:
            specs.append(ParamSpec(name=name, apply=lambda ctx, v, param=name: apply_nom_param(ctx, param, v)))
        else:
            print(f"    [WARN] Unrecognized parameter column {name}; skipping")
    return specs


def apply_row_to_particle(cfg: ProjectConfig, gage_id: str, pid: int, row: pd.Series) -> Tuple[List[List[ParamSpec]], List[int], np.ndarray, List[float]]:
    tile_values, tile_weights = parse_row_to_tile_values(row)
    specs_by_tile: List[List[ParamSpec]] = []
    tile_counts: List[int] = []
    flat_values: List[float] = []

    for tile_idx, tile_root in enumerate(cfg.model_roots, start=1):
        work_root = pwork(tile_root, gage_id, pid)
        ctx = TileContext(cfg.update_mptable, tile_root, gage_id, pid, work_root)
        per_tile = tile_values.get(tile_idx, {})
        specs = build_specs_for_row(ctx, per_tile)
        vals = [per_tile[s.name] for s in specs]
        if any(s.name.startswith(("log_alpha_L", "log_Ks_L", "n_L", "theta_e_1", "theta_e_L")) for s in specs):
            ctx.ensure_local_soil()
        for spec, val in zip(specs, vals):
            spec.apply(ctx, float(val))
        specs_by_tile.append(specs)
        tile_counts.append(len(specs))
        flat_values.extend(vals)

    weights = [1.0 / len(cfg.model_roots)] * len(cfg.model_roots)
    if len(cfg.model_roots) == 2:
        w = tile_weights.get(1)
        if w is not None:
            weights = [float(w), 1.0 - float(w)]
    return specs_by_tile, tile_counts, np.array(flat_values, dtype=float), weights


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

    cfg_lasam_dir = os.path.join(configs_root, "lasam")
    cfg_pet_dir = os.path.join(configs_root, "pet")
    cfg_nom_dir = os.path.join(configs_root, "noahowp")
    cfg_nom_param = os.path.join(cfg_nom_dir, "parameters")

    forms = rz.get("global", {}).get("formulations", [])
    for form in forms:
        params = form.get("params", {})
        modules = params.get("modules", [])
        for module in modules:
            p = module.get("params", {})
            init_cfg = p.get("init_config", "")
            if not init_cfg:
                continue

            model_type = (p.get("model_type_name") or "").upper()
            fname = Path(init_cfg).name
            src = init_cfg if os.path.isabs(init_cfg) else os.path.join(base_out_dir, init_cfg)

            if any(alias in model_type for alias in ("LASAM", "LGAR")) or "/configs/lasam/" in init_cfg:
                os.makedirs(cfg_lasam_dir, exist_ok=True)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(cfg_lasam_dir, fname))
                p["init_config"] = os.path.abspath(os.path.join(cfg_lasam_dir, fname))

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


def run_one_particle(
    cfg: ProjectConfig,
    gage_id: str,
    pid: int,
    label: str,
    specs_by_tile: List[List[ParamSpec]],
    tile_counts: List[int],
    params: np.ndarray,
    weights: List[float],
) -> Dict[str, object]:
    spinup_start = pd.Timestamp(cfg.time_cfg["spinup_start"])
    cal_start = pd.Timestamp(cfg.time_cfg["cal_start"])
    cal_end = pd.Timestamp(cfg.time_cfg["cal_end"])
    val_end = pd.Timestamp(cfg.time_cfg["val_end"])

    n_tiles = len(cfg.model_roots)
    params = np.array(params, dtype=float)
    if n_tiles == 2 and len(weights) == 2 and len(params) >= sum(tile_counts):
        run_weights = list(weights)
    else:
        run_weights = [1.0 / n_tiles] * n_tiles

    offset = 0
    for tile_idx, tile_root in enumerate(cfg.model_roots):
        n = tile_counts[tile_idx]
        tile_vals = np.array(params[offset:offset + n], dtype=float)
        offset += n

        work_root = pwork(tile_root, gage_id, pid)
        json_dir = os.path.join(work_root, "json")
        os.makedirs(json_dir, exist_ok=True)
        json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
        if not json_files:
            raise FileNotFoundError(f"No realization JSON found in {json_dir}")
        realization_path = os.path.join(json_dir, sorted(json_files)[0])

        base_out_dir = os.path.join(tile_root, "out", gage_id)
        retarget_realization_paths(realization_path, work_root, base_out_dir)

        with open(realization_path, "r") as f:
            realization = json.load(f)
        realization["time"]["start_time"] = cfg.time_cfg["spinup_start"]
        realization["time"]["end_time"] = cfg.time_cfg["val_end"]
        with open(realization_path, "w") as f:
            json.dump(realization, f, indent=4)

        tile_ctx = TileContext(cfg.update_mptable, tile_root, gage_id, pid, work_root)
        for spec, val in zip(specs_by_tile[tile_idx], tile_vals):
            spec.apply(tile_ctx, float(val))

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

        tile_sandbox_config = os.path.join(cfg.project_root, "configs", f"sandbox_config_tile{tile_idx+1}.yaml")
        env = os.environ.copy()
        env["NGEN_CONCURRENT_PARTICLES"] = "1"
        env["NGEN_PARTICLE_ID"] = str(pid)
        env["NGEN_REALIZATION_PATH"] = realization_path

        ret = subprocess.call(
            ["python", cfg.sandbox_path, "-i", tile_sandbox_config, "-run", "--gage_id", gage_id],
            cwd=tile_root,
            env=env,
        )
        if ret != 0:
            raise RuntimeError(f"Hydrology failed: gage {gage_id} | pid {pid} | tile {tile_idx}")

    router_tile_root = cfg.model_roots[0]
    router_work = pwork(router_tile_root, gage_id, pid)
    weighted_div_dir = os.path.join(router_work, "outputs", "div_weighted")
    if os.path.exists(weighted_div_dir):
        shutil.rmtree(weighted_div_dir)
    os.makedirs(weighted_div_dir, exist_ok=True)

    src_div_dirs = [resolve_div_dir(root, gage_id, pid) for root in cfg.model_roots]
    files = []
    for div_dir in src_div_dirs:
        if os.path.isdir(div_dir):
            candidates = [f for f in os.listdir(div_dir) if (f.startswith("cat-") or f.startswith("nex-")) and f.endswith(".csv")]
            if candidates:
                files = candidates
                break
    if not files:
        raise RuntimeError(f"No divide CSVs found for routing for {gage_id} pid {pid}")

    if len(cfg.model_roots) == 1:
        src = src_div_dirs[0]
        for fname in files:
            shutil.copy2(os.path.join(src, fname), os.path.join(weighted_div_dir, fname))
    else:
        for fname in files:
            dfs = []
            df_ref = None
            for tile_idx, div_dir in enumerate(src_div_dirs):
                fp = os.path.join(div_dir, fname)
                if os.path.exists(fp):
                    if fname.startswith("nex-"):
                        df = pd.read_csv(fp, header=None)
                        df.columns = ["Time Step", "Time", "q_out"]
                    else:
                        df = pd.read_csv(fp)
                    if df_ref is None:
                        df_ref = df.copy()
                    dfs.append(df["q_out"] * run_weights[tile_idx])
            if dfs and df_ref is not None:
                out_df = df_ref.copy()
                out_df["q_out"] = sum(dfs)
                out_df["Time"] = pd.to_datetime(out_df["Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                outpath = os.path.join(weighted_div_dir, fname)
                if fname.startswith("nex-"):
                    out_df.to_csv(outpath, index=False, header=False)
                else:
                    out_df.to_csv(outpath, index=False)

    troute_path = os.path.join(router_work, "configs", "troute_config.yaml")
    with open(troute_path, "r") as f:
        troute_cfg = yaml.safe_load(f)

    nts = int((val_end - spinup_start) / pd.Timedelta(seconds=300))
    troute_cfg["compute_parameters"]["restart_parameters"]["start_datetime"] = spinup_start.strftime("%Y-%m-%d_%H:%M:%S")
    troute_cfg["compute_parameters"]["forcing_parameters"]["nts"] = nts
    troute_cfg["compute_parameters"]["forcing_parameters"]["qlat_input_folder"] = weighted_div_dir

    particle_troute_dir = os.path.join(router_work, "troute")
    os.makedirs(particle_troute_dir, exist_ok=True)
    op = troute_cfg.setdefault("output_parameters", {})
    so = op.setdefault("stream_output", {})
    so["stream_output_directory"] = particle_troute_dir

    mask_path = so.get("mask_output")
    if not (isinstance(mask_path, str) and os.path.isfile(mask_path)):
        orig_mask = os.path.join(router_tile_root, "out", gage_id, "configs", "mask_output.yaml")
        if os.path.isfile(orig_mask):
            so["mask_output"] = orig_mask
        else:
            so.pop("mask_output", None)

    with open(troute_path, "w") as f:
        yaml.safe_dump(troute_cfg, f)

    env = os.environ.copy()
    env["NGEN_CONCURRENT_PARTICLES"] = "1"
    env["NGEN_PARTICLE_ID"] = str(pid)
    ret = subprocess.call(["python3", "-m", "nwm_routing", "-f", "-V4", troute_path], env=env)
    if ret != 0:
        raise RuntimeError(f"Routing failed: gage {gage_id} | pid {pid}")

    postproc_dir = os.path.join(router_work, "postproc")
    os.makedirs(postproc_dir, exist_ok=True)
    output_path = os.path.join(postproc_dir, f"{gage_id}_{label}.csv")

    get_hydrograph_path = os.path.join(cfg.project_root, "model_assessment", "util", "get_hydrograph.py")
    summary_csv = os.path.join(cfg.project_root, "model_assessment", "util", "downstream_flowpath_summary.csv")
    troute_output_dir = os.path.join(router_work, "troute")

    env = os.environ.copy()
    env["NGEN_CONCURRENT_PARTICLES"] = "1"
    env["NGEN_PARTICLE_ID"] = str(pid)
    ret = subprocess.call(
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
    if ret != 0:
        raise RuntimeError(f"Hydrograph extraction failed: gage {gage_id} | pid {pid}")

    sim_df = (
        pd.read_csv(output_path, parse_dates=["current_time"])
        .set_index("current_time")["flow"]
        .resample("1h")
        .mean()
    )
    obs_path = os.path.join(cfg.observed_q_root, "successful_sites_resampled", f"{gage_id}.csv")
    obs_df = pd.read_csv(obs_path, parse_dates=["value_time"]).set_index("value_time")["flow_m3_per_s"]

    cal_start = pd.Timestamp(cfg.time_cfg["cal_start"])
    cal_end = pd.Timestamp(cfg.time_cfg["cal_end"])
    val_start = pd.Timestamp(cfg.time_cfg["val_start"])
    val_end = pd.Timestamp(cfg.time_cfg["val_end"])

    sim_cal = sim_df[cal_start:cal_end].dropna()
    obs_cal = obs_df[cal_start:cal_end].dropna()
    sim_cal, obs_cal = sim_cal.align(obs_cal, join="inner")
    if len(sim_cal) > 0:
        sim_cal.iloc[-1] += 1e-8
    if len(obs_cal) > 0:
        obs_cal.iloc[-1] += 1e-8
    cal_metrics = cfg.compute_metrics(sim_cal, obs_cal, event_threshold=1e-2)

    sim_val = sim_df[val_start:val_end].dropna()
    obs_val = obs_df[val_start:val_end].dropna()
    sim_val, obs_val = sim_val.align(obs_val, join="inner")
    if len(sim_val) > 0:
        sim_val.iloc[-1] += 1e-8
    if len(obs_val) > 0:
        obs_val.iloc[-1] += 1e-8
    val_metrics = cfg.compute_metrics(sim_val, obs_val, event_threshold=1e-2)

    return {
        "gage_id": gage_id,
        "particle": pid,
        "label": label,
        "output_path": output_path,
        "kge_calibration": cal_metrics.get("kge", np.nan),
        "mappe_calibration": cal_metrics.get("mappe", np.nan),
        "pareto_kge_mappe_calibration": cal_metrics.get("pareto_kge_mappe", np.nan),
        "kge_validation": val_metrics.get("kge", np.nan),
        "mappe_validation": val_metrics.get("mappe", np.nan),
        "pareto_kge_mappe_validation": val_metrics.get("pareto_kge_mappe", np.nan),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run best KGE and best MAPPE LASAM parameter sets once each.")
    p.add_argument("--params-root", required=True, help="Directory containing per-gage calibration CSVs")
    p.add_argument("--project-root", default=os.getcwd(), help="NextGenSandboxHub root")
    p.add_argument("--gage-id", action="append", default=None, help="Optional gage(s) to process")
    p.add_argument("--kge-column", default="kge_calibration", help="Column used to select p0. Default: kge_calibration")
    p.add_argument("--mappe-column", default="mappe_calibration", help="Column used to select p1. Default: mappe_calibration")
    p.add_argument("--summary-csv", default=None, help="Optional summary CSV output path")
    p.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of gages to process concurrently. Each worker runs p0 then p1 sequentially for its gage.",
    )
    return p.parse_args()


def process_one_gage(task: Tuple[str, str, str, str]) -> Dict[str, object]:
    project_root_str, csv_fp_str, kge_column, mappe_column = task
    gage_id = Path(csv_fp_str).stem
    try:
        cfg = load_project_config(Path(project_root_str).resolve())
        df = pd.read_csv(csv_fp_str)
        row_kge = choose_best_row(df, kge_column, direction="max")
        row_mappe = choose_best_row(df, mappe_column, direction="min")

        specs0, counts0, params0, weights0 = apply_row_to_particle(cfg, gage_id, 0, row_kge)
        specs1, counts1, params1, weights1 = apply_row_to_particle(cfg, gage_id, 1, row_mappe)

        res0 = run_one_particle(cfg, gage_id, 0, "best_kge", specs0, counts0, params0, weights0)
        res1 = run_one_particle(cfg, gage_id, 1, "best_mappe", specs1, counts1, params1, weights1)

        res0["source_row_metric"] = row_kge.get(kge_column, np.nan)
        res1["source_row_metric"] = row_mappe.get(mappe_column, np.nan)

        return {
            "gage_id": gage_id,
            "ok": True,
            "results": [res0, res1],
            "message": f"p0 -> {res0['output_path']} | p1 -> {res1['output_path']}",
        }
    except Exception as exc:
        return {
            "gage_id": gage_id,
            "ok": False,
            "results": [],
            "message": str(exc),
        }


def main() -> int:
    args = parse_args()
    params_root = Path(args.params_root).resolve()
    project_root = Path(args.project_root).resolve()

    if not params_root.is_dir():
        print(f"[ERROR] params-root does not exist: {params_root}")
        return 1
    if not project_root.is_dir():
        print(f"[ERROR] project-root does not exist: {project_root}")
        return 1
    if args.n_workers < 1:
        print("[ERROR] --n-workers must be at least 1")
        return 1

    csv_files = list_gage_csvs(params_root, args.gage_id)
    if not csv_files:
        print(f"[ERROR] No calibration CSVs found in {params_root}")
        return 1

    tasks = [(str(project_root), str(csv_fp), args.kge_column, args.mappe_column) for csv_fp in csv_files]
    summary_path = args.summary_csv or str(project_root / "model_assessment" / "util" / "best_kge_mappe_run_summary.csv")

    print(
        f"[INFO] Starting {len(tasks)} gage(s) with {args.n_workers} worker(s). "
        "Each worker runs p0 then p1 sequentially for its assigned gage."
    )

    results: List[Dict[str, object]] = []
    failures = 0

    if args.n_workers == 1:
        outcomes = map(process_one_gage, tasks)
    else:
        ctx = get_context("spawn")
        pool = ctx.Pool(processes=args.n_workers)
        outcomes = pool.imap_unordered(process_one_gage, tasks)

    try:
        for outcome in outcomes:
            gage_id = outcome["gage_id"]
            if outcome["ok"]:
                results.extend(outcome["results"])
                print(f"[OK] {gage_id}: {outcome['message']}")
            else:
                failures += 1
                print(f"[FAIL] {gage_id}: {outcome['message']}")
    finally:
        if args.n_workers != 1:
            pool.close()
            pool.join()

    if results:
        pd.DataFrame(results).to_csv(summary_path, index=False)
        print(f"[INFO] Wrote summary CSV: {summary_path}")

    if failures:
        print(f"[WARN] {failures} gage(s) failed")
        return 1

    print("[INFO] Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
