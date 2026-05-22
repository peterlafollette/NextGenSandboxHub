#!/usr/bin/env python3
"""
Launch concurrent PET/NOM calibration variants for CFE and/or CASAM.

This helper intentionally runs only:
  1. sandbox.py -conf
  2. model_assessment/calib_scripts/pso_calibration_*.py

It never runs -subset or -forc.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

VARIANTS = {
    ("casam", "pet"): {
        "config": "configs/sandbox_config_casam.yaml",
        "script": "model_assessment/calib_scripts/pso_calibration_casam.py",
    },
    ("casam", "nom"): {
        "config": "configs/sandbox_config_nom_casam.yaml",
        "script": "model_assessment/calib_scripts/pso_calibration_casam.py",
    },
    ("cfe", "pet"): {
        "config": "configs/sandbox_config_cfe.yaml",
        "script": "model_assessment/calib_scripts/pso_calibration_cfe.py",
    },
    ("cfe", "nom"): {
        "config": "configs/sandbox_config_nom_cfe.yaml",
        "script": "model_assessment/calib_scripts/pso_calibration_cfe.py",
    },
}

FAILURE_PATTERNS = (
    "Traceback",
    "RuntimeError",
    "No NetCDF",
    "No successful particle",
    "Hydrology run failed",
    "Hydrology failed",
    "Final T-route failed",
)


@dataclass(frozen=True)
class Variant:
    model: str
    formulation_variant: str
    config: Path
    script: Path

    @property
    def label(self) -> str:
        return f"{self.model}_{self.formulation_variant}"


@dataclass
class Job:
    name: str
    cmd: list[str]
    env: dict[str, str]
    log_path: Path


def split_choices(value: str, valid: set[str], label: str) -> list[str]:
    choices = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not choices:
        raise SystemExit(f"No {label} selected.")
    bad = sorted(set(choices) - valid)
    if bad:
        raise SystemExit(f"Unsupported {label}: {', '.join(bad)}. Valid values: {', '.join(sorted(valid))}")
    return list(dict.fromkeys(choices))


def flatten_gage_args(values: Iterable[str]) -> list[str]:
    gages: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                gages.append(item)
    return list(dict.fromkeys(gages))


def read_gages_from_csv(path: Path) -> list[str]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "gage_id" not in (reader.fieldnames or []):
            raise SystemExit(f"{path} must contain a gage_id column.")
        gages = [str(row["gage_id"]).strip() for row in reader if str(row.get("gage_id", "")).strip()]
    if not gages:
        raise SystemExit(f"No gage IDs found in {path}.")
    return list(dict.fromkeys(gages))


def write_selected_basin_csv(gages: list[str], logs_dir: Path) -> Path:
    selected_csv = logs_dir / "selected_gages.csv"
    with selected_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gage_id", "num_divides"])
        writer.writeheader()
        for gage in gages:
            writer.writerow({"gage_id": gage, "num_divides": 0})
    return selected_csv


def require_path_arg(name: str, value: str | None) -> Path:
    if not value:
        raise SystemExit(f"{name} is required. Pass it as an argument or set the matching environment variable.")
    return Path(os.path.expandvars(value)).expanduser().resolve()


def resolve_output_dir(config_path: Path, env: dict[str, str]) -> Path:
    with config_path.open("r") as f:
        config = yaml.safe_load(f) or {}
    output_dir = config.get("output_dir")
    if not output_dir:
        raise SystemExit(f"{config_path} does not define output_dir.")
    expanded = os.path.expandvars(str(output_dir))
    for key, value in env.items():
        expanded = expanded.replace("${" + key + "}", value)
        expanded = expanded.replace("$" + key, value)
    return Path(expanded).expanduser().resolve()


def build_base_env(args: argparse.Namespace, basin_csv: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["NGSH_ROOT"] = str(REPO_ROOT)
    env["CIROH_INPUT_DIR"] = str(args.input_dir)
    env["CIROH_HF_GPKG"] = str(args.hf_gpkg)
    env["NGEN_DIR"] = str(args.ngen_dir)
    env["NGEN_MODEL_ROOT"] = str(args.ngen_model_root)
    env["BASIN_CSV"] = str(basin_csv)
    env["DOWNSTREAM_FLOWPATH_SUMMARY"] = str(args.downstream_flowpath_summary)
    env["NGEN_JOB_CORES"] = str(args.job_cores)
    env["NGEN_TROUTE_CPU_POOL"] = str(args.troute_cpu_pool)
    env["OMP_NUM_THREADS"] = str(args.omp_num_threads)
    return env


def make_variants(models: list[str], formulation_variants: list[str]) -> list[Variant]:
    variants: list[Variant] = []
    for model in models:
        for formulation_variant in formulation_variants:
            meta = VARIANTS[(model, formulation_variant)]
            variants.append(
                Variant(
                    model=model,
                    formulation_variant=formulation_variant,
                    config=(REPO_ROOT / meta["config"]).resolve(),
                    script=(REPO_ROOT / meta["script"]).resolve(),
                )
            )
    return variants


def variant_env(base_env: dict[str, str], variant: Variant) -> dict[str, str]:
    env = base_env.copy()
    env["NGEN_SANDBOX_CONFIG"] = str(variant.config)
    return env


def make_conf_jobs(args: argparse.Namespace, variants: list[Variant], base_env: dict[str, str]) -> list[Job]:
    jobs: list[Job] = []
    for variant in variants:
        cmd = [
            args.python,
            str(REPO_ROOT / "sandbox.py"),
            "-i",
            str(variant.config),
            "-conf",
            "--concurrent-particles",
            "--num-particles",
            str(args.n_particles),
        ]
        jobs.append(
            Job(
                name=f"conf_{variant.label}",
                cmd=cmd,
                env=variant_env(base_env, variant),
                log_path=args.launcher_logs_dir / f"conf_{variant.label}.log",
            )
        )
    return jobs


def add_time_args(cmd: list[str], args: argparse.Namespace) -> None:
    cmd.extend(
        [
            "--spinup-start",
            args.spinup_start,
            "--cal-start",
            args.cal_start,
            "--cal-end",
            args.cal_end,
            "--val-start",
            args.val_start,
            "--val-end",
            args.val_end,
        ]
    )


def make_calibration_jobs(
    args: argparse.Namespace,
    variants: list[Variant],
    gages: list[str],
    base_env: dict[str, str],
) -> list[Job]:
    jobs: list[Job] = []
    for variant in variants:
        for gage in gages:
            cmd = [
                args.python,
                str(variant.script),
                "--sandbox-config",
                str(variant.config),
                "--gage-id",
                str(gage),
                "--n-particles",
                str(args.n_particles),
                "--n-iterations",
                str(args.n_iterations),
                "--max-particle-procs",
                str(args.max_particle_procs),
                "--max-gage-procs",
                str(args.max_gage_procs),
            ]
            add_time_args(cmd, args)
            jobs.append(
                Job(
                    name=f"calib_{variant.label}_{gage}",
                    cmd=cmd,
                    env=variant_env(base_env, variant),
                    log_path=args.launcher_logs_dir / f"calib_{variant.label}_{gage}.log",
                )
            )
    return jobs


def format_cmd(cmd: list[str]) -> str:
    return " ".join(cmd)


def run_jobs(jobs: list[Job], max_concurrent: int, dry_run: bool) -> bool:
    if not jobs:
        return True
    max_concurrent = max(1, max_concurrent)
    if dry_run:
        for job in jobs:
            print(f"[dry-run] {job.name}")
            print(f"  log: {job.log_path}")
            print(f"  cmd: {format_cmd(job.cmd)}")
        return True

    pending = list(jobs)
    running: list[tuple[Job, subprocess.Popen, object]] = []
    ok = True

    while pending or running:
        while pending and len(running) < max_concurrent:
            job = pending.pop(0)
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh = job.log_path.open("w")
            print(f"[start] {job.name}")
            print(f"        log: {job.log_path}")
            proc = subprocess.Popen(
                job.cmd,
                cwd=REPO_ROOT,
                env=job.env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
            running.append((job, proc, log_fh))

        time.sleep(1)
        still_running: list[tuple[Job, subprocess.Popen, object]] = []
        for job, proc, log_fh in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((job, proc, log_fh))
                continue
            log_fh.close()
            status = "ok" if ret == 0 else f"failed exit={ret}"
            print(f"[done]  {job.name}: {status}")
            ok = ok and ret == 0
        running = still_running

    return ok


def scan_launcher_logs(log_dir: Path) -> list[str]:
    hits: list[str] = []
    for log_path in sorted(log_dir.glob("*.log")):
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            continue
        for pattern in FAILURE_PATTERNS:
            if pattern in text:
                hits.append(f"{log_path}: {pattern}")
                break
    return hits


def summarize_outputs(variants: list[Variant], gages: list[str], env: dict[str, str]) -> bool:
    print("\nOutput summary:")
    ok = True
    for variant in variants:
        output_dir = resolve_output_dir(variant.config, env)
        for gage in gages:
            log_csv = output_dir / gage / "logging" / f"{gage}.csv"
            best_csv = output_dir / gage / "particles" / "p0" / "postproc" / f"{gage}_best.csv"
            troute_dir = output_dir / gage / "particles" / "p0" / "troute"
            troute_files = sorted(troute_dir.glob("*.nc")) if troute_dir.exists() else []

            row_text = ""
            if log_csv.exists():
                try:
                    with log_csv.open(newline="") as f:
                        rows = list(csv.DictReader(f))
                    if rows:
                        row = rows[-1]
                        row_text = (
                            f" kge_val={row.get('kge_validation', '')}"
                            f" wall_s={row.get('total_wall_time_seconds', row.get('elapsed_wall_time_seconds', ''))}"
                            f" core_h={row.get('total_core_hours', row.get('core_hours_to_row', ''))}"
                        )
                except Exception as exc:
                    row_text = f" could_not_read_log_csv={exc}"

            present = log_csv.exists() and best_csv.exists() and bool(troute_files)
            ok = ok and present
            flag = "OK" if present else "MISSING"
            print(f"  {flag} {variant.label} {gage}:{row_text}")
            print(f"       log:   {log_csv}")
            print(f"       best:  {best_csv}")
            print(f"       route: {troute_files[0] if troute_files else troute_dir}")
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run concurrent PET/NOM calibration variants without Slurm."
    )
    parser.add_argument("--models", default="casam,cfe", help="Comma-separated: casam,cfe")
    parser.add_argument(
        "--formulation-variants",
        default="pet,nom",
        help="Comma-separated formulation variants: pet,nom",
    )
    parser.add_argument(
        "--forcings",
        dest="formulation_variants_alias",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gage-id",
        action="append",
        default=[],
        help="Gage ID to run. Repeat or comma-separate. Defaults to all gages in --basin-csv.",
    )
    parser.add_argument("--basin-csv", default=os.environ.get("BASIN_CSV"))
    parser.add_argument("--input-dir", default=os.environ.get("CIROH_INPUT_DIR"))
    parser.add_argument("--hf-gpkg", default=os.environ.get("CIROH_HF_GPKG"))
    parser.add_argument("--ngen-dir", default=os.environ.get("NGEN_DIR"))
    parser.add_argument("--ngen-model-root", default=os.environ.get("NGEN_MODEL_ROOT"))
    parser.add_argument(
        "--downstream-flowpath-summary",
        default=os.environ.get(
            "DOWNSTREAM_FLOWPATH_SUMMARY",
            str(REPO_ROOT / "model_assessment" / "util" / "downstream_flowpath_summary.csv"),
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--n-particles", type=int, default=2)
    parser.add_argument("--n-iterations", type=int, default=2)
    parser.add_argument("--max-particle-procs", type=int, default=1)
    parser.add_argument("--max-gage-procs", type=int, default=1)
    parser.add_argument("--max-concurrent-conf", type=int, default=1)
    parser.add_argument(
        "--max-concurrent-calibrations",
        type=int,
        default=0,
        help="0 means launch all selected calibration jobs concurrently.",
    )
    parser.add_argument("--job-cores", type=int, default=int(os.environ.get("NGEN_JOB_CORES", "1")))
    parser.add_argument("--troute-cpu-pool", type=int, default=int(os.environ.get("NGEN_TROUTE_CPU_POOL", "1")))
    parser.add_argument("--omp-num-threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "1")))
    parser.add_argument("--spinup-start", default=os.environ.get("NGEN_SPINUP_START", "2010-10-01"))
    parser.add_argument("--cal-start", default=os.environ.get("NGEN_CAL_START", "2011-10-01"))
    parser.add_argument("--cal-end", default=os.environ.get("NGEN_CAL_END", "2012-09-30"))
    parser.add_argument("--val-start", default=os.environ.get("NGEN_VAL_START", "2012-10-01"))
    parser.add_argument("--val-end", default=os.environ.get("NGEN_VAL_END", "2013-09-30"))
    parser.add_argument("--skip-conf", action="store_true", help="Reuse existing -conf outputs.")
    parser.add_argument("--conf-only", action="store_true", help="Run -conf and stop before calibration.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()

    args.models = split_choices(args.models, {"casam", "cfe"}, "models")
    if args.formulation_variants_alias:
        args.formulation_variants = args.formulation_variants_alias
    args.formulation_variants = split_choices(
        args.formulation_variants,
        {"pet", "nom"},
        "formulation variants",
    )
    args.input_dir = require_path_arg("--input-dir / CIROH_INPUT_DIR", args.input_dir)
    args.hf_gpkg = require_path_arg("--hf-gpkg / CIROH_HF_GPKG", args.hf_gpkg)
    args.ngen_dir = require_path_arg("--ngen-dir / NGEN_DIR", args.ngen_dir)
    args.ngen_model_root = require_path_arg("--ngen-model-root / NGEN_MODEL_ROOT", args.ngen_model_root)
    args.downstream_flowpath_summary = require_path_arg(
        "--downstream-flowpath-summary / DOWNSTREAM_FLOWPATH_SUMMARY",
        args.downstream_flowpath_summary,
    )
    args.basin_csv = Path(os.path.expandvars(args.basin_csv)).expanduser().resolve() if args.basin_csv else None
    args.launcher_logs_dir = args.ngen_model_root / "out" / "launcher_logs"
    args.launcher_logs_dir.mkdir(parents=True, exist_ok=True)
    return args


def main() -> int:
    args = parse_args()
    variants = make_variants(args.models, args.formulation_variants)

    requested_gages = flatten_gage_args(args.gage_id)
    if requested_gages:
        gages = requested_gages
        basin_csv = write_selected_basin_csv(gages, args.launcher_logs_dir)
    else:
        if not args.basin_csv:
            raise SystemExit("Pass --gage-id or set/pass --basin-csv.")
        gages = read_gages_from_csv(args.basin_csv)
        basin_csv = args.basin_csv

    base_env = build_base_env(args, basin_csv)
    max_calibrations = args.max_concurrent_calibrations or len(variants) * len(gages)

    print("Concurrent variant launcher")
    print("This script never calls -subset or -forc.")
    print(f"repo:       {REPO_ROOT}")
    print(f"root:       {args.ngen_model_root}")
    print(f"gages:      {', '.join(gages)}")
    print(f"variants:   {', '.join(v.label for v in variants)}")
    print(f"logs:       {args.launcher_logs_dir}")

    if not args.skip_conf:
        conf_jobs = make_conf_jobs(args, variants, base_env)
        if not run_jobs(conf_jobs, args.max_concurrent_conf, args.dry_run):
            print("One or more -conf jobs failed.")
            return 1

    if args.conf_only:
        return 0

    calibration_jobs = make_calibration_jobs(args, variants, gages, base_env)
    if not run_jobs(calibration_jobs, max_calibrations, args.dry_run):
        print("One or more calibration jobs failed.")
        return 1

    if args.dry_run:
        return 0

    log_hits = scan_launcher_logs(args.launcher_logs_dir)
    if log_hits:
        print("\nFailure signatures found in launcher logs:")
        for hit in log_hits:
            print(f"  {hit}")
        return 1

    return 0 if summarize_outputs(variants, gages, base_env) else 1


if __name__ == "__main__":
    raise SystemExit(main())
