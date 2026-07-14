#!/usr/bin/env python3
"""Create a compact, reviewable calibration result directory.

This is intended for Slurm jobs that run in node-local scratch. The calibration
scripts can keep their full particle workspaces while running; this utility
copies only the final/best outputs that should be retained long term.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore_dangling_symlinks=True)


def gzip_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as f_in, gzip.open(dst, "wb", compresslevel=6) as f_out:
        shutil.copyfileobj(f_in, f_out)
    shutil.copystat(src, dst, follow_symlinks=True)


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def is_cfe_nom_variant(variant: str) -> bool:
    """Match PSO and algorithm-prefixed variants such as dds_cfe_nom."""
    return str(variant).strip().lower().split("_")[-2:] == ["cfe", "nom"]


def final_realization_time_window(
    best_root: Path, warnings: list[str]
) -> tuple[Path, str, str] | None:
    """Read the finalized best-particle realization used by ngen."""
    json_dir = best_root / "json"
    candidates = sorted(json_dir.glob("realization*.json"))
    windows: list[tuple[Path, str, str]] = []

    for path in candidates:
        try:
            realization = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"Could not read final realization {path}: {exc}")
            continue

        time_config = realization.get("time")
        if not isinstance(time_config, dict):
            warnings.append(f"Final realization has no time object: {path}")
            continue
        start_time = time_config.get("start_time")
        end_time = time_config.get("end_time")
        if not start_time or not end_time:
            warnings.append(f"Final realization has incomplete time window: {path}")
            continue
        windows.append((path, str(start_time), str(end_time)))

    if not windows:
        warnings.append(f"No finalized realization time window found under {json_dir}")
        return None

    distinct = {(start, end) for _, start, end in windows}
    if len(distinct) != 1:
        warnings.append(
            "Final best-particle realizations disagree on the time window: "
            + ", ".join(f"{start} -> {end}" for start, end in sorted(distinct))
        )
        return None

    return windows[0]


def synchronize_realization_times(
    best_root: Path, start_time: str, end_time: str, warnings: list[str]
) -> int:
    """Set the final time window in every retained realization copy."""
    synchronized = 0
    for directory in (best_root / "json", best_root / "shared_json"):
        for path in sorted(directory.glob("realization*.json")):
            try:
                realization = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                warnings.append(f"Could not synchronize realization {path}: {exc}")
                continue

            time_config = realization.get("time")
            if not isinstance(time_config, dict):
                warnings.append(f"Retained realization has no time object: {path}")
                continue
            time_config["start_time"] = start_time
            time_config["end_time"] = end_time
            path.write_text(json.dumps(realization, indent=4) + "\n")
            synchronized += 1
    return synchronized


def yaml_scalar(path: Path, key: str) -> str:
    pattern = re.compile(
        rf"^[ \t]*{re.escape(key)}[ \t]*:[ \t]*(.*?)[ \t]*$", re.MULTILINE
    )
    matches = pattern.findall(path.read_text())
    if len(matches) != 1:
        raise ValueError(f"Expected one {key!r} entry in {path}, found {len(matches)}")
    return matches[0]


def replace_yaml_scalar(path: Path, key: str, value: str) -> None:
    text = path.read_text()
    pattern = re.compile(
        rf"^([ \t]*{re.escape(key)}[ \t]*:[ \t]*).*?$", re.MULTILINE
    )
    updated, count = pattern.subn(lambda match: f"{match.group(1)}{value}", text)
    if count != 1:
        raise ValueError(f"Expected one {key!r} entry in {path}, found {count}")
    path.write_text(updated)


def synchronize_troute_times(best_root: Path, warnings: list[str]) -> int:
    """Copy final particle t-route timing scalars into all retained copies."""
    final_path = best_root / "configs" / "troute_config.yaml"
    if not final_path.is_file():
        warnings.append(f"No finalized t-route config found at {final_path}")
        return 0

    try:
        start_datetime = yaml_scalar(final_path, "start_datetime")
        nts = yaml_scalar(final_path, "nts")
    except (OSError, ValueError) as exc:
        warnings.append(f"Could not read finalized t-route timing: {exc}")
        return 0

    synchronized = 0
    for path in (
        best_root / "configs" / "troute_config.yaml",
        best_root / "shared_configs" / "troute_config.yaml",
    ):
        if not path.is_file():
            continue
        try:
            replace_yaml_scalar(path, "start_datetime", start_datetime)
            replace_yaml_scalar(path, "nts", nts)
        except (OSError, ValueError) as exc:
            warnings.append(f"Could not synchronize t-route config {path}: {exc}")
            continue
        synchronized += 1
    return synchronized


def format_noahowp_time(value: str) -> str:
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y%m%d%H%M")
        except ValueError:
            pass
    raise ValueError(f"Cannot format NoahOWP time from {value!r}")


def synchronize_noahowp_times(
    best_root: Path, start_time: str, end_time: str, warnings: list[str]
) -> int:
    """Set final startdate/enddate in every retained NoahOWP input copy."""
    try:
        startdate = format_noahowp_time(start_time)
        enddate = format_noahowp_time(end_time)
    except ValueError as exc:
        warnings.append(str(exc))
        return 0

    synchronized = 0
    paths: set[Path] = set()
    for directory in (
        best_root / "configs" / "noahowp",
        best_root / "shared_configs" / "noahowp",
    ):
        if directory.is_dir():
            paths.update(path for path in directory.glob("*.input") if path.is_file())

    for path in sorted(paths):
        try:
            lines = path.read_text().splitlines(keepends=True)
        except OSError as exc:
            warnings.append(f"Could not read retained NoahOWP input {path}: {exc}")
            continue
        found_start = False
        found_end = False
        updated_lines: list[str] = []
        for line in lines:
            stripped = line.strip().lower()
            newline = "\n" if line.endswith("\n") else ""
            indent = line[: len(line) - len(line.lstrip())]
            if stripped.startswith("startdate"):
                updated_lines.append(f'{indent}startdate      = "{startdate}"  {newline}')
                found_start = True
            elif stripped.startswith("enddate"):
                updated_lines.append(f'{indent}enddate      = "{enddate}"  {newline}')
                found_end = True
            else:
                updated_lines.append(line)

        if not found_start or not found_end:
            warnings.append(f"Could not find startdate/enddate in retained NoahOWP input: {path}")
            continue
        try:
            path.write_text("".join(updated_lines))
        except OSError as exc:
            warnings.append(f"Could not synchronize retained NoahOWP input {path}: {exc}")
            continue
        synchronized += 1
    return synchronized


def synchronize_retained_timing(best_root: Path, warnings: list[str]) -> dict[str, object]:
    """Normalize timing metadata in duplicate compact-archive configurations."""
    final_window = final_realization_time_window(best_root, warnings)
    if final_window is None:
        return {}

    realization_path, start_time, end_time = final_window
    return {
        "source_realization": str(realization_path.relative_to(best_root.parent)),
        "start_time": start_time,
        "end_time": end_time,
        "realizations": synchronize_realization_times(best_root, start_time, end_time, warnings),
        "troute_configs": synchronize_troute_times(best_root, warnings),
        "noahowp_inputs": synchronize_noahowp_times(
            best_root, start_time, end_time, warnings
        ),
    }


def find_best_hydrograph(source: Path, gage_id: str) -> Path | None:
    preferred = source / "particles" / "p0" / "postproc" / f"{gage_id}_best.csv"
    if preferred.is_file():
        return preferred

    candidates = list(source.glob(f"particles/*/postproc/{gage_id}_best.csv"))
    candidates += list(source.glob(f"postproc/{gage_id}_best.csv"))
    candidates += list(source.glob(f"**/{gage_id}_best.csv"))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)[0]


def particle_root_for(best_hydrograph: Path, source: Path) -> Path | None:
    parts = best_hydrograph.relative_to(source).parts
    if len(parts) >= 4 and parts[0] == "particles":
        return source / parts[0] / parts[1]
    return None


def copy_logging(source: Path, dest: Path, compress_logs: bool) -> list[str]:
    copied: list[str] = []
    src_logging = source / "logging"
    dst_logging = dest / "logging"
    if not src_logging.is_dir():
        return copied

    for item in sorted(src_logging.iterdir()):
        if item.is_dir():
            copy_tree(item, dst_logging / item.name)
            copied.append(str((dst_logging / item.name).relative_to(dest)))
        elif compress_logs and item.suffix == ".log":
            gzip_copy(item, dst_logging / f"{item.name}.gz")
            copied.append(str((dst_logging / f"{item.name}.gz").relative_to(dest)))
        elif item.is_file():
            copy_file(item, dst_logging / item.name)
            copied.append(str((dst_logging / item.name).relative_to(dest)))
    return copied


def compact_outputs(
    source: Path,
    dest: Path,
    gage_id: str,
    variant: str,
    mode: str,
    keep_best_qlat: bool,
    compress_logs: bool,
) -> int:
    source = source.resolve()
    dest = dest.resolve()
    warnings: list[str] = []
    copied: list[str] = []
    timing_synchronization: dict[str, object] = {}

    if not source.is_dir():
        print(f"Source gage directory does not exist: {source}", file=sys.stderr)
        return 2

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    copied.extend(copy_logging(source, dest, compress_logs))

    best_hydrograph = find_best_hydrograph(source, gage_id)
    if best_hydrograph is None:
        warnings.append(f"No {gage_id}_best.csv found under {source}")
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": str(source),
            "gage_id": gage_id,
            "variant": variant,
            "retain_mode": mode,
            "warnings": warnings,
            "copied": copied,
        }
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return 2

    particle_root = particle_root_for(best_hydrograph, source)
    best_root = dest / "best"

    copy_file(best_hydrograph, best_root / "hydrograph" / best_hydrograph.name)
    copied.append(str((best_root / "hydrograph" / best_hydrograph.name).relative_to(dest)))

    if mode != "minimal":
        if particle_root is not None:
            for name in ("troute", "configs", "json"):
                src = particle_root / name
                if src.exists():
                    copy_tree(src, best_root / name)
                    copied.append(str((best_root / name).relative_to(dest)))
                else:
                    warnings.append(f"Missing best particle {name} directory: {src}")

            if keep_best_qlat:
                for name in ("div_weighted", "div"):
                    src = particle_root / "outputs" / name
                    if src.exists():
                        copy_tree(src, best_root / "qlat" / name)
                        copied.append(str((best_root / "qlat" / name).relative_to(dest)))
        else:
            warnings.append("Best hydrograph was not inside a particle workspace")

        for name in ("configs", "json"):
            src = source / name
            if src.exists():
                copy_tree(src, best_root / f"shared_{name}")
                copied.append(str((best_root / f"shared_{name}").relative_to(dest)))

        # CFE+NOM realizations keep NoahOWP init_config paths at the gage level,
        # while CFE paths are retargeted into the particle workspace. Replace
        # the unused particle NoahOWP copy with the files the final ngen run
        # actually referenced. Remove the destination first so particle-only
        # files cannot survive the overlay.
        if is_cfe_nom_variant(variant):
            shared_noahowp = source / "configs" / "noahowp"
            best_noahowp = best_root / "configs" / "noahowp"
            if shared_noahowp.is_dir():
                if best_noahowp.exists():
                    shutil.rmtree(best_noahowp)
                copy_tree(shared_noahowp, best_noahowp)
                copied.append(str(best_noahowp.relative_to(dest)))
            else:
                warnings.append(f"Missing shared CFE+NOM NoahOWP directory: {shared_noahowp}")

        timing_synchronization = synchronize_retained_timing(best_root, warnings)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "gage_id": gage_id,
        "variant": variant,
        "retain_mode": mode,
        "best_particle": particle_root.name if particle_root else None,
        "best_hydrograph": str(best_hydrograph.relative_to(source)),
        "keep_best_qlat": keep_best_qlat,
        "compress_logs": compress_logs,
        "copied": copied,
        "warnings": warnings,
    }
    if timing_synchronization:
        manifest["timing_synchronization"] = timing_synchronization
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    size_before = directory_size(source)
    size_after = directory_size(dest)
    print(
        "Compacted calibration outputs: "
        f"{size_before / 1024**2:.1f} MiB -> {size_after / 1024**2:.1f} MiB"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Full gage output directory to compact")
    parser.add_argument("--dest", required=True, type=Path, help="Destination compact gage output directory")
    parser.add_argument("--gage-id", required=True)
    parser.add_argument("--variant", required=True, help="Variant label, e.g. casam_nom")
    parser.add_argument("--mode", choices=("compact", "minimal"), default="compact")
    parser.add_argument("--keep-best-qlat", default="false")
    parser.add_argument("--compress-logs", default="true")
    args = parser.parse_args()

    return compact_outputs(
        source=args.source,
        dest=args.dest,
        gage_id=args.gage_id,
        variant=args.variant,
        mode=args.mode,
        keep_best_qlat=parse_bool(args.keep_best_qlat),
        compress_logs=parse_bool(args.compress_logs),
    )


if __name__ == "__main__":
    raise SystemExit(main())
