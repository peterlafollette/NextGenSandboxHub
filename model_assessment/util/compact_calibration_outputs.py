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
