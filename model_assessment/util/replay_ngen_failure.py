#!/usr/bin/env python3
"""Replay an NGen failure bundle without modifying its archived files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _find_library(repo: Path, stem: str) -> Path:
    candidates = [
        repo / "build" / f"{stem}.so",
        repo / "build" / f"{stem}.dylib",
        repo / "cmake_build" / f"{stem}.so",
        repo / "cmake_build" / f"{stem}.dylib",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find {stem}.so or {stem}.dylib under {repo}")


def _replace_paths(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [_replace_paths(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_paths(item, replacements) for key, item in value.items()}
    return value


def _parameter_blocks(value: Any):
    """Yield nested BMI parameter blocks, including bmi_multi modules."""
    if isinstance(value, dict):
        if "model_type_name" in value:
            yield value
        for item in value.values():
            yield from _parameter_blocks(item)
    elif isinstance(value, list):
        for item in value:
            yield from _parameter_blocks(item)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _patch_text_configs(config_root: Path, replacements: dict[str, str]) -> None:
    for path in config_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        updated = text
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--ngen", required=True, help="Current NGen executable")
    parser.add_argument("--lgarto-repo", required=True, help="Current LGAR-C source/build tree")
    parser.add_argument("--input-root", required=True, help="Root corresponding to archived CIROH_INPUT_DIR")
    parser.add_argument("--noah-library", help="Optional local Noah-OWP BMI library")
    parser.add_argument("--sloth-library", help="Optional local SLoTH BMI library")
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--remove-workdir", action="store_true")
    args = parser.parse_args()

    bundle = Path(args.bundle).resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    references = json.loads((bundle / "input_references.json").read_text())
    workdir = Path(tempfile.mkdtemp(prefix=f"ngen-replay-{manifest['gage_id']}-"))
    staged = workdir / "particle_workspace"
    shutil.copytree(bundle / "particle_workspace", staged)

    # Remap every archived particle-local path into the disposable replay tree.
    replacements: dict[str, str] = {manifest["original_work_root"]: str(staged)}
    if manifest.get("original_work_root_resolved"):
        replacements[manifest["original_work_root_resolved"]] = str(staged)
    input_root = Path(args.input_root).resolve()
    gpkg: Path | None = None
    for reference in references:
        old = reference.get("original_path")
        bundled = reference.get("bundled_path")
        relative = reference.get("relative_to_input_root")
        if bundled:
            local = bundle / bundled
        elif relative:
            local = input_root / relative
        else:
            continue
        if not local.is_file():
            raise FileNotFoundError(f"Required replay input is missing: {local}")
        expected_hash = reference.get("sha256")
        if expected_hash and _sha256(local) != expected_hash:
            raise RuntimeError(f"Replay input does not match the archived checksum: {local}")
        replacements[old] = str(local.resolve())
        if local.suffix.lower() == ".gpkg":
            gpkg = local.resolve()

    realization_files = sorted((staged / "json").glob("*.json"))
    if not realization_files:
        raise FileNotFoundError("Bundle contains no realization JSON")
    realization_path = realization_files[0]
    realization = json.loads(realization_path.read_text())

    lgarto_library = _find_library(Path(args.lgarto_repo), "liblasambmi")
    for params in _parameter_blocks(realization):
        model_type = str(params.get("model_type_name", "")).upper()
        old_init = params.get("init_config")
        old_library = params.get("library_file")
        if old_library and ("LASAM" in model_type or "LGAR" in model_type or "CASAM" in model_type):
            replacements[old_library] = str(lgarto_library)
            params["library_file"] = str(lgarto_library)
        elif old_library and "NOAH" in model_type and args.noah_library:
            replacements[old_library] = str(Path(args.noah_library).resolve())
            params["library_file"] = str(Path(args.noah_library).resolve())
        elif old_library and "SLOTH" in model_type and args.sloth_library:
            replacements[old_library] = str(Path(args.sloth_library).resolve())
            params["library_file"] = str(Path(args.sloth_library).resolve())
        if old_init:
            matches = list((staged / "configs").rglob(Path(old_init).name))
            if matches:
                replacements[old_init] = str(matches[0].resolve())
                params["init_config"] = str(matches[0].resolve())

    realization = _replace_paths(realization, replacements)
    output_root = workdir / "outputs"
    output_root.mkdir()
    realization["output_root"] = str(output_root)
    realization_path.write_text(json.dumps(realization, indent=2))
    _patch_text_configs(staged / "configs", replacements)

    if gpkg is None:
        bundled_gpkgs = list((bundle / "input" / "geopackage").glob("*.gpkg"))
        if bundled_gpkgs:
            gpkg = bundled_gpkgs[0].resolve()
    if gpkg is None:
        raise FileNotFoundError("Bundle contains no catchment geopackage")

    command = [str(Path(args.ngen).resolve()), str(gpkg), "all", str(gpkg), "all", str(realization_path)]
    print("Replay workspace:", workdir)
    print("Command:", shlex.join(command), flush=True)
    log_path = workdir / "ngen_replay.log"
    try:
        with log_path.open("w") as log:
            result = subprocess.run(
                command,
                cwd=staged,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout_seconds or None,
                check=False,
            )
        print(f"NGen return code: {result.returncode}")
        print(f"Log: {log_path}")
        if result.returncode != 0:
            lines = log_path.read_text(errors="replace").splitlines()[-80:]
            if lines:
                print("Final replay log lines:\n" + "\n".join(lines))
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"NGen exceeded {args.timeout_seconds:g} seconds; log: {log_path}", file=sys.stderr)
        return 124
    finally:
        if args.remove_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"Replay workspace retained at {workdir}")


if __name__ == "__main__":
    raise SystemExit(main())
