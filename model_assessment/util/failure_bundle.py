"""Capture an exact particle workspace before PSO reuses it after an NGen failure."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TRUE_VALUES = {"1", "true", "yes", "on"}


def env_flag(name: str, default: bool = False) -> bool:
    """Read a conventional boolean environment variable."""
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in TRUE_VALUES


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_reference(path: Path, input_root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    reference: dict[str, Any] = {"original_path": str(resolved), "exists": resolved.is_file()}
    if not resolved.is_file():
        return reference
    stat = resolved.stat()
    reference.update({"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": _sha256(resolved)})
    if input_root is not None:
        try:
            reference["relative_to_input_root"] = str(resolved.relative_to(input_root.resolve()))
        except ValueError:
            pass
    return reference


def _git_state(repo: Path) -> dict[str, Any]:
    state: dict[str, Any] = {"path": str(repo.resolve())}
    try:
        for key, args in (
            ("commit", ["rev-parse", "HEAD"]),
            ("branch", ["rev-parse", "--abbrev-ref", "HEAD"]),
            ("status", ["status", "--short"]),
            ("tracked_diff", ["diff", "--no-ext-diff", "--binary"]),
        ):
            result = subprocess.run(
                ["git", "-C", str(repo), *args],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                state[key] = result.stdout.rstrip()
    except (OSError, subprocess.SubprocessError):
        pass
    return state


def _containing_repo(path: Path) -> Path | None:
    candidate = path.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / ".git").exists():
            return directory
    return None


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open() as source:
            value = json.load(source)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _last_nonempty_line(path: Path) -> str:
    try:
        with path.open("rb") as source:
            source.seek(0, os.SEEK_END)
            position = source.tell()
            data = b""
            while position > 0 and data.count(b"\n") < 2:
                take = min(8192, position)
                position -= take
                source.seek(position)
                data = source.read(take) + data
        lines = [line for line in data.decode("utf-8", errors="replace").splitlines() if line.strip()]
        return lines[-1][:1000] if lines else ""
    except OSError:
        return ""


def _write_output_inventory(work_root: Path, destination: Path) -> None:
    outputs = work_root / "outputs"
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "size_bytes", "mtime_ns", "last_nonempty_line"))
        writer.writeheader()
        if not outputs.is_dir():
            return
        for path in sorted(item for item in outputs.rglob("*") if item.is_file()):
            stat = path.stat()
            writer.writerow(
                {
                    "path": str(path.relative_to(work_root)),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "last_nonempty_line": _last_nonempty_line(path) if path.suffix.lower() == ".csv" else "",
                }
            )


def _safe_name(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in text)


def _copy_if_file(source: Path, destination: Path) -> bool:
    if not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def capture_failure_bundle(
    *,
    destination_root: str | os.PathLike[str],
    gage_id: str,
    iteration: Any,
    particle_idx: int,
    tile_idx: int,
    stage: str,
    work_root: str | os.PathLike[str],
    realization_path: str | os.PathLike[str],
    params: Any,
    param_names: list[str] | None,
    sandbox_config: str | os.PathLike[str],
    sandbox_returncode: int,
    ngen_log_path: str | os.PathLike[str],
    ngen_metadata_path: str | os.PathLike[str],
    project_root: str | os.PathLike[str],
    include_forcing: bool = False,
) -> Path:
    """Create a durable, atomic bundle containing the exact failed particle inputs."""
    now = datetime.now(timezone.utc)
    root = Path(destination_root).expanduser()
    gage_root = root / _safe_name(gage_id)
    gage_root.mkdir(parents=True, exist_ok=True)
    suffix = uuid.uuid4().hex[:8]
    name = (
        f"{_safe_name(gage_id)}_iter-{_safe_name(iteration)}_particle-{particle_idx}"
        f"_{now.strftime('%Y%m%d_%H%M%S_%f')}_{suffix}"
    )
    final_path = gage_root / name
    temporary = gage_root / f".{name}.tmp"
    work_literal = Path(work_root).absolute()
    work = work_literal.resolve()
    realization = Path(realization_path).resolve()
    project = Path(project_root).resolve()
    input_root_value = os.environ.get("CIROH_INPUT_DIR")
    input_root = Path(input_root_value).resolve() if input_root_value else None

    try:
        temporary.mkdir()
        snapshot = temporary / "particle_workspace"
        snapshot.mkdir()
        for dirname in ("configs", "json"):
            source = work / dirname
            if source.is_dir():
                shutil.copytree(source, snapshot / dirname)

        _copy_if_file(Path(ngen_log_path), temporary / "ngen.log")
        _copy_if_file(Path(ngen_metadata_path), temporary / "ngen_run_metadata.json")
        _copy_if_file(Path(sandbox_config), temporary / "workflow" / "sandbox_config.yaml")
        _write_output_inventory(work, temporary / "output_inventory.csv")

        workflow_sources = (
            project / "model_assessment" / "calib_scripts" / "pso_calibration_casam.py",
            project / "src" / "python" / "runner.py",
            project / "src" / "python" / "configuration.py",
            project / "sandbox.py",
        )
        for source in workflow_sources:
            if source.is_file():
                _copy_if_file(source, temporary / "workflow" / source.name)

        replay_source = Path(__file__).with_name("replay_ngen_failure.py")
        _copy_if_file(replay_source, temporary / "replay_ngen_failure.py")

        metadata = _read_json(Path(ngen_metadata_path))
        realization_data = _read_json(realization)
        candidate_inputs: set[Path] = set()
        gpkg = metadata.get("gpkg_file")
        if gpkg:
            candidate_inputs.add(Path(gpkg))
        for value in _walk_strings(realization_data):
            expanded = Path(os.path.expandvars(value))
            if expanded.suffix.lower() in {".nc", ".csv", ".gpkg"} and expanded.is_file():
                candidate_inputs.add(expanded)

        references: list[dict[str, Any]] = []
        for source in sorted(candidate_inputs, key=lambda item: str(item)):
            reference = _file_reference(source, input_root)
            suffix_lower = source.suffix.lower()
            if suffix_lower == ".gpkg" or (include_forcing and suffix_lower in {".nc", ".csv"}):
                subdir = "geopackage" if suffix_lower == ".gpkg" else "forcing"
                local = temporary / "input" / subdir / source.name
                _copy_if_file(source, local)
                reference["bundled_path"] = str(local.relative_to(temporary))
            references.append(reference)
        with (temporary / "input_references.json").open("w") as stream:
            json.dump(references, stream, indent=2)

        named_params = {
            name: value
            for name, value in zip(param_names or [], _jsonable(params))
        }
        with (temporary / "parameters.json").open("w") as stream:
            json.dump(
                {"names": param_names or [], "values": _jsonable(params), "by_name": named_params},
                stream,
                indent=2,
            )

        runtime_files: list[dict[str, Any]] = []
        runtime_artifacts: list[Path] = []
        for key in ("ngen_executable", "realization_path"):
            value = metadata.get(key)
            if value:
                artifact = Path(value)
                runtime_artifacts.append(artifact)
                runtime_files.append(_file_reference(artifact))
        for value in _walk_strings(realization_data):
            expanded = Path(os.path.expandvars(value))
            if expanded.suffix.lower() in {".so", ".dylib"} and expanded.is_file():
                runtime_artifacts.append(expanded)
                runtime_files.append(_file_reference(expanded))

        source_repositories = [_git_state(project)]
        seen_repositories = {project.resolve()}
        for artifact in runtime_artifacts:
            repository = _containing_repo(artifact)
            if repository is not None and repository.resolve() not in seen_repositories:
                seen_repositories.add(repository.resolve())
                source_repositories.append(_git_state(repository))

        safe_environment = {
            key: os.environ[key]
            for key in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_JOB_NODELIST",
                "SLURM_CPUS_PER_TASK",
                "NGEN_DIR",
                "NGEN_SANDBOX_CONFIG",
                "CIROH_INPUT_DIR",
                "NGEN_MODEL_ROOT",
            )
            if key in os.environ
        }
        manifest = {
            "bundle_format_version": 1,
            "captured_at_utc": now.isoformat(),
            "gage_id": gage_id,
            "iteration": iteration,
            "particle": particle_idx,
            "tile": tile_idx,
            "stage": stage,
            "sandbox_returncode": sandbox_returncode,
            "ngen_returncode": metadata.get("returncode"),
            "ngen_signal": metadata.get("signal"),
            "original_work_root": str(work_literal),
            "original_work_root_resolved": str(work),
            "original_realization_path": str(realization),
            "forcing_files_included": include_forcing,
            "runtime": {"python": sys.version, "platform": platform.platform(), "environment": safe_environment},
            "source_repositories": source_repositories,
            "runtime_files": runtime_files,
            "ngen_run": metadata,
        }
        with (temporary / "manifest.json").open("w") as stream:
            json.dump(manifest, stream, indent=2)

        with (temporary / "REPLAY.md").open("w") as stream:
            stream.write(
                "# Replay this NGen failure\n\n"
                "The bundle contains the exact failed particle realization and configuration files. "
                "Forcing is referenced by checksum by default rather than duplicated.\n\n"
                "```bash\n"
                "python3 replay_ngen_failure.py \\\n+  --ngen /path/to/ngen \\\n+  --lgarto-repo /path/to/LGAR-C \\\n+  --input-root /path/to/inhf22/in\n"
                "```\n"
            )

        os.replace(temporary, final_path)
        return final_path
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
