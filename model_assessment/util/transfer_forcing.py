"""
transfer_forcing.py

Usage (from repo root, e.g., NextGenSandboxHub):
  python model_assessment/util/transfer_forcing.py

What it does:
- Reads basin_IDs/basin_IDs.csv and collects gage_id values.
- Copies forcing trees from SOURCE_BASE/<gage_id> to DEST_BASE/<gage_id>.
- Default SOURCE_BASE: /projects/standard/nieberj/shared/plafolle/in_long
- Default DEST_BASE:   /tmp/in
- Only copies files that are missing or have different size/mtime (simple sync).
- Parallelized across gages (default workers=4).

You can override paths/workers via env vars:
  TRANSFER_SOURCE_BASE, TRANSFER_DEST_BASE, TRANSFER_WORKERS

For one-gage Slurm jobs, set TRANSFER_GAGE_ID, NGEN_GAGE_ID, or GAGE_ID.
Set BASIN_CSV to use a different gage CSV when no single gage is provided.
"""

import os
import sys
import csv
import time
import math
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------- Config (env-overridable) --------
SOURCE_BASE = Path(os.environ.get("TRANSFER_SOURCE_BASE", "/projects/standard/nieberj/shared/plafolle/in_long"))
DEST_BASE   = Path(os.environ.get("TRANSFER_DEST_BASE", "/tmp/in"))
N_WORKERS   = int(os.environ.get("TRANSFER_WORKERS", "4"))

# -------- Helpers --------
def project_root() -> Path:
    # This file lives at model_assessment/util/transfer_forcing.py
    return Path(__file__).resolve().parents[2]

def csv_path() -> Path:
    return Path(os.environ.get("BASIN_CSV", project_root() / "basin_IDs" / "basin_IDs.csv"))

def gage_ids_from_env() -> list[str]:
    value = (
        os.environ.get("TRANSFER_GAGE_ID")
        or os.environ.get("NGEN_GAGE_ID")
        or os.environ.get("GAGE_ID")
    )
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]

def human(nbytes: int) -> str:
    units = ["B","KiB","MiB","GiB","TiB"]
    i = 0
    x = float(nbytes)
    while x >= 1024 and i < len(units)-1:
        x /= 1024.0; i += 1
    return f"{x:.1f} {units[i]}"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def same_file(src: Path, dst: Path) -> bool:
    """Return True if dst exists and appears identical by size and mtime."""
    if not dst.exists():
        return False
    try:
        s_stat = src.stat()
        d_stat = dst.stat()
        return (s_stat.st_size == d_stat.st_size) and (int(s_stat.st_mtime) == int(d_stat.st_mtime))
    except FileNotFoundError:
        return False

def copy_file(src: Path, dst: Path):
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)  # preserves mtime/perm where possible

def copy_tree_incremental(src_root: Path, dst_root: Path) -> tuple[int, int]:
    """
    Incremental copy:
      - creates dirs as needed
      - copies files that are missing or different (size/mtime)
    Returns: (files_copied, bytes_copied)
    """
    files_copied = 0
    bytes_copied = 0
    for dirpath, dirnames, filenames in os.walk(src_root):
        rel = Path(dirpath).relative_to(src_root)
        dst_dir = dst_root / rel
        ensure_dir(dst_dir)

        for name in filenames:
            s = Path(dirpath) / name
            d = dst_dir / name
            if not same_file(s, d):
                sz = s.stat().st_size if s.exists() else 0
                copy_file(s, d)
                files_copied += 1
                bytes_copied += sz
    return files_copied, bytes_copied

def disk_free(path: Path) -> int:
    """Return free bytes for the filesystem containing path."""
    usage = shutil.disk_usage(path)
    return usage.free

def transfer_one_gage(gage_id: str) -> dict:
    src = SOURCE_BASE / gage_id
    dst = DEST_BASE / gage_id
    t0 = time.time()
    out = {
        "gage_id": gage_id,
        "status": "OK",
        "files": 0,
        "bytes": 0,
        "secs": 0.0,
        "note": "",
    }

    if not src.exists():
        out["status"] = "MISSING_SRC"
        out["note"] = f"Source not found: {src}"
        return out

    try:
        ensure_dir(DEST_BASE)
        # Optional: quick free-space sanity (warn if < 1.1x source size)
        try:
            # fast-ish estimate by walking once for sizes
            total_src = 0
            for dirpath, _, filenames in os.walk(src):
                for f in filenames:
                    p = Path(dirpath) / f
                    try:
                        total_src += p.stat().st_size
                    except FileNotFoundError:
                        pass
            free = disk_free(DEST_BASE)
            if free < total_src * 1.1:
                out["note"] = f"Low free space on {DEST_BASE} (free {human(free)}, needs ~{human(total_src)})"

        except Exception:
            # If estimating fails, continue anyway.
            pass

        files, bytes_ = copy_tree_incremental(src, dst)
        out["files"] = files
        out["bytes"] = bytes_
        out["secs"] = time.time() - t0
        return out

    except Exception as e:
        out["status"] = "ERROR"
        out["note"] = f"{type(e).__name__}: {e}"
        out["secs"] = time.time() - t0
        return out

def read_gage_ids(csvfile: Path) -> list[str]:
    ids: list[str] = []
    with csvfile.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        # Accept common variations in header
        key = None
        for k in rdr.fieldnames or []:
            if k.strip().lower() in ("gage_id","gageid","staid","id"):
                key = k
                break
        if key is None:
            raise RuntimeError(f"No gage_id-like column found in {csvfile}")
        for row in rdr:
            gid = (row.get(key, "") or "").strip()
            if gid:
                ids.append(gid)
    return ids

def main():
    root = project_root()
    csvfile = csv_path()

    print("=== transfer_forcing.py ===")
    print(f"Repo root:  {root}")
    print(f"CSV path:   {csvfile}")
    print(f"Source base:{SOURCE_BASE}")
    print(f"Dest base:  {DEST_BASE}")
    print(f"Workers:    {N_WORKERS}")
    print()

    env_gages = gage_ids_from_env()
    if env_gages:
        gages = env_gages
    elif not csvfile.exists():
        print(f"ERROR: CSV not found: {csvfile}", file=sys.stderr)
        sys.exit(1)
    else:
        try:
            gages = read_gage_ids(csvfile)
        except Exception as e:
            print(f"ERROR reading CSV: {e}", file=sys.stderr)
            sys.exit(1)

    if not gages:
        print("No gage IDs found in CSV.")
        return

    print(f"Found {len(gages)} gage(s). Starting transfers...\n")
    ensure_dir(DEST_BASE)

    results = []
    with ThreadPoolExecutor(max_workers=max(1, N_WORKERS)) as ex:
        futs = {ex.submit(transfer_one_gage, gid): gid for gid in gages}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            gid = res["gage_id"]
            if res["status"] == "OK":
                print(f"[OK]  {gid}: copied {res['files']} file(s), {human(res['bytes'])} in {res['secs']:.1f}s"
                      + (f"  ({res['note']})" if res["note"] else ""))
            elif res["status"] == "MISSING_SRC":
                print(f"[SKIP] {gid}: {res['note']}")
            else:
                print(f"[ERR] {gid}: {res['note']}")

    # Summary
    ok = [r for r in results if r["status"] == "OK"]
    miss = [r for r in results if r["status"] == "MISSING_SRC"]
    err = [r for r in results if r["status"] == "ERROR"]
    tot_bytes = sum(r["bytes"] for r in ok)
    tot_files = sum(r["files"] for r in ok)

    print("\n=== SUMMARY ===")
    print(f"Success: {len(ok)}  | Missing src: {len(miss)}  | Errors: {len(err)}")
    print(f"Copied:  {tot_files} file(s), {human(tot_bytes)} total")
    if err:
        print("Some transfers failed. See messages above.")

if __name__ == "__main__":
    main()
