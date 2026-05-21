#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def _read_basin_rows(basin_csv: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Reads basin_IDs.csv as DictReader and returns (fieldnames, rows).
    Preserves all columns (e.g., gage_id,num_divides).
    """
    with basin_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return [], []
        rows = [r for r in reader if any((v or "").strip() for v in r.values())]
        return list(reader.fieldnames), rows


def _write_basin_rows(out_csv: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _last_nonempty_row_first_two_fields(log_csv: Path) -> Tuple[str, str]:
    """
    Returns (field0, field1) from the last non-empty CSV row.
    Assumes first two columns are iteration and particle.
    """
    last = None
    with log_csv.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if all((c or "").strip() == "" for c in row):
                continue
            last = row

    if last is None:
        return "", ""

    c0 = (last[0] if len(last) > 0 else "").strip()
    c1 = (last[1] if len(last) > 1 else "").strip()
    return c0, c1


def is_complete_for_gage(
    gage_id: str,
    tmp_out_root: Path,
    logging_dir: Path,
) -> Tuple[bool, List[str]]:
    """
    Complete iff:
      1) /tmp/model1/out/<gage>/particles/p0/postproc/<gage>_best.csv exists
      2) <logging_dir>/<gage>.csv exists AND last non-empty row has
         first two fields == "FINAL" and "BEST"
    """
    reasons: List[str] = []

    best_csv = tmp_out_root / gage_id / "particles" / "p0" / "postproc" / f"{gage_id}_best.csv"
    if not best_csv.is_file():
        reasons.append(f"missing best hydrograph: {best_csv}")

    log_csv = logging_dir / f"{gage_id}.csv"
    if not log_csv.is_file():
        reasons.append(f"missing log csv: {log_csv}")
    else:
        c0, c1 = _last_nonempty_row_first_two_fields(log_csv)
        if c0 != "FINAL" or c1 != "BEST":
            reasons.append(f"log last row not FINAL,BEST (got {c0!r},{c1!r}): {log_csv}")

    return (len(reasons) == 0), reasons


def main() -> int:
    # Defaults match your layout; can override via env if desired.
    basin_csv = Path(os.environ.get("BASIN_CSV", "basin_IDs/basin_IDs.csv")).resolve()
    retry_csv = Path(os.environ.get("RETRY_BASIN_CSV", str(basin_csv.parent / "basin_IDs_retry.csv"))).resolve()

    tmp_out_root = Path(os.environ.get("TMP_OUT_ROOT", "/tmp/model1/out")).resolve()
    logging_dir = Path(os.environ.get("LOGGING_DIR", "logging")).resolve()

    if not basin_csv.is_file():
        print(f"[audit] ERROR: basin csv not found: {basin_csv}", file=sys.stderr)
        return 0

    fieldnames, rows = _read_basin_rows(basin_csv)
    if not fieldnames or not rows:
        print(f"[audit] WARNING: basin csv empty or missing header: {basin_csv}")
        _write_basin_rows(retry_csv, fieldnames or ["gage_id", "num_divides"], [])
        print(f"[audit] wrote empty retry csv: {retry_csv}")
        return 0

    if "gage_id" not in fieldnames:
        print(f"[audit] ERROR: basin csv missing 'gage_id' column: {basin_csv}", file=sys.stderr)
        return 0

    incomplete_rows: List[Dict[str, str]] = []
    reasons_by_gage: Dict[str, List[str]] = {}

    for r in rows:
        g = (r.get("gage_id") or "").strip()
        if not g:
            continue
        ok, reasons = is_complete_for_gage(g, tmp_out_root=tmp_out_root, logging_dir=logging_dir)
        if not ok:
            incomplete_rows.append(r)
            reasons_by_gage[g] = reasons

    _write_basin_rows(retry_csv, fieldnames, incomplete_rows)

    print(f"[audit] basin csv: {basin_csv}")
    print(f"[audit] tmp_out_root: {tmp_out_root}")
    print(f"[audit] logging_dir: {logging_dir}")
    print(f"[audit] total gages: {len(rows)}")
    print(f"[audit] incomplete: {len(incomplete_rows)}")
    print(f"[audit] retry csv: {retry_csv}")

    reasons_txt = retry_csv.with_suffix(".reasons.txt")
    with reasons_txt.open("w") as f:
        for r in incomplete_rows:
            g = (r.get("gage_id") or "").strip()
            if not g:
                continue
            for reason in reasons_by_gage.get(g, []):
                f.write(f"{g}\t{reason}\n")
    print(f"[audit] reasons file: {reasons_txt}")

    # Exit 0 so the batch job itself doesn't fail just because some gages did.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
