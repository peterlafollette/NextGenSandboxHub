#!/bin/bash

set -euo pipefail

# Agate convenience wrapper for a production-sized PSO calibration submission.
# Run from any Agate login/desktop shell:
#   bash /users/4/plafolle/infil_proj/NextGenSandboxHub/slurm/submit_agate_pso_production.sh
#
# Override any setting by exporting it before calling this script, for example:
#   MODELS=casam MAX_ARRAY_CONCURRENT=200 bash slurm/submit_agate_pso_production.sh

export NGSH_ROOT="${NGSH_ROOT:-/users/4/plafolle/infil_proj/NextGenSandboxHub}"
export PYTHON_VENV="${PYTHON_VENV:-/users/4/plafolle/.venv_sandbox_py3.11/bin/activate}"

cd "$NGSH_ROOT"

if [ -f "$PYTHON_VENV" ]; then
    source "$PYTHON_VENV"
else
    echo "Python venv activation file not found: $PYTHON_VENV" >&2
    exit 1
fi

export BASIN_CSV="${BASIN_CSV:-$NGSH_ROOT/basin_IDs/basin_IDs.csv}"
export SHARED_INPUT_DIR="${SHARED_INPUT_DIR:-/projects/standard/nieberj/shared/plafolle/inhf22}"
export CIROH_HF_GPKG="${CIROH_HF_GPKG:-/users/4/plafolle/infil_proj/conus_nextgen_updated_fixed.gpkg}"
export NGEN_DIR="${NGEN_DIR:-/users/4/plafolle/CIROH_project/ngen}"
export DEST_MODEL_ROOT="${DEST_MODEL_ROOT:-/projects/standard/nieberj/shared/plafolle/infil_proj/out}"
export DOWNSTREAM_FLOWPATH_SUMMARY="${DOWNSTREAM_FLOWPATH_SUMMARY:-$NGSH_ROOT/model_assessment/util/downstream_flowpath_summary.csv}"

# Model arrays. With both models, each model gets its own Slurm array.
export CALIBRATION_ALGORITHM="pso"
export MODELS="${MODELS:-casam,cfe}"
export CASAM_MODE="${CASAM_MODE:-standard}"

# Per-task resources. Four CPUs lets PSO run four particles at a time.
export CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
export MEM_PER_CPU="${MEM_PER_CPU:-5G}"
export TMP_DISK="${TMP_DISK:-20G}"
export TIME_LIMIT="${TIME_LIMIT:-4-0}"

# Production PSO controls: 12 particles x 42 iterations = about 504 objective calls.
export N_PARTICLES="${N_PARTICLES:-12}"
export N_ITERATIONS="${N_ITERATIONS:-42}"
export MAX_PARTICLE_PROCS="${MAX_PARTICLE_PROCS:-4}"

# This throttle applies per model array. With MODELS=casam,cfe and CPUS_PER_TASK=4,
# MAX_ARRAY_CONCURRENT=100 can allocate up to about 800 cores total if both arrays run.
export MAX_ARRAY_CONCURRENT="${MAX_ARRAY_CONCURRENT:-100}"

# Default long-window run. Adjust these if you want a different calibration/validation split.
export SPINUP_START="${SPINUP_START:-2010-01-01}"
export CAL_START="${CAL_START:-2010-10-01}"
export CAL_END="${CAL_END:-2017-09-30}"
export VAL_START="${VAL_START:-2017-10-01}"
export VAL_END="${VAL_END:-2020-09-30}"

export TRANSFER_WORKERS="${TRANSFER_WORKERS:-1}"
export OUTPUT_RETAIN_MODE="${OUTPUT_RETAIN_MODE:-compact}"
export KEEP_BEST_QLAT="${KEEP_BEST_QLAT:-false}"
export COPY_FAILED_OUTPUTS="${COPY_FAILED_OUTPUTS:-true}"
export COMPRESS_JOB_LOG="${COMPRESS_JOB_LOG:-true}"
# Opt in to exact failed-particle snapshots; false preserves the current workflow.
export CAPTURE_FAILURE_BUNDLES="${CAPTURE_FAILURE_BUNDLES:-false}"
export FAILURE_BUNDLE_INCLUDE_FORCING="${FAILURE_BUNDLE_INCLUDE_FORCING:-false}"

if [ ! -f "$BASIN_CSV" ]; then
    echo "Missing BASIN_CSV: $BASIN_CSV" >&2
    exit 1
fi

N_GAGES=$(awk -F',' 'NR>1 && $1!="" {c++} END{print c+0}' "$BASIN_CSV")
IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
N_MODELS="${#MODEL_LIST[@]}"
TOTAL_TASKS=$(( N_GAGES * 2 * N_MODELS ))
MAX_CORES=$(( MAX_ARRAY_CONCURRENT * CPUS_PER_TASK * N_MODELS ))

cat <<EOF
=== NGSH Agate PSO production submission ===
NGSH_ROOT:                    $NGSH_ROOT
BASIN_CSV:                    $BASIN_CSV
Gages:                        $N_GAGES
Models:                       $MODELS
CASAM mode:                   $CASAM_MODE
Calibration algorithm:        $CALIBRATION_ALGORITHM
Total Slurm array tasks:      $TOTAL_TASKS
Max array concurrency/model:  $MAX_ARRAY_CONCURRENT
Approx max allocated cores:   $MAX_CORES
CPUs/task:                    $CPUS_PER_TASK
Mem/CPU:                      $MEM_PER_CPU
Tmp/task:                     $TMP_DISK
Time limit:                   $TIME_LIMIT
Particles / iterations:       $N_PARTICLES / $N_ITERATIONS
Max particle procs/task:      $MAX_PARTICLE_PROCS
Time windows:                 spinup=$SPINUP_START cal=$CAL_START..$CAL_END val=$VAL_START..$VAL_END
Destination output root:      $DEST_MODEL_ROOT
Output retain mode:           $OUTPUT_RETAIN_MODE
Keep best qlat:               $KEEP_BEST_QLAT
Copy failed outputs:          $COPY_FAILED_OUTPUTS
Capture failure bundles:      $CAPTURE_FAILURE_BUNDLES
Copy forcing into bundles:    $FAILURE_BUNDLE_INCLUDE_FORCING
EOF

bash "$NGSH_ROOT/slurm/submit_calibration_arrays.sh"
