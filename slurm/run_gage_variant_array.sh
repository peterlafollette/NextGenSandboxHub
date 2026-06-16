#!/bin/bash -l

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --tmp=20G
#SBATCH -t 4-0
#SBATCH -p msismall
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --constraint=genoa
#SBATCH --mail-user=plafolle@umn.edu

set -euo pipefail

# Required by submit helper or sbatch --export.
MODEL="${MODEL:?Set MODEL to casam or cfe}"
CALIBRATION_ALGORITHM="${CALIBRATION_ALGORITHM:-pso}"

# HPC path defaults. Override with sbatch --export or environment variables.
NGSH_ROOT="${NGSH_ROOT:-/users/4/plafolle/infil_proj/NextGenSandboxHub}"
SHARED_INPUT_DIR="${SHARED_INPUT_DIR:-/projects/standard/nieberj/shared/plafolle/inhf22}"
CIROH_HF_GPKG="${CIROH_HF_GPKG:-/users/4/plafolle/infil_proj/conus_nextgen_updated_fixed.gpkg}"
NGEN_DIR="${NGEN_DIR:-/users/4/plafolle/CIROH_project/ngen}"
DEST_MODEL_ROOT="${DEST_MODEL_ROOT:-/projects/standard/nieberj/shared/plafolle/infil_proj/out}"
DEST_OUTPUT_ROOT="${DEST_OUTPUT_ROOT:-$DEST_MODEL_ROOT}"
BASIN_CSV="${BASIN_CSV:-$NGSH_ROOT/basin_IDs/basin_IDs.csv}"
DOWNSTREAM_FLOWPATH_SUMMARY="${DOWNSTREAM_FLOWPATH_SUMMARY:-$NGSH_ROOT/model_assessment/util/downstream_flowpath_summary.csv}"
PYTHON_VENV="${PYTHON_VENV:-/users/4/plafolle/.venv_sandbox_py3.11/bin/activate}"

# Short test defaults. Override these for production runs.
N_PARTICLES="${N_PARTICLES:-2}"
N_ITERATIONS="${N_ITERATIONS:-2}"
MAX_PARTICLE_PROCS="${MAX_PARTICLE_PROCS:-4}"
SPINUP_START="${SPINUP_START:-2010-10-01}"
CAL_START="${CAL_START:-2011-01-01}"
CAL_END="${CAL_END:-2011-12-31}"
VAL_START="${VAL_START:-2012-01-01}"
VAL_END="${VAL_END:-2012-06-30}"

TRANSFER_WORKERS="${TRANSFER_WORKERS:-1}"
OUTPUT_RETAIN_MODE="${OUTPUT_RETAIN_MODE:-compact}"
KEEP_BEST_QLAT="${KEEP_BEST_QLAT:-false}"
COPY_FAILED_OUTPUTS="${COPY_FAILED_OUTPUTS:-true}"
COMPRESS_JOB_LOG="${COMPRESS_JOB_LOG:-true}"

case "$CALIBRATION_ALGORITHM" in
    pso|dds)
        ;;
    *)
        echo "Unsupported CALIBRATION_ALGORITHM: $CALIBRATION_ALGORITHM" >&2
        exit 1
        ;;
esac

if [ -f "$PYTHON_VENV" ]; then
    source "$PYTHON_VENV"
else
    echo "Python venv activation file not found: $PYTHON_VENV" >&2
    exit 1
fi

cd "$NGSH_ROOT"

mapfile -t GAGES < <(awk -F',' 'NR>1 {gsub(/\r/,"",$1); if ($1!="") print $1}' "$BASIN_CSV")
N_GAGES="${#GAGES[@]}"
if [ "$N_GAGES" -eq 0 ]; then
    echo "No gages found in $BASIN_CSV" >&2
    exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be run as a Slurm array job.}"
GAGE_INDEX=$(( TASK_ID / 2 ))
VARIANT_INDEX=$(( TASK_ID % 2 ))

if [ "$GAGE_INDEX" -ge "$N_GAGES" ]; then
    echo "Array task $TASK_ID maps to gage index $GAGE_INDEX, but only $N_GAGES gages exist." >&2
    exit 1
fi

GAGE_ID="${GAGES[$GAGE_INDEX]}"
if [ "$VARIANT_INDEX" -eq 0 ]; then
    FORMULATION_VARIANT="pet"
else
    FORMULATION_VARIANT="nom"
fi

case "${CALIBRATION_ALGORITHM}:${MODEL}:${FORMULATION_VARIANT}" in
    pso:casam:pet)
        SANDBOX_CONFIG="$NGSH_ROOT/configs/sandbox_config_casam.yaml"
        CALIB_SCRIPT="$NGSH_ROOT/model_assessment/calib_scripts/pso_calibration_casam.py"
        ;;
    pso:casam:nom)
        SANDBOX_CONFIG="$NGSH_ROOT/configs/sandbox_config_nom_casam.yaml"
        CALIB_SCRIPT="$NGSH_ROOT/model_assessment/calib_scripts/pso_calibration_casam.py"
        ;;
    pso:cfe:pet)
        SANDBOX_CONFIG="$NGSH_ROOT/configs/sandbox_config_cfe.yaml"
        CALIB_SCRIPT="$NGSH_ROOT/model_assessment/calib_scripts/pso_calibration_cfe.py"
        ;;
    pso:cfe:nom)
        SANDBOX_CONFIG="$NGSH_ROOT/configs/sandbox_config_nom_cfe.yaml"
        CALIB_SCRIPT="$NGSH_ROOT/model_assessment/calib_scripts/pso_calibration_cfe.py"
        ;;
    dds:casam:pet)
        SANDBOX_CONFIG="$NGSH_ROOT/configs/sandbox_config_casam.yaml"
        CALIB_SCRIPT="$NGSH_ROOT/model_assessment/calib_scripts/dds_calibration_casam.py"
        ;;
    dds:casam:nom)
        SANDBOX_CONFIG="$NGSH_ROOT/configs/sandbox_config_nom_casam.yaml"
        CALIB_SCRIPT="$NGSH_ROOT/model_assessment/calib_scripts/dds_calibration_casam.py"
        ;;
    dds:cfe:pet)
        SANDBOX_CONFIG="$NGSH_ROOT/configs/sandbox_config_cfe.yaml"
        CALIB_SCRIPT="$NGSH_ROOT/model_assessment/calib_scripts/dds_calibration_cfe.py"
        ;;
    dds:cfe:nom)
        SANDBOX_CONFIG="$NGSH_ROOT/configs/sandbox_config_nom_cfe.yaml"
        CALIB_SCRIPT="$NGSH_ROOT/model_assessment/calib_scripts/dds_calibration_cfe.py"
        ;;
    *)
        echo "Unsupported CALIBRATION_ALGORITHM/MODEL/FORMULATION_VARIANT: ${CALIBRATION_ALGORITHM}/${MODEL}/${FORMULATION_VARIANT}" >&2
        exit 1
        ;;
esac

if [ "$CALIBRATION_ALGORITHM" = "pso" ]; then
    VARIANT_LABEL="${MODEL}_${FORMULATION_VARIANT}"
    WORK_VARIANT_LABEL="$VARIANT_LABEL"
else
    VARIANT_LABEL="${CALIBRATION_ALGORITHM}_${MODEL}_${FORMULATION_VARIANT}"
    # sandbox.py and the calibration scripts still write under the model/formulation
    # label; keep DDS separated only at the final destination path.
    WORK_VARIANT_LABEL="${MODEL}_${FORMULATION_VARIANT}"
fi
if [ -z "${CONF_PARTICLES:-}" ]; then
    if [ "$CALIBRATION_ALGORITHM" = "dds" ]; then
        CONF_PARTICLES=1
    else
        CONF_PARTICLES="$N_PARTICLES"
    fi
fi
NODE_TMP="${SLURM_TMPDIR:-/tmp/${USER}/ngsh_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
TMP_INPUT_DIR="$NODE_TMP/in"
TMP_MODEL_ROOT="$NODE_TMP/model_root"
TMP_BASIN_CSV="$NODE_TMP/basin_IDs_${GAGE_ID}.csv"
TMP_VARIANT_GAGE_DIR="$TMP_MODEL_ROOT/out/$WORK_VARIANT_LABEL/$GAGE_ID"
TMP_COMPACT_GAGE_DIR="$NODE_TMP/compact/$VARIANT_LABEL/$GAGE_ID"
DEST_VARIANT_DIR="$DEST_OUTPUT_ROOT/$VARIANT_LABEL"
DEST_VARIANT_GAGE_DIR="$DEST_VARIANT_DIR/$GAGE_ID"
FAILED_DEST_GAGE_DIR="$DEST_VARIANT_DIR/_failed/${GAGE_ID}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$TMP_INPUT_DIR" "$TMP_VARIANT_GAGE_DIR/logging" "$DEST_VARIANT_DIR"
printf 'gage_id,num_divides\n%s,0\n' "$GAGE_ID" > "$TMP_BASIN_CSV"

JOB_LOG="$TMP_VARIANT_GAGE_DIR/logging/slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${VARIANT_LABEL}_${GAGE_ID}.log"

copy_back_outputs() {
    status=$?
    set +e
    if [ -d "$TMP_VARIANT_GAGE_DIR" ]; then
        if [ "$status" -eq 0 ]; then
            COPY_SOURCE="$TMP_VARIANT_GAGE_DIR"
            if [ "$OUTPUT_RETAIN_MODE" != "full" ]; then
                echo "Compacting successful outputs with OUTPUT_RETAIN_MODE=$OUTPUT_RETAIN_MODE ..."
                python "$NGSH_ROOT/model_assessment/util/compact_calibration_outputs.py" \
                    --source "$TMP_VARIANT_GAGE_DIR" \
                    --dest "$TMP_COMPACT_GAGE_DIR" \
                    --gage-id "$GAGE_ID" \
                    --variant "$VARIANT_LABEL" \
                    --mode "$OUTPUT_RETAIN_MODE" \
                    --keep-best-qlat "$KEEP_BEST_QLAT" \
                    --compress-logs "$COMPRESS_JOB_LOG"
                if [ "$?" -eq 0 ] && [ -d "$TMP_COMPACT_GAGE_DIR" ]; then
                    COPY_SOURCE="$TMP_COMPACT_GAGE_DIR"
                else
                    echo "Output compaction failed; falling back to full output copy."
                fi
            fi
            rm -rf "$DEST_VARIANT_GAGE_DIR"
            mkdir -p "$DEST_VARIANT_GAGE_DIR"
            cp -a "$COPY_SOURCE/." "$DEST_VARIANT_GAGE_DIR/"
        elif [ "$COPY_FAILED_OUTPUTS" = "true" ]; then
            mkdir -p "$FAILED_DEST_GAGE_DIR"
            cat > "$TMP_VARIANT_GAGE_DIR/FAILED_RUN.txt" <<EOF
status=$status
job_id=${SLURM_JOB_ID:-}
array_task_id=${SLURM_ARRAY_TASK_ID:-}
model=$MODEL
formulation_variant=$FORMULATION_VARIANT
gage_id=$GAGE_ID
date=$(date)
EOF
            cp -a "$TMP_VARIANT_GAGE_DIR/." "$FAILED_DEST_GAGE_DIR/"
        fi
    fi
    exit "$status"
}
trap copy_back_outputs EXIT

exec > "$JOB_LOG" 2>&1

echo "=== one-gage calibration array task ==="
echo "date:                    $(date)"
echo "job id:                  ${SLURM_JOB_ID:-}"
echo "array task id:           $TASK_ID"
echo "algorithm:               $CALIBRATION_ALGORITHM"
echo "model:                   $MODEL"
echo "formulation variant:     $FORMULATION_VARIANT"
echo "gage id:                 $GAGE_ID"
echo "node tmp:                $NODE_TMP"
echo "shared input dir:        $SHARED_INPUT_DIR"
echo "tmp input dir:           $TMP_INPUT_DIR"
echo "tmp model root:          $TMP_MODEL_ROOT"
echo "destination output root: $DEST_OUTPUT_ROOT"
echo "destination gage dir:    $DEST_VARIANT_GAGE_DIR"
echo "sandbox config:          $SANDBOX_CONFIG"
echo "calibration script:      $CALIB_SCRIPT"
echo "particles / iterations:  $N_PARTICLES / $N_ITERATIONS"
echo "conf particles:          $CONF_PARTICLES"
echo "time windows:            spinup=$SPINUP_START cal=$CAL_START..$CAL_END val=$VAL_START..$VAL_END"
echo "output retain mode:      $OUTPUT_RETAIN_MODE"
echo "keep best qlat:          $KEEP_BEST_QLAT"
echo "copy failed outputs:     $COPY_FAILED_OUTPUTS"
echo

export NGSH_ROOT
export CIROH_INPUT_DIR="$TMP_INPUT_DIR"
export CIROH_HF_GPKG
export NGEN_DIR
export NGEN_MODEL_ROOT="$TMP_MODEL_ROOT"
export BASIN_CSV="$TMP_BASIN_CSV"
export DOWNSTREAM_FLOWPATH_SUMMARY
export NGEN_JOB_CORES="${SLURM_CPUS_PER_TASK:-1}"
export NGEN_TROUTE_CPU_POOL=1
export OMP_NUM_THREADS=1
export NGEN_SANDBOX_CONFIG="$SANDBOX_CONFIG"

echo "Staging selected gage input to node-local tmp..."
TRANSFER_SOURCE_BASE="$SHARED_INPUT_DIR" \
TRANSFER_DEST_BASE="$TMP_INPUT_DIR" \
TRANSFER_WORKERS="$TRANSFER_WORKERS" \
TRANSFER_GAGE_ID="$GAGE_ID" \
python "$NGSH_ROOT/model_assessment/util/transfer_forcing.py"

echo
echo "Running -conf..."
python "$NGSH_ROOT/sandbox.py" \
    -i "$SANDBOX_CONFIG" \
    -conf \
    --gage_id "$GAGE_ID" \
    --concurrent-particles \
    --num-particles "$CONF_PARTICLES"

echo
echo "Running calibration..."
CALIBRATION_CMD=(
    python "$CALIB_SCRIPT"
    --sandbox-config "$SANDBOX_CONFIG"
    --gage-id "$GAGE_ID"
    --n-iterations "$N_ITERATIONS"
    --max-gage-procs 1
    --spinup-start "$SPINUP_START"
    --cal-start "$CAL_START"
    --cal-end "$CAL_END"
    --val-start "$VAL_START"
    --val-end "$VAL_END"
)
if [ "$CALIBRATION_ALGORITHM" = "pso" ]; then
    CALIBRATION_CMD+=(--n-particles "$N_PARTICLES")
    CALIBRATION_CMD+=(--max-particle-procs "$MAX_PARTICLE_PROCS")
fi
"${CALIBRATION_CMD[@]}"

echo
echo "Completed successfully at $(date). Outputs will be copied back by the EXIT trap."
