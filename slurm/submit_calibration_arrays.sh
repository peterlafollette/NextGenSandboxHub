#!/bin/bash

set -euo pipefail

# Submit one Slurm array per requested hydro model. Each array task runs exactly
# one gage and one formulation variant:
#   task 0 -> gage 0 PET
#   task 1 -> gage 0 NOM
#   task 2 -> gage 1 PET
#   task 3 -> gage 1 NOM
#   ...

NGSH_ROOT="${NGSH_ROOT:-/users/4/plafolle/infil_proj/NextGenSandboxHub}"
BASIN_CSV="${BASIN_CSV:-$NGSH_ROOT/basin_IDs/basin_IDs.csv}"
MODELS="${MODELS:-casam,cfe}"
CALIBRATION_ALGORITHM="${CALIBRATION_ALGORITHM:-pso}"
CASAM_MODE="${CASAM_MODE:-standard}"
MAX_ARRAY_CONCURRENT="${MAX_ARRAY_CONCURRENT:-50}"
ARRAY_SCRIPT="${ARRAY_SCRIPT:-$NGSH_ROOT/slurm/run_gage_variant_array.sh}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM_PER_CPU="${MEM_PER_CPU:-5G}"
TMP_DISK="${TMP_DISK:-20G}"
TIME_LIMIT="${TIME_LIMIT:-4-0}"

if [ ! -f "$BASIN_CSV" ]; then
    echo "Missing BASIN_CSV: $BASIN_CSV" >&2
    exit 1
fi

N_GAGES=$(awk -F',' 'NR>1 && $1!="" {c++} END{print c+0}' "$BASIN_CSV")
if [ "$N_GAGES" -eq 0 ]; then
    echo "No gages found in $BASIN_CSV" >&2
    exit 1
fi

LAST_TASK=$(( N_GAGES * 2 - 1 ))

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
for model in "${MODEL_LIST[@]}"; do
    model="$(echo "$model" | tr '[:upper:]' '[:lower:]' | xargs)"
    case "$model" in
        casam|cfe)
            if [ "$model" = "casam" ]; then
                case "$CASAM_MODE" in
                    standard|dual_fd)
                        ;;
                    *)
                        echo "Unsupported CASAM_MODE: $CASAM_MODE (expected standard or dual_fd)" >&2
                        exit 1
                        ;;
                esac
            fi
            job_model_label="$model"
            if [ "$model" = "casam" ] && [ "$CASAM_MODE" = "dual_fd" ]; then
                job_model_label="dual_fd_casam"
            fi
            if [ "$CALIBRATION_ALGORITHM" = "pso" ]; then
                JOB_NAME="ngsh_${job_model_label}"
            else
                JOB_NAME="ngsh_${CALIBRATION_ALGORITHM}_${job_model_label}"
            fi
            echo "Submitting $model array: $N_GAGES gages x PET/NOM = $((N_GAGES * 2)) tasks"
            echo "  algorithm=${CALIBRATION_ALGORITHM}; throttle=${MAX_ARRAY_CONCURRENT}; cpus/task=${CPUS_PER_TASK}; mem/cpu=${MEM_PER_CPU}; tmp/task=${TMP_DISK}; time=${TIME_LIMIT}"
            sbatch \
                --job-name="$JOB_NAME" \
                --array="0-${LAST_TASK}%${MAX_ARRAY_CONCURRENT}" \
                --cpus-per-task="$CPUS_PER_TASK" \
                --mem-per-cpu="$MEM_PER_CPU" \
                --tmp="$TMP_DISK" \
                --time="$TIME_LIMIT" \
                --export=ALL,CALIBRATION_ALGORITHM="$CALIBRATION_ALGORITHM",CASAM_MODE="$CASAM_MODE",MODEL="$model",NGSH_ROOT="$NGSH_ROOT",BASIN_CSV="$BASIN_CSV" \
                "$ARRAY_SCRIPT"
            ;;
        *)
            echo "Skipping unsupported model: $model" >&2
            ;;
    esac
done
