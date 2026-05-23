# NextGenSandboxHub

[NextGen](https://github.com/NOAA-OWP/ngen) is NOAA-OWP's model-agnostic hydrologic modeling framework. This branch has been refactored for HF 2.2 calibration runs where each job handles one gage/catchment, with independent output directories for CASAM/CFE and PET/NOM formulation variants.

<div align="center">
<img src="https://github.com/user-attachments/assets/d06b3cf9-6019-4ebd-86f1-e797b4debbae" style="width:800px; height:400px;"/>
</div>

## Current Workflow

This branch is intended for pre-existing HF 2.2 inputs. In normal calibration runs, do not run `sandbox.py -subset` or `sandbox.py -forc`; those steps regenerate input data and can overwrite or modify geopackage/forcing products.

The expected input layout is:

```text
${CIROH_INPUT_DIR}/${GAGE_ID}/data/gage_${GAGE_ID}.gpkg
${CIROH_INPUT_DIR}/${GAGE_ID}/data/forcing/2010_to_2022/*_corrected.nc
${NGSH_ROOT}/model_assessment/USGS_streamflow/successful_sites_resampled/${GAGE_ID}.csv
```

The one-gage-per-job pattern is:

1. Generate or update `model_assessment/util/downstream_flowpath_summary.csv` from the exact full HF 2.2 geopackage and the current `basin_IDs.csv`.
2. Run `sandbox.py -conf` for one gage, one hydro model, and one formulation variant.
3. Run the matching calibration script for that same gage.
4. On the HPC, repeat this through Slurm arrays where each array task is one gage plus one formulation variant.

## Important Paths

The code is driven mostly by environment variables:

```bash
export NGSH_ROOT=/path/to/NextGenSandboxHub
export BASIN_CSV=$NGSH_ROOT/basin_IDs/basin_IDs.csv
export CIROH_INPUT_DIR=/path/to/inhf22
export CIROH_HF_GPKG=/path/to/conus_nextgen_updated_fixed.gpkg
export NGEN_DIR=/path/to/ngen
export NGEN_MODEL_ROOT=/path/to/run/root
export DOWNSTREAM_FLOWPATH_SUMMARY=$NGSH_ROOT/model_assessment/util/downstream_flowpath_summary.csv
```

`NGEN_MODEL_ROOT` controls where local/manual outputs go. The variant config files place outputs under:

```text
${NGEN_MODEL_ROOT}/out/casam_pet/${GAGE_ID}
${NGEN_MODEL_ROOT}/out/casam_nom/${GAGE_ID}
${NGEN_MODEL_ROOT}/out/cfe_pet/${GAGE_ID}
${NGEN_MODEL_ROOT}/out/cfe_nom/${GAGE_ID}
```

On the HPC Slurm workflow, `DEST_MODEL_ROOT` is the final shared output location. Jobs run in node-local tmp first, then copy the finished variant/gage directory back to:

```text
${DEST_MODEL_ROOT}/${MODEL}_${FORMULATION_VARIANT}/${GAGE_ID}
```

For Agate, use a shared/project path for `DEST_MODEL_ROOT`, not home space.

## Configuration Files

Use the variant-specific sandbox configs for calibration:

| Config | Formulation | Output subdir |
| --- | --- | --- |
| `configs/sandbox_config_casam.yaml` | `PET, CASAM, T-route` | `out/casam_pet` |
| `configs/sandbox_config_nom_casam.yaml` | `NOM, CASAM, T-route` | `out/casam_nom` |
| `configs/sandbox_config_cfe.yaml` | `PET, CFE, T-route` | `out/cfe_pet` |
| `configs/sandbox_config_nom_cfe.yaml` | `NOM, CFE, T-route` | `out/cfe_nom` |

`configs/sandbox_config.yaml` is still a generic/manual config, but the concurrent launcher and Slurm scripts use the variant-specific files above.

HF 2.2 should leave divide-attribute computation disabled:

```yaml
# compute_divide_attributes: true
```

## Generate The HF 2.2 Downstream Summary

Run this after changing `basin_IDs/basin_IDs.csv`, or when switching to a different full hydrofabric geopackage:

```bash
cd /users/4/plafolle/infil_proj/NextGenSandboxHub
source /users/4/plafolle/.venv_sandbox_py3.11/bin/activate

export NGSH_ROOT=/users/4/plafolle/infil_proj/NextGenSandboxHub
export BASIN_CSV=$NGSH_ROOT/basin_IDs/basin_IDs.csv
export CIROH_HF_GPKG=/users/4/plafolle/infil_proj/conus_nextgen_updated_fixed.gpkg
export DOWNSTREAM_FLOWPATH_SUMMARY=$NGSH_ROOT/model_assessment/util/downstream_flowpath_summary.csv

python model_assessment/util/get_penult_ids.py \
  --hf-gpkg "$CIROH_HF_GPKG" \
  --gage-file "$BASIN_CSV"
```

The output file is used by calibration and t-route post-processing to identify the downstream nexus/flowpaths for the gage of interest.

## Manual One-Gage Run Outside Slurm

Example for CASAM + PET:

```bash
cd /users/4/plafolle/infil_proj/NextGenSandboxHub
source /users/4/plafolle/.venv_sandbox_py3.11/bin/activate

export NGSH_ROOT=/users/4/plafolle/infil_proj/NextGenSandboxHub
export GAGE_ID=08158927
export BASIN_CSV=$NGSH_ROOT/basin_IDs/basin_IDs.csv
export CIROH_INPUT_DIR=/projects/standard/nieberj/shared/plafolle/inhf22
export CIROH_HF_GPKG=/users/4/plafolle/infil_proj/conus_nextgen_updated_fixed.gpkg
export NGEN_DIR=/users/4/plafolle/CIROH_project/ngen
export NGEN_MODEL_ROOT=/projects/standard/nieberj/shared/plafolle/infil_proj/out/manual_test
export DOWNSTREAM_FLOWPATH_SUMMARY=$NGSH_ROOT/model_assessment/util/downstream_flowpath_summary.csv

export NGEN_JOB_CORES=4
export NGEN_TROUTE_CPU_POOL=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python sandbox.py \
  -i configs/sandbox_config_casam.yaml \
  -conf \
  --gage_id "$GAGE_ID" \
  --concurrent-particles \
  --num-particles 2

python model_assessment/calib_scripts/pso_calibration_casam.py \
  --sandbox-config configs/sandbox_config_casam.yaml \
  --gage-id "$GAGE_ID" \
  --n-particles 2 \
  --n-iterations 2 \
  --max-particle-procs 2 \
  --max-gage-procs 1 \
  --spinup-start 2010-10-01 \
  --cal-start 2011-01-01 \
  --cal-end 2011-12-31 \
  --val-start 2012-01-01 \
  --val-end 2012-06-30
```

Swap the config/script pair for other variants, keeping the same calibration arguments shown above:

```bash
# CASAM + NOM
python sandbox.py -i configs/sandbox_config_nom_casam.yaml -conf --gage_id "$GAGE_ID" --concurrent-particles --num-particles 2
python model_assessment/calib_scripts/pso_calibration_casam.py --sandbox-config configs/sandbox_config_nom_casam.yaml --gage-id "$GAGE_ID"

# CFE + PET
python sandbox.py -i configs/sandbox_config_cfe.yaml -conf --gage_id "$GAGE_ID" --concurrent-particles --num-particles 2
python model_assessment/calib_scripts/pso_calibration_cfe.py --sandbox-config configs/sandbox_config_cfe.yaml --gage-id "$GAGE_ID"

# CFE + NOM
python sandbox.py -i configs/sandbox_config_nom_cfe.yaml -conf --gage_id "$GAGE_ID" --concurrent-particles --num-particles 2
python model_assessment/calib_scripts/pso_calibration_cfe.py --sandbox-config configs/sandbox_config_nom_cfe.yaml --gage-id "$GAGE_ID"
```

`--concurrent-particles` only creates particle-local workspaces. It does not allocate cores by itself. Actual particle concurrency is controlled by `--max-particle-procs`.

## Local Concurrent Smoke Test

`run_concurrent_variants.py` automates small local tests across multiple models, formulation variants, and gages. It never runs `-subset` or `-forc`.

```bash
cd /Users/peterlafollette/CIROH_single_catch_per_job_refactor/NextGenSandboxHub
source /Users/peterlafollette/.venv_sandbox_py3.11/bin/activate

python model_assessment/calib_scripts/run_concurrent_variants.py \
  --models casam,cfe \
  --formulation-variants pet,nom \
  --gage-id 08158927,01311810 \
  --input-dir /Volumes/OWCEnvoyProFX/inhf22/in \
  --hf-gpkg /Users/peterlafollette/standardized_CIROH_project/conus_nextgen_updated_fixed.gpkg \
  --ngen-dir /Users/peterlafollette/CIROH_project/ngen \
  --ngen-model-root /Users/peterlafollette/CIROH_single_catch_per_job_refactor/local_concurrency_test \
  --downstream-flowpath-summary /Users/peterlafollette/CIROH_single_catch_per_job_refactor/NextGenSandboxHub/model_assessment/util/downstream_flowpath_summary.csv \
  --n-particles 2 \
  --n-iterations 2 \
  --max-particle-procs 2 \
  --max-gage-procs 1 \
  --max-concurrent-conf 1 \
  --max-concurrent-calibrations 4 \
  --job-cores 4 \
  --troute-cpu-pool 1 \
  --omp-num-threads 1 \
  --spinup-start 2010-10-01 \
  --cal-start 2011-01-01 \
  --cal-end 2011-12-31 \
  --val-start 2012-01-01 \
  --val-end 2012-06-30
```

Use `--conf-only` to stop after configuration, or `--skip-conf` to reuse existing config outputs.

## Slurm Array Workflow

The Slurm helper submits one array per hydro model. Each array task handles exactly one gage and one formulation variant:

```text
task 0 -> gage 0 PET
task 1 -> gage 0 NOM
task 2 -> gage 1 PET
task 3 -> gage 1 NOM
...
```

For 100 gages and `MODELS=casam`, this submits one array with 200 tasks. For `MODELS=casam,cfe`, it submits two arrays, each with 200 tasks. Outputs do not clobber because every task writes to a distinct model/formulation/gage directory.

### Agate Test Submission

```bash
cd /users/4/plafolle/infil_proj/NextGenSandboxHub
source /users/4/plafolle/.venv_sandbox_py3.11/bin/activate

export NGSH_ROOT=/users/4/plafolle/infil_proj/NextGenSandboxHub
export BASIN_CSV=$NGSH_ROOT/basin_IDs/basin_IDs.csv
export SHARED_INPUT_DIR=/projects/standard/nieberj/shared/plafolle/inhf22
export CIROH_HF_GPKG=/users/4/plafolle/infil_proj/conus_nextgen_updated_fixed.gpkg
export NGEN_DIR=/users/4/plafolle/CIROH_project/ngen
export DEST_MODEL_ROOT=/projects/standard/nieberj/shared/plafolle/infil_proj/out
export DOWNSTREAM_FLOWPATH_SUMMARY=$NGSH_ROOT/model_assessment/util/downstream_flowpath_summary.csv

export MODELS=casam,cfe
export CPUS_PER_TASK=4
export MEM_PER_CPU=8G
export TMP_DISK=20G
export TIME_LIMIT=1:00:00

export N_PARTICLES=2
export N_ITERATIONS=2
export MAX_PARTICLE_PROCS=2
export MAX_ARRAY_CONCURRENT=4

bash slurm/submit_calibration_arrays.sh
```

For a full production PSO run, increase the calibration controls, for example:

```bash
export TIME_LIMIT=4-0
export N_PARTICLES=12
export N_ITERATIONS=42
export MAX_PARTICLE_PROCS=4
export MAX_ARRAY_CONCURRENT=200
```

`MAX_ARRAY_CONCURRENT` throttles how many array tasks run at the same time. It should be chosen based on scheduler/account limits and how many total cores you want active. With `CPUS_PER_TASK=4` and `MAX_ARRAY_CONCURRENT=200`, one model array can use up to 800 allocated cores if the scheduler starts all allowed tasks.

### What The Slurm Task Does

`slurm/run_gage_variant_array.sh`:

1. Resolves `GAGE_ID` and PET/NOM from `SLURM_ARRAY_TASK_ID`.
2. Creates a node-local tmp input and model-root directory.
3. Copies only the selected gage from `SHARED_INPUT_DIR` to node-local tmp with `model_assessment/util/transfer_forcing.py`.
4. Runs `sandbox.py -conf --concurrent-particles`.
5. Runs the PSO calibration script for CASAM or CFE.
6. Copies the completed variant/gage output back to `DEST_MODEL_ROOT`.

The script sets Slurm stdout/stderr to `/dev/null`; run logs are written internally under the gage output directory.

DDS calibration scripts are available, but the current Slurm helper is wired for PSO.

## Outputs And Logs

For local/manual runs:

```text
${NGEN_MODEL_ROOT}/out/${VARIANT}/${GAGE_ID}/
```

For Slurm runs:

```text
${DEST_MODEL_ROOT}/${VARIANT}/${GAGE_ID}/
```

where `VARIANT` is one of `casam_pet`, `casam_nom`, `cfe_pet`, or `cfe_nom`.

Important files:

```text
logging/${GAGE_ID}.csv
logging/${GAGE_ID}_errors.log
logging/${GAGE_ID}_incomplete.csv
logging/slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${VARIANT}_${GAGE_ID}.log
particles/p0/postproc/${GAGE_ID}_best.csv
particles/p*/json/
particles/p*/troute/
particles/p*/postproc/
```

The calibration CSV includes parameter values, calibration/validation metrics, status, wall time, `job_cores`, `particle_pool_size`, `core_hours_to_row`, and final `total_core_hours`. The objective function defaults to KGE.

## Calibration Notes

- CFE calibration uses the upstream CFE-S parameter set and ranges. The generated CFE configs use Schaake/CFE-S, not CFE-X.
- CASAM calibration uses `vG_params_stat_nom_ordered.dat` for CASAM soil parameters.
- CASAM config generation initializes `lateral_flow_psi_threshold=500.0` and `lateral_flow_factor=1.0`.
- CASAM PSO and DDS use `pso_calibration_casam.py`'s `CALIBRATION_REQUEST`, which currently includes `log10_lateral_flow_psi_threshold` and `log10_lateral_flow_factor`. They are searched in log10 space and written back to CASAM as actual values.
- To keep those CASAM lateral-flow parameters static, remove or comment their two entries in `CALIBRATION_REQUEST` before running CASAM PSO or DDS calibration.
- NOM parameters are appended when NOM is present in the formulation. NOM slope/aspect initialization is aligned with the upstream flat-domain approach.
- t-route is configured with `NGEN_TROUTE_CPU_POOL=1` in the Slurm workflow unless you explicitly change it.

Available calibration scripts:

```bash
python model_assessment/calib_scripts/pso_calibration_casam.py
python model_assessment/calib_scripts/dds_calibration_casam.py
python model_assessment/calib_scripts/pso_calibration_cfe.py
python model_assessment/calib_scripts/dds_calibration_cfe.py
python model_assessment/calib_scripts/pso_calibration_lasam.py
python model_assessment/calib_scripts/dds_calibration_lasam.py
```

## Operational Commands

Check Slurm jobs:

```bash
squeue -u $USER
```

Cancel only these calibration arrays, while leaving an interactive or virtual desktop job alone:

```bash
scancel --name=ngsh_casam
scancel --name=ngsh_cfe
```

Pull HPC outputs to the local machine:

```bash
mkdir -p /Users/peterlafollette/CIROH_single_catch_per_job_refactor/HPC_test_results

rsync -avh --progress -z \
  "plafolle@ahl03.agate.msi.umn.edu:/projects/standard/nieberj/shared/plafolle/infil_proj/out/" \
  "/Users/peterlafollette/CIROH_single_catch_per_job_refactor/HPC_test_results/out/"
```

## Legacy Setup And Data Preparation

The original project setup scripts are still present, but the current calibration workflow assumes HF 2.2 geopackages, forcing files, streamflow observations, ngen, CFE, CASAM/LGAR-C, NOM, PET, and t-route already exist.

Use these only when intentionally preparing or replacing data/builds:

```bash
./utils/build_sandbox.sh
./utils/build_models.sh
python sandbox.py -subset
python sandbox.py -forc
```

Again, `-subset` and `-forc` are not part of the normal one-gage calibration workflow.
