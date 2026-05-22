# Next-Generation Framework Sandbox Hub (NextGenSandboxHub)
[NextGen](https://github.com/NOAA-OWP/ngen), Next-Generation Water Resources Modeling Framework, developed by the NOAA's Office of Water Prediction is a standards-based language- and model-agnostic framework, which allows to run a mosaic of surface and subsurface models in a single basin comprised of 10s-100s sub-catchments. 

## Schematic of the NextGenSandboxHub Workflow

<div align="center">
<img src="https://github.com/user-attachments/assets/d06b3cf9-6019-4ebd-86f1-e797b4debbae" style="width:800px; height:400px;"/>
</div>

## Current HF 2.2 Single-Gage Calibration Workflow

This branch is configured for a one-gage-per-job calibration workflow. The intended HPC pattern is:

1. Use pre-existing HF 2.2 geopackages and forcing files.
2. Run `sandbox.py -conf` for exactly one gage to create model configs and particle-local workspaces.
3. Run one calibration script for that same gage.
4. Launch many independent Slurm array tasks, where each task handles one gage ID.

Do not run `-subset` or `-forc` in this workflow unless you explicitly intend to regenerate input data. The expected input layout is:

```text
${CIROH_INPUT_DIR}/${GAGE_ID}/data/gage_${GAGE_ID}.gpkg
${CIROH_INPUT_DIR}/${GAGE_ID}/data/forcing/
${NGSH_ROOT}/model_assessment/USGS_streamflow/successful_sites_resampled/${GAGE_ID}.csv
```

### Required Environment

Activate the Python environment and set the path variables before running configuration or calibration:

```bash
source /Users/peterlafollette/.venv_sandbox_py3.11/bin/activate

cd /path/to/NextGenSandboxHub

export NGSH_ROOT="$PWD"
export GAGE_ID=01089100
export CIROH_INPUT_DIR="/path/to/standardized_CIROH_project/inhf22"
export CIROH_HF_GPKG="/path/to/standardized_CIROH_project/conus_nextgen_updated_fixed.gpkg"
export NGEN_DIR="/path/to/ngen"
export NGEN_MODEL_ROOT="/tmp/ngen_${USER}_${GAGE_ID}"
export BASIN_CSV="$NGSH_ROOT/basin_IDs/basin_IDs.csv"

export NGEN_JOB_CORES=1
export NGEN_TROUTE_CPU_POOL=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

`NGEN_MODEL_ROOT` should be unique per gage/job, especially on the HPC. The calibration scripts read and write under:

```text
${NGEN_MODEL_ROOT}/out/${GAGE_ID}/
```

### Choose The Formulation

Select the model formulation in [configs/sandbox_config.yaml](configs/sandbox_config.yaml):

```yaml
formulation:
  models: "PET, CASAM, T-route"
  # models: "NOM, CASAM, T-route"
  # models: "PET, CFE, T-route"
  # models: "NOM, CFE, T-route"
```

HF 2.2 should leave divide-attribute computation disabled:

```yaml
# compute_divide_attributes: true
```

### Configure One Gage

Generate the downstream-flowpath summary for the exact HF 2.2 hydrofabric, then generate configs for the target gage:

```bash
export GAGE_ID=01089100
export NGEN_GAGE_ID="$GAGE_ID"
export DOWNSTREAM_FLOWPATH_SUMMARY="${NGEN_MODEL_ROOT}/downstream_flowpath_summary.csv"

mkdir -p "$NGEN_MODEL_ROOT/out"

python model_assessment/util/get_penult_ids.py \
  --gage-id "$GAGE_ID" \
  --hf-gpkg "$CIROH_HF_GPKG"

python sandbox.py -conf -i configs/sandbox_config.yaml \
  --gage_id "$GAGE_ID" \
  --concurrent-particles \
  --num-particles 15
```

`--concurrent-particles` only creates particle-local workspaces such as:

```text
${NGEN_MODEL_ROOT}/out/${GAGE_ID}/particles/p0/
${NGEN_MODEL_ROOT}/out/${GAGE_ID}/particles/p1/
...
```

It does not by itself allocate cores or run particles. For a one-core job, cap the calibration with `--max-particle-procs 1`.

### Run Calibration Outside Slurm

Run one of the calibration scripts for the configured gage:

```bash
# CASAM PSO
python model_assessment/calib_scripts/pso_calibration_casam.py \
  --gage-id "$GAGE_ID" \
  --n-particles 15 \
  --n-iterations 50 \
  --max-particle-procs 1 \
  --max-gage-procs 1 \
  --sandbox-config configs/sandbox_config.yaml

# CFE PSO
python model_assessment/calib_scripts/pso_calibration_cfe.py \
  --gage-id "$GAGE_ID" \
  --n-particles 15 \
  --n-iterations 50 \
  --max-particle-procs 1 \
  --max-gage-procs 1 \
  --sandbox-config configs/sandbox_config.yaml

# CASAM DDS
python model_assessment/calib_scripts/dds_calibration_casam.py \
  --gage-id "$GAGE_ID" \
  --n-iterations 100 \
  --max-gage-procs 1 \
  --sandbox-config configs/sandbox_config.yaml

# CFE DDS
python model_assessment/calib_scripts/dds_calibration_cfe.py \
  --gage-id "$GAGE_ID" \
  --n-iterations 100 \
  --max-gage-procs 1 \
  --sandbox-config configs/sandbox_config.yaml
```

LASAM variants are also available:

```bash
python model_assessment/calib_scripts/pso_calibration_lasam.py --gage-id "$GAGE_ID"
python model_assessment/calib_scripts/dds_calibration_lasam.py --gage-id "$GAGE_ID"
```

The calibration objective defaults to KGE. Calibration CSV logs include elapsed wall time, core count, and core-hours.

### Outputs

The compact calibration log is written to:

```text
${NGSH_ROOT}/logging/${GAGE_ID}.csv
```

Failure-only diagnostics may be written to:

```text
${NGSH_ROOT}/logging/${GAGE_ID}_errors.log
${NGSH_ROOT}/logging/${GAGE_ID}_incomplete.csv
```

The final routed hydrograph for the best parameter set is written under the particle-0 workspace:

```text
${NGEN_MODEL_ROOT}/out/${GAGE_ID}/particles/p0/postproc/${GAGE_ID}_best.csv
```

Particle workspaces also contain the particle-local configs, realization JSONs, divide outputs, t-route NetCDFs, and intermediate hydrographs.

### Slurm Pattern

On the HPC, prefer a Slurm array where each array task resolves one `GAGE_ID` from `BASIN_CSV`, creates a unique `NGEN_MODEL_ROOT`, runs `get_penult_ids.py`, runs `sandbox.py -conf`, and then runs exactly one calibration script. Use `--array=1-N%M` to control the number of simultaneous gage jobs.

## Setup And Optional Data Preparation

### <ins>  Step 1. Build Sandbox Workflow
  - `git clone https://github.com/ajkhattak/NextGenSandboxHub && cd NextGenSandboxHub`
  - `git submodule update --init`
  - Run `./utils/build_sandbox.sh` (this will install python env required for the workflow, t-route, and ngen)
  
### <ins>  Step 2. Hydrofabric Installation
Ensure R and Rtools are already installed before proceeding. There are two ways to install the required packages:
  #### Option 1: Using RStudio
  1. Open RStudio
  2. Load and run the installation script by sourcing it:
     - Open `<path_to_sandboxhub>/src/R/install_load_libs.R` in RStudio.
     - Click Source to execute the script.
     - Alternatively, run the following command in the RStudio Console:
       ```
       source("~/<path_to_sandboxhub>/src/R/install_load_libs.R")
       ```
  #### Option 2: Using the Command Line
  Run the following command in a terminal or command prompt:
  ```
   Rscript <path_to_sandboxhub>/src/R/install_load_libs.R
  ```

### <ins> Step 3. Hydrofabric Subsetting
For the current HF 2.2 calibration workflow, this step is normally skipped because geopackages already exist under `CIROH_INPUT_DIR`. Only run this step when intentionally creating or replacing basin geopackages.

  - Dependency: Step 2
  - Download domain (CONUS or oCONUS) from [lynker-spatial](https://www.lynker-spatial.com/data?path=hydrofabric%2Fv2.2%2F), for instance conus/conus_nextgen.gpkg
  - open `<path_to_sandboxhub>/configs/sandbox_config.yaml` [here](configs/sandbox_config.yaml) and adjust sandbox_dir, input_dir, output_dir, and subsetting according to your local settings
  - Now there are two options to proceed:
      - run `python <path_to_sandboxhub>/sandbox.py -subset`
      - or open `<path_to_sandboxhub>/src/R/main.R` in RStudio and source on main.R. Note Set file name `infile_config` [here](https://github.com/ajkhattak/NextGenSandboxHub/blob/main/src/R/main.R#L53) 
    
    Either one will install the hydrofabric and several other libraries, and if everything goes well, a basin geopackage will be subsetted and stored under `<input_dir>/<basin_id>/data/gage_<basin_id>.gpkg`

### <ins> Step 4. Forcing Data Download
For the current HF 2.2 calibration workflow, this step is normally skipped because forcing files already exist under `CIROH_INPUT_DIR`. Only run this step when intentionally downloading or replacing forcing data.

The workflow uses [CIROH_DL_NextGen](https://github.com/ajkhattak/CIROH_DL_NextGen) forcing_prep tool to download atmospheric forcing data. It uses a Python environment (`~/.venv_forcing`) that is created during the workflow setup step (Step 1). To download the forcing data run:
```
   python <path_to_sandboxhub>/sandbox.py -forc
```

====================================================================================
### Note: Steps 5 and 6 require both the ngen and models builds. Please follow the instructions in the [build_models](https://github.com/ajkhattak/NextGenSandboxHub/blob/main/utils/build_models.sh) script to build ngen and models.
====================================================================================

Note: The sandbox workflow assumes that [ngen](https://github.com/NOAA-OWP/ngen) and models including [t-route](https://github.com/NOAA-OWP/t-route) have been built in the Python virtual environment created in Step 1.

 ### <ins>  Step 5a. Determine Nexus Used In Calibration
For HF 2.2, generate this from the exact full hydrofabric used by the run. In one-gage-per-job mode, prefer a per-job summary file:
 ```
    export DOWNSTREAM_FLOWPATH_SUMMARY="${NGEN_MODEL_ROOT}/downstream_flowpath_summary.csv"
    python model_assessment/util/get_penult_ids.py --gage-id "$GAGE_ID" --hf-gpkg "$CIROH_HF_GPKG"
 ```

### <ins>  Step 5b. Generate Configuration and Realization Files
To generate configuration and realization files, set up the `formulation` block in the sandbox config file [here](configs/sandbox_config.yaml), then run `-conf` for one gage:
 ```
    python sandbox.py -conf -i configs/sandbox_config.yaml \
      --gage_id "$GAGE_ID" \
      --concurrent-particles \
      --num-particles "$N_PARTICLES"
 ```
The current calibration scripts expect the particle-local workspaces created by `--concurrent-particles`.

### <ins> Step 6. Run Calibration/Validation Simulations
Run the calibration script for the same gage. For a one-core job, cap particle concurrency:
 ```
    python model_assessment/calib_scripts/pso_calibration_casam.py \
      --gage-id "$GAGE_ID" \
      --n-particles "$N_PARTICLES" \
      --n-iterations "$N_ITERATIONS" \
      --max-particle-procs 1 \
      --max-gage-procs 1 \
      --sandbox-config configs/sandbox_config.yaml
 ```

#### Summary
1. Use existing HF 2.2 geopackage and forcing data.
2. Generate the per-gage downstream-flowpath summary.
3. Run `sandbox.py -conf --gage_id "$GAGE_ID" --concurrent-particles --num-particles "$N_PARTICLES"`.
4. Run one calibration script for that gage.
5. On the HPC, repeat this through a Slurm array with one gage per task.

Available calibration scripts:

```bash
python model_assessment/calib_scripts/pso_calibration_casam.py
python model_assessment/calib_scripts/dds_calibration_casam.py
python model_assessment/calib_scripts/pso_calibration_cfe.py
python model_assessment/calib_scripts/dds_calibration_cfe.py
python model_assessment/calib_scripts/pso_calibration_lasam.py
python model_assessment/calib_scripts/dds_calibration_lasam.py
```

The compact per-gage calibration log is written to `logging/${GAGE_ID}.csv`.
