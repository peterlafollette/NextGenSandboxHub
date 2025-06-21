
# `dds_calibration_cfe.py` Function Documentation

This script calibrates a NextGen formulation using the CFE model with the **Dynamically Dimensioned Search (DDS)** optimization algorithm. It supports:

- CFE + PET + T-Route or CFE + T-Route + NOM configurations
- Optional calibration of NOM parameters if present in the formulation
- Logging of calibration and validation metrics per gage
- Per-gage parallel execution using multiprocessing
- Final full-period validation run with best parameters

-------------------------------------------------------------------------------

## Global Configuration

- `n_iterations` : Number of DDS iterations to perform
- `max_cores_for_gages` : Number of gages processed in parallel
- `metric_to_calibrate_on` : Metric to minimize (default: "kge")
- Periods for spinup, calibration, and validation are loaded from `time_config.yaml`

-------------------------------------------------------------------------------

## Helper Functions

### `clear_terminal()`
Clears the terminal screen.

### `reflect_bounds(x, low, high)`
Ensures a perturbed parameter stays within bounds by reflecting values that exceed the limits.

-------------------------------------------------------------------------------

## Class: `DDS`

### Constructor Parameters

- `bounds` : List of parameter (min, max) bounds
- `n_iterations` : Number of optimization iterations
- `gage_id` : USGS gage ID
- `config_path` : Path to the initial CFE config file
- `observed_path` : Path to the observed flow CSV
- `postproc_base_path` : Directory where postprocessed output will be stored (hydrograph from the correct nexus)
- `init_params` : Initial parameter values (from config)
- `metric_to_calibrate_on` : Metric to optimize (default: "kge")
- `sigma` : DDS hyperparameter, standard deviation factor for perturbations (default: 0.2)

### Method: `optimize()`

Runs the DDS optimization procedure:

1. Performs initial objective function evaluation and logs it
2. Iteratively perturbs a random subset of parameters
3. Evaluates each candidate parameter set
4. If a candidate improves the objective, it becomes the new best
5. At the end of all iterations:
   - Regenerates the model config with best parameters
   - Runs the full-period model simulation (including validation period)
   - Saves the final metrics (calibration + validation)
   - Writes all metrics and parameters to CSV

-------------------------------------------------------------------------------

## Function: `calibrate_gage_dds(gage_id)`

Calibrates a single gage:

- Constructs paths for model config and observation file
- Extracts initial CFE (and NOM if included) parameters
- Runs DDS with defined bounds and settings

-------------------------------------------------------------------------------

## Main Execution Block

### `if __name__ == "__main__"`

- Loads the gage list from `basins_passed_custom.csv`
- Uses multiprocessing to run `calibrate_gage_dds()` in parallel for each gage

-------------------------------------------------------------------------------

## Notes

- Results are written to `logging/{gage_id}.csv`
- NOM parameter calibration is handled transparently if included in the model
- Final best hydrograph is saved to `{gage_id}_best.csv` via `get_hydrograph.py`
- Output metrics are computed using `compute_metrics()`

-------------------------------------------------------------------------------
