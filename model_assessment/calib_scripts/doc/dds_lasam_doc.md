# `dds_calibration_lasam.py` Function Documentation

This script calibrates a nextgen formulation with lasam using the **Dynamically Dimensioned Search (DDS)** optimization algorithm. It supports:
- 1- or 2-tile model configurations (each tile corresponds to a different parameter set of the lasam based formulation)
- Optional calibration of NOM parameters (if NOM is in the formulation)
- Per-gage parallel execution using multiprocessing
- Logging of calibration and validation metrics
- Live saving of best parameters and final full-period validation metrics

---

## Global Configuration

- `n_iterations`: Number of DDS iterations
- `max_cores_for_gages`: Number of gages calibrated in parallel
- `project_root`, `observed_q_root`, `logging_dir`, `model_roots`: All defined via `path_config`
- `n_tiles`: Number of model tiles (typically 1 or 2)

---

## Helper Functions

### `reflect_bounds(x, low, high)`
Ensures a perturbed parameter stays within bounds by reflecting it back inside the valid range.

---

## Class: `DDS`
Performs DDS optimization for a single gage.

### **Constructor Parameters**
- `bounds`: List of `(min, max)` tuples for each parameter
- `n_iterations`: Total number of DDS iterations
- `gage_id`: USGS gage ID
- `observed_q_root`: Root path for observed streamflow
- `base_roots`: List of model roots (e.g., for tiled model setup)
- `init_params`: Initial parameter vector (typically from config files)
- `metric_to_calibrate_on`: Metric to minimize (e.g., `kge`, `log_kge`, etc.)
- `sigma`: Standard deviation multiplier for perturbations (DDS hyperparameter)
- `include_nom`: Whether NOM parameters are included
- `n_tiles`: Number of LASAM tiles
- `weights`: Tile weighting scheme
- `param_names`: List of parameter names (used for logging)

### **Method: `optimize()`**
- Initializes log file for metrics and parameters
- Evaluates objective function with initial parameters
- At each iteration:
  - Computes perturbation probability
  - Randomly perturbs a subset of parameters
  - Evaluates the new candidate
  - Updates best if objective improves
  - Logs all values to CSV
- After all iterations:
  - Runs final validation over full calibration+validation window
  - Appends final metrics and parameters to log file
  - Returns the best parameter vector, best metric value, and total wall time

---

## Function: `calibrate_gage_dds(gage_id)`
Orchestrates DDS optimization for a single gage.

### Steps:
1. Initializes lists for parameter values, bounds, and names
2. For each tile:
   - Reads LASAM config file to extract initial parameters
   - Detects and includes NOM parameters if present
   - Constructs bounds for soil, LASAM, and NOM parameters
3. Optionally adds a tile weighting parameter if using 2 tiles
4. Instantiates and runs the `DDS` optimizer

---

## Main Execution Block

### `if __name__ == "__main__"`
- Reads gage list from `cfg.gages_file`
- Initializes multiprocessing pool using `spawn` context
- Runs `calibrate_gage_dds()` across all gages in parallel
- Prints total wall clock time at the end

---

## Notes

- Calibration and validation metrics are logged to `logging/{gage_id}.csv`
- Best parameter set is used to perform a final validation run across the full simulation period
- NOM parameter calibration is optional and auto-detected per tile
- Tile weighting (`tile1_weight`) is added when `n_tiles = 2`

---

