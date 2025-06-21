# `pso_calibration_cfe.py` Function Documentation

This script calibrates the CFE hydrologic model using Particle Swarm Optimization (PSO). It supports:
- CFE + PET + T-Route or CFE + T-Route + NOM configurations
- 1-tile model structure only (for now)
- Optional inclusion of NOM parameter calibration (now supported if NOM is detected in the model formulation)
- Logging and saving of best hydrographs
- Final model run across full calibration + validation period
- Parallel calibrations across multiple gages

---

## Global Configuration

- `n_particles`, `n_iterations`: Number of particles and iterations in PSO
- `max_cores_for_gages`: Number of gages calibrated in parallel
- `metric_to_calibrate_on`: Calibration target (e.g., `kge`, `log_kge`, `event_kge`)
- `param_bounds`: Parameter search space
- `nom_param_names`, `nom_param_bounds`: Used if NOM is included in the model. Appended to `param_names` and `param_bounds`.

---

## Function Descriptions

### `clear_terminal()`
Clears the terminal screen to reduce output clutter.

---

### `check_for_stop_signal_or_low_disk()`
- Checks for a stop file (`STOP_NOW.txt`) in the project root
- Monitors available disk space and returns `True` if below 100 GB

---

### `transform_params(params)`
- Converts parameters stored in log10 space (e.g., `Cgw`, `satdk`) back to their linear scale

---

### `extract_initial_cfe_params(config_path)`
- Reads initial parameter values from a `cfe_config_cat` config file
- Supports mapping from config keys to model parameters
- Applies log10 transform for parameters marked in `log_scale_params`

---

### `get_observed_q(observed_path)`
- Loads observed USGS streamflow data from file
- Returns a `pandas.Series` indexed by `value_time`

---

### `regenerate_cfe_config(config_dir, params)`
- Rewrites all `cfe_config_cat*` files in `config_dir` with updated parameter values

---

### `objective_function(args)`
- Executes the full modeling workflow for one particle:
  - Updates CFE config (and NOM if included) and t-route realization JSON for calibration window
  - Runs the model via `sandbox.py`
  - Creates hydrograph for the correct nexus with `get_hydrograph.py`
  - Computes calibration and validation metrics (but validation metrics will just be dummy values)
- Returns:
  - Negative calibration metric (objective function), for example minimizes negative KGE
  - Validation metric dictionary
  - Calibration metric dictionary


---

### `extract_initial_nom_params(nom_file_path)`
- Parses initial values of NOM parameters from `MPTABLE.TBL` if NOM is included in the formulation

---

### `update_mptable(original_file, output_file, updated_params, verbose=False)`
- Rewrites the MPTABLE.TBL file to include new NOM parameter values (e.g., `MFSNO`, `RSURF_SNOW`, `HVT`, etc.)
---

## Class Descriptions

### `Particle`
- Represents a single PSO particle
- Tracks position, velocity, best values, stagnation count
- Methods:
  - `reset(bounds)`: Reinitialize to random point within bounds
  - `update_velocity()`: Applies standard PSO update rule
  - `update_position()`: Moves particle and enforces bounds

---

### `PSO`
- Main optimization engine
- Tracks particles, global best, and logs
- Methods:
  - `optimize()`: Runs PSO iterations and final validation run
    - At each iteration:
      - Evaluates all particles
      - Updates velocities and positions
      - Logs metrics and parameters
      - Saves best hydrograph to `{gage_id}_best.csv`
    - After all iterations:
      - Re-runs model over full calibration + validation window
      - Logs final summary row to CSV
    - If NOM is enabled:
      - NOM parameters are appended to the search space
      - `update_mptable()` is called after each improved global best
      - NOM parameter values are logged to CSV and applied in the final model run

---

## Calibration Wrapper

### `calibrate_gage(gage_id)`
- Sets up file paths for a single gage
- Extracts initial parameters and config path
- Instantiates and runs the PSO engine

---

## Main Execution

### `if __name__ == "__main__"`
- Loads gage list from `basins_passed_custom.csv`
- Launches multiprocessing pool with up to `max_cores_for_gages`
- Runs `calibrate_gage()` in parallel for each gage
- Prints total wall clock time

---

## Notes

- Final hydrograph is saved to `postproc/{gage_id}_best.csv`
- Full-period metrics (including validation) are re-computed at the end
- Parameters are logged in-place using the same format as LASAM calibration

