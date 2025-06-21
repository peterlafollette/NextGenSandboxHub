# `pso_calibration_lasam.py` Function Documentation

This script calibrates the LASAM hydrologic model using Particle Swarm Optimization (PSO). It supports:
- 1 or 2 tiled model structures
- NOM parameter calibration (if present)
- Live saving of best hydrographs
- Disk space monitoring and early termination
- Parallel execution across multiple gages

---

## Global Configuration

- `n_particles`, `n_iterations`: PSO settings
- `max_cores_for_gages`: Number of gages calibrated in parallel
- `time_config.yaml`: Defines calibration/validation periods
- `path_config.py`: Defines all path locations
-  parameter bounds set in calibrate_gage

---

## Function Descriptions

### `clear_terminal()`
Clears the terminal. Helps reduce clutter during model execution.

---

### `check_for_stop_signal_or_low_disk(project_root, threshold_gb=100)`
- Checks if `STOP_NOW.txt` exists and halts execution
- Monitors disk space and halts if below threshold

---

### `get_observed_q(observed_path)`
- Reads observed USGS streamflow data
- Parses hourly `flow_m3_per_s` indexed by `value_time`

---

### `extract_tile_params(full_params, tile_idx, n_tiles)`
- Extracts a tile-specific parameter slice from the full parameter vector

---

### `extract_initial_params(example_config_path)`
- Extracts initial LASAM and NOM parameters from a config file
- Includes:
  - Soil layer properties
  - `a`, `b`, `frac_to_GW`, `spf_factor`, `theta_e_1`, etc.
  - NOM parameters (if present)

---

### `update_lasam_files_for_divide(config_path, params, include_nom)`
- Updates LASAM config and soil files with new parameters for a single divide

---

### `objective_function_tiled(args, ...)`
- Runs the LASAM based formulation for all tiles with specified parameters
- Steps:
  - Update LASAM and NOM config and parameter files (both normally and log normally parameters supported)
  - Run model via `sandbox.py -run`
  - Create hydrograph at the correct nexus via `get_hydrograph.py`
  - Compute calibration metric (e.g., KGE)
- Returns:
  - Negative calibration metric (will attempt to minimize negative KGE)
  - Validation metric (vestigial, will just be a dummy value)
  - Full metric dictionaries

---

### `run_validation_with_best(...)`
- Runs one final model simulation using the best parameter set discovered for the calibration period
- Extends the model period to include validation
- Saves the best hydrograph to `{gage_id}_best.csv`

---

## Class Descriptions

### `Particle`
- Represents a single PSO particle
- Methods:
  - `reset()`: Reinitialize if stagnant
  - `update_velocity()`: Apply PSO velocity equation
  - `update_position()`: Move particle and enforce bounds

---

### `PSO`
- Core optimization engine
- Initializes particle swarm, performs updates
- Monitors stagnation and resets particles as needed
- Saves live best hydrographs during calibration
- At end, runs final validation simulation
- Logs all parameters and metrics to CSV

---

### `calibrate_gage(gage_id)`
- Initializes config for a specific gage
- Reads initial parameters from LASAM + NOM config and parameter tables
- Constructs bounds
- Instantiates and runs the PSO optimizer

---

### `if __name__ == "__main__"`
- Loads list of gages from `cfg.gages_file`
- Spawns multiprocessing pool
- Runs `calibrate_gage()` across gages in parallel
- Logs total runtime

---

