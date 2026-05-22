import os
from pathlib import Path

import yaml

# Define base paths
project_root = os.environ.get(
    "NGSH_ROOT",
    str(Path(__file__).resolve().parents[2])
)
model_assessment_root = os.path.join(project_root, "model_assessment")

# Define commonly used paths
logging_dir = os.path.join(project_root, "logging") ###legacy fallback for calibration logs
sandbox_path = os.path.join(project_root, "sandbox.py") ###path to the sandbox script which is used to run models in the calibration scripts
calib_scripts_dir = os.path.join(model_assessment_root, "calib_scripts") ###path to where the calibration scripts live
observed_q_root = os.path.join(project_root, "model_assessment", "USGS_streamflow") ###where USGS streamflow observations live 
usgs_output_dir = os.path.join(observed_q_root, "successful_sites") ###where raw USGS streamflow data will go
gages_file = os.environ.get(
    "BASIN_CSV",
    os.path.join(project_root, "basin_IDs", "basin_IDs.csv")
) ###this is the list of USGS gages that will be included in calibration 


# Tile model roots
### for each model root, "out" subdirectory must be created, and also each model root iteself must be created 
model_roots = [
    os.environ.get("NGEN_MODEL_ROOT", "/tmp/model1")
    # single-tile calibration only
]


def active_output_dir(model_root=None, sandbox_config=None):
    """Return the sandbox output_dir, falling back to <model_root>/out."""
    root = model_root or model_roots[0]
    config_path = sandbox_config or os.environ.get("NGEN_SANDBOX_CONFIG")
    if config_path:
        config_path = os.path.abspath(os.path.expandvars(config_path))
        try:
            with open(config_path, "r") as f:
                sandbox_cfg = yaml.safe_load(f) or {}
            output_dir = sandbox_cfg.get("output_dir")
            if output_dir:
                return os.path.abspath(os.path.expandvars(str(output_dir)))
        except Exception:
            pass
    return os.path.join(root, "out")


def gage_output_dir(gage_id, model_root=None, sandbox_config=None):
    """Return the active ngen output directory for one gage."""
    return os.path.join(active_output_dir(model_root, sandbox_config), str(gage_id))


def gage_logging_dir(gage_id, model_root=None, sandbox_config=None):
    """Return the calibration log directory colocated with one gage output."""
    log_dir = os.path.join(gage_output_dir(gage_id, model_root, sandbox_config), "logging")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


# Ensure logging directory exists
os.makedirs(logging_dir, exist_ok=True)
