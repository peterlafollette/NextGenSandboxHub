import os
from pathlib import Path

# Define base paths
project_root = os.environ.get(
    "NGSH_ROOT",
    str(Path(__file__).resolve().parents[2])
)
model_assessment_root = os.path.join(project_root, "model_assessment")

# Define commonly used paths
logging_dir = os.path.join(project_root, "logging") ###this will be the directory where parameter logs will be stored as calibration progresses
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

# Ensure logging directory exists
os.makedirs(logging_dir, exist_ok=True)
