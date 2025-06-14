import os

# Define base paths
project_root = "/Users/peterlafollette/CIROH_project/NextGenSandboxHub"
model_assessment_root = os.path.join(project_root, "model_assessment")

# Define commonly used paths
logging_dir = os.path.join(project_root, "logging") ###this will be the directory where parameter logs will be stored as calibration progresses
sandbox_path = os.path.join(project_root, "sandbox.py") ###path to the sandbox script which is used to run models in the calibration scripts
calib_scripts_dir = os.path.join(model_assessment_root, "calib_scripts") ###path to where the calibration scripts live
observed_q_root = os.path.join(project_root, "model_assessment", "USGS_streamflow") ###where USGS streamflow observations live 
# gages_file = os.path.join(project_root, "out", "basins_passed_custom.csv") ###this is the list of USGS gages that will be included in calibration 
gages_file = os.path.join("/Users/peterlafollette/CIROH_project/model1/out/basins_passed_custom.csv") ###this is the list of USGS gages that will be included in calibration 


# Tile model roots
### for each model root, "out" and "postproc" subdirectories must exist 
model_roots = [
    "/Users/peterlafollette/CIROH_project/model1"
    # Add a second tile here
    # "/Users/peterlafollette/CIROH_project/model2"
]

# Ensure logging directory exists
os.makedirs(logging_dir, exist_ok=True)
