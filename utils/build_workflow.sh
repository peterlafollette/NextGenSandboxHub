###############################################################
# Author : Ahmad Jan Khattak [ahmad.jan.khattak@noaa.gov | September 10, 2024]
# Contributor : Sifan A. Koriche [sakoriche@ua.edu | December 18, 2024]

# If running on AWS EC2 instance, run setup_ec2.sh before bulding models to setup the EC2 instance

# Clone NextGenSandboxHub and NextGen GitHub repositories
# Step 1: Clone NextGenSandboxHub
#         - git clone https://github.com/ajkhattak/NextGenSandboxHub && cd NextGenSandboxHub
# Step 2: Setup bash file
#         - Refer to the instructions here: (utils/setup_ec2.sh, line 23)

###############################################################

BUILD_WORKFLOW=ON

# Notes:
# If vevn_forcing failed or forcing downloader is failing, that could be due to inconsistent
# versions of packages, try buidling env based on doc/env/venv_forcing.piplist
#####################################################


build_workflow()
{

    mkdir ~/.venv_ngen_py3.11
    python3.11 -m venv ~/.venv_ngen_py3.11
    source ~/.venv_ngen_py3.11/bin/activate
    pip install -U pip==24.0
    
    pip install -r ./utils/requirements.txt
    
    git submodule update --init
    git submodule update --remote extern/ngen-cal
    git submodule update --remote extern/CIROH_DL_NextGen
    
    pip install 'extern/ngen-cal/python/ngen_cal[netcdf]'
    pip install extern/ngen-cal/python/ngen_config_gen
    pip install hydrotools.events
    pip install -e ./extern/ngen_cal_plugins

    deactivate
    
    mkdir ~/.venv_forcing
    python3.11 -m venv ~/.venv_forcing
    source ~/.venv_forcing/bin/activate
    pip install -U pip==24.0
    pip install -r ./doc/env/requirements_forcing.txt
    # or run the below two steps
    # pip install -r extern/CIROH_DL_NextGen/forcing_prep/requirements.txt
    # pip install zarr==2.18.2
    deactivate
}


if [ "$BUILD_WORKFLOW" == "ON" ]; then
    echo "Workflow build: ${BUILD_WORKFLOW}"
    build_workflow
fi



