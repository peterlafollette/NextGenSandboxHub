###############################################################
# Author : Ahmad Jan Khattak [ahmad.jan.khattak@noaa.gov | September 10, 2024]
# Contributor : Sifan A. Koriche [sakoriche@ua.edu | December 18, 2024]

# If running on AWS EC2 instance, run setup_ec2.sh before bulding models to setup the EC2 instance

# Clone NextGenSandboxHub and NextGen GitHub repositories
# Step 1: Clone NextGenSandboxHub
#         - git clone https://github.com/ajkhattak/NextGenSandboxHub && cd NextGenSandboxHub
# Step 2: Clone NextGen
#         - git clone https://github.com/NOAA-OWP/ngen && cd ngen
#         - git submodule update --init --recursive
# Step 3: Setup bash file
#         - Refer to the instructions here: (utils/setup_ec2.sh, line 23)

# Order of building options
# 1st build T-ROUTE >> this helps to create t-route based environment which will also be handy for NGEN
# 2nd build NGEN
# 3rd build MODELS
###############################################################

export wkdir=$(pwd)
export builddir="cmake_build"
cd ${wkdir}

#####################################################

BUILD_TROUTE=OFF
BUILD_NGEN=OFF
BUILD_MODELS=OFF

ngen_dir= <path_to_nextgen_repo>  #/Users/ahmadjankhattak/Code/ngen/ngen

# Notes:
# If vevn_forcing failed or forcing downloader is failing, that could be due to inconsistent
# versions of packages, try buidling env based on doc/env/venv_forcing.piplist
#####################################################

build_ngen()
{
    pushd $ngen_dir

    rm -rf ${builddir}
    cmake -DCMAKE_BUILD_TYPE=Release \
	  -DNGEN_WITH_BMI_FORTRAN=ON \
	  -DNGEN_WITH_NETCDF=ON \
	  -DNGEN_WITH_SQLITE=ON \
	  -DNGEN_WITH_ROUTING=ON \
	  -DNGEN_WITH_EXTERN_ALL=ON  \
	  -DNGEN_WITH_TESTS=ON \
          -DNGEN_QUIET=ON \
	  -DNGEN_WITH_MPI=ON \
	  -DNetCDF_ROOT=/usr/local/lib \
	  -B ${builddir} \
	  -S .
    
    make -j8 -C ${builddir}
    # run the following if ran into tests timeout issues
    #cmake -j4 --build cmake_build --target ngen
    #cmake --build cmake_build --tartget ngen -j8
    popd
}


build_troute()
{
    pushd $ngen_dir/extern/t-route
    git checkout master
    git pull

    # these are no longer needed as workflow env is already built using the same packages/versions
    #mkdir ~/vevn_ngen_py3.11
    #python3.11 -m venv ~/vevn_ngen_py3.11
    #source ~/vevn_ngen_py3.11/bin/activate
    #pip install -U pip==24.0
    #sed -i 's/netcdf4/netcdf4<=1.6.3/g' extern/t-route/requirements.txt
    #pip install -r extern/t-route/requirements.txt 
    ##hot patch nc config to nf config
    #sed -i 's/nc-config/nf-config/g' src/kernel/reservoir/makefile

    ./compiler.sh no-e
    popd
}

build_models()
{
    pushd $ngen_dir

    for model in cfe evapotranspiration SoilFreezeThaw SoilMoistureProfiles LGAR; do
	rm -rf extern/$model/${builddir}
	if [ "$model" == "cfe" ] || [ "$model" == "SoilFreezeThaw" ] || [ "$model" == "SoilMoistureProfiles" ]; then
	    git submodule update --remote extern/${model}/${model}
	    cmake -B extern/${model}/${model}/${builddir} -S extern/${model}/${model} -DNGEN=ON -DCMAKE_BUILD_TYPE=Release
	    make -C extern/${model}/${model}/${builddir}
	fi
	
	if [ "$model" == "LGAR" ]; then
	    git clone https://github.com/NOAA-OWP/LGAR-C extern/${model}/${model}
	    cmake -B extern/${model}/${model}/${builddir} -S extern/${model}/${model} -DNGEN=ON -DCMAKE_BUILD_TYPE=Release
	    make -C extern/${model}/${model}/${builddir}
	fi
	if [ "$model" == "evapotranspiration" ]; then
	    git submodule update --remote extern/${model}/${model}
	    cmake -B extern/${model}/${model}/${builddir} -S extern/${model}/${model} -DCMAKE_BUILD_TYPE=Release
	    make -C extern/${model}/${model}/${builddir}
	fi
    done

    popd
}


if [ "$BUILD_NGEN" == "ON" ]; then
    echo "NextGen build: ${BUILD_NGEN}"
    build_ngen
fi
if [ "$BUILD_TROUTE" == "ON" ]; then
    echo "Troute build: ${BUILD_TROUTE}"
    build_troute
fi
if [ "$BUILD_MODELS" == "ON" ]; then
    echo "Models build: ${BUILD_MODELS}"
    build_models
fi

#if [ "$model" == "ngen-cal" ] && [ "$BUILD_CALIB" == "ON" ]; then
#    git clone https://github.com/NOAA-OWP/ngen-cal extern/${model}
#    pip install -e extern/${model}/python/ngen_cal
#    # or try installing this way
#    #pip install "git+https://github.com/noaa-owp/ngen-cal@master#egg=ngen_cal&subdirectory=python/ngen_cal"
#    #pip install "git+https://github.com/aaraney/ngen-cal@forcing-hotfix#egg=ngen_cal&subdirectory=python/ngen_cal"
#    #cd ${wkdir}
#fi



