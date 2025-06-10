###############################################################
# Author      : Ahmad Jan Khattak [ahmad.jan.khattak@noaa.gov | September 10, 2024]
# Contributor : Sifan A. Koriche [sakoriche@ua.edu | December 18, 2024]

# If running on AWS EC2 instance, run setup_ec2.sh before bulding models to setup the EC2 instance

# Step 1: Clone NextGen
#         - git clone https://github.com/NOAA-OWP/ngen && cd ngen
#         - git submodule update --init --recursive
# Step 2: Setup bash file
#         - Refer to the instructions here: (utils/setup_ec2.sh, line 23)


# 1st build NGEN
# 2nd build MODELS
# 3rd build T-ROUTE


###############################################################

export wkdir=$(pwd)
export builddir="cmake_build"
cd ${wkdir}

#####################################################

BUILD_NGEN=OFF
BUILD_MODELS=OFF
BUILD_TROUTE=ON

ngen_dir=/Users/peterlafollette/CIROH_project/ngen

#####################################################

# build_ngen()
# {
#     pushd $ngen_dir

#     rm -rf ${builddir}
#     cmake -DCMAKE_BUILD_TYPE=Release \
# 	  -DNGEN_WITH_BMI_FORTRAN=ON \
# 	  -DNGEN_WITH_NETCDF=ON \
# 	  -DNGEN_WITH_SQLITE=ON \
# 	  -DNGEN_WITH_ROUTING=ON \
# 	  -DNGEN_WITH_EXTERN_ALL=ON  \
# 	  -DNGEN_WITH_TESTS=ON \
#           -DNGEN_QUIET=ON \
# 	  -DNGEN_WITH_MPI=ON \
# 	  -DNetCDF_ROOT=${NETCDF_ROOT}/lib \
# 	  -B ${builddir} \
# 	  -S .
    
#     make -j8 -C ${builddir}
#     # run the following if ran into tests timeout issues
#     #cmake -j4 --build cmake_build --target ngen
#     #cmake --build cmake_build --tartget ngen -j8
#     popd
# }

###update for M-series Apple CPU
build_ngen()
{
    pushd $ngen_dir

    rm -rf ${builddir}

    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_C_FLAGS="-I/opt/homebrew/include" \
          -DCMAKE_CXX_FLAGS="-I/opt/homebrew/include" \
          -DCMAKE_EXE_LINKER_FLAGS="-L/opt/homebrew/lib" \
          -DCMAKE_SHARED_LINKER_FLAGS="-L/opt/homebrew/lib" \
          -DNGEN_WITH_BMI_FORTRAN=ON \
          -DNGEN_WITH_NETCDF=ON \
          -DNGEN_WITH_SQLITE=ON \
          -DNGEN_WITH_ROUTING=ON \
          -DNGEN_WITH_EXTERN_ALL=ON  \
          -DNGEN_WITH_TESTS=ON \
          -DNGEN_QUIET=ON \
          -DNGEN_WITH_MPI=ON \
          -DNetCDF_ROOT=${NETCDF_ROOT}/lib \
          -DUDUNITS2_INCLUDE_DIR=/opt/homebrew/include \
          -DUDUNITS2_LIBRARY=/opt/homebrew/lib/libudunits2.dylib \
          -B ${builddir} \
          -S .

    make -j8 -C ${builddir}
    popd
}


build_troute()
{
    pushd $ngen_dir/extern/t-route
    git checkout master
    git pull

    ##hot patch nc config to nf config
    #sed -i 's/nc-config/nf-config/g' src/kernel/reservoir/makefile

    if [[ "$(uname)" == "Darwin" ]]; then
	NETCDF=$(brew --prefix netcdf-fortran)/include LIBRARY_PATH=$(brew --prefix gcc)/lib/gcc/current/:$(brew --prefix)/lib:$LIBRARY_PATH FC=$FC CC=$CC F90=$FC ./compiler.sh no-e
    else
	export NETCDF=${NETCDF_ROOT}/include
	./compiler.sh no-e
    fi

    popd
}

# #####using this version because there is a bug when trying to use t-route with hydrofabric version 2.2 -- there ends up being two "id" columns per subset geopacakge which t-route does not like.
# #####this fixes that and is compatible with both v 2.1.1 and 2.2.
# #####however, cloning directly from this repo seems to slow routing down. So my recommendation is to just use build_troute as above and then copy in the fix linked below if using hydrofabric v 2.2
# #####see https://github.com/shorvath-noaa/t-route/blob/400fd8ce80be509f21e7b896e51cce655cb78950/src/troute-network/troute/HYFeaturesNetwork.py#L97-L117
# build_troute()
# {
#     pushd $ngen_dir/extern

#     # If t-route already exists, remove it (optional but ensures a clean version)
#     rm -rf t-route

#     git clone https://github.com/shorvath-noaa/t-route.git
#     cd t-route

#     # Checkout the specific commit or a branch containing the changes
#     git checkout 400fd8ce80be509f21e7b896e51cce655cb78950

#     ## Optional: hot patch (if still needed)
#     #sed -i 's/nc-config/nf-config/g' src/kernel/reservoir/makefile

#     if [[ "$(uname)" == "Darwin" ]]; then
#         NETCDF=$(brew --prefix netcdf-fortran)/include LIBRARY_PATH=$(brew --prefix gcc)/lib/gcc/current/:$(brew --prefix)/lib:$LIBRARY_PATH FC=$FC CC=$CC F90=$FC ./compiler.sh no-e
#     else
#         export NETCDF=${NETCDF_ROOT}/include
#         ./compiler.sh no-e
#     fi

#     popd
# }


build_models()
{
    pushd $ngen_dir

    for model in noah-owp-modular cfe evapotranspiration SoilFreezeThaw SoilMoistureProfiles LGAR; do
	rm -rf extern/$model/${builddir}
	if [ "$model" == "noah-owp-modular" ]; then
	    git submodule update --remote extern/${model}/${model}
	    cmake -B extern/${model}/${builddir} -S extern/${model} -DCMAKE_BUILD_TYPE=Release -DNGEN_IS_MAIN_PROJECT=ON -DCMAKE_C_FLAGS="-I/opt/homebrew/include" -DCMAKE_CXX_FLAGS="-I/opt/homebrew/include" -DCMAKE_EXE_LINKER_FLAGS="-L/opt/homebrew/lib" -DCMAKE_SHARED_LINKER_FLAGS="-L/opt/homebrew/lib" 
	    make -C extern/${model}/${builddir}
	fi
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
if [ "$BUILD_MODELS" == "ON" ]; then
    echo "Models build: ${BUILD_MODELS}"
    build_models
fi
if [ "$BUILD_TROUTE" == "ON" ]; then
    echo "Troute build: ${BUILD_TROUTE}"
    build_troute
fi


#if [ "$model" == "ngen-cal" ] && [ "$BUILD_CALIB" == "ON" ]; then
#    git clone https://github.com/NOAA-OWP/ngen-cal extern/${model}
#    pip install -e extern/${model}/python/ngen_cal
#    # or try installing this way
#    #pip install "git+https://github.com/noaa-owp/ngen-cal@master#egg=ngen_cal&subdirectory=python/ngen_cal"
#    #pip install "git+https://github.com/aaraney/ngen-cal@forcing-hotfix#egg=ngen_cal&subdirectory=python/ngen_cal"
#    #cd ${wkdir}
#fi



