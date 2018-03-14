#!/bin/bash

if [ $# != 1  ]; then
    echo "Usage: $0 [alvra|bernina]"
    exit 1
fi
    
dest=$1

echo "Loading psi-python36/4.4.0"
module load psi-python36/4.4.0

DIR=/sf/${dest}/anaconda/4.4.0

echo "Creating jungfrau_client Conda env"
#conda create -c paulscherrerinstitute --copy -p /sf/${dest}/config/jungfrau/envs/jungfrau_client detector_integration_api ipython setuptools h5py numpy matplotlib numba jungfrau_utils
#mkdir -p /sf/${dest}/anaconda/4.4.0

conda create -p $DIR ipython setuptools h5py numpy matplotlib numba dask colorama

source activate $DIR
conda install -c paulscherrerinstitute  detector_integration_api jungfrau_utils pyepics pyscan cam_server
conda install --copy -c conda-forge pyFAI
