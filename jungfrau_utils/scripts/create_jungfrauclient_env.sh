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
conda create -c paulscherrerinstitute --copy -p $DIR/jungfrau_utils detector_integration_api ipython setuptools h5py numpy matplotlib numba jungfrau_utils

source activate $DIR/jungfrau_client
conda install --copy -c conda-forge pyFAI
