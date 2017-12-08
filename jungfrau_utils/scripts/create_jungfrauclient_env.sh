#!/bin/bash

if [ $# != 1  ]; then
    echo "Usage: $0 [alvra|bernina]"
    exit 1
fi
    
dest=$1

echo "Loading psi-python36"
module load psi-python36

echo "Creating jungfrau_client Conda env"
conda create -c paulscherrerinstitute --copy -p /sf/${dest}/config/jungfrau/envs/jungfrau_client detector_integration_api ipython setuptools h5py numpy dask matplotlib numba

source activate jungfrau_client
conda install --copy -c conda-forge pyFAI wxpython
