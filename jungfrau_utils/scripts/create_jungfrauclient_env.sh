#!/bin/bash

if [ $# != 2  ]; then
    echo error
    exit 1
fi
    
dest=$2

module load psi-python34
conda create -y -c paulscherrerinstitute -p /sf/bernina/jungfrau/envs/jungfrau_client detector_integration_api ipython setuptools h5py numpy dask matplotlib
