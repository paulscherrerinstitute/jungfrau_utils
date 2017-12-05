#!/bin/bash

if [ $# != 1  ]; then
    echo "Usage: $0 [alvra|bernina]"
    exit 1
fi
    
dest=$1

echo Loading PSI Anaconda Python distribution 3.6
module load psi-python36

echo Activating Conda environment
source activate /sf/${dest}/jungfrau/envs/jungfrau_client

echo Starting Interactive Python session
#export QT_XKB_CONFIG_ROOT=/sf/${dest}/jungfrau/envs/jungfrau_client/lib
if [ $dest == "alvra" ]; then
    str="api_address = 'http://sf-daq-2:10000'; client_4p5M = DetectorIntegrationClient(api_address); print('\nJungfrau 4.5M Integration API on %s' % api_address);"
elif [ $dest == "bernina" ]; then
    str="api_address = 'http://sf-daq-1:10000'; client_1p5M = DetectorIntegrationClient(api_address); print('\nJungfrau 1.5M Integration API on %s' % api_address);"
else
    echo "Please select either alvra or bernina"
    exit
fi
    
ipython -i -c "from detector_integration_api import DetectorIntegrationClient;"${str}";import h5py;import numpy as np;import matplotlib.pyplot as plt;import dask.array as da;print('Imported matplotlib (as plt), h5py, numpy (as np), dask.array (as da)');print('Jungfrau client available as client. Try: client.get_status()\n')"
