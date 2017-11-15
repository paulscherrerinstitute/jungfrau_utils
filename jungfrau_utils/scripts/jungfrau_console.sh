#!/bin/bash

if [ $# != 1  ]; then
    echo "Usage: $0 [alvra|bernina]"
    exit 1
fi
    
dest=$1

echo Loading PSI Anaconda Python distribution 3.4
module load psi-python34

echo Activating Conda environment
source activate /sf/${dest}/jungfrau/envs/jungfrau_client

echo Starting Interactive Python session
export QT_XKB_CONFIG_ROOT=/sf/${dest}/jungfrau/envs/jungfrau_client/lib
ipython -i -c "from detector_integration_api import DetectorIntegrationClient; api_address = 'http://sf-daq-1:10000'; client = DetectorIntegrationClient(api_address); print('\nJungfrau Integration API on %s' % api_address);import h5py;import numpy as np;import matplotlib.pyplot as plt;print('Imported matplotlib (as plt), h5py, numpy (as np)');print('Jungfrau client available as client. Try: client.get_status()\n')"
