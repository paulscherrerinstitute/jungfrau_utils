#!/bin/bash

if [ $# != 1  ]; then
    echo "Usage: $0 [alvra|bernina]"
    exit 1
fi
    
dest=$1

echo Loading PSI Anaconda Python distribution 3.4
module load psi-python34

echo Activating Conda environment
cd /sf/${dest}/jungfrau/envs/
source activate jungfrau_client/
cd -
