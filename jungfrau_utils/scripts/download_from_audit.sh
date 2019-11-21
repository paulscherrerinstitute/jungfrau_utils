#!/bin/bash
array=( $@ )
len=${#array[@]}
_domain=${array[$len]}
_args=${array[@]:0:$len}

export PATH=/home/writer/miniconda3/bin:$PATH

echo "Activating conda environment 'bsread'."

# Activate the bsread conda environment.
source deactivate
source activate bsread

echo "Done."

echo "Starting data-api downloader."

python download_from_audit.py $_args
