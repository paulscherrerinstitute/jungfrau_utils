#!/bin/bash

BL=bernina
check=`which conda 2> /dev/null | grep /$BL/anaconda/4.4.0`

if [ "$check" == "" ]; then
    source /sf/$BL/bin/anaconda_env
fi

MATPLOT_TRY='exec("try: import matplotlib.pyplot as plt\nexcept: sys.exit(1)");'

CONFIG='import sys; from detector_integration_api import DetectorIntegrationClient; api_address = "http://sf-daq-1:10000"; client_1p5 = DetectorIntegrationClient(api_address); print("\nJungfrau Integration API on %s" % api_address);import h5py;import numpy as np;'$MATPLOT_TRY'import dask.array as da;print("Imported matplotlib (as plt), h5py, numpy (as np), dask.array (as da)"); exec("""if "client_1p5" in locals(): print("Jungfrau 1.5M client available as client_1p5. Try: client_1p5.get_status() ") """);'

echo Starting Interactive Python session
export QT_XKB_CONFIG_ROOT=/sf/alvra/anaconda/4.4.0/jungfrau_utils/lib
if [ "$1" == "nox" ]; then
    export QT_QPA_PLATFORM='offscreen'
    echo "Starting no-graphics version"
else
    echo If you get \"Could not connect to display\" or similar, try: \"jungfrau_console nox\"
fi
ipython -i -c "$CONFIG"

