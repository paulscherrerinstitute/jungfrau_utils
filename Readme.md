# Description

`jungrau_utils` collects a set of scripts for operating and doing first analysis of the Jungfrau detectors. This includes:

* a python environment called `jungfrau_client`, for controllig the detector from Alvra and Bernina
* a set of scripts for calibrations (running a pedestal run, converting gain maps, ...)
* some examples (soon)

# Usage

`jungfrau_utils` is provided on the Alvra, Bernina beamlines already in a conda environment. To get into the environment, execute e.g.:

```
source /sf/bernina/jungfrau/bin/jungfrau_env.sh
```


Then, to open an IPython shell already configured for the Jungfrau detector at the beamline:

```
jungfrau_console.sh
```


**Example:** starting a data acquisition with a Jungfrau 1.5M at Bernina
```
In [1]: writer_config = {"output_file": "/sf/bernina/data/raw/p16582/test.h5", "process_uid": 16582, "process_gid": 16582, "dataset_nam
   ...: e": "jungfrau/data"}

In [2]: detector_config = {"period": 1, "exptime": 0.01, "frames": 1000}

In [3]: backend_config = {"n_frames": 1000}

In [4]: client.reset()

In [5]: client.set_config(writer_config=writer_config, backend_config=backend_config, detector_config=detector_config)

In [6]: client.start()

```

# Installation

The package is provided with a conda recipe, and uploaded on Anaconda Cloud. The easiest way to install is:

```
conda install -c paulscherrerinstitute jungfrau_utils
```

For testing, the git repo can also simply be cloned.
