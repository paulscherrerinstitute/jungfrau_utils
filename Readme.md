# Description

`jungrau_utils` collects a set of scripts for operating and doing first analysis of the Jungfrau detectors. This includes:

* a python environment called `jungfrau_client`, for controllig the detector from Alvra and Bernina
* a set of scripts for calibrations (running a pedestal run, converting gain maps, ...)
* some examples (soon)

For more information about the Detector Integration Api please visit:

* https://github.com/datastreaming/detector_integration_api

# Usage

`jungfrau_utils` is provided on the Alvra, Bernina beamlines already in a dedicated analysis-friendly Anaconda environment. To get into the environment, execute e.g.:

```bash
source anaconda_env
```


Then, to open an IPython shell already configured for the Jungfrau detector at the beamline:

```bash
jungfrau_console
```

In case you do not have graphical access, please do:

```bash
jungfrau_console nox
```


**Example Alvra**:
Start the anaconda environment.
```bash
source anaconda_env
ipython
```

Use the client to communicate with DIA:
```python
from detector_integration_api import DetectorIntegrationClient

client = DetectorIntegrationClient("http://sf-daq-2:10000")

detector_config = {"exptime": 0.00001, "cycles": 1000, "dr":16}
backend_config = {"n_frames": 1000, "bit_depth":16}
writer_config = {"n_frames": 1000, "user_id": 16581, "output_file": "/sf/alvra/data/p16581/raw/test_writer.h5"}
bsread_config = {"user_id": 16581, "output_file": "/sf/alvra/data/p16581/raw/test_bsread.h5"}

FORMAT_PARAMETERS = {"general/user": "p16581", 
                     "general/instrument": "Alvra, JF 4.5M", 
                     "general/created": "today", 
                     "general/process": "detector integration api"}

writer_config.update(FORMAT_PARAMETERS)
bsread_config.update(FORMAT_PARAMETERS)

configuration = {"detector": detector_config, 
                 "backend": backend_config, 
                 "writer": writer_config, 
                 "bsread": bsread_config}

client.reset()

client.set_config(configuration)

client.start()
```


**Example:** starting a data acquisition with a Jungfrau 1.5M at Bernina
```
In [1]: writer_config = {"output_file": "/sf/bernina/data/p16582/raw/test_data.h5", "process_uid": 16582, "process_gid": 16582, "dataset_name": "jungfrau/data", "n_messages": 1000}

In [2]: detector_config = {"timing": "trigger", "exptime": 0.00001, "cycles": 1000}

In [3]: backend_config = {"n_frames": 1000, "gain_corrections_filename": "/sf/bernina/data/p16582/res/gains.h5", "gain_corrections_dataset": "gains", "pede_corrections_filename": "/sf/bernina//data/p16582/res/JF_pedestal/pedestal_20171124_1646_res.h5", "pede_corrections_dataset": 
   ...: "gains", "activate_corrections_preview": True}

In [4]: default_channels_list = jungfrau_utils.load_default_channel_list()

In [5]: bsread_config = {'output_file': '/sf/bernina/data/p16582/raw/test_bsread.h5', 'process_uid': 16582, 'process_gid': 16582, 'channels': default_channels_list, 'n_pulses':550}

In [6]: client.reset()

In [7]: configuration = {"writer": writer_config, "backend": backend_config, "detector": detector_config, "bsread": bsread_config}

In [8]: client.set_config(configuration)

In [9]: client.start()

```

You can load a default list with `ju.load_default_channel_list()`

## Commissioning 2017-11-19

```
backend_config = {"n_frames": 100000, "pede_corrections_filename": "/sf/bernina/data/p16582/res/pedestal_20171119_1027_res.h5", "pede_corrections_dataset": "gains", "gain_corrections_filename": "/sf/bernina/data/p16582/res/gains.h5", "gain_corrections_dataset": "gains", "activate_corrections_preview": True, "pede_mask_dataset": "pixel_mask"}
detector_config = {"exptime": 0.00001, "cycles":20000, "timing": "trigger", "frames": 1} 

client.reset()
writer_config = {'dataset_name': 'jungfrau/data','output_file': '/sf/bernina/data/p16582/raw/Bi11_pp_delayXXPP_tests.h5','process_gid': 16582,   'process_uid': 16582, "disable_processing": False};
configuration = {"writer": writer_config, "backend": backend_config, "detector": detector_config}
client.set_config(configuration); 
client.start()

client.get_status()

## only if it is {'state': 'ok', 'status': 'IntegrationStatus.DETECTOR_STOPPED'}
client.reset()
```

## Taking a pedestal

```
# This records a pedestal run
jungfrau_run_pedestals --numberFrames 3000 --period 0.05

# This analyses and creates a pedestal correction file, in this case /sf/bernina/data/p16582/res/pedestal_20171124_1646_res.h5
jungfrau_create_pedestals -f /sf/bernina/data/p16582/raw/pedestal_20171124_1646.h5 -v 3 -o /sf/bernina/data/p16582/res/
```

## Correct data on file

One utility `jungfrau_utils` provides is a pede and gain subtraction routine. Eg.:

```
In [2]: import jungfrau_utils as ju
In [3]: f = h5py.File("/sf/bernina/data/p16582/raw/AgBeNH_dtz60_run3.h5")
In [4]: fp = h5py.File("/sf/bernina/data/p16582/res/pedestal_20171119_0829_res_merge.h5")
In [5]: fg = h5py.File("/sf/bernina/data/p16582/res/gains.h5")
In [6]: images = f["jungfrau/data"]
In [7]: G = fg["gains"][:]
In [8]: P = fp["gains"][:]
In [9]: corrected_image = ju.apply_gain_pede(images[2], G, P, pixel_mask=fp["pixelMask"][:])

```

## Restart services

There are 4 services running on `sf-daq-1`:
* `detector_integration_api` : controls detector, backend and writer
* `detector_backend`: controls the data acquisition
* `writer`: writes data
* `detector_visualization`: controls the live visualization
* 

These services can be restarted from `sf-daq-1` with the user `dbe` with:
```
sudo systemctl stop <SERVICE_NAME>
sudo systemctl start <SERVICE_NAME>
```
where `<SERVICE_NAME>` is one of the above.



# Installation

The package is provided with a conda recipe, and uploaded on Anaconda Cloud. The easiest way to install is:

```
conda install -c paulscherrerinstitute jungfrau_utils
```

For testing, the git repo can also simply be cloned.
