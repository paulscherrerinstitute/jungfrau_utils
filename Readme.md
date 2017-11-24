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
In [1]: writer_config = {"output_file": "/sf/bernina/data/raw/p16582/test.h5", "process_uid": 16582, "process_gid": 16582, "dataset_name": "jungfrau/data"}

In [2]: detector_config = {"timing": "trigger", "exptime": 0.0001, "cycles": 1000}

In [3]: backend_config = {"n_frames": 1000, "gain_corrections_filename": "/sf/bernina/data/res/p16582/gains.h5", "gain_corrections_dataset": "gains", "pede_corrections_filename": "/sf/bernina/data/res/p16582/pedestal_20171115_1100_res_merge.h5", "pede_corrections_dataset": 
   ...: "gains", "activate_corrections_preview": True}

In [4]:bsread_config = {'output_file': '/sf/bernina/data/raw/p16582/test_bsread.h5', 'process_uid': 16582, 'process_gid': 16582, 'channels': ['SAROP21-CVME-PBPS2:Lnk9Ch7-BG-DATA',
    ...:   'SAROP21-CVME-PBPS2:Lnk9Ch7-BG-DATA-CALIBRATED']}

In [5]: client.reset()

In [6]: client.set_config(writer_config=writer_config, backend_config=backend_config, detector_config=detector_config, bsread_config=bsread_config)

In [7]: client.start()

```

## Commissioning 2017-11-19

```
backend_config = {"n_frames": 100000, "pede_corrections_filename": "/sf/bernina/data/res/p16582/pedestal_20171119_1027_res.h5", "pede_corrections_dataset": "gains", "gain_corrections_filename": "/sf/bernina/data/res/p16582/gains.h5", "gain_corrections_dataset": "gains", "activate_corrections_preview": True, "pede_mask_dataset": "pixel_mask"}
detector_config = {"exptime": 0.0001, "cycles":20000, "timing": "trigger", "frames": 1} 

client.reset()
writer_config = {'dataset_name': 'jungfrau/data','output_file': '/gpfs/sf-data/bernina/raw/p16582/Bi11_pp_delayXXPP_tests.h5','process_gid': 16582,   'process_uid': 16582, "disable_processing": False};
client.set_config(writer_config=writer_config,backend_config=backend_config, detector_config=detector_config); 
client.start()

client.get_status()

## only if it is {'state': 'ok', 'status': 'IntegrationStatus.DETECTOR_STOPPED'}
client.reset()
```

## Correct data on file

One utility `jungfrau_utils` provides is a pede and gain subtraction routine. Eg.:

```
In [2]: import jungfrau_utils as ju
In [3]: f = h5py.File("/gpfs/sf-data/bernina/raw/p16582/AgBeNH_dtz60_run3.h5")
In [4]: fp = h5py.File("/sf/bernina/data/res/p16582/pedestal_20171119_0829_res_merge.h5")
In [5]: fg = h5py.File("/sf/bernina/data/res/p16582/gains.h5")
In [6]: images = f["jungfrau/data"]
In [7]: G = fg["gains"][:]
In [8]: P = fp["gains"][:]
In [9]: corrected_image = ju.apply_gain_pede(images[2], G, P, pixel_mask=None)

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
