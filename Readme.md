[![Deployment](https://github.com/paulscherrerinstitute/jungfrau_utils/actions/workflows/deployment.yaml/badge.svg)](https://github.com/paulscherrerinstitute/jungfrau_utils/actions/workflows/deployment.yaml)
[![Documentation Status](https://readthedocs.org/projects/jungfrau-utils/badge/?version=latest)](https://jungfrau-utils.readthedocs.io/en/latest/?badge=latest)

# Description

The **`jungfrau_utils`** library provides tools for processing of data from the Jungfrau detectors.

The most common way to utilize it is to open a raw Jungfrau file with `jungfrau_utils.File()`
wrapper, which provides a similar interface to `h5py.File()`. To read images one can directly
index or slice the resulting object. Any metadata entry is accessed via the same object, but using
the metadata's name as a key in a standart python dictionary style.

A typical extraction patterns for data and metadata:
```python
import jungfrau_utils as ju

with ju.File("<path_to_a_file>") as juf:
    first_image = juf[0]                    # a first image in a file
    roi_all_images = juf[:, 100:200, :100]  # all images, rows from 100 to 200, first 100 columns
    all_pulse_ids = juf["pulse_id"][:]      # all `pulse_id` values
```

By default, it will try to auto-locate the corresponding pedestal (and gain) file, based on the file
creation time stamp, but the pedestal path can also be provided explicitly:
```python
with ju.File("<path_to_a_file>", pedestal_file="<gain_file_path>") as juf:
    ...
```

Finally, it's possible to control conversion (adc to keV), masking, gap pixels and geometry
corrections via the corresponding flags:
```python
with ju.File("<path_to_a_file>", conversion=True, mask=True, gap_pixels=True, geometry=True) as juf:
    ...
```

The full reference could be found here: https://jungfrau-utils.readthedocs.io/en/latest/

# Installation

The package is created with a conda recipe, and uploaded to the `paulscherrerinstitute` Anaconda
channel. Thus, the easiest way to install is:

```
conda install -c paulscherrerinstitute jungfrau_utils
```

For testing, the git repo can also be simply cloned.

## Problem with bitshuffle

There are several channels (conda-forge, paulscherrerinstitute) where bitshuffle package is released. It's better to not use bitshuffle from paulscherrerinstitute channel (have compatibility problem with h5py/hdf). Starting from 3.14.4 version, bitshuffle version >=0.5 is requested for jungfrau_utils. This version may have a problem on nodes, since compiled with AVX512 instructions, which are not available for processors like (Intel(R) Xeon(R) CPU E5-2697A v4) or (AMD EPYC 7453). Problem with such bitshuffle will be "Illegal instruction" message when reading or writing of compressed files via jungfrau_utils is done. To fix this problem, reinstall bitshuffle in your conda environment with the recompilation:
```
conda  remove bitshuffle --force              # remove bitshuffle from conda environment
pip install bitshuffle --no-binary=bitshuffle # install bitshuffle with recompilation in the current conda environment
```
To check if bitshuffle compiled with AVX512 support:
```
python -c "import bitshuffle; print(bitshuffle.using_AVX512())"
```
List of supported processors instructions can be found in /proc/cpuinfo
```
grep -i avx512 /proc/cpuinfo
```

# Build

The build is triggered via the Github Actions script upon pushing a tag into a master branch.
It builds a package and uploads it to `paulscherrerinstitute` Anaconda channel. A tagged release
commit can be created with `make_release.py` script.
