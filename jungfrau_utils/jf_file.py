from datetime import timedelta
from pathlib import Path

import h5py
import numpy as np
from bitshuffle.h5 import H5_COMPRESS_LZ4, H5FILTER  # pylint: disable=no-name-in-module

from .data_handler import JFDataHandler

# bitshuffle hdf5 filter params
BLOCK_SIZE = 0
compargs = {'compression': H5FILTER, 'compression_opts': (BLOCK_SIZE, H5_COMPRESS_LZ4)}


class File:
    """Jungfrau file"""

    def __init__(
        self, file_path, gain_file=None, pedestal_file=None, convertion=True, geometry=True
    ):
        """Create a new Jungfrau file wrapper

        Args:
            file_path (str): path to Jungfrau file
            gain_file (str, optional): path to gain file. Auto-locate if None. Defaults to None.
            pedestal_file (str, optional): path to pedestal file. Auto-locate if None.
                Defaults to None.
            convertion (bool, optional): Apply gain conversion and pedestal correction.
                Defaults to True.
            geometry (bool, optional): Apply geometry correction. Defaults to True.
        """
        self.file_path = Path(file_path)

        self.jf_file = h5py.File(self.file_path, 'r')
        self.jf_handler = JFDataHandler(self.jf_file['/general/detector_name'][()].decode())

        self.convertion = convertion
        self.geometry = geometry

        # Gain file
        if gain_file is None:
            gain_file = self._locate_gain_file()
            print(f'Auto-located gain file: {gain_file}')

        self.jf_handler.gain_file = gain_file.as_posix()

        # Pedestal file (with a pixel mask)
        if pedestal_file is None:
            pedestal_file, mtime_diff = self._locate_pedestal_file()
            print(f'Auto-located pedestal file: {pedestal_file}')
            if mtime_diff < 0:
                # timedelta doesn't work nicely with negative values
                # https://docs.python.org/3/library/datetime.html#datetime.timedelta.resolution
                tdelta_str = '-' + str(timedelta(seconds=-mtime_diff))
            else:
                tdelta_str = str(timedelta(seconds=mtime_diff))
            print('    mtime difference: ' + tdelta_str)

        self.jf_handler.pedestal_file = pedestal_file.as_posix()

        if 'module_map' in self.jf_file[f'/data/{self.detector_name}']:
            # Pick only the first row (module_map of the first frame), because it is not expected
            # that module_map ever changes during a run. In fact, it is forseen in the future that
            # this data will be saved as a single row for the whole run.
            module_map = self.jf_file[f'/data/{self.detector_name}/module_map'][0, :]
        else:
            module_map = None

        self.jf_handler.module_map = module_map

        # TODO: Here we use daq_rec only of the first pulse within an hdf5 file, however its
        # value can be different for later pulses and this needs to be taken care of. Currently,
        # _allow_n_images decorator applies a function in a loop, making it impossible to change
        # highgain for separate images in a 3D stack.
        daq_rec = self.jf_file[f'/data/{self.detector_name}/daq_rec'][0]

        self.jf_handler.highgain = daq_rec & 0b1

    @property
    def detector_name(self):
        """Detector name (readonly)"""
        return self.jf_handler.detector_name

    @property
    def gain_file(self):
        """Gain file path (readonly)"""
        return self.jf_handler.gain_file

    @property
    def pedestal_file(self):
        """Pedestal file path (readonly)"""
        return self.jf_handler.pedestal_file

    @property
    def convertion(self):
        """A flag for applying pedestal correction and gain conversion"""
        return self.jf_handler.convertion

    @convertion.setter
    def convertion(self, value):
        if self._processed:
            print("The file is already processed, setting 'convertion' to False")
            value = False

        self.jf_handler.convertion = value

    @property
    def geometry(self):
        """A flag for applying geometry"""
        return self.jf_handler.geometry

    @geometry.setter
    def geometry(self, value):
        if self._processed:
            print("The file is already processed, setting 'geometry' to False")
            value = False

        self.jf_handler.geometry = value

    @property
    def _processed(self):
        # TODO: generalize this check for data reduction case, where dtype can be different
        return self.jf_file[f'/data/{self.detector_name}/data'].dtype == np.float32

    def save_as(self, dest, roi_x=(None,), roi_y=(None,), compress=False, factor=None, dtype=None):
        """Save data in a separate hdf5 file

        Args:
            dest (str): Destination file path
            roi_x (tuple, optional): ROI to save along x axis. Defaults to (None,).
            roi_y (tuple, optional): ROI to save along y axis. Defaults to (None,).
            compress (bool, optional): Apply bitshuffle+lz4 compression. Defaults to False.
            factor (float, optional): Divide all values by a factor. Defaults to None.
            dtype (np.dtype, optional): Resulting image data type. Defaults to None.
        """

        def copy_objects(name, obj):
            if isinstance(obj, h5py.Group):
                h5_dest.create_group(name)

            elif isinstance(obj, h5py.Dataset):
                dset_source = self.jf_file[name]

                args = {
                    k: getattr(dset_source, k)
                    for k in (
                        'shape',
                        'dtype',
                        'chunks',
                        'compression',
                        'compression_opts',
                        'scaleoffset',
                        'shuffle',
                        'fletcher32',
                        'fillvalue',
                    )
                }

                if dset_source.shape != dset_source.maxshape:
                    args['maxshape'] = dset_source.maxshape

                if name == f'data/{self.detector_name}/data':  # compress and copy
                    data = self[:, roi_y, roi_x]
                    if factor:
                        data = np.round(data / factor)

                    args['shape'] = data.shape
                    args['maxshape'] = data.shape

                    if data.ndim == 3:
                        args['chunks'] = (1, *data.shape[1:])
                    else:
                        args['chunks'] = data.shape

                    if dtype is None:
                        args['dtype'] = data.dtype
                    else:
                        args['dtype'] = dtype

                    if compress:
                        args.update(compargs)

                    dset_dest = h5_dest.create_dataset(name, **args)
                    dset_dest[:] = data

                else:  # copy
                    h5_dest.create_dataset(name, data=dset_source, **args)

            # copy attributes
            for key, value in self.jf_file[name].attrs.items():
                h5_dest[name].attrs[key] = value

        roi_x = slice(*roi_x)
        roi_y = slice(*roi_y)

        with h5py.File(dest, 'w') as h5_dest:
            self.jf_file.visititems(copy_objects)

    def _locate_gain_file(self):
        # the default gain file location is
        # '/sf/<beamline>/config/jungfrau/gainMaps/<detector>/gains.h5'
        if self.file_path.parts[1] != 'sf':
            raise Exception(f'Gain file needs to be specified explicitly.')

        gain_path = Path(*self.file_path.parts[:3]).joinpath('config', 'jungfrau', 'gainMaps')
        gain_file = gain_path.joinpath(self.detector_name, 'gains.h5')

        if not gain_file.is_file():
            raise Exception(f'No gain file in default location: {gain_path}')

        return gain_file

    def _locate_pedestal_file(self):
        # the default processed pedestal files path for a particula p-group is
        # '/sf/<beamline>/data/<p-group>/res/JF_pedestals/'
        if self.file_path.parts[1] != 'sf':
            raise Exception(f'Pedestal file needs to be specified explicitly.')

        pedestal_path = Path(*self.file_path.parts[:5]).joinpath('res', 'JF_pedestals')

        # find a pedestal file, which was created closest in time to the jungfrau file
        jf_file_mtime = self.file_path.stat().st_mtime
        nearest_pedestal_file = None
        min_time_diff = float('inf')
        for entry in pedestal_path.iterdir():
            if entry.is_file() and self.detector_name in entry.name:
                time_diff = jf_file_mtime - entry.stat().st_mtime
                if abs(time_diff) < abs(min_time_diff):
                    min_time_diff = time_diff
                    nearest_pedestal_file = entry

        pedestal_file = nearest_pedestal_file

        if pedestal_file is None:
            raise Exception(f'No pedestal file in default location: {pedestal_path}')

        return pedestal_file, min_time_diff

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getitem__(self, item):
        if isinstance(item, str):
            # metadata entry (lazy)
            return self.jf_file[f'/data/{self.detector_name}/{item}']

        elif isinstance(item, (int, slice)):
            # single image index or slice, no roi
            ind, roi = item, ()

        else:
            # image index and roi
            ind, roi = item[0], item[1:]

        jf_data = self.jf_file[f'/data/{self.detector_name}/data'][ind]
        jf_data = self.jf_handler.process(jf_data)

        if roi:
            if jf_data.ndim == 3:
                roi = (slice(None), *roi)
            jf_data = jf_data[roi]

        return jf_data

    def __repr__(self):
        if self.jf_file.id:
            r = f'<Jungfrau file "{self.file_path.name}">'
        else:
            r = '<Closed Jungfrau file>'
        return r

    def close(self):
        """Close Jungfrau file"""
        if self.jf_file.id:
            self.jf_file.close()

    @property
    def shape(self):
        """Data shape in Jungfrau file"""
        return self['data'].shape

    @property
    def size(self):
        """Size of data in Jungfrau file"""
        return self['data'].size

    @property
    def ndim(self):
        """Number of dimensions of data in Jungfrau file"""
        return len(self.shape)

    def __len__(self):
        """Number of images in Jungfrau file"""
        return self.shape[0]
