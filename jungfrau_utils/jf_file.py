from pathlib import Path

import h5py

from .corrections import JFDataHandler


class File:
    """ Jungfrau file """

    def __init__(self, file_path, gain_file=None, pedestal_file=None, raw=False, geometry=True):
        self.raw = raw
        self.geometry = geometry

        self.file_path = Path(file_path)

        self.jf_file = h5py.File(self.file_path, 'r')
        self.detector_name = self.jf_file['/general/detector_name'][()].decode()

        # TODO: Here we use daq_rec only of the first pulse within an hdf5 file, however its
        # value can be different for later pulses and this needs to be taken care of. Currently,
        # _allow_n_images decorator applies a function in a loop, making it impossible to change
        # highgain for separate images in a 3D stack.
        self.daq_rec = self.jf_file[f'/data/{self.detector_name}/daq_rec'][0]

        if 'module_map' in self.jf_file[f'/data/{self.detector_name}']:
            # Pick only the first row (module_map of the first frame), because it is not expected
            # that module_map ever changes during a run. In fact, it is forseen in the future that
            # this data will be saved as a single row for the whole run.
            self.module_map = self.jf_file[f'/data/{self.detector_name}/module_map'][0, :]
        else:
            self.module_map = None

        # Gain file
        if gain_file is None:
            gain_file = self._locate_gain_file()
            print(f'Auto-located gain file: {gain_file}')

        try:
            with h5py.File(gain_file, 'r') as h5gain:
                gain = h5gain['/gains'][:]
        except:
            print('Error reading gain file:', gain_file)
            raise
        else:
            self.gain_file = gain_file

        # Pedestal file (with a pixel mask)
        if pedestal_file is None:
            pedestal_file = self._locate_pedestal_file()
            print(f'Auto-located pedestal file: {pedestal_file}')

        try:
            with h5py.File(pedestal_file, 'r') as h5pedestal:
                pedestal = h5pedestal['/gains'][:]
                pixel_mask = h5pedestal['/pixel_mask'][:]
        except:
            print('Error reading pedestal file:', pedestal_file)
            raise
        else:
            self.pedestal_file = pedestal_file

        self.jf_handler = JFDataHandler(self.detector_name, gain, pedestal, pixel_mask)

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
                time_diff = abs(entry.stat().st_mtime - jf_file_mtime)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    nearest_pedestal_file = entry

        pedestal_file = nearest_pedestal_file

        if pedestal_file is None:
            raise Exception(f'No pedestal file in default location: {pedestal_path}')

        return pedestal_file

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

        if self.jf_handler.highgain != self.daq_rec & 0b1:
            self.jf_handler.highgain = self.daq_rec & 0b1

        if self.module_map is not None:
            if (self.jf_handler.module_map != self.module_map).any():
                self.jf_handler.module_map = self.module_map

        if not self.raw:  # apply gain and pedestal corrections
            jf_data = self.jf_handler.apply_gain_pede(jf_data)

        if self.geometry:  # apply detector geometry corrections
            jf_data = self.jf_handler.apply_geometry(jf_data)

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
        if self.jf_file.id:
            self.jf_file.close()

    @property
    def shape(self):
        return self['data'].shape

    @property
    def size(self):
        return self['data'].size

    @property
    def ndim(self):
        return len(self.shape)

    def __len__(self):
        return self.shape[0]
