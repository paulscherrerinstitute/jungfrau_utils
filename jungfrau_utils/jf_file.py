from pathlib import Path

import h5py

from jungfrau_utils.corrections import apply_gain_pede


class File():
    """ Jungfrau file """
    def __init__(self, file_path, gain_file=None, pedestal_file=None):
        file_path = Path(file_path)
        self.file_path = file_path

        self.jf_file = h5py.File(file_path, 'r')
        self.detector_name = self.jf_file['/general/detector_name'][()].decode()  #pylint: disable=E1101

        # Gain file
        if gain_file is None:
            # the default gain file location is
            # '/sf/<beamline>/config/jungfrau/gainMaps/<detector>/gains.h5'
            gain_file = Path(*file_path.parts[:3]).joinpath(
                'config', 'jungfrau', 'gainMaps', self.detector_name, 'gains.h5'
            )

        try:
            with h5py.File(gain_file, 'r') as h5gain:
                gain = h5gain['/gains'][:]
        except:
            print('Error reading gain file:', gain_file)
            raise

        self.gain = gain

        # Pedestal file (with a pixel mask)
        if pedestal_file is None:
            # the default processed pedestal files path for a particula p-group is
            # '/sf/<beamline>/data/<p-group>/res/JF_pedestal/'
            pedestal_path = Path(*file_path.parts[:5]).joinpath(
                'res', 'JF_pedestal'
            )

            # find a pedestal file, which was created closest in time to the jungfrau file
            jf_file_mtime = file_path.stat().st_mtime
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
                raise Exception('No pedestal file in default location: {}'.format(pedestal_path))

        try:
            with h5py.File(pedestal_file, 'r') as h5pedestal:
                pedestal = h5pedestal['/gains'][:]
                pixel_mask = h5pedestal['/pixel_mask'][:]
        except:
            print('Error reading pedestal file:', pedestal_file)
            raise

        self.pedestal = pedestal
        self.pixel_mask = pixel_mask

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getitem__(self, item):
        jf_data = self.jf_file['/data/{}/data'.format(self.detector_name)][item]
        # apply gain and pedestal corrections
        jf_proc = apply_gain_pede(jf_data, G=self.gain, P=self.pedestal, pixel_mask=self.pixel_mask)
        return jf_proc

    def __repr__(self):
        if self.jf_file.id:
            r = '<Jungfrau file "{}">'.format(self.file_path.name)
        else:
            r = '<Closed Jungfrau file>'
        return r

    def close(self):
        if self.jf_file.id:
            self.jf_file.close()
