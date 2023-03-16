import warnings

import h5py
import numpy as np

from jungfrau_utils.data_handler import JFDataHandler
from jungfrau_utils.swissfel_helpers import (
    get_single_detector_name,
    locate_gain_file,
    locate_pedestal_file,
)

warnings.filterwarnings("default", category=DeprecationWarning)


class EscapeAdapter:
    """Adapter to interface jungfrau data handler with escape library.

    Args:
        file_path (str): Path to Jungfrau file, which metadata should be used for jungfrau data
            handler setup.
        detector_name (str, optional): Name of a detector, which data should be processed if there
            are multiple detectors' data present in the file. If empty, the file must contain data
            for a single detector only. Defaults to ''.
        gain_file (str, optional): Path to gain file. Auto-locate if empty. Defaults to "".
        pedestal_file (str, optional): Path to pedestal file. Auto-locate if empty. Defaults to "".
    """

    def __init__(self, file_path, *, detector_name="", gain_file="", pedestal_file=""):
        if not detector_name:
            detector_name = get_single_detector_name(file_path)

        self.handler = JFDataHandler(detector_name)

        # Gain file
        if not gain_file:
            gain_file = locate_gain_file(file_path, detector_name=detector_name)

        self.handler.gain_file = gain_file

        # Pedestal file (with a pixel mask)
        if not pedestal_file:
            pedestal_file = locate_pedestal_file(file_path, detector_name=detector_name)

        self.handler.pedestal_file = pedestal_file

        with h5py.File(file_path, "r") as h5f:
            data_group = h5f[f"data/{detector_name}"]
            meta_group = (
                data_group["meta"]
                if "meta" in data_group and isinstance(data_group["meta"], h5py.Group)
                else data_group
            )

            if "module_map" in meta_group:
                module_map = meta_group["module_map"][:]
                if module_map.ndim == 2:
                    # This is an old format. Pick only the first row (module_map of the first frame),
                    # because it is not expected that module_map ever changes during a run.
                    module_map = module_map[0, :]
            else:
                module_map = None

            self.handler.module_map = module_map

            # TODO: Here we use daq_rec only of the first pulse, where is_good_frame is True, within
            # an hdf5 file, however its value can be different for later pulses and this needs to be
            # taken care of.
            good_frame_idx = np.nonzero(data_group["is_good_frame"])[0]
            if good_frame_idx.size > 0:
                daq_rec = data_group["daq_rec"][good_frame_idx[0]]
            else:
                warnings.warn("The file doesn't contain good frames. Highgain is set to False.")
                daq_rec = 0

            self.handler.highgain = daq_rec & 0b1

    @property
    def process(self):
        """Escape represents jungfrau files as a single array and only needs the process function."""
        return self.handler.process

    @property
    def gain_file(self):
        """Gain file path (readonly)."""
        return self.handler.gain_file

    @property
    def pedestal_file(self):
        """Pedestal file path (readonly)."""
        return self.handler.pedestal_file
