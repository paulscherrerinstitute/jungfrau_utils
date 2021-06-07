import h5py

from jungfrau_utils.data_handler import JFDataHandler
from jungfrau_utils.swissfel_helpers import locate_gain_file, locate_pedestal_file


class EscapeAdapter:
    """Adapter to interface jungfrau data handler with escape library.

    Args:
        file_path (str): Path to Jungfrau file, which metadata should be used for jungfrau data
            handler setup.
        gain_file (str, optional): Path to gain file. Auto-locate if empty. Defaults to "".
        pedestal_file (str, optional): Path to pedestal file. Auto-locate if empty. Defaults to "".
    """

    def __init__(self, file_path, *, gain_file="", pedestal_file=""):
        with h5py.File(file_path, "r") as h5f:
            detector_name = h5f["/general/detector_name"][()].decode()
            self.handler = JFDataHandler(detector_name)

            if "module_map" in h5f[f"/data/{detector_name}"]:
                # Pick only the first row (module_map of the first frame), because it is not
                # expected that module_map ever changes during a run. In fact, it is forseen in the
                # future that this data will be saved as a single row for the whole run.
                module_map = h5f[f"/data/{detector_name}/module_map"][0, :]
            else:
                module_map = None

            self.handler.module_map = module_map

            # TODO: Here we use daq_rec only of the first pulse within an hdf5 file, however its
            # value can be different for later pulses and this needs to be taken care of.
            daq_rec = h5f[f"/data/{detector_name}/daq_rec"][0]
            self.handler.highgain = daq_rec & 0b1

        # Gain file
        if not gain_file:
            gain_file = locate_gain_file(file_path)

        self.handler.gain_file = gain_file

        # Pedestal file (with a pixel mask)
        if not pedestal_file:
            pedestal_file = locate_pedestal_file(file_path)

        self.handler.pedestal_file = pedestal_file

    @property
    def process(self):
        """Escape represents jungfrau files as a single array and only needs the process function.
        """
        return self.handler.process

    @property
    def gain_file(self):
        """Gain file path (readonly).
        """
        return self.handler.gain_file

    @property
    def pedestal_file(self):
        """Pedestal file path (readonly).
        """
        return self.handler.pedestal_file
