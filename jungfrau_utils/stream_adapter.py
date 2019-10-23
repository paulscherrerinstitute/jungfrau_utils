import logging

import numpy as np

from .data_handler import JFDataHandler

logger = logging.getLogger(__name__)


class StreamAdapter:
    def __init__(self):
        # a placeholder for jf data handler to be initiated with detector name
        self._handler = None

    @staticmethod
    def get_gains(image):
        if image.dtype != np.uint16:
            raise TypeError(
                f"Expected image type is 'uint16', provided data has type '{image.dtype}'"
            )
        return image >> 14

    def process(self, image, metadata):
        # as a first step, try to set the detector_name
        detector_name = metadata.get('detector_name')
        # check if jungfrau data handler is already set for this detector
        if self._handler is None or self._handler.detector_name != detector_name:
            try:
                self._handler = JFDataHandler(detector_name)
            except KeyError:
                logging.exception(f"Error creating data handler for detector {detector_name}")
                self._handler = None

        # return a copy of input image if
        # 1) its data type differs from 'uint16' (probably, it is already been processed)
        # 2) jf data handler failed to be created for that detector_name
        if image.dtype != np.uint16 or self._handler is None:
            return np.copy(image)

        # parse metadata
        self._update_handler(metadata)

        # skip conversion step if jungfrau data handler cannot do it, thus avoiding Exception raise
        self._handler.convertion = self._handler.can_convert()

        return self._handler.process(image)

    def _update_handler(self, md_dict):
        # gain file
        gain_file = md_dict.get('gain_file')
        try:
            self._handler.gain_file = gain_file
        except Exception:
            logging.exception(f"Error loading gain file {gain_file}")
            self._handler.gain_file = None

        # pedestal file
        pedestal_file = md_dict.get('pedestal_file')
        try:
            self._handler.pedestal_file = pedestal_file
        except Exception:
            logging.exception(f"Error loading pedestal file {pedestal_file}")
            self._handler.pedestal_file = None

        # module map
        module_map = md_dict.get('module_map')
        self._handler.module_map = None if (module_map is None) else np.array(module_map)

        # highgain
        daq_rec = md_dict.get('daq_rec')
        self._handler.highgain = False if (daq_rec is None) else bool(daq_rec & 0b1)
