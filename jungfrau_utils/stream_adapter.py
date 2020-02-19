import logging

import numpy as np

from .data_handler import JFDataHandler

logger = logging.getLogger(__name__)


class StreamAdapter:
    def __init__(self):
        # a placeholder for jf data handler to be initiated with detector name
        self.handler = None

    def process(self, image, metadata):
        # as a first step, try to set the detector_name, skip if detector_name is empty
        detector_name = metadata.get("detector_name")
        if detector_name:
            # check if jungfrau data handler is already set for this detector
            if self.handler is None or self.handler.detector_name != detector_name:
                try:
                    self.handler = JFDataHandler(detector_name)
                except KeyError:
                    logging.exception(f"Error creating data handler for detector {detector_name}")
                    self.handler = None
        else:
            self.handler = None

        # return a copy of input image if
        # 1) its data type differs from 'uint16' (probably, it is already been processed)
        # 2) jf data handler failed to be created for that detector_name
        if image.dtype != np.uint16 or self.handler is None:
            return np.copy(image)

        # parse metadata
        self._update_handler(metadata)

        # skip conversion step if jungfrau data handler cannot do it, thus avoiding Exception raise
        conversion = self.handler.can_convert()

        return self.handler.process(image, conversion=conversion)

    def _update_handler(self, md_dict):
        # gain file
        gain_file = md_dict.get("gain_file", "")
        try:
            self.handler.gain_file = gain_file
        except Exception:
            logging.exception(f"Error loading gain file {gain_file}")
            self.handler.gain_file = ""

        # pedestal file
        pedestal_file = md_dict.get("pedestal_file", "")
        try:
            self.handler.pedestal_file = pedestal_file
        except Exception:
            logging.exception(f"Error loading pedestal file {pedestal_file}")
            self.handler.pedestal_file = ""

        # module map
        module_map = md_dict.get("module_map")
        self.handler.module_map = None if (module_map is None) else np.array(module_map)

        # highgain
        daq_rec = md_dict.get("daq_rec")
        self.handler.highgain = False if (daq_rec is None) else bool(daq_rec & 0b1)
