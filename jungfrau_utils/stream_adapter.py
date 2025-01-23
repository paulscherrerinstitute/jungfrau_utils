from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from jungfrau_utils.data_handler import JFDataHandler

logger = logging.getLogger(__name__)


class StreamAdapter:
    def __init__(self) -> None:
        # a placeholder for jf data handler to be initiated with detector name
        self.handler: JFDataHandler | None = None

    def process(self, image: NDArray, metadata: dict, **kwargs) -> NDArray:
        """Perform jungfrau detector data processing on an image received via stream.

        Args:
            image (ndarray): An image to be processed.
            metadata (dict): A corresponding image metadata.
            **kwargs: Extra arguments for JFDataHandler.process() call.

        Returns:
            ndarray: Resulting image.
        """
        # as a first step, try to set the detector_name, skip if detector_name is empty
        detector_name = metadata.get("detector_name")
        if detector_name:
            # check if jungfrau data handler is already set for this detector
            if self.handler is None or self.handler.detector_name != detector_name:
                try:
                    self.handler = JFDataHandler(detector_name)
                except ValueError:
                    logging.exception(f"Error creating data handler for detector {detector_name}")
                    self.handler = None
        else:
            self.handler = None

        # return a copy of input image if
        # 1) its data type differs from 'uint16' (probably, it is already been processed)
        # 2) jf data handler failed to be created for that detector_name
        if image.dtype != np.uint16 or self.handler is None:
            return np.copy(image)

        # gain file
        gain_file = metadata.get("gain_file", "")
        try:
            self.handler.gain_file = gain_file
        except Exception:
            logging.exception(f"Error loading gain file {gain_file}")
            self.handler.gain_file = ""

        # pedestal file
        pedestal_file = metadata.get("pedestal_file", "")
        try:
            self.handler.pedestal_file = pedestal_file
        except Exception:
            logging.exception(f"Error loading pedestal file {pedestal_file}")
            self.handler.pedestal_file = ""

        # module map
        module_map = metadata.get("module_map")
        self.handler.module_map = None if (module_map is None) else np.array(module_map)

        # highgain
        daq_rec = metadata.get("daq_rec")
        self.handler.highgain = False if (daq_rec is None) else bool(daq_rec & 0b1)

        if "conversion" not in kwargs:
            # skip conversion step if jungfrau data handler cannot do it, thus avoiding Exception
            # raise
            kwargs["conversion"] = self.handler.can_convert()

        if "mask" not in kwargs:
            # skip masking step if pixel_mask is None
            kwargs["mask"] = self.handler.pixel_mask is not None

        return self.handler.process(image, **kwargs)
