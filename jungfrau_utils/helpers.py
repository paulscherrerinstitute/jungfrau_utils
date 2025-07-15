import warnings

import numpy as np
from numpy.typing import NDArray

from jungfrau_utils.data_handler import (
    CHIP_GAP_X,
    CHIP_GAP_Y,
    CHIP_NUM_X,
    CHIP_NUM_Y,
    CHIP_SIZE_X,
    CHIP_SIZE_Y,
    MODULE_SIZE_Y,
    JFDataHandler,
)


def reverse_pixel_mask(
    detector_name: str, mask: NDArray, gap_pixels: bool = True, geometry: bool = True
) -> NDArray:
    """Get raw pixel mask from processed mask.

    Args:
        detector_name (str): Name of the detector.
        mask (ndarray): A processed pixel mask to reverse.
        gap_pixels (bool, optional): Whether processed pixel mask has gap_pixels applied.
            Defaults to True.
        geometry (bool, optional): Whether processed pixel mask has geometry applied.
            Defaults to True.

    Returns:
        raw_mask (ndarray): Raw pixel mask.
    """
    if detector_name == "JF02T09V01":
        raise ValueError("JF02T09V01 detector is not supported")

    if mask.ndim != 2:
        raise ValueError(f"Expected mask dimensions 2, provided {mask.ndim}.")

    handler = JFDataHandler(detector_name=detector_name)
    if handler.detector_geometry.is_stripsel and gap_pixels:
        warnings.warn("'gap_pixels' flag has no effect on stripsel detectors", RuntimeWarning)
        gap_pixels = False

    out_shape = handler.get_shape_out(gap_pixels=gap_pixels, geometry=geometry)

    if mask.shape != out_shape:
        raise ValueError(f"Expected pixel_mask shape {out_shape}, provided {mask.shape}.")

    raw_mask = np.zeros(handler._shape_in, dtype=bool)

    if handler.detector_geometry.is_stripsel and geometry:
        # create a map of pixel indexes for detector geometry
        _index = np.arange(np.prod(handler._shape_in)).reshape(handler._shape_in)
        # gap_pixels is False and geometry is True here
        index_map = (
            handler.process(_index + 1, conversion=False, mask=False, gap_pixels=False).ravel() - 1
        )
        # empty space has an index_map value of -1
        _data_index = index_map != -1
        raw_mask.ravel()[index_map[_data_index]] = mask.ravel()[_data_index]

    else:
        # module_map has to be default because the raw mask includes all modules
        for i, m in enumerate(handler.module_map):
            # m is always equal to i here (enumerate is kept for consistency)
            oy, oy_end, ox, ox_end, rot90 = handler._get_module_coords(m, i, geometry, gap_pixels)
            mask_module = np.rot90(mask[oy:oy_end, ox:ox_end], k=-rot90)
            if gap_pixels:
                for i1 in range(CHIP_NUM_Y):
                    out_y = m * MODULE_SIZE_Y + i1 * CHIP_SIZE_Y
                    in_y = i1 * (CHIP_SIZE_Y + CHIP_GAP_Y)
                    for i2 in range(CHIP_NUM_X):
                        out_x = i2 * CHIP_SIZE_X
                        in_x = i2 * (CHIP_SIZE_X + CHIP_GAP_X)
                        raw_mask[out_y : out_y + CHIP_SIZE_Y, out_x : out_x + CHIP_SIZE_X] = (
                            mask_module[in_y : in_y + CHIP_SIZE_Y, in_x : in_x + CHIP_SIZE_X]
                        )
            else:
                raw_mask[m * MODULE_SIZE_Y : (m + 1) * MODULE_SIZE_Y, :] = mask_module

    return np.invert(raw_mask)
