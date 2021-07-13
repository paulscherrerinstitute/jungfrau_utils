import re
import warnings
from collections import namedtuple
from functools import wraps

import h5py
import numpy as np
from numba import njit, prange

from jungfrau_utils.geometry import detector_geometry

CHIP_SIZE_X = 256
CHIP_SIZE_Y = 256

CHIP_NUM_X = 4
CHIP_NUM_Y = 2

MODULE_SIZE_X = CHIP_NUM_X * CHIP_SIZE_X
MODULE_SIZE_Y = CHIP_NUM_Y * CHIP_SIZE_Y

CHIP_GAP_X = 2
CHIP_GAP_Y = 2

# 256 not divisible by 3, so we round up to 86
# the last 4 pixels can be omitted, so the final height is (256 - 4) / 3 = 84

# 18 since we have 6 more pixels in H per gap
STRIPSEL_SIZE_X = 1024 * 3 + 18  # = 3090
STRIPSEL_SIZE_Y = 84


def _allow_2darray(func):
    @wraps(func)
    def wrapper(self, array, *args, **kwargs):
        if array.ndim == 2:
            is_2darray = True
            array = array[np.newaxis]
        else:
            is_2darray = False

        array = func(self, array, *args, **kwargs)

        if is_2darray:
            array = array[0]

        return array

    return wrapper


class JFDataHandler:
    """A class to perform jungfrau detector data handling like pedestal correction,
    gain conversion, pixel mask, module map, etc.

    Args:
        detector_name (str): name of a detector in the form ``JF<id>T<nmod>V<version>``
    """

    def __init__(self, detector_name):
        # detector_name needs to be a valid name
        if detector_name in detector_geometry:
            self._detector_name = detector_name
            self._detector_geometry = detector_geometry[detector_name]
        else:
            raise KeyError(f"Geometry for '{detector_name}' detector is not present.")

        self._gain_file = ""
        self._pedestal_file = ""

        # these values store the original gains/pedestal values
        self._gain = None
        self._pedestal = None
        self._pixel_mask = None

        self._factor = None
        self._highgain = False

        # gain and pedestal arrays with better memory layout for the actual data conversion
        self._g_all = {True: None, False: None}
        self._p_all = {True: None, False: None}

        self._module_map = np.arange(self.detector.n_modules)

        self._mask_all = {True: None, False: None}

    @property
    def detector_name(self):
        """Detector name (readonly).
        """
        return self._detector_name

    @property
    def detector_geometry(self):
        """Detector geometry configuration (readonly).
        """
        return self._detector_geometry

    def is_stripsel(self):
        """Return true if detector is a stripsel.
        """
        return self.detector_geometry.is_stripsel

    @property
    def detector(self):
        """A namedtuple of detector parameters extracted from its name (readonly).
        """
        det = namedtuple("Detector", ["id", "n_modules", "version"])
        return det(*(int(d) for d in re.findall(r"\d+", self.detector_name)))

    def _get_shape_n_modules(self, n):
        if self.detector_name == "JF02T09V01":  # a special case
            shape_y, shape_x = MODULE_SIZE_Y, MODULE_SIZE_X * n
        else:
            shape_y, shape_x = MODULE_SIZE_Y * n, MODULE_SIZE_X

        return shape_y, shape_x

    @property
    def _number_active_modules(self):
        return np.sum(self.module_map != -1)

    @property
    def _shape_full(self):
        return self._get_shape_n_modules(self.detector.n_modules)

    @property
    def _shape_in(self):
        return self._get_shape_n_modules(self._number_active_modules)

    def _get_shape_out(self, gap_pixels, geometry):
        """Return the image shape of a detector without an optional post-processing rotation step,
        based on gap_pixel and geometry flags

        Args:
            gap_pixels (bool, optional): Add gap pixels between detector chips. Defaults to True.
            geometry (bool, optional): Apply detector geometry corrections. Defaults to True.

        Returns:
            tuple: Height and width of a resulting image.
        """
        if self.is_stripsel():
            return self._get_stripsel_shape_out(geometry=geometry)

        if geometry and gap_pixels:
            modules_orig_y = self.detector_geometry.origin_y
            modules_orig_x = self.detector_geometry.origin_x
            shape_x = max(modules_orig_x) + MODULE_SIZE_X + (CHIP_NUM_X - 1) * CHIP_GAP_X
            shape_y = max(modules_orig_y) + MODULE_SIZE_Y + (CHIP_NUM_Y - 1) * CHIP_GAP_Y

        elif geometry and not gap_pixels:
            modules_orig_y = self.detector_geometry.origin_y
            modules_orig_x = self.detector_geometry.origin_x
            shape_x = max(modules_orig_x) + MODULE_SIZE_X
            shape_y = max(modules_orig_y) + MODULE_SIZE_Y

        elif not geometry and gap_pixels:
            shape_y, shape_x = self._shape_in
            if self.detector_name == "JF02T09V01":
                shape_x += (CHIP_NUM_X - 1) * CHIP_GAP_X * self._number_active_modules
                shape_y += (CHIP_NUM_Y - 1) * CHIP_GAP_Y
            else:
                shape_x += (CHIP_NUM_X - 1) * CHIP_GAP_X
                shape_y += (CHIP_NUM_Y - 1) * CHIP_GAP_Y * self._number_active_modules

        else:  # not geometry and not gap_pixels:
            shape_y, shape_x = self._shape_in

        return shape_y, shape_x

    def get_shape_out(self, *, gap_pixels=True, geometry=True):
        """Return the final image shape of a detector, based on gap_pixel and geometry flags

        Args:
            gap_pixels (bool, optional): Add gap pixels between detector chips. Defaults to True.
            geometry (bool, optional): Apply detector geometry corrections. Defaults to True.

        Returns:
            tuple: Height and width of a resulting image.
        """
        shape_y, shape_x = self._get_shape_out(gap_pixels, geometry)
        if geometry and self.detector_geometry.rotate90 % 2:
            shape_y, shape_x = shape_x, shape_y

        return shape_y, shape_x

    def _get_stripsel_shape_out(self, geometry):
        if geometry:
            modules_orig_y = self.detector_geometry.origin_y
            modules_orig_x = self.detector_geometry.origin_x
            shape_x = max(modules_orig_x) + STRIPSEL_SIZE_X
            shape_y = max(modules_orig_y) + STRIPSEL_SIZE_Y
        else:
            shape_y, shape_x = self._shape_in

        return shape_y, shape_x

    def get_dtype_out(self, dtype_in, *, conversion=True):
        """Resulting image dtype of a detector, based on input dtype and a conversion flag.

        Args:
            dtype_in (dtype): dtype of an input data.
            conversion (bool, optional): Whether data is expected to be converted to keV (apply gain
                and pedestal corrections). Defaults to True.

        Returns:
            dtype: dtype of a resulting image.
        """
        if conversion:
            if dtype_in != np.uint16:
                raise TypeError(f"Only images of dtype {np.dtype(np.uint16)} can be converted.")

            if self.factor is None:
                dtype_out = np.dtype(np.float32)
            else:
                dtype_out = np.dtype(np.int32)
        else:
            dtype_out = dtype_in

        return dtype_out

    @property
    def gain_file(self):
        """Return current gain filepath.
        """
        return self._gain_file

    @gain_file.setter
    def gain_file(self, filepath):
        if not filepath:
            self._gain_file = ""
            self.gain = None
            return

        if filepath == self._gain_file:
            return

        with h5py.File(filepath, "r") as h5f:
            gains = h5f["/gains"][:]

        self._gain_file = filepath
        self.gain = gains

    @property
    def gain(self):
        """Current gain values.
        """
        return self._gain

    @gain.setter
    def gain(self, value):
        if value is None:
            self._gain = None
            return

        if value.ndim != 3:
            raise ValueError(f"Expected gain dimensions 3, provided gain dimensions {value.ndim}.")

        g_shape = (4, *self._shape_full)
        if value.shape != g_shape:
            raise ValueError(f"Expected gain shape {g_shape}, provided gain shape {value.shape}.")

        # convert _gain values to float32
        self._gain = value.astype(np.float32, copy=False)
        self._update_g_all()

    def _update_g_all(self):
        if self.factor is None:
            _g = 1 / self.gain
        else:
            # self.factor is one number and self.gain is a large array, so this order of division
            # will avoid double broadcasting
            _g = 1 / self.factor / self.gain

        self._g_all[True] = np.tile(_g[3], (4, 1, 1))

        _g[3] = _g[2]
        self._g_all[False] = _g

    @property
    def pedestal_file(self):
        """Return current pedestal filepath.
        """
        return self._pedestal_file

    @pedestal_file.setter
    def pedestal_file(self, filepath):
        if not filepath:
            self._pedestal_file = ""
            self.pedestal = None
            self.pixel_mask = None
            return

        if filepath == self._pedestal_file:
            return

        with h5py.File(filepath, "r") as h5f:
            pedestal = h5f["/gains"][:]
            pixel_mask = h5f["/pixel_mask"][:]

        self._pedestal_file = filepath
        self.pedestal = pedestal
        self.pixel_mask = pixel_mask

    @property
    def pedestal(self):
        """Current pedestal values.
        """
        return self._pedestal

    @pedestal.setter
    def pedestal(self, value):
        if value is None:
            self._pedestal = None
            return

        if value.ndim != 3:
            raise ValueError(
                f"Expected pedestal dimensions 3, provided pedestal dimensions {value.ndim}."
            )

        p_shape = (4, *self._shape_full)
        if value.shape != p_shape:
            raise ValueError(
                f"Expected pedestal shape {p_shape}, provided pedestal shape {value.shape}."
            )

        # convert _pedestal values to float32
        self._pedestal = value.astype(np.float32, copy=False)

        _p = self._pedestal.copy()

        self._p_all[True] = np.tile(_p[3], (4, 1, 1))

        _p[3] = _p[2]
        self._p_all[False] = _p

    @property
    def factor(self):
        """A factor value.

        If conversion is True, use this factor to divide converted values. The output values are
        also rounded and casted to np.int32 dtype. Keep the original values if None.
        """
        return self._factor

    @factor.setter
    def factor(self, value):
        if value is not None:
            value = float(value)

        if self._factor == value:
            return

        self._factor = value

        if self.gain is not None:
            self._update_g_all()

    @property
    def highgain(self):
        """Current flag for highgain.
        """
        return self._highgain

    @highgain.setter
    def highgain(self, value):
        if not isinstance(value, bool):
            value = bool(value)

        self._highgain = value

    @property
    def _g(self):
        return self._g_all[self.highgain]

    @property
    def _p(self):
        return self._p_all[self.highgain]

    @property
    def pixel_mask(self):
        """Current raw pixel mask values.
        """
        return self._pixel_mask

    @pixel_mask.setter
    def pixel_mask(self, value):
        if value is None:
            self._pixel_mask = None
            return

        if value.ndim != 2:
            raise ValueError(
                f"Expected pixel_mask dimensions 2, provided pixel_mask dimensions {value.ndim}."
            )

        pm_shape = self._shape_full
        if value.shape != pm_shape:
            raise ValueError(
                f"Expected pixel_mask shape {pm_shape}, provided pixel_mask shape {value.shape}."
            )

        self._pixel_mask = value

        # self._mask_all[False] -> original mask
        mask = np.invert(value.astype(bool, copy=True))
        self._mask_all[False] = mask.copy()

        # self._mask_all[True] -> original + double pixels mask
        if not self.is_stripsel():
            for m in range(self.detector.n_modules):
                module_mask = self._get_module_slice(mask, m)
                for n in range(1, CHIP_NUM_X):
                    module_mask[:, CHIP_SIZE_X * n - 1] = False
                    module_mask[:, CHIP_SIZE_X * n] = False

                for n in range(1, CHIP_NUM_Y):
                    module_mask[CHIP_SIZE_Y * n - 1, :] = False
                    module_mask[CHIP_SIZE_Y * n, :] = False

        self._mask_all[True] = mask

    def get_pixel_mask(self, *, gap_pixels=True, double_pixels="keep", geometry=True):
        """Return pixel mask, shaped according to gap_pixel and geometry flags.

        Args:
            gap_pixels (bool, optional): Add gap pixels between detector chips. Defaults to True.
            double_pixels (str, optional): A method to handle double pixels in-between ASICs. Can be
                "keep", "mask", or "interp". Defaults to "keep".
            geometry (bool, optional): Apply detector geometry corrections. Defaults to True.

        Returns:
            ndarray: Resulting pixel mask, where True values correspond to valid pixels.
        """
        if double_pixels == "interp" and not gap_pixels:
            raise RuntimeError("Double pixel interpolation requires 'gap_pixels' to be True.")

        if self.is_stripsel() and gap_pixels:
            warnings.warn("'gap_pixels' flag has no effect on stripsel detectors", RuntimeWarning)
            gap_pixels = False

        if self._pixel_mask is None:
            return None

        _mask = self._mask(double_pixels)[np.newaxis]

        res_shape = self._get_shape_out(gap_pixels, geometry)
        res = np.zeros((1, *res_shape), dtype=bool)

        for i, m in enumerate(self.module_map):
            if m == -1:
                continue

            oy, ox = self._get_final_module_coordinates(m, i, geometry, gap_pixels)

            mod = self._get_module_slice(_mask, i, geometry)
            mod_res = res[:, oy:, ox:]

            if self.is_stripsel() and geometry:
                _reshape_stripsel(mod_res, mod)
            else:
                # this will just copy data to the correct place
                _adc_to_energy(mod_res, mod, None, None, None, None, gap_pixels)
                if double_pixels == "interp":
                    _inplace_mask_dp(mod_res)

        # rotate mask according to a geometry configuration (e.g. for alvra JF06 detectors)
        if geometry and self.detector_geometry.rotate90:
            res = np.rot90(res, k=self.detector_geometry.rotate90, axes=(1, 2))

        res = res[0]

        return res

    def _mask(self, double_pixels):
        if double_pixels in ("keep", "interp"):
            return self._mask_all[False]
        elif double_pixels == "mask":
            return self._mask_all[True]
        else:
            raise ValueError("'double_pixels' can only be 'keep', 'mask', or 'interp'")

    @property
    def module_map(self):
        """Current module map.
        """
        return self._module_map

    @module_map.setter
    def module_map(self, value):
        n_modules = self.detector.n_modules
        if value is None:
            # support legacy data by emulating 'all modules are present'
            self._module_map = np.arange(n_modules)
            return

        if len(value) != n_modules:
            raise ValueError(
                f"Expected module_map length {n_modules}, provided module_map length {len(value)}."
            )

        if min(value) < -1 or n_modules <= max(value):
            raise ValueError(f"Valid module_map values are integers between -1 and {n_modules-1}.")

        self._module_map = value

    @_allow_2darray
    def process(
        self,
        images,
        *,
        conversion=True,
        mask=True,
        gap_pixels=True,
        double_pixels="keep",
        geometry=True,
        parallel=False,
        out=None,
    ):
        """Perform jungfrau detector data processing like pedestal correction, gain conversion,
        applying pixel mask, module map, etc.

        Args:
            images (ndarray): Image stack or single image to be processed
            conversion (bool, optional): Convert to keV (apply gain and pedestal corrections).
                Defaults to True.
            mask (bool, optional): Perform masking of bad pixels (set those values to 0).
                Defaults to True.
            gap_pixels (bool, optional): Add gap pixels between detector chips. Defaults to True.
            double_pixels (str, optional): A method to handle double pixels in-between ASICs. Can be
                "keep", "mask", or "interp". Defaults to "keep".
            geometry (bool, optional): Apply detector geometry corrections. Defaults to True.
            parallel (bool, optional): Parallelize image stack processing. Defaults to False.
            out (ndarray, optional): If provided, the destination to place the result. The shape
                must be correct, matching that of what the function would have returned if no out
                argument were specified. Defaults to None.

        Returns:
            ndarray: Resulting image stack or single image
        """
        image_shape = images.shape[-2:]
        if image_shape != self._shape_in:
            raise ValueError(
                f"Expected image shape {self._shape_in}, provided image shape {image_shape}."
            )

        if not (conversion or mask or gap_pixels or geometry):
            # no need to continue, return unchanged images
            if out is not None:
                out[:] = images
            return images

        if conversion and not self.can_convert():
            raise RuntimeError("Gain and/or pedestal values are not set.")

        if mask and self._pixel_mask is None:
            raise RuntimeError("Pixel mask values are not set.")

        if double_pixels not in ("keep", "mask", "interp"):
            raise ValueError("'double_pixels' can only be 'keep', 'mask', or 'interp'")

        if not mask and double_pixels == "mask":
            warnings.warn(
                '\'double_pixels="mask"\' has no effect when "mask"=False', RuntimeWarning
            )

        if double_pixels == "interp" and not gap_pixels:
            raise RuntimeError("Double pixel interpolation requires 'gap_pixels' to be True.")

        if double_pixels == "interp" and self.factor is not None:
            raise ValueError("Unsupported mode: double_pixels='interp' with a factor value.")

        if self.is_stripsel() and gap_pixels:
            warnings.warn("'gap_pixels' flag has no effect on stripsel detectors", RuntimeWarning)
            gap_pixels = False

        if self.is_stripsel() and double_pixels == "mask":
            warnings.warn(
                "Masking double pixels has no effect on stripsel detectors", RuntimeWarning
            )
            double_pixels = "keep"

        if out is None:
            # get the shape without an optional rotation
            out_shape = self._get_shape_out(gap_pixels, geometry)
            out_dtype = self.get_dtype_out(images.dtype, conversion=conversion)
            out = np.zeros((images.shape[0], *out_shape), dtype=out_dtype)
        else:
            out_shape = self.get_shape_out(gap_pixels=gap_pixels, geometry=geometry)
            if out.shape[-2:] != out_shape:
                raise ValueError(f"Expected 'out' shape is {out_shape}, provided is {out.shape}")

            # reshape inplace in case of a detector with an odd detector_geometry.rotate90 value
            # it will be rotated to the original shape at the end of this function
            if geometry and self.detector_geometry.rotate90 % 2:
                out.shape = out_shape[1], out_shape[0]

        self._process(out, images, conversion, mask, gap_pixels, double_pixels, geometry, parallel)

        # rotate image stack according to a geometry configuration (e.g. for alvra JF06 detectors)
        if geometry and self.detector_geometry.rotate90:
            out = np.rot90(out, k=self.detector_geometry.rotate90, axes=(1, 2))

        return out

    def can_convert(self):
        """Whether all data for gain/pedestal conversion is present.

        Returns:
            bool: Return true if all data for gain/pedestal conversion is present.
        """
        return (self.gain is not None) and (self.pedestal is not None)

    def _process(
        self, res, images, conversion, mask, gap_pixels, double_pixels, geometry, parallel
    ):
        _adc_to_energy = _adc_to_energy_jit[parallel]
        factor = self.factor
        _mask = self._mask(double_pixels)

        for i, m in enumerate(self.module_map):
            if m == -1:
                continue

            oy, ox = self._get_final_module_coordinates(m, i, geometry, gap_pixels)
            mod = self._get_module_slice(images, m, geometry)

            if mask:
                mod_mask = self._get_module_slice(_mask, i, geometry)
            else:
                mod_mask = None

            if conversion:
                mod_g = self._get_module_slice(self._g, i, geometry)
                mod_p = self._get_module_slice(self._p, i, geometry)
            else:
                mod_g = None
                mod_p = None

            mod_res = res[:, oy:, ox:]
            if self.is_stripsel() and geometry:
                mod_tmp_shape = (images.shape[0], *self._get_shape_n_modules(1))
                mod_tmp_dtype = self.get_dtype_out(images.dtype, conversion=conversion)
                mod_tmp = np.zeros(shape=mod_tmp_shape, dtype=mod_tmp_dtype)

                _adc_to_energy(mod_tmp, mod, mod_g, mod_p, mod_mask, factor, gap_pixels)
                _reshape_stripsel_jit[parallel](mod_res, mod_tmp)
            else:
                _adc_to_energy(mod_res, mod, mod_g, mod_p, mod_mask, factor, gap_pixels)
                if double_pixels == "interp":
                    _inplace_interp_dp_jit[parallel](mod_res)

    def _get_final_module_coordinates(self, m, i, geometry, gap_pixels):
        if geometry:
            oy = self.detector_geometry.origin_y[i]
            ox = self.detector_geometry.origin_x[i]

        elif gap_pixels:
            if self.detector_name == "JF02T09V01":
                oy = 0
                ox = m * (MODULE_SIZE_X + (CHIP_NUM_X - 1) * CHIP_GAP_X)
            else:
                oy = m * (MODULE_SIZE_Y + (CHIP_NUM_Y - 1) * CHIP_GAP_Y)
                ox = 0

        else:
            if self.detector_name == "JF02T09V01":
                oy = 0
                ox = m * MODULE_SIZE_X
            else:
                oy = m * MODULE_SIZE_Y
                ox = 0

        return oy, ox

    @_allow_2darray
    def _get_module_slice(self, data, m, geometry=False):
        if self.detector_name == "JF02T09V01":
            out = data[:, :, m * MODULE_SIZE_X : (m + 1) * MODULE_SIZE_X]
        else:
            out = data[:, m * MODULE_SIZE_Y : (m + 1) * MODULE_SIZE_Y, :]

        if geometry and self.detector_name in ("JF02T09V02", "JF02T01V02"):
            out = np.rot90(out, 2, axes=(1, 2))

        return out

    def get_gains(self, images, *, mask=True, gap_pixels=True, geometry=True):
        """Return gain values of images, based on mask, gap_pixel and geometry flags.

        Args:
            images (ndarray): Images to be processed.
            mask (bool, optional): Perform masking of bad pixels (set those values to 0).
                Defaults to True.
            gap_pixels (bool, optional): Add gap pixels between detector chips. Defaults to True.
            geometry (bool, optional): Apply detector geometry corrections. Defaults to True.

        Returns:
            ndarray: Gain values of pixels.
        """
        if images.dtype != np.uint16:
            raise TypeError(f"Expected image type {np.uint16}, provided data type {images.dtype}.")

        gains = images >> 14
        gains = self.process(
            gains, conversion=False, mask=mask, gap_pixels=gap_pixels, geometry=geometry
        )

        return gains

    def get_saturated_pixels(self, images, *, mask=True, gap_pixels=True, geometry=True):
        """Return coordinates of saturated pixels, based on mask, gap_pixel and geometry flags.

        Args:
            images (ndarray): Images to be processed.
            mask (bool, optional): Perform masking of bad pixels (set those values to 0).
                Defaults to True.
            gap_pixels (bool, optional): Add gap pixels between detector chips. Defaults to True.
            geometry (bool, optional): Apply detector geometry corrections. Defaults to True.

        Returns:
            tuple: Indices of saturated pixels.
        """
        if images.dtype != np.uint16:
            raise TypeError(f"Expected image type {np.uint16}, provided data type {images.dtype}.")

        if self.highgain:
            saturated_value = 0b0011111111111111  # = 16383
        else:
            saturated_value = 0b1100000000000000  # = 49152

        saturated_pixels = images == saturated_value
        saturated_pixels = self.process(
            saturated_pixels, conversion=False, mask=mask, gap_pixels=gap_pixels, geometry=geometry
        )

        saturated_pixels_coordinates = np.nonzero(saturated_pixels)

        return saturated_pixels_coordinates


@njit(cache=True)
def _adc_to_energy(res, image, gain, pedestal, mask, factor, gap_pixels):
    num, size_y, size_x = image.shape
    # TODO: remove after issue is fixed: https://github.com/PyCQA/pylint/issues/2910
    for i1 in prange(num):  # pylint: disable=not-an-iterable
        for i2 in range(size_y):
            for i3 in range(size_x):
                if mask is not None and not mask[i2, i3]:
                    continue

                ri2 = i2 + i2 // CHIP_SIZE_Y * CHIP_GAP_Y * gap_pixels
                ri3 = i3 + i3 // CHIP_SIZE_X * CHIP_GAP_X * gap_pixels

                if gain is None or pedestal is None:
                    res[i1, ri2, ri3] = image[i1, i2, i3]
                else:
                    gm = np.right_shift(image[i1, i2, i3], 14)
                    val = np.float32(image[i1, i2, i3] & 0x3FFF)
                    tmp_res = (val - pedestal[gm, i2, i3]) * gain[gm, i2, i3]

                    if factor is None:
                        res[i1, ri2, ri3] = tmp_res
                    else:
                        res[i1, ri2, ri3] = round(tmp_res)


@njit(cache=True, parallel=True)
def _adc_to_energy_parallel(res, image, gain, pedestal, mask, factor, gap_pixels):
    num, size_y, size_x = image.shape
    # TODO: remove after issue is fixed: https://github.com/PyCQA/pylint/issues/2910
    for i1 in prange(num):  # pylint: disable=not-an-iterable
        for i2 in range(size_y):
            for i3 in range(size_x):
                if mask is not None and not mask[i2, i3]:
                    continue

                ri2 = i2 + i2 // CHIP_SIZE_Y * CHIP_GAP_Y * gap_pixels
                ri3 = i3 + i3 // CHIP_SIZE_X * CHIP_GAP_X * gap_pixels

                if gain is None or pedestal is None:
                    res[i1, ri2, ri3] = image[i1, i2, i3]
                else:
                    gm = np.right_shift(image[i1, i2, i3], 14)
                    val = np.float32(image[i1, i2, i3] & 0x3FFF)
                    tmp_res = (val - pedestal[gm, i2, i3]) * gain[gm, i2, i3]

                    if factor is None:
                        res[i1, ri2, ri3] = tmp_res
                    else:
                        res[i1, ri2, ri3] = round(tmp_res)


@njit(cache=True)
def _reshape_stripsel(res, image):
    num = image.shape[0]
    # TODO: remove after issue is fixed: https://github.com/PyCQA/pylint/issues/2910
    for ind in prange(num):  # pylint: disable=not-an-iterable
        # first we fill the normal pixels, the gap ones will be overwritten later
        for yin in range(252):
            for xin in range(1024):
                ichip = xin // 256
                xout = (ichip * 774) + (xin % 256) * 3 + yin % 3
                # 774 is the chip period, 256*3+6
                yout = yin // 3
                res[ind, yout, xout] = image[ind, yin, xin]

        # now the gap pixels
        for igap in range(3):
            for yin in range(252):
                yout = (yin // 6) * 2

                # if we want a proper normalization (the area of those pixels is double,
                # so they see 2x the signal)

                # first the left side of gap
                xin = igap * 256 + 255
                xout = igap * 774 + 765 + yin % 6
                res[ind, yout, xout] = image[ind, yin, xin] / 2
                res[ind, yout + 1, xout] = image[ind, yin, xin] / 2

                # then the right side is mirrored
                xin = igap * 256 + 255 + 1
                xout = igap * 774 + 765 + 11 - yin % 6
                res[ind, yout, xout] = image[ind, yin, xin] / 2
                res[ind, yout + 1, xout] = image[ind, yin, xin] / 2


@njit(cache=True, parallel=True)
def _reshape_stripsel_parallel(res, image):
    num = image.shape[0]
    # TODO: remove after issue is fixed: https://github.com/PyCQA/pylint/issues/2910
    for ind in prange(num):  # pylint: disable=not-an-iterable
        # first we fill the normal pixels, the gap ones will be overwritten later
        for yin in range(252):
            for xin in range(1024):
                ichip = xin // 256
                xout = (ichip * 774) + (xin % 256) * 3 + yin % 3
                # 774 is the chip period, 256*3+6
                yout = yin // 3
                res[ind, yout, xout] = image[ind, yin, xin]

        # now the gap pixels
        for igap in range(3):
            for yin in range(252):
                yout = (yin // 6) * 2

                # if we want a proper normalization (the area of those pixels is double,
                # so they see 2x the signal)

                # first the left side of gap
                xin = igap * 256 + 255
                xout = igap * 774 + 765 + yin % 6
                res[ind, yout, xout] = image[ind, yin, xin] / 2
                res[ind, yout + 1, xout] = image[ind, yin, xin] / 2

                # then the right side is mirrored
                xin = igap * 256 + 255 + 1
                xout = igap * 774 + 765 + 11 - yin % 6
                res[ind, yout, xout] = image[ind, yin, xin] / 2
                res[ind, yout + 1, xout] = image[ind, yin, xin] / 2


@njit(cache=True)
def _inplace_interp_dp(res):
    for i1 in prange(res.shape[0]):
        # corner quad pixels
        for ri2 in (255, 257):
            for ri3 in (255, 257, 513, 515, 771, 773):
                shift_y = 0 if ri2 == 255 else 1
                shift_x = 0 if ri3 == 255 or ri3 == 513 or ri3 == 771 else 1

                shared_val = res[i1, ri2 + shift_y, ri3 + shift_x] / 4
                res[i1, ri2, ri3] = shared_val
                res[i1, ri2 + 1, ri3] = shared_val
                res[i1, ri2, ri3 + 1] = shared_val
                res[i1, ri2 + 1, ri3 + 1] = shared_val

        # rows of double pixels
        ri2 = 255
        for x_start, x_end in ((0, 255), (259, 513), (517, 771), (775, 1030)):
            for ri3 in range(x_start, x_end):
                v1 = res[i1, ri2 - 1, ri3]
                v2 = res[i1, ri2, ri3]
                v3 = res[i1, ri2 + 3, ri3]
                v4 = res[i1, ri2 + 4, ri3]

                if v1 == 0 and v3 == 0:
                    shared_val = v2 / 2
                    res[i1, ri2, ri3] = shared_val
                    res[i1, ri2 + 1, ri3] = shared_val
                else:
                    res[i1, ri2, ri3] *= (4 * v1 + v3) / (6 * v1 + 3 * v3)
                    res[i1, ri2 + 1, ri3] = v2 - res[i1, ri2, ri3]

                if v4 == 0 and v2 == 0:
                    shared_val = v3 / 2
                    res[i1, ri2 + 3, ri3] = shared_val
                    res[i1, ri2 + 2, ri3] = shared_val
                else:
                    res[i1, ri2 + 3, ri3] *= (4 * v4 + v2) / (6 * v4 + 3 * v2)
                    res[i1, ri2 + 2, ri3] = v3 - res[i1, ri2 + 3, ri3]

        # columns of double pixels
        for ri3 in (255, 513, 771):
            for y_start, y_end in ((0, 255), (259, 514)):
                for ri2 in range(y_start, y_end):
                    v1 = res[i1, ri2, ri3 - 1]
                    v2 = res[i1, ri2, ri3]
                    v3 = res[i1, ri2, ri3 + 3]
                    v4 = res[i1, ri2, ri3 + 4]

                    if v1 == 0 and v3 == 0:
                        shared_val = v2 / 2
                        res[i1, ri2, ri3] = shared_val
                        res[i1, ri2, ri3 + 1] = shared_val
                    else:
                        res[i1, ri2, ri3] *= (4 * v1 + v3) / (6 * v1 + 3 * v3)
                        res[i1, ri2, ri3 + 1] = v2 - res[i1, ri2, ri3]

                    if v4 == 0 and v2 == 0:
                        shared_val = v3 / 2
                        res[i1, ri2, ri3 + 3] = shared_val
                        res[i1, ri2, ri3 + 2] = shared_val
                    else:
                        res[i1, ri2, ri3 + 3] *= (4 * v4 + v2) / (6 * v4 + 3 * v2)
                        res[i1, ri2, ri3 + 2] = v3 - res[i1, ri2, ri3 + 3]


@njit(cache=True, parallel=True)
def _inplace_interp_dp_parallel(res):
    for i1 in prange(res.shape[0]):
        # corner quad pixels
        for ri2 in (255, 257):
            for ri3 in (255, 257, 513, 515, 771, 773):
                shift_y = 0 if ri2 == 255 else 1
                shift_x = 0 if ri3 == 255 or ri3 == 513 or ri3 == 771 else 1

                shared_val = res[i1, ri2 + shift_y, ri3 + shift_x] / 4
                res[i1, ri2, ri3] = shared_val
                res[i1, ri2 + 1, ri3] = shared_val
                res[i1, ri2, ri3 + 1] = shared_val
                res[i1, ri2 + 1, ri3 + 1] = shared_val

        # rows of double pixels
        ri2 = 255
        for x_start, x_end in ((0, 255), (259, 513), (517, 771), (775, 1030)):
            for ri3 in range(x_start, x_end):
                v1 = res[i1, ri2 - 1, ri3]
                v2 = res[i1, ri2, ri3]
                v3 = res[i1, ri2 + 3, ri3]
                v4 = res[i1, ri2 + 4, ri3]

                if v1 == 0 and v3 == 0:
                    shared_val = v2 / 2
                    res[i1, ri2, ri3] = shared_val
                    res[i1, ri2 + 1, ri3] = shared_val
                else:
                    res[i1, ri2, ri3] *= (4 * v1 + v3) / (6 * v1 + 3 * v3)
                    res[i1, ri2 + 1, ri3] = v2 - res[i1, ri2, ri3]

                if v4 == 0 and v2 == 0:
                    shared_val = v3 / 2
                    res[i1, ri2 + 3, ri3] = shared_val
                    res[i1, ri2 + 2, ri3] = shared_val
                else:
                    res[i1, ri2 + 3, ri3] *= (4 * v4 + v2) / (6 * v4 + 3 * v2)
                    res[i1, ri2 + 2, ri3] = v3 - res[i1, ri2 + 3, ri3]

        # columns of double pixels
        for ri3 in (255, 513, 771):
            for y_start, y_end in ((0, 255), (259, 514)):
                for ri2 in range(y_start, y_end):
                    v1 = res[i1, ri2, ri3 - 1]
                    v2 = res[i1, ri2, ri3]
                    v3 = res[i1, ri2, ri3 + 3]
                    v4 = res[i1, ri2, ri3 + 4]

                    if v1 == 0 and v3 == 0:
                        shared_val = v2 / 2
                        res[i1, ri2, ri3] = shared_val
                        res[i1, ri2, ri3 + 1] = shared_val
                    else:
                        res[i1, ri2, ri3] *= (4 * v1 + v3) / (6 * v1 + 3 * v3)
                        res[i1, ri2, ri3 + 1] = v2 - res[i1, ri2, ri3]

                    if v4 == 0 and v2 == 0:
                        shared_val = v3 / 2
                        res[i1, ri2, ri3 + 3] = shared_val
                        res[i1, ri2, ri3 + 2] = shared_val
                    else:
                        res[i1, ri2, ri3 + 3] *= (4 * v4 + v2) / (6 * v4 + 3 * v2)
                        res[i1, ri2, ri3 + 2] = v3 - res[i1, ri2, ri3 + 3]


@njit(cache=True)
def _inplace_mask_dp(res):
    for row in (256, 257):
        res[:, row, :1030] = True
    for col in (256, 257, 514, 515, 772, 773):
        res[:, :514, col] = True


# Numba functions
_adc_to_energy_jit = {
    True: _adc_to_energy_parallel,
    False: _adc_to_energy,
}

_reshape_stripsel_jit = {
    True: _reshape_stripsel_parallel,
    False: _reshape_stripsel,
}

_inplace_interp_dp_jit = {
    True: _inplace_interp_dp_parallel,
    False: _inplace_interp_dp,
}
