import re
import warnings
from collections import namedtuple
from functools import lru_cache

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

MODULE_FULL_SIZE_X = MODULE_SIZE_X + (CHIP_NUM_X - 1) * CHIP_GAP_X
MODULE_FULL_SIZE_Y = MODULE_SIZE_Y + (CHIP_NUM_Y - 1) * CHIP_GAP_Y

# 256 not divisible by 3, so we round up to 86
# the last 4 pixels can be omitted, so the final height is (256 - 4) / 3 = 84
# 18 since we have 6 more pixels in H per gap
STRIPSEL_SIZE_X = 1024 * 3 + 18  # = 3090
STRIPSEL_SIZE_Y = 84


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

        g_shape = (4, *self._shape_in_full)
        g_shape_v2 = (6, *self._shape_in_full)
        if value.shape != g_shape and value.shape != g_shape_v2:
            raise ValueError(
                f"Expected gain shape {g_shape} or {g_shape_v2}, provided gain shape {value.shape}."
            )

        # convert _gain values to float32
        self._gain = value.astype(np.float32, copy=False)
        self._update_g_all()

    def _update_g_all(self):
        if self.factor is None:
            _g = 1 / self._gain
        else:
            # self.factor is one number and self.gain is a large array, so this order of division
            # will avoid double broadcasting
            _g = 1 / self.factor / self._gain

        self._g_all[False] = np.stack((_g[0], _g[1], _g[2], _g[2]))

        g_shape = (4, *self._shape_in_full)  # g_shape_v2 = (6, *self._shape_in_full)
        if _g.shape == g_shape:
            self._g_all[True] = np.stack((_g[3], _g[3], _g[3], _g[3]))
        else:  # _g.shape == g_shape_v2
            self._g_all[True] = np.stack((_g[3], _g[4], _g[5], _g[5]))

    @property
    def _g(self):
        return self._g_all[self.highgain]

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

        p_shape = (4, *self._shape_in_full)
        p_shape_v2 = (3, *self._shape_in_full)
        if value.shape != p_shape and value.shape != p_shape_v2:
            raise ValueError(
                f"Expected pedestal shapes {p_shape} or {p_shape_v2}, provided pedestal shape {value.shape}."
            )

        # convert _pedestal values to float32
        self._pedestal = value.astype(np.float32, copy=False)

        _p = self._pedestal

        self._p_all[False] = np.stack((_p[0], _p[1], _p[2], _p[2]))

        if _p.shape == p_shape:
            self._p_all[True] = np.stack((_p[3], _p[3], _p[3], _p[3]))
        else:  # _p.shape == p_shape_v2
            self._p_all[True] = np.stack((_p[0], _p[1], _p[2], _p[2]))

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
            self.get_pixel_mask.cache_clear()
            return

        if value.ndim != 2:
            raise ValueError(
                f"Expected pixel_mask dimensions 2, provided pixel_mask dimensions {value.ndim}."
            )

        pm_shape = self._shape_in_full
        if value.shape != pm_shape:
            raise ValueError(
                f"Expected pixel_mask shape {pm_shape}, provided pixel_mask shape {value.shape}."
            )

        self._pixel_mask = value
        self.get_pixel_mask.cache_clear()

        # original mask -> self._mask_all[False]
        mask = np.invert(value.astype(bool, copy=True))
        self._mask_all[False] = mask.copy()

        # original + double pixels mask -> self._mask_all[True]
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

    def _mask(self, double_pixels):
        if double_pixels in ("keep", "interp"):
            return self._mask_all[False]
        elif double_pixels == "mask":
            return self._mask_all[True]
        else:
            raise ValueError("'double_pixels' can only be 'keep', 'mask', or 'interp'")

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
    def module_map(self):
        """Current module map.
        """
        return self._module_map

    @module_map.setter
    def module_map(self, value):
        if np.array_equal(self._module_map, value):
            return

        n_modules = self.detector.n_modules
        if value is None:
            # support legacy data by emulating 'all modules are present'
            value = np.arange(n_modules)

        if len(value) != n_modules:
            raise ValueError(
                f"Expected module_map length {n_modules}, provided module_map length {len(value)}."
            )

        if min(value) < -1 or n_modules <= max(value):
            raise ValueError(f"Valid module_map values are integers between -1 and {n_modules-1}.")

        self._module_map = value
        self.get_pixel_mask.cache_clear()

    def _get_shape_in_n_modules(self, n):
        if self.detector_name == "JF02T09V01":  # a special case
            shape_y, shape_x = MODULE_SIZE_Y, MODULE_SIZE_X * n
        else:
            shape_y, shape_x = MODULE_SIZE_Y * n, MODULE_SIZE_X

        return shape_y, shape_x

    @property
    def _shape_in_full(self):
        return self._get_shape_in_n_modules(self.detector.n_modules)

    @property
    def _shape_in(self):
        return self._get_shape_in_n_modules(self._n_active_modules)

    @property
    def _n_active_modules(self):
        return np.sum(self.module_map != -1)

    def _get_shape_out(self, gap_pixels, geometry):
        """Return the image shape of a detector without a full-detector rotation step.

        Args:
            gap_pixels (bool, optional): Add gap pixels between detector chips. Defaults to True.
            geometry (bool, optional): Apply detector geometry corrections. Defaults to True.

        Returns:
            tuple: Height and width of a resulting image.
        """
        if self.is_stripsel():
            return self._get_stripsel_shape_out(geometry=geometry)

        if geometry:
            module_size_x = MODULE_FULL_SIZE_X if gap_pixels else MODULE_SIZE_X
            module_size_y = MODULE_FULL_SIZE_Y if gap_pixels else MODULE_SIZE_Y
            origin_y = self.detector_geometry.origin_y
            origin_x = self.detector_geometry.origin_x

            oy_end_all = []
            ox_end_all = []
            for i, (oy, ox) in enumerate(zip(origin_y, origin_x)):
                if len(self.detector_geometry.mod_rot90) == 1:
                    mod_rot90 = self.detector_geometry.mod_rot90[0]
                else:
                    mod_rot90 = self.detector_geometry.mod_rot90[i]

                if mod_rot90 == 1:
                    oy -= module_size_x
                elif mod_rot90 == 2:
                    oy -= module_size_y
                    ox -= module_size_x
                elif mod_rot90 == 3:
                    ox -= module_size_y

                if mod_rot90 % 2 == 0:
                    oy_end = oy + module_size_y
                    ox_end = ox + module_size_x
                else:
                    oy_end = oy + module_size_x
                    ox_end = ox + module_size_y

                oy_end_all.append(oy_end)
                ox_end_all.append(ox_end)

            shape_y = max(oy_end_all)
            shape_x = max(ox_end_all)

        else:
            shape_y, shape_x = self._shape_in
            if gap_pixels:
                if self.detector_name == "JF02T09V01":
                    shape_y += (CHIP_NUM_Y - 1) * CHIP_GAP_Y
                    shape_x += (CHIP_NUM_X - 1) * CHIP_GAP_X * self._n_active_modules
                else:
                    shape_y += (CHIP_NUM_Y - 1) * CHIP_GAP_Y * self._n_active_modules
                    shape_x += (CHIP_NUM_X - 1) * CHIP_GAP_X

        return shape_y, shape_x

    def get_shape_out(self, *, gap_pixels=True, geometry=True):
        """Return the final image shape of a detector.

        Args:
            gap_pixels (bool, optional): Add gap pixels between detector chips. Defaults to True.
            geometry (bool, optional): Apply detector geometry corrections. Defaults to True.

        Returns:
            tuple: Height and width of a resulting image.
        """
        shape_y, shape_x = self._get_shape_out(gap_pixels, geometry)
        if geometry and self.detector_geometry.det_rot90 % 2:
            shape_y, shape_x = shape_x, shape_y

        return shape_y, shape_x

    def _get_stripsel_shape_out(self, geometry):
        if geometry:
            shape_x = max(self.detector_geometry.origin_x) + STRIPSEL_SIZE_X
            shape_y = max(self.detector_geometry.origin_y) + STRIPSEL_SIZE_Y
        else:
            shape_y, shape_x = self._shape_in

        return shape_y, shape_x

    def get_dtype_out(self, dtype_in, *, conversion=True):
        """Return resulting image dtype of a detector.

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

    def _get_module_coords(self, m, i, geometry, gap_pixels):
        module_size_y = MODULE_FULL_SIZE_Y if gap_pixels else MODULE_SIZE_Y
        module_size_x = MODULE_FULL_SIZE_X if gap_pixels else MODULE_SIZE_X

        if geometry:  # gap pixels are already accounted for in module geometry coordinates
            oy = self.detector_geometry.origin_y[i]
            ox = self.detector_geometry.origin_x[i]

            det_rot90 = self.detector_geometry.det_rot90
            if det_rot90:
                shape_y, shape_x = self._get_shape_out(gap_pixels, geometry)
                if det_rot90 == 1:  # (x, y) -> (y, -x)
                    ox, oy = oy, shape_x - ox
                elif det_rot90 == 2:  # (x, y) -> (-x, -y)
                    ox, oy = shape_x - ox, shape_y - oy
                elif det_rot90 == 3:  # (x, y) -> (-y, x)
                    ox, oy = shape_y - oy, ox

            if len(self.detector_geometry.mod_rot90) == 1:
                mod_rot90 = self.detector_geometry.mod_rot90[0]
            else:
                mod_rot90 = self.detector_geometry.mod_rot90[i]

            mod_rot90 = (mod_rot90 + det_rot90) % 4

            if mod_rot90 == 1:
                oy -= module_size_x
            elif mod_rot90 == 2:
                oy -= module_size_y
                ox -= module_size_x
            elif mod_rot90 == 3:
                ox -= module_size_y

        else:
            mod_rot90 = 0
            if self.detector_name == "JF02T09V01":
                oy = 0
                ox = m * module_size_x
            else:
                oy = m * module_size_y
                ox = 0

        if mod_rot90 % 2 == 0:
            oy_end = oy + module_size_y
            ox_end = ox + module_size_x
        else:
            oy_end = oy + module_size_x
            ox_end = ox + module_size_y

        return oy, oy_end, ox, ox_end, mod_rot90

    def _get_module_slice(self, data, m):
        if self.detector_name == "JF02T09V01":
            out = data[..., :, m * MODULE_SIZE_X : (m + 1) * MODULE_SIZE_X]
        else:
            out = data[..., m * MODULE_SIZE_Y : (m + 1) * MODULE_SIZE_Y, :]

        return out

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

        # handle 2D data by adding a singleton dimension, making it 3D
        if images.ndim == 2:
            is_2darray = True
            images = images[np.newaxis]
        else:
            is_2darray = False

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

        if self.is_stripsel() and double_pixels != "keep":
            warnings.warn(
                "Handling double pixels has no effect on stripsel detectors", RuntimeWarning
            )
            double_pixels = "keep"

        out_shape = self.get_shape_out(gap_pixels=gap_pixels, geometry=geometry)
        full_out_shape = (images.shape[0], *out_shape)
        if out is None:
            out_dtype = self.get_dtype_out(images.dtype, conversion=conversion)
            out = np.zeros(full_out_shape, dtype=out_dtype)
        else:
            if out.shape != full_out_shape:
                raise ValueError(f"Expected 'out' shape: {full_out_shape}, provided: {out.shape}")

        self._process(out, images, conversion, mask, gap_pixels, double_pixels, geometry, parallel)

        # remove the singleton dimension if input was 2D
        if is_2darray:
            out = out[0]

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
        if parallel:
            adc_to_energy = _adc_to_energy_parallel_jit
            reshape_stripsel = _reshape_stripsel_parallel_jit
            inplace_interp_double_pixels = _inplace_interp_double_pixels_parallel_jit
        else:
            adc_to_energy =_adc_to_energy_jit
            reshape_stripsel = _reshape_stripsel_jit
            inplace_interp_double_pixels = _inplace_interp_double_pixels_jit

        factor = self.factor
        _mask = self._mask(double_pixels)

        for i, m in enumerate(self.module_map):
            if m == -1:
                continue

            oy, oy_end, ox, ox_end, rot90 = self._get_module_coords(m, i, geometry, gap_pixels)
            mod = self._get_module_slice(images, m)

            if mask:
                mod_mask = self._get_module_slice(_mask, i)
            else:
                mod_mask = None

            if conversion:
                mod_g = self._get_module_slice(self._g, i)
                mod_p = self._get_module_slice(self._p, i)
            else:
                mod_g = None
                mod_p = None

            mod_res = res[:, oy:oy_end, ox:ox_end]
            if self.is_stripsel() and geometry:
                mod_tmp_shape = (images.shape[0], MODULE_SIZE_Y, MODULE_SIZE_X)
                mod_tmp_dtype = self.get_dtype_out(images.dtype, conversion=conversion)
                mod_tmp = np.zeros(shape=mod_tmp_shape, dtype=mod_tmp_dtype)

                adc_to_energy(mod_tmp, mod, mod_g, mod_p, mod_mask, factor, gap_pixels)
                reshape_stripsel(mod_res, mod_tmp)
            else:
                mod_res = np.rot90(mod_res, k=-rot90, axes=(1, 2))
                adc_to_energy(mod_res, mod, mod_g, mod_p, mod_mask, factor, gap_pixels)
                if double_pixels == "interp":
                    inplace_interp_double_pixels(mod_res)

    @lru_cache(maxsize=8)
    def get_pixel_mask(self, *, gap_pixels=True, double_pixels="keep", geometry=True):
        """Return pixel mask.

        Args:
            gap_pixels (bool, optional): Add gap pixels between detector chips. Defaults to True.
            double_pixels (str, optional): A method to handle double pixels in-between ASICs. Can be
                "keep", "mask", or "interp". Defaults to "keep".
            geometry (bool, optional): Apply detector geometry corrections. Defaults to True.

        Returns:
            ndarray: Resulting pixel mask, where True values correspond to valid pixels.
        """
        if double_pixels not in ("keep", "mask", "interp"):
            raise ValueError("'double_pixels' can only be 'keep', 'mask', or 'interp'")

        if double_pixels == "interp" and not gap_pixels:
            raise RuntimeError("Double pixel interpolation requires 'gap_pixels' to be True.")

        if self.is_stripsel() and gap_pixels:
            warnings.warn("'gap_pixels' flag has no effect on stripsel detectors", RuntimeWarning)
            gap_pixels = False

        if self.is_stripsel() and double_pixels != "keep":
            warnings.warn(
                "Handling double pixels has no effect on stripsel detectors", RuntimeWarning
            )
            double_pixels = "keep"

        if self._pixel_mask is None:
            return None

        _mask = self._mask(double_pixels)[np.newaxis]

        res_shape = self.get_shape_out(gap_pixels=gap_pixels, geometry=geometry)
        res = np.zeros((1, *res_shape), dtype=bool)

        for i, m in enumerate(self.module_map):
            if m == -1:
                continue

            oy, oy_end, ox, ox_end, rot90 = self._get_module_coords(m, i, geometry, gap_pixels)

            mod = self._get_module_slice(_mask, i)
            mod_res = res[:, oy:oy_end, ox:ox_end]
            if self.is_stripsel() and geometry:
                _reshape_stripsel_jit(mod_res, mod)
            else:
                mod_res = np.rot90(mod_res, k=-rot90, axes=(1, 2))
                # this will just copy data to the correct place
                _adc_to_energy_jit(mod_res, mod, None, None, None, None, gap_pixels)
                if double_pixels == "interp":
                    _inplace_mask_double_pixels_jit(mod_res)

        res = res[0]

        return res

    def get_pixel_coordinates(self):
        """Return arrays (x, y, z) of final coordinates for pixels in raw data.

        The shape of the result is the same as the raw input data (equivalently, gap_pixels=False,
        geometry=False), but the coordinates represent pixel positions after gap_pixel and geometry
        corrections (gap_pixels=True, double_pixels="keep", geometry=True).
        """
        if self.detector_geometry.is_stripsel is True:
            raise RuntimeError("Stripsel detectors are currently unsupported.")

        if any(self.detector_geometry.mod_rot90):
            raise RuntimeError("Detectors with rotated modules are currently unsupported.")

        if self.detector_geometry.det_rot90:
            raise RuntimeError("Detectors with rotated geometry are currently unsupported.")

        _y = np.arange(MODULE_SIZE_Y, dtype=np.float64)
        for n in range(1, CHIP_NUM_Y):
            _y[n * CHIP_SIZE_Y :] += CHIP_GAP_Y
            # shift for double pixels
            _y[n * CHIP_SIZE_Y - 1] += 0.5
            _y[n * CHIP_SIZE_Y] -= 0.5

        _x = np.arange(MODULE_SIZE_X, dtype=np.float64)
        for n in range(1, CHIP_NUM_X):
            _x[n * CHIP_SIZE_X :] += CHIP_GAP_X
            # shift for double pixels
            _x[n * CHIP_SIZE_X - 1] += 0.5
            _x[n * CHIP_SIZE_X] -= 0.5

        y_mod_grid, x_mod_grid = np.meshgrid(_y, _x, indexing="ij")

        shape_out = self.get_shape_out(gap_pixels=False, geometry=False)
        x_coord = np.zeros(shape=shape_out)
        y_coord = np.zeros(shape=shape_out)
        z_coord = np.zeros(shape=shape_out)

        for i, m in enumerate(self.module_map):
            if m == -1:
                continue

            oy, _, ox, _, _ = self._get_module_coords(m, i, geometry=True, gap_pixels=True)
            y_mod = self._get_module_slice(y_coord, i)
            x_mod = self._get_module_slice(x_coord, i)
            y_mod[:] = y_mod_grid + oy
            x_mod[:] = x_mod_grid + ox

        # apply final detector rotation
        if self.detector_geometry.det_rot90 == 1:  # (x, y) -> (y, -x)
            x_coord, y_coord = y_coord, np.max(x_coord) - x_coord
        elif self.detector_geometry.det_rot90 == 2:  # (x, y) -> (-x, -y)
            x_coord, y_coord = np.max(x_coord) - x_coord, np.max(y_coord) - y_coord
        elif self.detector_geometry.det_rot90 == 3:  # (x, y) -> (-y, x)
            x_coord, y_coord = np.max(y_coord) - y_coord, x_coord

        return x_coord, y_coord, z_coord

    def get_gains(self, images, *, mask=True, gap_pixels=True, double_pixels="keep", geometry=True):
        """Return gain values of images.

        Args:
            images (ndarray): Images to be processed.
            mask (bool, optional): Perform masking of bad pixels (set those values to 0).
                Defaults to True.
            gap_pixels (bool, optional): Add gap pixels between detector chips. Defaults to True.
            double_pixels (str, optional): A method to handle double pixels in-between ASICs. Can be
                "keep", "mask", or "interp" (resolves into "keep"). Defaults to "keep".
            geometry (bool, optional): Apply detector geometry corrections. Defaults to True.

        Returns:
            ndarray: Gain values of pixels.
        """
        if images.dtype != np.uint16:
            raise TypeError(f"Expected image type {np.uint16}, provided data type {images.dtype}.")

        if double_pixels == "interp":
            # interpolation makes sense only for final keV values
            double_pixels = "keep"

        gains = images >> 14
        gains = self.process(
            gains,
            conversion=False,
            mask=mask,
            gap_pixels=gap_pixels,
            double_pixels=double_pixels,
            geometry=geometry,
        )

        return gains

    def get_saturated_pixels(
        self, images, *, mask=True, gap_pixels=True, double_pixels="keep", geometry=True
    ):
        """Return coordinates of saturated pixels.

        Args:
            images (ndarray): Images to be processed.
            mask (bool, optional): Perform masking of bad pixels (set those values to 0).
                Defaults to True.
            gap_pixels (bool, optional): Add gap pixels between detector chips. Defaults to True.
            double_pixels (str, optional): A method to handle double pixels in-between ASICs. Can be
                "keep", "mask", or "interp" (resolves into "keep"). Defaults to "keep".
            geometry (bool, optional): Apply detector geometry corrections. Defaults to True.

        Returns:
            tuple: Indices of saturated pixels.
        """
        if images.dtype != np.uint16:
            raise TypeError(f"Expected image type {np.uint16}, provided data type {images.dtype}.")

        if double_pixels == "interp":
            # interpolation makes sense only for final keV values
            double_pixels = "keep"

        if self.highgain:
            saturated_value = 0b0011111111111111  # = 16383
        else:
            saturated_value = 0b1100000000000000  # = 49152

        saturated_pixels = images == saturated_value
        saturated_pixels = self.process(
            saturated_pixels,
            conversion=False,
            mask=mask,
            gap_pixels=gap_pixels,
            double_pixels=double_pixels,
            geometry=geometry,
        )

        saturated_pixels_coordinates = np.nonzero(saturated_pixels)

        return saturated_pixels_coordinates


@njit(cache=True)
def _adc_to_energy_jit(res, image, gain, pedestal, mask, factor, gap_pixels):
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
def _adc_to_energy_parallel_jit(res, image, gain, pedestal, mask, factor, gap_pixels):
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
def _reshape_stripsel_jit(res, image):
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
def _reshape_stripsel_parallel_jit(res, image):
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
def _inplace_interp_double_pixels_jit(res):
    for i1 in prange(res.shape[0]):  # pylint: disable=not-an-iterable
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

                div = 6 * v1 + 3 * v3
                if div == 0:
                    shared_val = v2 / 2
                    res[i1, ri2, ri3] = shared_val
                    res[i1, ri2 + 1, ri3] = shared_val
                else:
                    res[i1, ri2, ri3] *= (4 * v1 + v3) / div
                    res[i1, ri2 + 1, ri3] = v2 - res[i1, ri2, ri3]

                div = 6 * v4 + 3 * v2
                if div == 0:
                    shared_val = v3 / 2
                    res[i1, ri2 + 3, ri3] = shared_val
                    res[i1, ri2 + 2, ri3] = shared_val
                else:
                    res[i1, ri2 + 3, ri3] *= (4 * v4 + v2) / div
                    res[i1, ri2 + 2, ri3] = v3 - res[i1, ri2 + 3, ri3]

        # columns of double pixels
        for ri3 in (255, 513, 771):
            for y_start, y_end in ((0, 255), (259, 514)):
                for ri2 in range(y_start, y_end):
                    v1 = res[i1, ri2, ri3 - 1]
                    v2 = res[i1, ri2, ri3]
                    v3 = res[i1, ri2, ri3 + 3]
                    v4 = res[i1, ri2, ri3 + 4]

                    div = 6 * v1 + 3 * v3
                    if div == 0:
                        shared_val = v2 / 2
                        res[i1, ri2, ri3] = shared_val
                        res[i1, ri2, ri3 + 1] = shared_val
                    else:
                        res[i1, ri2, ri3] *= (4 * v1 + v3) / div
                        res[i1, ri2, ri3 + 1] = v2 - res[i1, ri2, ri3]

                    div = 6 * v4 + 3 * v2
                    if div == 0:
                        shared_val = v3 / 2
                        res[i1, ri2, ri3 + 3] = shared_val
                        res[i1, ri2, ri3 + 2] = shared_val
                    else:
                        res[i1, ri2, ri3 + 3] *= (4 * v4 + v2) / div
                        res[i1, ri2, ri3 + 2] = v3 - res[i1, ri2, ri3 + 3]


@njit(cache=True, parallel=True)
def _inplace_interp_double_pixels_parallel_jit(res):
    for i1 in prange(res.shape[0]):  # pylint: disable=not-an-iterable
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

                div = 6 * v1 + 3 * v3
                if div == 0:
                    shared_val = v2 / 2
                    res[i1, ri2, ri3] = shared_val
                    res[i1, ri2 + 1, ri3] = shared_val
                else:
                    res[i1, ri2, ri3] *= (4 * v1 + v3) / div
                    res[i1, ri2 + 1, ri3] = v2 - res[i1, ri2, ri3]

                div = 6 * v4 + 3 * v2
                if div == 0:
                    shared_val = v3 / 2
                    res[i1, ri2 + 3, ri3] = shared_val
                    res[i1, ri2 + 2, ri3] = shared_val
                else:
                    res[i1, ri2 + 3, ri3] *= (4 * v4 + v2) / div
                    res[i1, ri2 + 2, ri3] = v3 - res[i1, ri2 + 3, ri3]

        # columns of double pixels
        for ri3 in (255, 513, 771):
            for y_start, y_end in ((0, 255), (259, 514)):
                for ri2 in range(y_start, y_end):
                    v1 = res[i1, ri2, ri3 - 1]
                    v2 = res[i1, ri2, ri3]
                    v3 = res[i1, ri2, ri3 + 3]
                    v4 = res[i1, ri2, ri3 + 4]

                    div = 6 * v1 + 3 * v3
                    if div == 0:
                        shared_val = v2 / 2
                        res[i1, ri2, ri3] = shared_val
                        res[i1, ri2, ri3 + 1] = shared_val
                    else:
                        res[i1, ri2, ri3] *= (4 * v1 + v3) / div
                        res[i1, ri2, ri3 + 1] = v2 - res[i1, ri2, ri3]

                    div = 6 * v4 + 3 * v2
                    if div == 0:
                        shared_val = v3 / 2
                        res[i1, ri2, ri3 + 3] = shared_val
                        res[i1, ri2, ri3 + 2] = shared_val
                    else:
                        res[i1, ri2, ri3 + 3] *= (4 * v4 + v2) / div
                        res[i1, ri2, ri3 + 2] = v3 - res[i1, ri2, ri3 + 3]


@njit(cache=True)
def _inplace_mask_double_pixels_jit(res):
    # gap_pixels is always True here
    for row in (256,):
        vals = res[:, row - 1, :1030] & res[:, row + 2, :1030]
        res[:, row, :1030] = vals
        res[:, row + 1, :1030] = vals

    for col in (256, 514, 772):
        vals = res[:, :514, col - 1] & res[:, :514, col + 2]
        res[:, :514, col] = vals
        res[:, :514, col + 1] = vals
