import ctypes
import os
import re
from collections import namedtuple

import h5py
import numpy as np
from numba import jit

from .geometry import modules_orig

try:
    import mkl
except ImportError:
    pass
else:
    mkl.set_num_threads(1)  # pylint: disable=no-member

NUM_GAINS = 3
HIGHGAIN_ORDER = {True: (3, 1, 2), False: (0, 1, 2)}

CHIP_SIZE_X = 256
CHIP_SIZE_Y = 256

CHIP_NUM_X = 4
CHIP_NUM_Y = 2

MODULE_SIZE_X = CHIP_NUM_X * CHIP_SIZE_X
MODULE_SIZE_Y = CHIP_NUM_Y * CHIP_SIZE_Y

CHIP_GAP_X = 2
CHIP_GAP_Y = 2

# 256 not divisible by 3, so we round up to 86
# 18 since we have 6 more pixels in H per gap
STRIPSEL_MODULE_SIZE_X = 1024 * 3 + 18  # = 3090
STRIPSEL_MODULE_SIZE_Y = 86


try:
    # TODO: make a proper external package integration
    mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    for entry in os.scandir(mod_path):
        if (
            entry.is_file()
            and entry.name.startswith('libcorrections')
            and entry.name.endswith('.so')
        ):
            _mod = ctypes.cdll.LoadLibrary(os.path.join(mod_path, entry))

    correct_mask = _mod.jf_apply_pede_gain_mask
    correct_mask.argtypes = (
        ctypes.c_uint32,
        np.ctypeslib.ndpointer(ctypes.c_uint16, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_bool, flags="C_CONTIGUOUS"),
    )
    correct_mask.restype = None
    correct_mask.__doc__ = """Apply gain/pedestal and pixel mask corrections
    Parameters
    ----------
    image_size : c_uint32
        number of pixels in the image array
    image : uint16_t array
        Jungfrau 2D array to be corrected
    GP : float32 array
        array containing combined gain and pedestal corrections
    res : float32 array
        2D array containing corrected image
    pixel_mask : array_like, int
        2D array containing pixels to be masked (tagged with a one)
    """

    correct = _mod.jf_apply_pede_gain
    correct.argtypes = (
        ctypes.c_uint32,
        np.ctypeslib.ndpointer(ctypes.c_uint16, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    )
    correct.restype = None
    correct.__doc__ = """Apply gain/pedestal corrections
    Parameters
    ----------
    image_size : c_uint32
        number of pixels in the image array
    image : uint16_t array
        Jungfrau 2D array to be corrected
    GP : float32 array
        array containing combined gain and pedestal corrections
    res : float32 array
        2D array containing corrected image
    """
except:
    raise ImportError('Could not load libcorrections.')


class JFDataHandler:
    def __init__(self, detector_name):
        """Create an object to perform jungfrau detector data handling like pedestal correction,
        gain conversion, pixel mask, module map, etc.

        Args:
            detector_name (str): name of a detector in the form JF<id>T<nmod>V<version>
        """
        # detector_name needs to be a valid name
        if detector_name in modules_orig:
            self._detector_name = detector_name
        else:
            raise KeyError(f"Geometry for '{detector_name}' detector is not present.")

        # array to be used for the actual data conversion
        # G and P values are interleaved for better CPU cache utilization
        self._GP = np.empty(
            (self._GP_shape[0], 2 * NUM_GAINS * self._GP_shape[1]), dtype=np.float32
        )

        # values that define processing pipeline
        self.convertion = True  # convert to keV (apply gain and pedestal corrections)
        self.gap_pixels = True  # add gap pixels between detector submodules
        self.geometry = True  # apply detector geometry corrections

        self._gain_file = None
        self._pedestal_file = None

        self._G = None
        self._P = None
        self._pixel_mask = None

        self._highgain = False
        self._module_map = None

    @property
    def detector_name(self):
        """Detector name (readonly)"""
        return self._detector_name

    def is_stripsel(self):
        """Return true if detector is a stripsel"""
        return self.detector_name.startswith(("JF05", "JF11"))

    @property
    def _detector(self):
        det = namedtuple('Detector', ['id', 'n_modules', 'version'])
        return det(*(int(d) for d in re.findall(r'\d+', self.detector_name)))

    def _get_n_modules_shape(self, n_modules):
        if self.detector_name == 'JF02T09V01':  # a special case
            shape_y, shape_x = MODULE_SIZE_Y, MODULE_SIZE_X * n_modules
        else:
            shape_y, shape_x = MODULE_SIZE_Y * n_modules, MODULE_SIZE_X

        return shape_y, shape_x

    @property
    def _GP_shape(self):
        n_modules = self._detector.n_modules
        return self._get_n_modules_shape(n_modules)

    @property
    def _raw_shape(self):
        n_active_modules = np.sum(self.module_map != -1)
        return self._get_n_modules_shape(n_active_modules)

    @property
    def _stripsel_shape(self):
        if self.geometry:
            modules_orig_y, modules_orig_x = modules_orig[self.detector_name]
            shape_x = max(modules_orig_x) + STRIPSEL_MODULE_SIZE_X
            shape_y = max(modules_orig_y) + STRIPSEL_MODULE_SIZE_Y
        else:
            shape_y, shape_x = self._raw_shape

        return shape_y, shape_x

    @property
    def shape(self):
        """Shape of image after geometry correction"""
        if self.is_stripsel():
            return self._stripsel_shape

        if self.geometry and self.gap_pixels:
            modules_orig_y, modules_orig_x = modules_orig[self.detector_name]
            shape_x = max(modules_orig_x) + MODULE_SIZE_X + (CHIP_NUM_X - 1) * CHIP_GAP_X
            shape_y = max(modules_orig_y) + MODULE_SIZE_Y + (CHIP_NUM_Y - 1) * CHIP_GAP_Y

        elif self.geometry and not self.gap_pixels:
            modules_orig_y, modules_orig_x = modules_orig[self.detector_name]
            shape_x = max(modules_orig_x) + MODULE_SIZE_X
            shape_y = max(modules_orig_y) + MODULE_SIZE_Y

        elif not self.geometry and self.gap_pixels:
            shape_y, shape_x = self._raw_shape
            shape_x += (CHIP_NUM_X - 1) * CHIP_GAP_X
            shape_y += (CHIP_NUM_Y - 1) * CHIP_GAP_Y * self._detector.n_modules

        elif not self.geometry and not self.gap_pixels:
            shape_y, shape_x = self._raw_shape

        return shape_y, shape_x

    @property
    def gain_file(self):
        """Return gain filepath"""
        return self._gain_file

    @gain_file.setter
    def gain_file(self, filepath):
        if filepath is None:
            self._gain_file = None
            self.G = None
            return

        if filepath == self._gain_file:
            return

        with h5py.File(filepath, 'r') as h5f:
            gains = h5f['/gains'][:]

        self._gain_file = filepath
        self.G = gains

    @property
    def G(self):
        """Current gain values"""
        return self._G

    @G.setter
    def G(self, value):
        if value is None:
            self._G = None
            return

        if value.ndim != 3:
            raise ValueError(f"G should have 3 dimensions, provided G has {value.ndim} dimensions.")

        if value.shape[0] != 4:
            raise ValueError(
                f"First dimension of G should have length 4, provided G has {value.shape[0]}."
            )

        if value.shape[1:] != self._GP_shape:
            raise ValueError(
                f"Expected G shape is {self._GP_shape}, while provided G has {value.shape[1:]}."
            )

        # make sure _G has type float32
        self._G = value.astype(np.float32, copy=False)
        for i, g in zip(range(NUM_GAINS), HIGHGAIN_ORDER[self.highgain]):
            self._GP[:, 2 * i :: NUM_GAINS * 2] = 1 / self._G[g]

    @property
    def pedestal_file(self):
        """Return pedestal filepath"""
        return self._pedestal_file

    @pedestal_file.setter
    def pedestal_file(self, filepath):
        if filepath is None:
            self._pedestal_file = None
            self.P = None
            self.pixel_mask = None
            return

        if filepath == self._pedestal_file:
            return

        with h5py.File(filepath, 'r') as h5f:
            pedestal = h5f['/gains'][:]
            pixel_mask = h5f['/pixel_mask'][:]

        self._pedestal_file = filepath
        self.P = pedestal
        self.pixel_mask = pixel_mask

    @property
    def P(self):
        """Current pedestal values"""
        return self._P

    @P.setter
    def P(self, value):
        if value is None:
            self._P = None
            return

        if value.ndim != 3:
            raise ValueError(f"P should have 3 dimensions, provided P has {value.ndim} dimensions.")

        if value.shape[0] != 4:
            raise ValueError(
                f"First dimension of P should have length 4, provided P has {value.shape[0]}."
            )

        if value.shape[1:] != self._GP_shape:
            raise ValueError(
                f"Expected P shape is {self._GP_shape}, while provided P has {value.shape[1:]}."
            )

        # make sure _P has type float32
        self._P = value.astype(np.float32, copy=False)
        for i, g in zip(range(NUM_GAINS), HIGHGAIN_ORDER[self.highgain]):
            self._GP[:, 2 * i + 1 :: NUM_GAINS * 2] = self._P[g]

    @property
    def highgain(self):
        """Current flag for highgain"""
        return self._highgain

    @highgain.setter
    def highgain(self, value):
        if self._highgain == value:
            return

        self._highgain = value
        first_gain = HIGHGAIN_ORDER[value][0]

        if self.G is not None:
            self._GP[:, :: NUM_GAINS * 2] = 1 / self._G[first_gain]

        if self.P is not None:
            self._GP[:, 1 :: NUM_GAINS * 2] = self._P[first_gain]

    @property
    def pixel_mask(self):
        """Current pixel mask"""
        return self._pixel_mask

    @pixel_mask.setter
    def pixel_mask(self, value):
        if value is None:
            self._pixel_mask = None
            return

        if value.ndim != 2:
            raise ValueError(
                f"Pixel mask should have 2 dimensions, provided pixel mask has {value.ndim}."
            )

        if value.shape != self._GP_shape:
            raise ValueError(
                f"Expected pixel mask shape is {self._GP_shape}, provided pixel mask has {value.shape} shape."
            )

        self._pixel_mask = value.astype(np.bool, copy=False)

    @property
    def shaped_pixel_mask(self):
        """Pixel mask with geometry/gap pixels based on the corresponding flags (readonly)"""
        if self.pixel_mask is None:
            return None

        # currently, it requeires a hack of cleaning convertion and module_map values for shaping
        _module_map = self._module_map
        conversion = self.convertion

        self._module_map = None
        self.convertion = False
        res = np.invert(self.process(np.invert(self.pixel_mask)))

        # restore module_map and convertion values
        self._module_map = _module_map
        self.convertion = conversion

        return res

    @property
    def module_map(self):
        """Current module map"""
        if self._module_map is None:
            # support legacy data by emulating 'all modules are present'
            return np.arange(self._detector.n_modules)
        return self._module_map

    @module_map.setter
    def module_map(self, value):
        if value is None:
            self._module_map = None
            return

        if len(value) != self._detector.n_modules:
            raise ValueError(
                f"Expected module_map length is {self._detector.n_modules}, provided value length is {len(value)}"
            )

        if min(value) < -1 or self._detector.n_modules <= max(value):
            raise ValueError(
                f"Valid module_map values are integers between -1 and {self._detector.n_modules-1}"
            )

        self._module_map = value

    def process(self, images):
        """Perform jungfrau detector data processing like pedestal correction, gain conversion,
        pixel mask, module map, etc.

        Args:
            images (ndarray): image stack or single image to be processed

        Returns:
            ndarray: resulting image stack or single image
        """
        if images.ndim == 2:
            remove_first_dim = True
            images = images[np.newaxis]
        else:
            remove_first_dim = False

        if self.convertion:
            images = self._convert(images)

        if self.is_stripsel():
            if self.geometry:
                images = self._apply_geometry_stripsel(images)
        else:
            if self.geometry:
                # this will also handle self.gap_pixels
                images = self._apply_geometry(images)
            elif self.gap_pixels:
                images = self._add_gap_pixels(images)

        if remove_first_dim:
            images = images[0]

        return images

    def can_convert(self):
        return (self.G is not None) and (self.P is not None)

    def _convert(self, image_stack):
        """Apply pedestal correction and gain conversion

        Args:
            image_stack (ndarray): image stack to be processed

        Returns:
            ndarray: resulting image stack or a single image
        """
        self._check_image_stack_shape(image_stack)

        if not self.can_convert():
            raise RuntimeError("Gain and/or pedestal values are not set")

        res = np.empty(shape=image_stack.shape, dtype=np.float32)
        module_size = np.uint32(MODULE_SIZE_X * MODULE_SIZE_Y)
        for i, m in enumerate(self.module_map):
            if m == -1:
                continue

            module = self._get_module_slice(image_stack, m)
            module_res = res[:, m * MODULE_SIZE_Y : (m + 1) * MODULE_SIZE_Y, :]
            module_GP = self._GP[i * MODULE_SIZE_Y : (i + 1) * MODULE_SIZE_Y, :]

            if self.pixel_mask is None:
                for mod, mod_res in zip(module, module_res):
                    correct(module_size, mod, module_GP, mod_res)
            else:
                mask_module = self.pixel_mask[i * MODULE_SIZE_Y : (i + 1) * MODULE_SIZE_Y, :]
                for mod, mod_res in zip(module, module_res):
                    correct_mask(module_size, mod, module_GP, mod_res, mask_module)

        return res

    def _apply_geometry(self, image_stack):
        """Rearrange image according to geometry of detector modules

        Args:
            image_stack (ndarray): image stack to be processed

        Returns:
            ndarray: resulting image_stack with modules on their actual places
        """
        self._check_image_stack_shape(image_stack)

        modules_orig_y, modules_orig_x = modules_orig[self.detector_name]

        res = np.zeros((image_stack.shape[0], *self.shape), dtype=image_stack.dtype)
        for m, oy, ox in zip(self.module_map, modules_orig_y, modules_orig_x):
            if m == -1:
                continue

            module = self._get_module_slice(image_stack, m)

            if self.detector_name in ('JF02T09V02', 'JF02T01V02'):
                module = np.rot90(module, 2, axes=(1, 2))

            if self.gap_pixels:
                for j in range(CHIP_NUM_Y):
                    for k in range(CHIP_NUM_X):
                        # reading positions
                        ry_s = j * CHIP_SIZE_Y
                        rx_s = k * CHIP_SIZE_X

                        # writing positions
                        wy_s = oy + ry_s + j * CHIP_GAP_Y
                        wx_s = ox + rx_s + k * CHIP_GAP_X

                        res[:, wy_s : wy_s + CHIP_SIZE_Y, wx_s : wx_s + CHIP_SIZE_X] = module[
                            :, ry_s : ry_s + CHIP_SIZE_Y, rx_s : rx_s + CHIP_SIZE_X
                        ]
            else:
                res[:, oy : oy + MODULE_SIZE_Y, ox : ox + MODULE_SIZE_X] = module

        # rotate image stack in case of alvra detector
        if self.detector_name.startswith('JF06'):
            res = np.rot90(res, axes=(1, 2))

        return res

    def _add_gap_pixels(self, image_stack):
        self._check_image_stack_shape(image_stack)

        res = np.zeros((image_stack.shape[0], *self.shape), dtype=image_stack.dtype)

        for m in range(np.sum(self.module_map != -1)):
            module = self._get_module_slice(image_stack, m)
            for j in range(CHIP_NUM_Y):
                for k in range(CHIP_NUM_X):
                    # reading positions
                    ry_s = j * CHIP_SIZE_Y
                    rx_s = k * CHIP_SIZE_X

                    # writing positions
                    wy_s = m * (MODULE_SIZE_Y + CHIP_GAP_Y) + ry_s + j * CHIP_GAP_Y
                    wx_s = rx_s + k * CHIP_GAP_X

                    res[:, wy_s : wy_s + CHIP_SIZE_Y, wx_s : wx_s + CHIP_SIZE_X] = module[
                        :, ry_s : ry_s + CHIP_SIZE_Y, rx_s : rx_s + CHIP_SIZE_X
                    ]

        return res

    def _apply_geometry_stripsel(self, image_stack):
        """Rearrange stripsel image according to geometry of detector modules

        Args:
            image_stack (ndarray): image stack to be processed

        Returns:
            ndarray: resulting image_stack with modules on their actual places
        """
        self._check_image_stack_shape(image_stack)

        modules_orig_y, modules_orig_x = modules_orig[self.detector_name]

        res = np.zeros((image_stack.shape[0], *self.shape), dtype=image_stack.dtype)
        for m, oy, ox in zip(self.module_map, modules_orig_y, modules_orig_x):
            if m == -1:
                continue

            module = self._get_module_slice(image_stack, m)

            for ind in range(module.shape[0]):
                res[
                    ind, oy : oy + STRIPSEL_MODULE_SIZE_Y, ox : ox + STRIPSEL_MODULE_SIZE_X
                ] = reshape_stripsel(module[ind])

        return res

    def _check_image_stack_shape(self, image_stack):
        image_shape = image_stack.shape[-2:]
        if image_shape != self._raw_shape:
            raise ValueError(
                f"Expected image shape {self._raw_shape}, provided image shape {image_shape}"
            )

    def _get_module_slice(self, images, index):
        # in case of a single image, Ellipsis will be ignored
        # in case of 3D image stack, Ellipsis will be parsed into slice(None, None)
        if self.detector_name == 'JF02T09V01':
            module = images[Ellipsis, :, index * MODULE_SIZE_X : (index + 1) * MODULE_SIZE_X]
        else:
            module = images[Ellipsis, index * MODULE_SIZE_Y : (index + 1) * MODULE_SIZE_Y, :]

        return module


@jit(nopython=True)
def reshape_stripsel(image):
    res = np.zeros((STRIPSEL_MODULE_SIZE_Y, STRIPSEL_MODULE_SIZE_X), dtype=image.dtype)

    # first we fill the normal pixels, the gap ones will be overwritten later
    for yin in range(256):
        for xin in range(1024):
            ichip = xin // 256
            xout = (ichip * 774) + (xin % 256) * 3 + yin % 3
            # 774 is the chip period, 256*3+6
            yout = yin // 3
            res[yout, xout] = image[yin, xin]

    # now the gap pixels
    for igap in range(3):
        for yin in range(256):
            yout = (yin // 6) * 2

            # first the left side of gap
            xin = igap * 64 + 63
            xout = igap * 774 + 765 + yin % 6
            res[yout, xout] = image[yin, xin]
            res[yout + 1, xout] = image[yin, xin]

            # then the right side is mirrored
            xin = igap * 64 + 63 + 1
            xout = igap * 774 + 765 + 11 - yin % 6
            res[yout, xout] = image[yin, xin]
            res[yout + 1, xout] = image[yin, xin]
            # if we want a proper normalization (the area of those pixels is double, so they see 2x
            # the signal)
            # res[yout,xout] = res[yout,xout]/2

    return res
