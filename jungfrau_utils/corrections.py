import ctypes
import os
import re
from collections import namedtuple
from functools import wraps
from time import time

import numpy as np
from numpy import ma

from .geometry import modules_orig

try:
    import mkl
except ImportError:
    pass
else:
    mkl.set_num_threads(1)

NUM_GAINS = 4

CHIP_SIZE_X = 256
CHIP_SIZE_Y = 256

CHIP_NUM_X = 4
CHIP_NUM_Y = 2

MODULE_SIZE_X = CHIP_NUM_X * CHIP_SIZE_X
MODULE_SIZE_Y = CHIP_NUM_Y * CHIP_SIZE_Y

CHIP_GAP_X = 2
CHIP_GAP_Y = 2


def _allow_n_images(method):
    """Allows any **method** that expects a single image as first argument to accept n images"""

    @wraps(method)
    def wrapper(this, images, *args, **kwargs):
        func = lambda *args, **kwargs: method(this, *args, **kwargs)  # hide the self argument
        if images.ndim == 3:
            return _apply_to_all_images(func, images, *args, **kwargs)
        else:
            return func(images, *args, **kwargs)

    return wrapper


def _apply_to_all_images(func, images, *args, **kwargs):
    """Apply func to all images forwarding args and kwargs"""
    nshots = len(images)
    one_image = func(images[0], *args, **kwargs)
    target_dtype = one_image.dtype
    target_shape = one_image.shape
    target_shape = [nshots] + list(target_shape)
    images_result = np.empty(shape=target_shape, dtype=target_dtype)
    for n, img in enumerate(images):
        images_result[n] = func(img, *args, **kwargs)
    return images_result


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
    print('Could not load libcorrections.')

    def correct(*args, **kwargs):
        raise NotImplementedError(
            "libcorrections is needed. python version of jf_apply_pede_gain() missing."
        )

    def correct_mask(*args, **kwargs):
        raise NotImplementedError(
            "libcorrections is needed. python version of jf_apply_pede_gain_mask() missing."
        )


def apply_gain_pede_np(image, G=None, P=None, pixel_mask=None):
    mask = int('0b' + 14 * '1', 2)
    mask2 = int('0b' + 2 * '1', 2)

    gain_mask = np.bitwise_and(np.right_shift(image, 14), mask2)
    data = np.bitwise_and(image, mask)

    m1 = gain_mask != 0
    m2 = gain_mask != 1
    m3 = gain_mask < 2
    if G is not None:
        g = (
            ma.array(G[0], mask=m1, dtype=np.float32).filled(0)
            + ma.array(G[1], mask=m2, dtype=np.float32).filled(0)
            + ma.array(G[2], mask=m3, dtype=np.float32).filled(0)
        )
    else:
        g = np.ones(data.shape, dtype=np.float32)
    if P is not None:
        p = (
            ma.array(P[0], mask=m1, dtype=np.float32).filled(0)
            + ma.array(P[1], mask=m2, dtype=np.float32).filled(0)
            + ma.array(P[2], mask=m3, dtype=np.float32).filled(0)
        )
    else:
        p = np.zeros(data.shape, dtype=np.float32)

    res = np.divide(data - p, g)

    if pixel_mask is not None:
        res[pixel_mask] = 0

    return res


try:
    from numba import jit

    @jit(nopython=True, nogil=True, cache=False)
    def apply_gain_pede_corrections_numba(m, n, image, G, P, mask, mask2, pede_mask, gain_mask):
        res = np.empty((m, n), dtype=np.float32)
        for i in range(m):
            for j in range(n):
                if pede_mask[i][j] != 0:
                    res[i][j] = 0
                    continue
                gm = gain_mask[i][j]
                # if i==0 and j==0:
                #    print(gm, image[i][j], P[gm][i][j], G[gm][i][j])
                if gm == 3:
                    gm = 2

                res[i][j] = (image[i][j] - P[gm][i][j]) / G[gm][i][j]
        return res

    def apply_gain_pede_numba(image, G=None, P=None, pixel_mask=None):

        mask = int('0b' + 14 * '1', 2)
        mask2 = int('0b' + 2 * '1', 2)
        gain_mask = np.bitwise_and(np.right_shift(image, 14), mask2)
        image = np.bitwise_and(image, mask)

        if G is None:
            G = np.ones((3, image.shape[0], image.shape[1]), dtype=np.float32)
        if P is None:
            P = np.zeros((3, image.shape[0], image.shape[1]), dtype=np.float32)
        if pixel_mask is None:
            pixel_mask = np.zeros(image.shape, dtype=np.int)

        return apply_gain_pede_corrections_numba(
            image.shape[0], image.shape[1], image, G, P, mask, mask2, pixel_mask, gain_mask
        )

    is_numba = True

except ImportError:
    is_numba = False


def apply_gain_pede(image, G=None, P=None, pixel_mask=None, highgain=False):
    """Apply gain corrections to Jungfrau image. Gain and Pedestal corrections are
    to be provided as a 3D array of shape (3, image.shape[0], image.shape[1]).
    The formula for the correction is: (image - P) / G

    If Numba is available, a Numba-optimized routine is used: otherwise, a Numpy based one.

    Parameters
    ----------
    image : array_like
        2D array to be corrected
    G : array_like
        3D array containing gain corrections
    P : array_like
        3D array containing pedestal corrections
    pixel_mask : array_like, int
        2D array containing pixels to be masked (tagged with a one)
    highgain : bool
        Are you using G0 or HG0? If the latter, then this should be True (default: False)

    Returns
    -------
    res : NDArray
        Corrected image

    Notes
    -----
    Performances for correcting a random image as of 2017-11-23, shape [1500, 1000]

    Numpy
    60 ms +- 72.7 us per loop (mean +- std. dev. of 7 runs, 10 loops each)

    Numba
    6.23 ms +- 7.22 us per loop (mean +- std. dev. of 7 runs, 100 loops each)
    """

    if G is not None:
        G = G.astype(np.float32)

    if P is not None:
        P = P.astype(np.float32)

    if highgain:
        G[0] = G[3]
        P[0] = P[3]

    func_to_use = apply_gain_pede_np
    if is_numba:
        func_to_use = apply_gain_pede_numba

    partial_func_to_use = lambda X: func_to_use(X, G=G, P=P, pixel_mask=pixel_mask)

    if image.ndim == 3:
        res = _apply_to_all_images(partial_func_to_use, image)
    else:
        res = partial_func_to_use(image)

    return res


def get_gain_data(image):
    """Return the Jungfrau gain map and data using as an input the 16 bit encoded raw data.
    RAW data is composed by the two MSB (most significant bits) encoding the gain, and 14
    bits containing the actual data counts. Possible gain levels are: 00, 01, 11.

    Parameters
    ----------
    image : array_like
        2D array to be corrected

    Returns
    -------
    gain_map : NDArray
        Array containing the gain levels of each pixel
    data : NDArray
        Array containing the data

    """
    mask = int('0b' + 14 * '1', 2)
    mask2 = int('0b' + 2 * '1', 2)

    gain_map = np.bitwise_and(np.right_shift(image, 14), mask2)
    data = np.bitwise_and(image, mask)

    return gain_map, data


def add_gap_pixels(image, modules, module_gap, chip_gap=[2, 2]):
    """Add module and pixel gaps to an image.

    Parameters
    ----------
    image : array_like
        2D array to be corrected
    modules : array_like
        number of modules, in the form [rows, columns]. E.g., for a 1.5M in vertical this is [3, 1]
    module_gap : array_like
        gap between the modules in pixels
    chip_gap : array_like
        gap between the chips in a module, default: [2, 2]

    Returns
    -------
    res : NDArray
        Corrected image

    Notes
    -----
    Performances for correcting a random image as of 2017-11-28, shape [3*512, 1024]

    4.47 ms ± 734 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    """
    chips = [2, 4]
    shape = image.shape
    mod_size = [256, 256]  # this is the chip size
    new_shape = [
        shape[i] + (module_gap[i]) * (modules[i] - 1) + (chips[i] - 1) * chip_gap[i] * modules[i]
        for i in range(2)
    ]

    res = np.zeros(new_shape)
    m = [module_gap[i] - chip_gap[i] for i in range(2)]

    for i in range(modules[0] * chips[0]):
        for j in range(modules[1] * chips[1]):
            disp = [
                int(i / chips[0]) * m[0] + i * chip_gap[0],
                int(j / chips[1]) * m[1] + j * chip_gap[1],
            ]
            init = [i * mod_size[0], j * mod_size[1]]
            end = [(1 + i) * mod_size[0], (1 + j) * mod_size[1]]
            res[disp[0] + init[0] : disp[0] + end[0], disp[1] + init[1] : disp[1] + end[1]] = image[
                init[0] : end[0], init[1] : end[1]
            ]

    return res


class JFDataHandler:
    def __init__(self, detector_name, G=None, P=None, pixel_mask=None, highgain=False):
        """A class to perform jungfrau detector data handling like pedestal correction, gain
        conversion, pixel mask, module map, etc.

        Args:
            detector_name (str): name of a detector
            G (ndarray, optional): 4d array with gain values
            P (ndarray, optional): 4d array with pedestal values
            pixel_mask (ndarray, optional): 2d array with non-zero values referring to bad pixels.
                When None, all pixels assumed to be good. Defaults to None.
            highgain (bool, optional): Highgain mode where G[3] is used for G[0]. Defaults to False.
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

        # this will also fill self._GP array with G and P values if they are not None
        self.G = G
        self.P = P

        self.pixel_mask = pixel_mask
        self.highgain = highgain
        self.module_map = None

    @property
    def detector_name(self):
        return self._detector_name

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
        if self.module_map is None:
            return self._GP_shape

        n_active_modules = np.sum(self.module_map != -1)
        return self._get_n_modules_shape(n_active_modules)

    @property
    def shape(self):
        modules_orig_y, modules_orig_x = modules_orig[self.detector_name]
        shape_x = max(modules_orig_x) + MODULE_SIZE_X + (CHIP_NUM_X - 1) * CHIP_GAP_X
        shape_y = max(modules_orig_y) + MODULE_SIZE_Y + (CHIP_NUM_Y - 1) * CHIP_GAP_Y

        return shape_y, shape_x

    @property
    def G(self):
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
        for i in range(NUM_GAINS):
            self._GP[:, 2 * i :: NUM_GAINS * 2] = 1 / self._G[i]

    @property
    def P(self):
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
        for i in range(NUM_GAINS):
            self._GP[:, (2 * i + 1) :: NUM_GAINS * 2] = self._P[i]

    @property
    def highgain(self):
        return self._highgain

    @highgain.setter
    def highgain(self, value):
        if value is True and self.G is None:
            raise ValueError(f"Gains are not defined.")

        self._highgain = value
        if self.G is not None:
            if value:
                self._GP[:, :: NUM_GAINS * 2] = 1 / self._G[3]
            else:
                self._GP[:, :: NUM_GAINS * 2] = 1 / self._G[0]

        if self.P is not None:
            if value:
                self._GP[:, 1 :: NUM_GAINS * 2] = self._P[3]
            else:
                self._GP[:, 1 :: NUM_GAINS * 2] = self._P[0]

    @property
    def pixel_mask(self):
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

    @_allow_n_images
    def apply_gain_pede(self, image):
        """Apply pedestal correction and gain conversion

        Args:
            image (ndarray): image to be processed

        Returns:
            ndarray: resulting image
        """
        if image.shape != self._raw_shape:
            raise ValueError(
                f"Expected image shape {self._raw_shape}, provided image shape {image.shape}"
            )

        if self.G is None:
            raise ValueError(f"Gains are not set")

        if self.P is None:
            raise ValueError(f"Pedestal values are not set")

        res = np.empty(shape=image.shape, dtype=np.float32)
        if self.module_map is None:
            if self.pixel_mask is None:
                correct(np.uint32(image.size), image, self._GP, res)
            else:
                correct_mask(np.uint32(image.size), image, self._GP, res, self.pixel_mask)

        else:
            for i, m in enumerate(self.module_map):
                if m == -1:
                    continue

                module = self._get_module(image, m)

                module_res = res[m * MODULE_SIZE_Y : (m + 1) * MODULE_SIZE_Y, :]
                module_GP = self._GP[i * MODULE_SIZE_Y : (i + 1) * MODULE_SIZE_Y, :]
                module_size = np.uint32(module.size)

                if self.pixel_mask is None:
                    correct(module_size, module, module_GP, module_res)
                else:
                    mask_module = self.pixel_mask[i * MODULE_SIZE_Y : (i + 1) * MODULE_SIZE_Y, :]
                    correct_mask(module_size, module, module_GP, module_res, mask_module)

        return res

    def apply_geometry(self, image):
        """Rearrange image according to geometry of detector modules.

        Args:
            image (ndarray): a single image or image stack to be processed

        Returns:
            ndarray: image with modules on their actual places
        """
        if image.shape[-2:] != self._raw_shape:
            raise ValueError(
                f"Expected image shape {self._raw_shape}, provided image shape {image.shape[-2:]}"
            )

        modules_orig_y, modules_orig_x = modules_orig[self.detector_name]

        if image.ndim == 3:
            res = np.zeros((image.shape[0], *self.shape), dtype=image.dtype)
            rot_axes = (1, 2)
        else:
            res = np.zeros(self.shape, dtype=image.dtype)
            rot_axes = (0, 1)

        if self.module_map is None:
            # emulate 'all modules are present'
            module_map = range(len(modules_orig_y))
        else:
            module_map = self.module_map

        for m, oy, ox in zip(module_map, modules_orig_y, modules_orig_x):
            if m == -1:
                continue

            module = self._get_module(image, m)

            if self.detector_name in ('JF02T09V02', 'JF02T01V02'):
                module = np.rot90(module, 2, axes=rot_axes)

            for j in range(CHIP_NUM_Y):
                for k in range(CHIP_NUM_X):
                    # reading positions
                    ry_s = j * CHIP_SIZE_Y
                    rx_s = k * CHIP_SIZE_X

                    # writing positions
                    wy_s = oy + ry_s + j * CHIP_GAP_Y
                    wx_s = ox + rx_s + k * CHIP_GAP_X

                    res[Ellipsis, wy_s : wy_s + CHIP_SIZE_Y, wx_s : wx_s + CHIP_SIZE_X] = module[
                        Ellipsis, ry_s : ry_s + CHIP_SIZE_Y, rx_s : rx_s + CHIP_SIZE_X
                    ]

        # rotate image in case of alvra detector
        if self.detector_name.startswith('JF06'):
            res = np.rot90(res, axes=rot_axes)

        return res

    def _get_module(self, image, index):
        # in case of a single image, Ellipsis will be ignored
        # in case of 3D image stack, Ellipsis will be parsed into slice(None, None)
        if self.detector_name == 'JF02T09V01':
            module = image[Ellipsis, :, index * MODULE_SIZE_X : (index + 1) * MODULE_SIZE_X]
        else:
            module = image[Ellipsis, index * MODULE_SIZE_Y : (index + 1) * MODULE_SIZE_Y, :]

        return module


def apply_geometry(image_in, detector_name):
    if detector_name in modules_orig:
        modules_orig_y, modules_orig_x = modules_orig[detector_name]
    else:
        return image_in

    image_out_shape_x = max(modules_orig_x) + MODULE_SIZE_X + (CHIP_NUM_X - 1) * CHIP_GAP_X
    image_out_shape_y = max(modules_orig_y) + MODULE_SIZE_Y + (CHIP_NUM_Y - 1) * CHIP_GAP_Y
    image_out = np.zeros((image_out_shape_y, image_out_shape_x), dtype=image_in.dtype)

    for i, (oy, ox) in enumerate(zip(modules_orig_y, modules_orig_x)):
        if detector_name == 'JF02T09V01':
            module_in = image_in[:, i * MODULE_SIZE_X : (i + 1) * MODULE_SIZE_X]
        elif detector_name == 'JF02T09V02' or detector_name == 'JF02T01V02':
            module_in = np.rot90(image_in[i * MODULE_SIZE_Y : (i + 1) * MODULE_SIZE_Y, :], 2)
        else:
            module_in = image_in[i * MODULE_SIZE_Y : (i + 1) * MODULE_SIZE_Y, :]

        for j in range(CHIP_NUM_Y):
            for k in range(CHIP_NUM_X):
                # reading positions
                ry_s = j * CHIP_SIZE_Y
                rx_s = k * CHIP_SIZE_X

                # writing positions
                wy_s = oy + ry_s + j * CHIP_GAP_Y
                wx_s = ox + rx_s + k * CHIP_GAP_X

                image_out[wy_s : wy_s + CHIP_SIZE_Y, wx_s : wx_s + CHIP_SIZE_X] = module_in[
                    ry_s : ry_s + CHIP_SIZE_Y, rx_s : rx_s + CHIP_SIZE_X
                ]

    # rotate image in case of alvra detector
    if detector_name.startswith('JF06'):
        image_out = np.rot90(image_out)  # check .copy()

    return image_out


def test():
    size_1 = 16384
    size_2 = 1024
    data = np.random.randint(0, 60000, size=[size_1, size_2], dtype=np.uint16)
    pede = 60000 * np.random.random(size=[4, size_1, size_2]).astype(np.float32)
    gain = 100 * np.random.random(size=[4, size_1, size_2]).astype(np.float32) + 1
    mask = np.random.randint(2, size=[size_1, size_2], dtype=np.bool)

    t_i = time()
    res1 = apply_gain_pede_np(data, gain, pede, mask)
    print("NP", time() - t_i)
    t_i = time()
    res2 = apply_gain_pede(data, gain, pede, mask)
    print("Numba", time() - t_i)
    t_i = time()
    res2 = apply_gain_pede(data, gain, pede, mask)
    print("Numba", time() - t_i)

    calib = JFDataHandler('JF06T32V01', G=gain, P=pede, pixel_mask=mask)
    t_i = time()
    res3 = calib.apply_gain_pede(data)
    print("C", time() - t_i)

    return (np.allclose(res1, res2, rtol=0.01), np.allclose(res1, res3))


if __name__ == "__main__":
    print(test())
