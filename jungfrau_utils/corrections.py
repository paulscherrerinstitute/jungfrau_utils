import ctypes
import os
from time import time

import numpy as np
from numpy import ma

from jungfrau_utils.geometry import modules_orig

is_numba = False

try:
    # TODO: make a proper external package integration
    mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    for entry in os.scandir(mod_path):
        if entry.is_file() and entry.name.startswith('libcorrections') and entry.name.endswith('.so'):
            _mod = ctypes.cdll.LoadLibrary(os.path.join(mod_path, entry))

    correct = _mod.jf_apply_pede_gain_mask
    correct.argtypes = (
        ctypes.c_uint16, ctypes.c_uint16,
        np.ctypeslib.ndpointer(ctypes.c_uint16, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
    )
    correct.restype = None
    correct.__doc__ = """Insert doc here
    Parameters
    ----------
    m : int
        size of the image array, rows
    n : int
        size of the image array, columns
    image : uint16_t array
        Jungfrau 2D array to be corrected
    GP : float32 array
        array containing combined gain and pedestal corrections
    res : float32 array
        2D array containing corrected image
    pixel_mask : array_like, int
        2D array containing pixels to be masked (tagged with a one)
    """
except:
    print('Could not load libcorrections.')


def apply_gain_pede_np(image, G=None, P=None, pixel_mask=None):
    mask = int('0b' + 14 * '1', 2)
    mask2 = int('0b' + 2 * '1', 2)

    gain_mask = np.bitwise_and(np.right_shift(image, 14), mask2)
    data = np.bitwise_and(image, mask)

    m1 = gain_mask != 0
    m2 = gain_mask != 1
    m3 = gain_mask < 2
    if G is not None:
        g = ma.array(G[0], mask=m1, dtype=np.float32).filled(0) + ma.array(G[1], mask=m2, dtype=np.float32).filled(0) + ma.array(G[2], mask=m3, dtype=np.float32).filled(0)
    else:
        g = np.ones(data.shape, dtype=np.float32)
    if P is not None:
        p = ma.array(P[0], mask=m1, dtype=np.float32).filled(0) + ma.array(P[1], mask=m2, dtype=np.float32).filled(0) + ma.array(P[2], mask=m3, dtype=np.float32).filled(0)
    else:
        p = np.zeros(data.shape, dtype=np.float32)
    if pixel_mask is not None:
        data = ma.array(data, mask=pixel_mask, dtype=data.dtype).filled(0)

    res = np.divide(data - p, g)
    return res


try:
    from numba import jit

    @jit(nopython=True, nogil=True, cache=False)
    def apply_gain_pede_corrections_numba(m, n, image, G, P, mask, mask2, pede_mask, gain_mask, inverse_gain=False):
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
                if inverse_gain is False:
                    res[i][j] = (image[i][j] - P[gm][i][j]) / G[gm][i][j]
                else:
                    res[i][j] = (image[i][j] - P[gm][i][j]) / G[gm][i][j]
        return res

    def apply_gain_pede_numba(image, G=None, P=None, pixel_mask=None, inverse_gain=False):

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

        return apply_gain_pede_corrections_numba(image.shape[0], image.shape[1], image, G, P, mask, mask2, pixel_mask, gain_mask, inverse_gain=inverse_gain)

    is_numba = True

except:
    print("[INFO][corrections] Numba not available, reverting to Numpy")
    #print(sys.exc_info())


def apply_gain_pede(image, G=None, P=None, pixel_mask=None, highgain=False, inverse_gain=False):
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
    inverse_gain : bool
        The inverse of the gain correction is provided

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

    Numba plus inverse
    6.15 ms +- 42.2 us per loop (mean +- std. dev. of 7 runs, 100 loops each)
    """

    G = G.astype(np.float32)
    P = P.astype(np.float32)

    if highgain:
        G[0] = G[3]
        P[0] = P[3]

    if is_numba:
        return apply_gain_pede_numba(image, G=G, P=P, pixel_mask=pixel_mask, inverse_gain=inverse_gain)
    return apply_gain_pede_np(image, G=G, P=P, pixel_mask=pixel_mask, inverse_gain=inverse_gain)


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
    new_shape = [shape[i] + (module_gap[i]) * (modules[i] - 1) + (chips[i] - 1) * chip_gap[i] * modules[i] for i in range(2)]

    res = np.zeros(new_shape)
    m = [module_gap[i] - chip_gap[i] for i in range(2)]

    for i in range(modules[0] * chips[0]):
        for j in range(modules[1] * chips[1]):
            disp = [int(i / chips[0]) * m[0] + i * chip_gap[0], int(j / chips[1]) * m[1] + j * chip_gap[1]]
            init = [i * mod_size[0], j * mod_size[1]]
            end = [(1 + i) * mod_size[0], (1 + j) * mod_size[1]]
            res[disp[0] + init[0]: disp[0] + end[0], disp[1] + init[1]:disp[1] + end[1]] = image[init[0]:end[0], init[1]:end[1]]

    return res


class JungfrauCalibration():

    def __init__(self, G, P, pixel_mask=None, highgain=False):
        """[summary]

        Parameters
        ----------
        G : [type]
            [description]
        P : [type]
            [description]

        """

        G = G.astype(np.float32)
        P = P.astype(np.float32)
        if highgain:
            G[0] = G[3]
            P[0] = P[3]

        self.GP = np.zeros(shape=[G.shape[1], 2 * G.shape[0] * (G.shape[2])], dtype=G.dtype)

        for i in range(G.shape[0]):
            self.GP[:, 2 * i::G.shape[0] * 2] = G[i, :, :]
            self.GP[:, (2 * i + 1)::G.shape[0]*2] = P[i, :, :]

        self.pixel_mask = pixel_mask
        if self.pixel_mask is None:
            self.pixel_mask = np.zeros([G.shape[1], G.shape[2]], dtype=np.int32)

    from numba import jit

    # FIXME add HG0
    @staticmethod
    @jit(nopython=True, nogil=True, cache=True)
    def apply_gain_pede_corrections_numba(m, n, image, GP, mask, mask2, pede_mask, gain_mask):
        res = np.empty((m, n), dtype=np.float32)
        for i in range(m):
            for j in range(n):
                if pede_mask[i][j] != 0:
                    res[i][j] = 0
                    continue
                gm = gain_mask[i][j]
                if gm == 3:
                    gm = 2
                res[i][j] = (image[i][j] - GP[i][2 * gm + 8 * j + 1]) / GP[i][2 * gm + 8 * j]
        return res

    def apply_gain_pede(self, image):
        res = np.empty(shape=image.shape, dtype=np.float32)
        correct(image.shape[0], image.shape[1], image, self.GP, res, self.pixel_mask)
        return res


def apply_geometry(image_in, detector_name):
    chip_shape_x = 256
    chip_shape_y = 256

    chip_gap_x = 2
    chip_gap_y = 2

    chip_num_x = 4
    chip_num_y = 2

    module_shape_x = 1024
    module_shape_y = 512

    if detector_name in modules_orig:
        modules_orig_y, modules_orig_x = modules_orig[detector_name]
    else:
        return image_in

    image_out_shape_x = max(modules_orig_x) + module_shape_x + (chip_num_x-1)*chip_gap_x
    image_out_shape_y = max(modules_orig_y) + module_shape_y + (chip_num_y-1)*chip_gap_y
    image_out = np.zeros((image_out_shape_y, image_out_shape_x), dtype=image_in.dtype)

    for i, (oy, ox) in enumerate(zip(modules_orig_y, modules_orig_x)):
        module_in = image_in[i*module_shape_y:(i+1)*module_shape_y, :]
        for j in range(chip_num_y):
            for k in range(chip_num_x):
                # reading positions
                ry_s = j*chip_shape_y
                rx_s = k*chip_shape_x

                # writing positions
                wy_s = oy + ry_s + j*chip_gap_y
                wx_s = ox + rx_s + k*chip_gap_x

                image_out[wy_s:wy_s+chip_shape_y, wx_s:wx_s+chip_shape_x] = \
                    module_in[ry_s:ry_s+chip_shape_y, rx_s:rx_s+chip_shape_x]

    # rotate image in case of alvra detector
    if detector_name == 'JF06T32V01':
        image_out = np.rot90(image_out)  # check .copy()

    return image_out


def test():
    size_1 = 4500
    size_2 = 4000
    data = np.random.randint(0, 60000, size=[size_1, size_2], dtype=np.uint16)
    pede = 60000 * np.random.random(size=[4, size_1, size_2]).astype(np.float16)
    gain = 100 * np.random.random(size=[4, size_1, size_2]).astype(np.float16)
    gain[gain > 1] = 3

    t_i = time()
    res1 = apply_gain_pede_np(data, gain, pede)
    print("NP", time() - t_i)
    t_i = time()
    res2 = apply_gain_pede(data, gain, pede)
    print("Numba", time() - t_i)
    t_i = time()
    res2 = apply_gain_pede(data, gain, pede)
    print("Numba", time() - t_i)

    calib = JungfrauCalibration(G=gain, P=pede)
    t_i = time()
    res3 = calib.apply_gain_pede(data)
    print("C", time() - t_i)

    return (np.allclose(res1, res2, rtol=0.01), np.allclose(res1, res3))


if __name__ == "__main__":
    print(test())
