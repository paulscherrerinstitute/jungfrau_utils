from functools import wraps

import numpy as np
from numba import jit
from numpy import ma

from .geometry import modules_orig

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
