import numpy as np
import sys
from time import time
import numpy.ma as ma


is_numba = False


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
    print(sys.exc_info())
    

def apply_gain_pede(image, G=None, P=None, pixel_mask=None, inverse_gain=False):
    r"""Apply gain corrections to Jungfrau image. Gain and Pedestal corrections are
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
    60 ms ± 72.7 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

    Numba
    6.23 ms ± 7.22 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

    Numba plus inverse
    6.15 ms ± 42.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    """
    if is_numba:
        return apply_gain_pede_numba(image, G=G, P=P, pixel_mask=pixel_mask, inverse_gain=inverse_gain)
    return apply_gain_pede_np(image, G=G, P=P, pixel_mask=pixel_mask, inverse_gain=inverse_gain)


def test():
    data = np.random.randint(0, 60000, size=[1500, 1000], dtype=np.uint16)
    pede = 60000 * np.random.random(size=[3, 1500, 1000])
    gain = 100 * np.random.random(size=[3, 1500, 1000])
    gain[gain > 1] = 3

    t_i = time()
    res1 = apply_gain_pede_np(data, gain, pede)
    print("NP", time() - t_i)
    t_i = time()
    res2 = apply_gain_pede_numba(data, gain, pede)
    print("Numba", time() - t_i)
    t_i = time()
    res2 = apply_gain_pede_numba(data, gain, pede)
    print("Numba", time() - t_i)
    t_i = time()
    gain2 = 1. / gain
    res2 = apply_gain_pede(data, gain2, pede, inverse_gain=True)
    print("Numba inverse", time() - t_i)
    t_i = time()
    res2 = apply_gain_pede(data, gain2, pede, inverse_gain=True)
    print("Numba inverse", time() - t_i)
    # print((res1 - res2 < 0.01).all())
    # print(res1[0:2, 0:2], res2[0:2, 0:2])


if __name__ == "__main__":
    test()
