from time import time

import numpy as np
import numpy.ma as ma
from numba import jit


#@profile
def test(image, g1, g2, g3, p1, p2, p3):

    gain_mask = np.right_shift(image, 14)
    #g1_c = g1.copy()
    #g2_c = g2.copy()
    #g3_c = g3.copy()

    #g1_c[gain_mask != 0] = 0
    #g2_c[gain_mask != 1] = 0
    #g3_c[gain_mask < 2] = 0

    #g = g1_c + g2_c + g3_c

    m1 = gain_mask != 0
    m2 = gain_mask != 1
    m3 = gain_mask < 2
    g = ma.array(g1, mask=m1, dtype=np.uint16).filled(0) + ma.array(g2, mask=m2, dtype=np.uint16).filled(0) + ma.array(g3, mask=m3, dtype=np.uint16).filled(0)
    p = ma.array(p1, mask=m1, dtype=np.uint16).filled(0) + ma.array(p2, mask=m2, dtype=np.uint16).filled(0) + ma.array(p3, mask=m3, dtype=np.uint16).filled(0)

    res = np.divide(image - p, g)
    return res


#@jit('void(u2[:], u2[:], i2, i2, u2[:,:], u2[:,:]', nopython=False, nogil=True, )
@jit(nopython=True, nogil=True)
def pseudo(m, n, image, G, P):
    #m, n = image.shape
    gain_mask = np.right_shift(image, 14)
    res = np.empty((m, n), dtype=np.uint16)

    for i in range(m):
        for j in range(n):
            gm = gain_mask[i][j]
            res[i][j] = (image[i][j] - P[gm - 1][i][j]) / G[gm - 1][i][j]



if __name__ == "__main__":
    size = [4096, 4096]
    G = np.ones([3, ] + size, dtype=np.uint16)
    G[0] = np.ones(size, dtype=np.uint16)
    G[1] = 2 * np.ones(size, dtype=np.uint16)
    G[2] = 3 * np.ones(size, dtype=np.uint16)

    image = np.random.randint(0, 2**16, size=size, dtype=np.uint16)
    P = np.ones([3, ] + size, dtype=np.uint16)
    P[0] = image - 0.1 * image
    P[1] = image - 0.12 * image
    P[2] = image - 0.13 * image

    m, n = image.shape
    print(image)
    #print(test(image, g1, g2, g3, p1, p2, p3))
    t_i = time()
    print(pseudo(image.shape[0], image.shape[1], image, G, P))
    print(time() - t_i)

    image = np.random.randint(0, 2**16, size=size, dtype=np.uint16)
    t_i = time()
    res = np.empty((m, n))

    print(image.shape[0])
    print(pseudo(image.shape[0], image.shape[1], image, G, P))
    print(time() - t_i)
