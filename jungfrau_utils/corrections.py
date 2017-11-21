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
        g = ma.array(G[0], mask=m1, dtype=np.float32).filled(0) +  ma.array(G[1], mask=m2, dtype=np.float32).filled(0) + ma.array(G[2], mask=m3, dtype=np.float32).filled(0)
    else:
        g = np.ones(data.shape, dtype=np.float32)
    if P is not None:
        p = ma.array(P[0], mask=m1, dtype=np.float32).filled(0) +         ma.array(P[1], mask=m2, dtype=np.float32).filled(0) +         ma.array(P[2], mask=m3, dtype=np.float32).filled(0)
    else:
        p = np.zeros(data.shape, dtype=np.float32)
    if pixel_mask is not None:
        data = ma.array(data, mask=pixel_mask, dtype=data.dtype).filled(0)

    res = np.divide(data - p, g)
    return res

try:
    from numba import jit

    @jit(nopython=True, nogil=True, cache=True)
    def apply_gain_pede_corrections_numba(m, n, image, G, P, mask, mask2, pede_mask, gain_mask):
        res = np.empty((m, n), dtype=np.float32)
        for i in range(m):
            for j in range(n):
                if pede_mask[i][j] != 0:
                    res[i][j] = 0
                    continue
                gm = gain_mask[i][j]
                if i==0 and j==0:
                    print(gm, image[i][j], P[gm][i][j], G[gm][i][j])
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

        return apply_gain_pede_corrections_numba(image.shape[0], image.shape[1], image, G, P, mask, mask2, pixel_mask, gain_mask)

    is_numba = True

except:
    print("Numba not available, reverting to Numpy")


def apply_gain_pede(image, G=None, P=None, pixel_mask=None):
    r"""A one-line summary that does not use variable names or the
    function name.
    Several sentences providing an extended description. Refer to
    variables using back-ticks, e.g. `var`.

    Parameters
    ----------
    var1 : array_like
        Descr
    Returns
    -------
    type
        Explanation of anonymous return value of type ``type``.
    describe : type
        Explanation of return value named `describe`.
    out : type
        Explanation of `out`.
    Other Parameters
    ----------------
    only_seldom_used_keywords : type
        Explanation
    common_parameters_listed_above : type
        Explanation
    Raises
    ------
    BadException
        Because you shouldn't have done that.
    See Also
    --------
    otherfunc : relationship (optional)
    newfunc : Relationship (optional), which could be fairly long, in which
              case the line wraps here.
    thirdfunc, fourthfunc, fifthfunc
    Notes
    -----
    Notes about the implementation algorithm (if needed).
    This can have multiple paragraphs.
    You may include some math:
    .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}
    And even use a greek symbol like :math:`omega` inline.
    References
    ----------
    Cite the relevant literature, e.g. [1]_.  You may also cite these
    references in the notes section above.
    .. [1] O. McNoleg, "The integration of GIS, remote sensing,
       expert systems and adaptive co-kriging for environmental habitat
       modelling of the Highland Haggis using object-oriented, fuzzy-logic
       and neural-network techniques," Computers & Geosciences, vol. 22,
       pp. 585-588, 1996.
    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.
    >>> a = [1, 2, 3]
    >>> print [x + 3 for x in a]
    [4, 5, 6]
    >>> print "a\n\nb"
    a
    b
    """
    if is_numba:
        return apply_gain_pede_numba(image, G=G, P=P, pixel_mask=pixel_mask)
    return apply_gain_pede_np(image, G=G, P=P, pixel_mask=pixel_mask)


def test():
    data = np.random.randint(0, 60000, size=[1500, 1000], dtype=np.uint16)
    pede = 60000 * np.random.random(size=[3, 1500, 1000])
    gain = 100 * np.random.random(size=[3, 1500, 1000])
    gain[gain>1] = 3

    t_i = time()
    res1 = apply_gain_pede_np(data, gain, pede)
    print(time() - t_i)
    t_i = time()
    res2 = apply_gain_pede_numba(data, gain, pede)    
    print(time() - t_i)
    t_i = time()
    res2 = apply_gain_pede(data, gain, pede)    
    print(time() - t_i)
    #print((res1 - res2 < 0.01).all())
    #print(res1[0:2, 0:2], res2[0:2, 0:2])


if __name__ == "__main__":
    test()
