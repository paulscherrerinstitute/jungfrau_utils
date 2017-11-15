import h5py
import numpy as np
import argparse
import os
from datetime import datetime


def merge_gainmaps(maps, shape, module_shape):
    if maps[0].shape != module_shape:
        print("[ERROR]: shape of the provided maps is not correct. Provided shape: %s, required shape: %s" % (maps[0].shape, module_shape))
    res = np.zeros([3, shape[0] * module_shape[1], shape[1] * module_shape[2]], dtype=np.float)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for z in range(module_shape[0]):
                ri = (i * module_shape[1], (i + 1) * module_shape[1])
                rj = (j * module_shape[2], (j + 1) * module_shape[2])
                res[z, ri[0]:ri[1], rj[0]:rj[1]] = maps[i + j][z]
    return res
