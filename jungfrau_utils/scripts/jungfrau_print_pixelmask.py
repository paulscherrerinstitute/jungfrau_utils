#!/usr/bin/env python

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("-f", default="pedestal_res.h5", help="file with the pixel mask")
args = parser.parse_args()

if not (os.path.isfile(args.f) and os.access(args.f, os.R_OK)):
    print("File {} not found, exit".format(args.f))
    exit()

f = h5py.File(args.f,"r")

pixelMask = f["pixelMask"]

(sh_y,sh_x) = pixelMask.shape

for y in range(sh_y):
    for x in range(sh_x):
        if pixelMask[y][x] != 0:
            print("Bad pixel (y,x) ({} {}) : {}, {} ".format(y,x,pixelMask[y][x], [ (int(pixelMask[y][x]) >> i & 1) for i in range(10)]))

