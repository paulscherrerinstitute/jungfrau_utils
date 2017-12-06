import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py


def h5_printname(name):
    print("  {}".format(name))


def forcedGainValue(i, n0, n1, n2, n3):
    if i <= n0 - 1:
        return 0
    if i <= (n0 + n1) - 1:
        return 1
    if i <= (n0 + n1 + n2) - 1:
        return 3
    if i <= (n0 + n1 + n2 + n3) - 1:
        return 4
    return 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", default="pedestal.h5", help="pedestal file")
    parser.add_argument("-N", type=int, default=-1, help="show frame number N and exit")
    parser.add_argument("-v", type=int, default=0, help="verbosity level (0 - silent)")
    parser.add_argument("-tX", type=int, default=0, help="x position of the test pixel")
    parser.add_argument("-tY", type=int, default=0, help="y position of the test pixel")
    parser.add_argument("-nFramesPede", type=int, default=1000, help="number of pedestal frames to average pedestal value")
    parser.add_argument("-numberGain0", type=int, default=0, help="force to treat pedestal run as first numberGain0 taken in gain0, then numberGain1 in gain1, and numberGain2 in gain2 and HG0")
    parser.add_argument("-numberGain1", type=int, default=0, help="force to treat pedestal run as first numberGain0 taken in gain0, then numberGain1 in gain1, and numberGain2 in gain2 and HG0")
    parser.add_argument("-numberGain2", type=int, default=0, help="force to treat pedestal run as first numberGain0 taken in gain0, then numberGain1 in gain1, and numberGain2 in gain2 and HG0")
    parser.add_argument("-numberGainH0", type=int, default=0, help="force to treat pedestal run as first numberGain0 taken in gain0, then numberGain1 in gain1, and numberGain2 in gain2 and Hg0") 
    parser.add_argument("-totalFrames", type=int, default=1000000, help="analyze only first TOTALFRAMES frame")
    parser.add_argument("-pedestalFrames", type=int, default=1000, help="for pedestal in each gain average over last PEDESTALFRAMES frames, reducing weight of previous")
    parser.add_argument("-o", default="./", help="Output directory where to store pixelmask and gain file")
    parser.add_argument("-gainModule", type=int, default=1, help="check that gain setting in each of the module correspnds to the general gain switch, (0 - dont check)")
    parser.add_argument("-showPixelMask", type=int, default=0, help=">0 - show pixel mask image at the end of the run (default: not)")
    parser.add_argument("-nBadModules", type=int, default=0, help="Number of bad modules (default 0)")
    args = parser.parse_args()

    if not (os.path.isfile(args.f) and os.access(args.f, os.R_OK)):
        print("Pedestal file {} not found, exit".format(args.f))
        exit()

    overwriteGain = False
    if (args.numberGain0 + args.numberGain1 + args.numberGain2) > 0:
        if args.v >= 1:
            print("Treat this run as taken with {} frames in gain0, then {} frames in gain1 and {} frames in gain2".format(args.numberGain0, args.numberGain1, args.numberGain2))
        overwriteGain = True

    f = h5py.File(args.f, "r")

    numberOfFrames = len(f["jungfrau/data"])
    (sh_y, sh_x) = f["jungfrau/data"][0].shape
    nModules = (sh_x * sh_y) // (1024 * 512)
    if (nModules * 1024 * 512) != (sh_x * sh_y):
        print("Something very strange in the data, Jungfrau consists of (1024x512) modules, while data has {}x{}".format(sh_x, sh_y))
        exit()

    (tX, tY) = (args.tX, args.tY)
    if tX < 0 or tX > (sh_x - 1):
        tX = 0
    if tY < 0 or tY > (sh_y - 1):
        tY = 0

    if args.v >= 4:
        print("test pixel is (xy): {}x{}".format(tX, tY))

    if args.v >= 2:
        print("In pedestal file {} there are {} frames".format(args.f, numberOfFrames + 1))
    if args.v >= 3:
        print("Following groups are available:")
        f.visit(h5_printname)
        print("    data has the following shape: {}, type: {}, {} modules".format(f["jungfrau/data"][0].shape, f["jungfrau/data"][0].dtype, nModules))

    if args.N != -1:
        frameToShow = args.N
        if frameToShow <= numberOfFrames and frameToShow >= 0:
            print("Show frame number {}".format(frameToShow))
            frameData = np.bitwise_and(f["jungfrau/data"][frameToShow], 0b0011111111111111)
            gainData = np.bitwise_and(f["jungfrau/data"][frameToShow], 0b1100000000000000) >> 14
            print("Number of channels in gain0 : {}; gain1 : {}; gain2 : {}; undefined gain : {}".format(np.sum(gainData == 0), np.sum(gainData == 1), np.sum(gainData == 3), np.sum(gainData == 2)))
            plt.imshow(frameData, vmax=25000, origin='lower')
            plt.colorbar()
            plt.show()
        else:
            print("You requested to show frame number {}, but valid number for this pedestal run a 0-{}".format(frameToShow, numberOfFrames))
            exit()

    pixelMask = np.zeros((sh_y, sh_x), dtype=np.int)

    adcValuesN = [np.zeros((sh_y, sh_x)), np.zeros((sh_y, sh_x)), np.zeros((sh_y, sh_x)), np.zeros((sh_y, sh_x)), np.zeros((sh_y, sh_x))]
    adcValuesNN = [np.zeros((sh_y, sh_x)), np.zeros((sh_y, sh_x)), np.zeros((sh_y, sh_x)), np.zeros((sh_y, sh_x)), np.zeros((sh_y, sh_x))]

    averagePedestalFrames = args.pedestalFrames

    nMgain = [0] * 5

    gainCheck = -1
    printFalseGain = False
    nGoodFrames = 0
    nGoodFramesGain = 0

    analyzeFrames = min(numberOfFrames, args.totalFrames)
    for n in range(analyzeFrames):

        if not f["jungfrau/is_good_frame"][n]:
            continue

        nGoodFrames += 1

        daq_rec = f["jungfrau/daq_rec"][n]

        frameData = (np.bitwise_and(f["jungfrau/data"][n], 0b0011111111111111)).astype(np.float64)  # without cast can't use easily self multiplication
        gainData = np.bitwise_and(f["jungfrau/data"][n], 0b1100000000000000) >> 14
        trueGain = forcedGainValue(n, args.numberGain0, args.numberGain1, args.numberGain2, args.numberGainH0) if overwriteGain else ( (daq_rec & 0b11000000000000) >> 12 )
        highG0 = (daq_rec & 0b1)

        gainGoodAllModules = True
        if args.gainModule > 0:
            daq_recs = f["jungfrau/daq_recs"][n]
            for i in range(len(daq_recs)):
                if trueGain != ((daq_recs[i] & 0b11000000000000) >> 12) or highG0 != (daq_recs[i] & 0b1):
                    gainGoodAllModules = False

        nFramesGain = np.sum(gainData==(trueGain))
        if nFramesGain < (nModules-0.5-args.nBadModules)*(1024*512):  # make sure that most are the modules are in correct gain 
            gainGoodAllModules = False
            if args.v >= 3:
                print("Too many bad pixels, skip the frame {}, true gain: {} ({});  gain0 : {}; gain1 : {}; gain2 : {}; undefined gain : {}".format(n, trueGain, nFramesGain, np.sum(gainData==0),np.sum(gainData==1),np.sum(gainData==3),np.sum(gainData==2)))

        if not gainGoodAllModules:
            if args.v >= 3:
                print("In Frame Number {} gain mismatch in modules and general settings, {} vs {} (or too many bad pixels)".format(n, trueGain, ((daq_recs & 0b11000000000000) >> 12)))
            continue
        nGoodFramesGain += 1

        if args.v >= 1:
            if gainData[tY][tX] != trueGain:
                if not printFalseGain:
                    print("Gain wrong for channel ({}x{}) should be {}, but {}. Frame {}. {} {}".format(tX, tY, trueGain, gainData[tY][tX], n, trueGain, daq_rec))
                    printFalseGain = True
            else:
                if gainCheck != -1 and printFalseGain:
                    print("Gain was wrong for channel ({}x{}) in previous frames, but now correct : {}. Frame {}.".format(tX, tY, gainData[tY, tX], n))
                printFalseGain = False

            if gainData[tY][tX] != gainCheck:
                print("Gain changed for ({}x{}) channel {} -> {}, frame number {}, match: {}".format(tX, tY, gainCheck, gainData[tY][tX], n, gainData[tY][tX] == trueGain))
                gainCheck = gainData[tY][tX]

        if gainGoodAllModules:
            pixelMask[gainData != trueGain] |= (1 << trueGain)

            trueGain += 4*highG0
            nMgain[trueGain] += 1

            if nMgain[trueGain] > averagePedestalFrames:
                adcValuesN[trueGain] -= adcValuesN[trueGain] / averagePedestalFrames
                adcValuesNN[trueGain] -= adcValuesNN[trueGain] / averagePedestalFrames

            adcValuesN[trueGain] += frameData
            adcValuesNN[trueGain] += frameData * frameData

    if args.v >= 1:
        print("{} frames analyzed, {} good frames, {} frames without gain mismatch. Gain frames distribution (0,1,2,3,HG0) : ({})".format(analyzeFrames, nGoodFrames, nGoodFramesGain, nMgain))

    fileNameIn = os.path.splitext(os.path.basename(args.f))[0]
    print("Output file with pedestal corrections in: %s" % (args.o + "/" + fileNameIn + "_res.h5"))
    outFile = h5py.File(args.o + "/" + fileNameIn + "_res.h5", "w")
    dset = outFile.create_dataset('pixel_mask', data=pixelMask)
    dset2 = outFile.create_dataset('pixelMask', data=pixelMask)

    gains = [None] * 4
    gainsRMS = [None] * 4

    for gain in range(5):
        numberFramesAverage = max(1, min(averagePedestalFrames, nMgain[gain]))
        mean = adcValuesN[gain] / numberFramesAverage
        mean2 = adcValuesNN[gain] / numberFramesAverage
        variance = mean2 - mean * mean
        stdDeviation = np.sqrt(variance)
        if args.v >= 3:
            print("gain {} values results (pixel ({},{}) : {} {}".format(gain, tY, tX, mean[tY][tX], stdDeviation[tY][tX]))
        if gain != 2:
            g = gain if gain < 3 else (gain-1)
            dset = outFile.create_dataset('gain' + str(g), data=mean)
            dset = outFile.create_dataset('gain' + str(g) + '_rms', data=stdDeviation)
            gains[g] = mean
            gainsRMS[g] = stdDeviation

    dset = outFile.create_dataset('gains', data=gains)
    dset = outFile.create_dataset('gainsRMS', data=gains)

    outFile.close()

    if args.showPixelMask > 0:
        plt.imshow(pixelMask, vmax=1.0, vmin=0.0, origin='lower')
        plt.show()

    if args.v >= 1:
        print("Number of good pixels: {} from {} in total ({} bad pixels)".format(np.sum(pixelMask == 0), sh_x * sh_y, (sh_x * sh_y - np.sum(pixelMask == 0))))


if __name__ == "__main__":
    main()
