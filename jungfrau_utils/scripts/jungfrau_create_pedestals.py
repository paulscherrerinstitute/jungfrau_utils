import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import logging

ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))

log = logging.getLogger("create_pedestals")
log.addHandler(ch)


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
    parser.add_argument("--filename", default="pedestal.h5", help="pedestal file")
    parser.add_argument("--frame_number", type=int, default=-1, help="show frame number N and exit")
    parser.add_argument("--verbosity", type=int, default=1, help="verbosity level (0 - silent, 2 - DEBUG, default=1)")
    parser.add_argument("--X_test_pixel", type=int, default=0, help="x position of the test pixel")
    parser.add_argument("--Y_test_pixel", type=int, default=0, help="y position of the test pixel")
    parser.add_argument("--nFramesPede", type=int, default=1000, help="number of pedestal frames to average pedestal value")
    parser.add_argument("--frames_G0", type=int, default=0, help="force to treat pedestal run as first frames_G0 taken in gain0, then frames_G1 in gain1, and frames_G2 in gain2 and HG0")
    parser.add_argument("--frames_G1", type=int, default=0, help="force to treat pedestal run as first frames_G0 taken in gain0, then frames_G1 in gain1, and frames_G2 in gain2 and HG0")
    parser.add_argument("--frames_G2", type=int, default=0, help="force to treat pedestal run as first frames_G0 taken in gain0, then frames_G1 in gain1, and frames_G2 in gain2 and HG0")
    parser.add_argument("--frames_HG0", type=int, default=0, help="force to treat pedestal run as first frames_G0 taken in gain0, then frames_G1 in gain1, and frames_G2 in gain2 and HG0") 
    parser.add_argument("--number_frames", type=int, default=1000000, help="analyze only first number_frames frames")
    parser.add_argument("--frames_average", type=int, default=1000, help="for pedestal in each gain average over last frames_average frames, reducing weight of previous")
    parser.add_argument("--directory", default="./", help="Output directory where to store pixelmask and gain file")
    parser.add_argument("--gain_check", type=int, default=1, help="check that gain setting in each of the module corresponds to the general gain switch, (0 - dont check)")
    parser.add_argument("--show_pixel_mask", type=int, default=0, help=">0 - show pixel mask image at the end of the run (default: not)")
    parser.add_argument("--number_bad_modules", type=int, default=0, help="Number of bad modules (default 0)")
    parser.add_argument("--jungfrau_name", type=str, default="JF", help="Name of the jungfrau detector, e.g. JF_1.5M")
    args = parser.parse_args()

    if not (os.path.isfile(args.filename) and os.access(args.filename, os.R_OK)):
        print("Pedestal file {} not found, exit".format(args.filename))
        exit()

    log.setLevel(args.verbosity)

    overwriteGain = False
    if (args.frames_G0 + args.frames_G1 + args.frames_G2) > 0:
        log.info("Treat this run as taken with {} frames in gain0, then {} frames in gain1 and {} frames in gain2".format(args.frames_G0, args.frames_G1, args.frames_G2))
        overwriteGain = True

    f = h5py.File(args.filename, "r")

    data_location = "data/" + args.jungfrau_name + "/data"
    daq_recs_location = "data/" + args.jungfrau_name + "/daq_rec"
    is_good_frame_location = "data/" + args.jungfrau_name + "/is_good_frame"

    numberOfFrames = len(f["data/" + args.jungfrau_name + "/data"])
    (sh_y, sh_x) = f[data_location][0].shape
    nModules = (sh_x * sh_y) // (1024 * 512)
    if (nModules * 1024 * 512) != (sh_x * sh_y):
        log.error("Something very strange in the data, Jungfrau consists of (1024x512) modules, while data has {}x{}".format(sh_x, sh_y))
        exit()

    (tX, tY) = (args.X_test_pixel, args.Y_test_pixel)
    if tX < 0 or tX > (sh_x - 1):
        tX = 0
    if tY < 0 or tY > (sh_y - 1):
        tY = 0

    log.debug("test pixel is ( x y ): {}x{}".format(tX, tY))
    log.info("In pedestal file {} there are {} frames".format(args.filename, numberOfFrames + 1))
    log.debug("Following groups are available:")
    if args.verbosity >= 3:
        f.visit(h5_printname)
    log.debug("    data has the following shape: {}, type: {}, {} modules".format(f["data/" + args.jungfrau_name + "/data"][0].shape, f["data/" + args.jungfrau_name + "/data"][0].dtype, nModules))

    if args.frame_number != -1:
        frameToShow = args.frame_number
        if frameToShow <= numberOfFrames and frameToShow >= 0:
            log.info("Show frame number {}".format(frameToShow))
            frameData = np.bitwise_and(f[data_location][frameToShow], 0b0011111111111111)
            gainData = np.bitwise_and(f[data_location][frameToShow], 0b1100000000000000) >> 14
            log.info("Number of channels in gain0 : {}; gain1 : {}; gain2 : {}; undefined gain : {}".format(np.sum(gainData == 0), np.sum(gainData == 1), np.sum(gainData == 3), np.sum(gainData == 2)))
            plt.imshow(frameData, vmax=25000, origin='lower')
            plt.colorbar()
            plt.show()
        else:
            log.error("You requested to show frame number {}, but valid number for this pedestal run a 0-{}".format(frameToShow, numberOfFrames))
            exit()

    pixelMask = np.zeros((sh_y, sh_x), dtype=np.int)

    adcValuesN = np.zeros((5, sh_y, sh_x))
    adcValuesNN = np.zeros((5, sh_y, sh_x))


    averagePedestalFrames = args.frames_average

    nMgain = [0] * 5

    gainCheck = -1
    highG0Check = 0
    printFalseGain = False
    nGoodFrames = 0
    nGoodFramesGain = 0

    analyzeFrames = min(numberOfFrames, args.number_frames)

    for n in range(analyzeFrames):

        if not f[is_good_frame_location][n]:
            continue

        nGoodFrames += 1

        daq_rec = (f[daq_recs_location][n])[0]

        image = f[data_location][n][:]
        frameData = (np.bitwise_and(image, 0b0011111111111111))
        gainData = np.bitwise_and(image, 0b1100000000000000) >> 14
        trueGain = forcedGainValue(n, args.framesG0, args.framesG1, args.framesG2, args.framesHG0) if overwriteGain else ( (daq_rec & 0b11000000000000) >> 12 )
        highG0 = (daq_rec & 0b1)

        gainGoodAllModules = True
        if args.gain_check > 0:
            daq_recs = f[daq_recs_location][n]
            for i in range(len(daq_recs)):
                if trueGain != ((daq_recs[i] & 0b11000000000000) >> 12) or highG0 != (daq_recs[i] & 0b1):
                    gainGoodAllModules = False

        if highG0 == 1 and trueGain != 0:
            gainGoodAllModules = False
            log.info("Jungfrau is in the high G0 mode ({}), but gain settings is strange: {}".format( highG0, trueGain))

        nFramesGain = np.sum(gainData==(trueGain))
        if nFramesGain < (nModules - 0.5 - args.number_bad_modules) * (1024 * 512):  # make sure that most are the modules are in correct gain 
            gainGoodAllModules = False
            log.debug("Too many bad pixels, skip the frame {}, true gain: {}(highG0: {}) ({});  gain0 : {}; gain1 : {}; gain2 : {}; undefined gain : {}".format(n, trueGain, highG0, nFramesGain, np.sum(gainData==0), np.sum(gainData==1), np.sum(gainData==3), np.sum(gainData==2)))

        if not gainGoodAllModules:
            log.debug("In Frame Number {} : mismatch in modules and general settings, Gain: {} vs {}; HighG0: {} vs {} (or too many bad pixels)".format(n, trueGain, ((daq_recs & 0b11000000000000) >> 12), highG0, (daq_recs & 0b1)))
            continue
        nGoodFramesGain += 1

        if gainData[tY][tX] != trueGain:
            if not printFalseGain:
                log.info("Gain wrong for channel ({}x{}) should be {}, but {}. Frame {}. {} {}".format(tX, tY, trueGain, gainData[tY][tX], n, trueGain, daq_rec))
                printFalseGain = True
        else:
            if gainCheck != -1 and printFalseGain:
                log.info("Gain was wrong for channel ({}x{}) in previous frames, but now correct : {}. Frame {}.".format(tX, tY, gainData[tY, tX], n))
            printFalseGain = False

        if gainData[tY][tX] != gainCheck or highG0Check != highG0:
            log.info("Gain changed for ({}x{}) channel {} -> {} (highG0 setting: {} -> {}), frame number {}, match: {}".format(tX, tY, gainCheck, gainData[tY][tX], highG0Check, highG0, n, gainData[tY][tX] == trueGain))
            gainCheck = gainData[tY][tX]
            highG0Check = highG0

        if gainGoodAllModules:

            pixelMask[gainData != trueGain] |= (1 << (trueGain+4*highG0))

            trueGain += 4 * highG0
        

            nMgain[trueGain] += 1

            if nMgain[trueGain] > averagePedestalFrames:
                adcValuesN[trueGain] -= adcValuesN[trueGain] / averagePedestalFrames
                adcValuesNN[trueGain] -= adcValuesNN[trueGain] / averagePedestalFrames

            adcValuesN[trueGain] += frameData
            adcValuesNN[trueGain] += np.float_power(frameData, 2)


    log.info("{} frames analyzed, {} good frames, {} frames without settings mismatch. Gain frames distribution (0,1,2,3,HG0) : ({})".format(analyzeFrames, nGoodFrames, nGoodFramesGain, nMgain))

    fileNameIn = os.path.splitext(os.path.basename(args.filename))[0]
    full_fileNameOut = args.directory + "/" + fileNameIn + "_res.h5"
    log.info("Output file with pedestal corrections in: %s" % full_fileNameOut)
    outFile = h5py.File(full_fileNameOut, "w")
    dset = outFile.create_dataset('pixel_mask', data=pixelMask)

    gains = [None] * 4
    gainsRMS = [None] * 4

    for gain in range(5):
        numberFramesAverage = max(1, min(averagePedestalFrames, nMgain[gain]))
        mean = adcValuesN[gain] / float(numberFramesAverage)
        mean2 = adcValuesNN[gain] / float(numberFramesAverage)
        variance = mean2 - np.float_power(mean, 2)
        stdDeviation = np.sqrt(variance)
        log.debug("gain {} values results (pixel ({},{}) : {} {}".format(gain, tY, tX, mean[tY][tX], stdDeviation[tY][tX]))
        if gain != 2:
            g = gain if gain < 3 else (gain-1)
            dset = outFile.create_dataset('gain' + str(g), data=mean)
            dset = outFile.create_dataset('gain' + str(g) + '_rms', data=stdDeviation)
            gains[g] = mean
            gainsRMS[g] = stdDeviation

    dset = outFile.create_dataset('gains', data=gains)
    dset = outFile.create_dataset('gainsRMS', data=gains)

    outFile.close()

    if args.show_pixel_mask > 0:
        plt.imshow(pixelMask, vmax=1.0, vmin=0.0, origin='lower')
        plt.show()

    log.info("Number of good pixels: {} from {} in total ({} bad pixels)".format(np.sum(pixelMask == 0), sh_x * sh_y, (sh_x * sh_y - np.sum(pixelMask == 0))))


if __name__ == "__main__":
    main()
