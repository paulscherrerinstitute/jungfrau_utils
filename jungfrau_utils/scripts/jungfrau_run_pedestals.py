from datetime import datetime
from time import sleep
import argparse
import os
import subprocess

from detector_integration_api import DetectorIntegrationClient


def reset_bits(client):
    sleep(0.1)
    print(client.set_detector_value("clearbit", "0x5d 0"))
    sleep(0.1)
    print(client.set_detector_value("clearbit", "0x5d 12"))
    sleep(0.1)
    print(client.set_detector_value("clearbit", "0x5d 13"))
    sleep(0.1)


def main():
    date_string = datetime.now().strftime("%Y%m%d_%H%M")

    parser = argparse.ArgumentParser(description="Create a pedestal file for Jungrau")
    parser.add_argument("--api", default="http://sf-daq-1:10000")
    parser.add_argument("--filename", default="pedestal_%s.h5" % date_string, help="Output file name")
    parser.add_argument("--directory", default="/sf/bernina/data/raw/p16582", help="Output directory")
    parser.add_argument("--uid", default=16582, help="User ID which needs to own the file", type=int)
    parser.add_argument("--period", default=0.01, help="Period (default is 10Hz - 0.01)", type=float)
    parser.add_argument("--exptime", default=0.000010, help="Integration time (default 0.000010 - 10us)", type=float)
    parser.add_argument("--numberFrames", default=10000, help="Integration time (default 10000)", type=int)
    parser.add_argument("--trigger", default=1, help="run with the trigger, PERIOD will be ignored in this case(default - 1(yes))", type=int)
    parser.add_argument("--analyze", default=False, help="Run the pedestal analysis (default False)", action="store_true")
    parser.add_argument("--nBadModules", default=0, help="Number of bad modules in the detector. Makes sense only together with --analyse (default 0)", 
                        action="store", type=int)
    args = parser.parse_args()

    api_address = args.api
    client = DetectorIntegrationClient(api_address)

    client.get_status()
    try:
        writer_config = {"output_file": args.directory + "/" + args.filename,
                         "user_id": args.uid,
                         "n_frames": args.numberFrames,
                         "general/user": str(args.uid),
                         "general/process": __name__,
                         "general/created": str(datetime.now()),
                         "general/instrument": "JF 4.5M"
                         }

        print(writer_config)
        if args.trigger == 0:
            detector_config = {"period": args.period,
                               "exptime": args.exptime,
                               "frames": args.numberFrames,
                               'cycles': 1}
        else:
            detector_config = {"period": args.period,
                               "exptime": args.exptime,
                               "frames": 1, 'cycles': args.numberFrames,
                               "timing": "trigger"
                               }

        backend_config = {"n_frames": args.numberFrames,
                          "bit_depth": 16
                          }

        bsread_config = {'output_file': "/dev/null",
                         'user_id': args.uid
                         }


        client.reset()

        configuration = {"writer": writer_config, "backend": backend_config,
                         "detector": detector_config, "bsread": bsread_config}

        client.set_config(configuration)
        print(client.get_config())

        sleepTime = args.numberFrames * args.period / 5

        print("Resetting gain bits on Jungfrau")
        reset_bits(client)

        print(client.set_detector_value("setbit", "0x5d 0"))
        sleep(5) # for the moment there is a delay to make sure detectory is in the highG0 mode
        print("Taking data at HG0")
        client.start()

        #subprocess.check_call(["caput", "SIN-TIMAST-TMA:Evt-24-Ena-Sel", "1"])
        sleep(sleepTime)

        print(client.set_detector_value("clearbit", "0x5d 0"))
        print("Taking data at G0")
        sleep(sleepTime)

        print(client.set_detector_value("setbit", "0x5d 12"))
        print("Taking data at G1")
        sleep(sleepTime)

        print(client.set_detector_value("setbit", "0x5d 13"))
        print("Taking data at G2")
        sleep(2 * sleepTime)

        #subprocess.check_call(["caput", "SIN-TIMAST-TMA:Evt-24-Ena-Sel", "0"])
        client.stop()
        client.reset()
        reset_bits(client)
    except KeyboardInterrupt:
        print("CTRL-C caught, stopping and resetting")
        client.stop()
        client.reset()
        reset_bits(client)

    print("Pedestal run data saved in %s" % writer_config["output_file"])
    if args.analyze:
        print("Running pedestal analysis, output file in %s", os.path.join(args.directory.replace("raw", "res"), ""))
        subprocess.call(["jungfrau_create_pedestals", "-f", writer_config["output_file"], "-o", os.path.join(args.directory.replace("raw", "res"), ""), "-v", "4", "-nBadModules", args.nBadModules])
    print("Done")

    
if __name__ == "__main__":
    main()
