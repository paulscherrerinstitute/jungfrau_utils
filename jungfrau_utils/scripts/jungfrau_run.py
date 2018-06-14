from datetime import datetime
from time import sleep
import argparse
import os
import subprocess

from detector_integration_api import DetectorIntegrationClient


def reset_bits(client):
    sleep(1)
    print(client.set_detector_value("clearbit", "0x5d 0"))
    sleep(1)
    print(client.set_detector_value("clearbit", "0x5d 12"))
    sleep(1)
    print(client.set_detector_value("clearbit", "0x5d 13"))
    sleep(1)


def run_jungfrau(n_frames, save=True, exptime=0.000010, outfile="", outdir="", uid=16852, api_address="http://sf-daq-1:10001", gain_filename="", pede_filename="", is_HG0=False, instrument=""):  # caput=False):
    client = DetectorIntegrationClient(api_address)

    client.get_status()

    #print("Resetting gain bits on Jungfrau")
    #reset_bits(client)

    writer_config = {"output_file": outdir + "/" + outfile,
                     "user_id": uid,
                     "n_frames": n_frames,
                     "general/user": str(uid),
                     "general/process": __name__,
                     "general/created": str(datetime.now()),
                     "general/instrument": instrument
                     }

    if not save:
        writer_config["output_file"] = "/dev/null"

    detector_config = {"exptime": exptime,
                       "frames": 1,
                       'cycles': n_frames,
                       "timing": "trigger",
                       "dr": 16
                       }

    backend_config = {"n_frames": n_frames,
                      "bit_depth": 16
                      }

    bsread_config = {'output_file': outdir + "/BSREAD_" + outfile,
                     'user_id': uid,
                     "general/user": str(uid),
                     "general/process": __name__,
                     "general/created": str(datetime.now()),
                     "general/instrument": instrument
                     }

    if gain_filename != "" or pede_filename != "":
        backend_config["gain_corrections_filename"] = gain_filename
        backend_config["gain_corrections_dataset"] = "gains"
        backend_config["pede_corrections_filename"] = pede_filename
        backend_config["pede_corrections_dataset"] = "gains"
        backend_config["pede_mask_dataset"] = "pixel_mask"
        backend_config["activate_corrections_preview"] = True
        print("Corrections in online viewer activated")
    
    if is_HG0:
        backend_config["is_HG0"] = True
        detector_config["setbit"] = "0x5d 0"
    else:
        print(client.set_detector_value("clearbit", "0x5d 0"))

    try:
        client.reset()
        
        configuration = {"writer": writer_config,
                         "backend": backend_config,
                         "detector": detector_config,
                         "bsread": bsread_config}

        client.set_config(configuration)

        print(client.get_config())

        print("Starting acquisition")
        client.start()

        try:
            client.wait_for_status(["IntegrationStatus.FINISHED"], polling_interval=0.1)
        except:
            print("Got IntegrationStatus ERROR")
            print(client.get_status())
            print(client.get_status_details())

        print("Stopping acquisition")
        client.reset()

        print("Done")
    except KeyboardInterrupt:
        print("Caught CTRL-C, resetting")
        client.reset()

def main():
    
    date_string = datetime.now().strftime("%Y%m%d_%H%M")

    parser = argparse.ArgumentParser(description="Create a pedestal file for Jungrau")
    parser.add_argument("--api", default="http://sf-daq-1:10000", required=True)
    parser.add_argument("--filename", default="run_%s.h5" % date_string, help="Output file name")
    parser.add_argument("--pede", default="", help="File containing pedestal corrections")
    parser.add_argument("--gain", default="", help="File containing gain corrections")
    parser.add_argument("--directory", default="/sf/bernina/data/raw/p16582", help="Output directory")
    parser.add_argument("--uid", default=16582, help="User ID which needs to own the file", type=int)
    parser.add_argument("--period", default=0.01, help="Period (default is 10Hz - 0.01)", type=float)
    parser.add_argument("--exptime", default=0.000010, help="Integration time (default 0.000010 - 10us)", type=float)
    parser.add_argument("--frames", default=10, help="Number of frames to take", type=int)
    parser.add_argument("--save", default=False, help="Save data file", action="store_true")
    parser.add_argument("--highgain", default=False, help="Enable High Gain (HG0)", action="store_true")
    parser.add_argument("--instrument", default="", help="Name of the instrument, e.g. Alvra")
    # parser.add_argument("--caput", default=False, help="Use the CAPUT trick (experts only!!!)", action="store_true")
    
    args = parser.parse_args()

    run_jungfrau(args.frames, args.save, args.exptime, outfile=args.filename, outdir=args.directory, uid=args.uid, api_address=args.api, gain_filename=args.gain, pede_filename=args.pede, is_HG0=args.highgain, instrument=args.instrument)

    
if __name__ == "__main__":
    main()
