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
    args = parser.parse_args()

    api_address = args.api
    client = DetectorIntegrationClient(api_address)

    client.get_status()

    print("Resetting gain bits on Jungfrau")
    reset_bits(client)

    writer_config = {"output_file": args.directory + "/" + args.filename, "process_uid": args.uid, "process_gid": args.uid, "dataset_name": "jungfrau/data", "disable_processing": False, "n_messages": args.numberFrames}
    print(writer_config)
    if args.trigger == 0:
        detector_config = {"period": args.period, "exptime": args.exptime, "frames": args.numberFrames}
    else:
        detector_config = {"period": args.period, "exptime": args.exptime, "frames": 1, 'cycles': args.numberFrames, "timing": "trigger"}
    backend_config = {"n_frames": args.numberFrames}

    bsread_config = {'output_file': "/dev/null", 'process_uid': args.uid, 'process_gid': args.uid, 'channels': []}

    client.reset()
    client.set_config(writer_config=writer_config, backend_config=backend_config, detector_config=detector_config, bsread_config=bsread_config)
    print(client.get_config())

    sleepTime = args.numberFrames * args.period / 4

    client.start()
    print("Taking data at G0")
    sleep(sleepTime)
    print(client.set_detector_value("setbit", "0x5d 12"))
    print("Taking data at G1")
    sleep(sleepTime)
    print(client.set_detector_value("setbit", "0x5d 13"))
    print("Taking data at G2")
    sleep(sleepTime)
    reset_bits(client)
    print(client.set_detector_value("setbit", "0x5d 1"))
    print("Taking data at HG0")
    sleep(sleepTime)
    client.stop()
    client.reset()
    reset_bits(client)

    print("Pedestal run data saved in %s" % writer_config["output_file"])
    if args.analyze:
        print("Running pedestal analysis, output file in %s", os.path.join(args.directory.replace("raw", "res"), "JF_pedestal"))
        subprocess.call(["jungfrau_create_pedestals", "-f", writer_config["output_file"], "-o", os.path.join(args.directory.replace("raw", "res"), "JF_pedestal"), "-v", "4"])
    print("Done")

    
if __name__ == "__main__":
    main()
