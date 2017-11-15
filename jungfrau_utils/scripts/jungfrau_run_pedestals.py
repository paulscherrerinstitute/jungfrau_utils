from datetime import datetime
from time import sleep
import argparse

from detector_integration_api import DetectorIntegrationClient


def reset_bits(client):
    client.set_detector_value("clearbit", "0x5d 12")
    client.set_detector_value("clearbit", "0x5d 13")
    client.set_detector_value("clearbit", "0x5d 0")


def main():
    date_string = datetime.now().strftime("%Y%m%d_%H%M")

    parser = argparse.ArgumentParser(description="Create a pedestal file for Jungrau")
    parser.add_argument("--api", default="http://sf-daq-1:10000")
    parser.add_argument("--filename", default="pedestal_%s.h5" % date_string, help="Output file name")
    parser.add_argument("--directory", default="/gpfs/sf-data/bernina/raw/p16582", help="Output directory")
    parser.add_argument("--uid", default=16582, help="User ID which needs to own the file", type=int)
    args = parser.parse_args()

    api_address = args.api
    client = DetectorIntegrationClient(api_address)

    client.get_status()

    print("Resetting gain bits on Jungfrau")
    reset_bits(client)
    

    writer_config = {"output_file": args.directory + "/" + args.filename, "process_uid": args.uid, "dataset_name": "jungfrau/data"}
    print(writer_config)
    detector_config = {"period": 0.01, "exptime": 0.001, "frames": 30000}
    backend_config = {"n_frames": 30000}

    client.reset()
    client.set_config(writer_config=writer_config, backend_config=backend_config, detector_config=detector_config)

    client.start()
    print("Taking data at G0")
    sleep(100)
    client.set_detector_value("setbit", "0x5d 12")
    print("Taking data at G1")
    sleep(100)
    client.set_detector_value("setbit", "0x5d 13")
    print("Taking data at G2")
    sleep(100)
    client.stop()
    client.reset()
    reset_bits()
    print("Done")

if __name__ == "__main__":
    main()
