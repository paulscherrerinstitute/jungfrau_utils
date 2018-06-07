import json
import argparse
from sf_databuffer_writer.writer import get_data_from_buffer, write_data_to_file
import os
from time import time


def get_requests(audit_fname, step_name):
    # The filename can be replaced with an .err file.
    with open(audit_fname) as audit_trail:
        lines = audit_trail.readlines()

    reqs = []
    params = []
    for line in lines:
        if line.find(step_name) == -1:
            continue

        timestamp = line[1:16]
        json_string = line[18:]
                                                                                                                                                                                        
        # This writing request can be manually submitted to the writer, to try writing the file manually.                                                                 
        writing_request = json.loads(json_string)                                                                                                                       
        data_api_request = json.loads(writing_request["data_api_request"])                                                                                                         
        
        par = json.loads(writing_request["parameters"]) 
        
        reqs.append(data_api_request)
        params.append(par)
    
    return reqs, params


def write_file(audit_fname, step_name):
    # You should load this from the audit trail or from the .err file.
    reqs, params = get_requests(audit_fname, step_name)
    
    for rp in zip(reqs, params):
        req, par = rp
        #par["output_file"] = "/tmp/test2.h5"
        req["channels"].append({'name': 'SLAAR21-LSCP1-FNS:CH3:VAL_GET'})
        req["channels"].append({'name': 'SLAAR21-LSCP1-LAS6991:CH3:2'})
        print("writing %s" % par)
        #print("req %s" % req)

        ti =  time()
        data = get_data_from_buffer(req)
        print("For file %s data retrieval took %.2f" % (step_name, time() - ti))
        write_data_to_file(par, data)
        ti = time()
        print("For file %s data writing took %.2f" % (step_name, time() - ti))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default="/var/log/sf_databuffer_audit.log", help="DataBuffer broker audit file, default: /var/log/sf_databuffer_audit.log")
    parser.add_argument("stepname", type=str, help="Name of the file to write / look for in the audit file, e.g. run035_Bi_time_scan_step0020")
    args = parser.parse_args()

    user_id = 17295

    os.setgid(user_id)
    os.setuid(user_id)
    write_file(args.audit, args.stepname)

