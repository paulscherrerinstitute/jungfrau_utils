import json
import argparse
from sf_databuffer_writer.writer import get_data_from_buffer, write_data_to_file
import os
from time import time


def get_requests(audit_fname, step_name, tfrom, tto):
    # The filename can be replaced with an .err file.
    with open(audit_fname) as audit_trail:
        lines = audit_trail.readlines()

    reqs = []
    params = []
    start_reading = False

    for line in lines:
        if step_name != "":
            if line.find(step_name) == -1:
                start_reading = True
            else:
                start_reading = False
            
        if tfrom != "" and not start_reading:
            if line.find(tfrom) != -1:
                start_reading = True
                
        if not start_reading:
            continue

        timestamp = line[1:16]
        json_string = line[18:]
                                                                                                                                                                                        
        # This writing request can be manually submitted to the writer, to try writing the file manually.                                                                 
        writing_request = json.loads(json_string)                                                                                                                       
        data_api_request = json.loads(writing_request["data_api_request"])                                                                                                         
        
        par = json.loads(writing_request["parameters"]) 
        
        reqs.append(data_api_request)
        params.append(par)

        if tto != "":
            if line.find(tto) != -1:
                break

    
    return reqs, params


def write_file(audit_fname, step_name, tfrom, tto):
    # You should load this from the audit trail or from the .err file.
    reqs, params = get_requests(audit_fname, step_name, tfrom, tto)
    
    for rp in zip(reqs, params):
        req, par = rp
        #par["output_file"] = "/tmp/" + par["output_file"].split("/")[-1]
        req["channels"].append({'name': 'SLAAR21-LSCP1-FNS:CH3:VAL_GET'})
        req["channels"].append({'name': 'SLAAR21-LSCP1-LAS6991:CH3:2'})
        print("writing %s" % par)
        #print("req %s" % req)

        ti =  time()
        data = get_data_from_buffer(req)
        print("For file %s data retrieval took %.2f" % (step_name, time() - ti))
        ti = time()
        write_data_to_file(par, data)
        print("For file %s data writing took %.2f" % (step_name, time() - ti))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default="/var/log/sf_databuffer_audit.log", help="DataBuffer broker audit file, default: /var/log/sf_databuffer_audit.log")
    parser.add_argument("--stepname", default="", type=str, help="Name of the file to write / look for in the audit file, e.g. run035_Bi_time_scan_step0020")
    parser.add_argument("--from_time", type=str, default="", help="timestamp to look for in the audit file, e.g. 20180607-213328")
    parser.add_argument("--to_time", type=str, default="", help="timestamp to look for in the audit file, e.g. 20180607-213328")
    args = parser.parse_args()

    user_id = 17295

    os.setgid(user_id)
    os.setuid(user_id)
    write_file(args.audit, args.stepname, args.from_time, args.to_time)

