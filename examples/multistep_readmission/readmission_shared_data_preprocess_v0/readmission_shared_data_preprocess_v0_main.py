import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Template Main Function')
    parser.add_argument('--input', type=str, help='input folder')
    parser.add_argument('--output', type=str, help='output folder')
    parser.add_argument('--vis', type=str, help='vis folder')
    args = parser.parse_args()

    cmd = "readmission_shared_data_preprocess_trainvalidationtest.sh"  # modify
    lib_param = {}
    lib_param["--input"] = args.input
    lib_param["--output"] = args.output
    lib_param["--vis"] = args.vis

    for k, v in lib_param.items():
        cmd = cmd + " " + str(k) + " " + str(v)
    
    cmd = "bash " + cmd
    print ("executing cmd: \n", cmd)
    os.system(cmd)  # modify
