import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Template Main Function')
    parser.add_argument('--input', type=str, help='input folder')
    parser.add_argument('--output', type=str, help='output folder')
    parser.add_argument('--vis', type=str, help='vis folder')
    parser.add_argument('--trainratio', type=float, help='trainratio')
    parser.add_argument('--validationratio', type=float, help='validationratio')
    parser.add_argument('--testratio', type=float, help='testratio')
    args = parser.parse_args()

    cmd = "CNN-trainvalidationtest.sh"  # modify
    lib_param = {}
    lib_param["--input"] = args.input
    lib_param["--output"] = args.output
    lib_param["--vis"] = args.vis
    lib_param["--trainratio"] = args.trainratio
    lib_param["--validationratio"] = args.validationratio
    lib_param["--testratio"] = args.testratio

    for k, v in lib_param.items():
        cmd = cmd + " " + str(k) + " " + str(v)
    
    cmd = "bash " + cmd
    print ("executing cmd: \n", cmd)
    os.system(cmd)  # modify
