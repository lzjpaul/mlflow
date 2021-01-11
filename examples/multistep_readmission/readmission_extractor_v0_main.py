import os
import argparse
import click
import mlflow

@click.command()
@click.option("--input")
@click.option("--output")
@click.option("--vis")
def readmission_extractor_v0(input, output, vis):
    with mlflow.start_run() as mlrun:
        cmd = "readmission_extractor_trainvalidationtest.sh"  # modify
        lib_param = {}
        lib_param["--input"] = input
        lib_param["--output"] = output
        lib_param["--vis"] = vis

        for k, v in lib_param.items():
            cmd = cmd + " " + str(k) + " " + str(v)
        
        cmd = "cd readmission_extractor_v0/ && bash " + cmd + " && cd .."
        print ("executing cmd: \n", cmd)
        os.system(cmd)  # modify

if __name__ == '__main__':
    readmission_extractor_v0()