import os
import argparse
import click
import mlflow

@click.command()
@click.option("--input")
@click.option("--output")
@click.option("--vis")
@click.option("--trainratio", default=1.0, type=float)
@click.option("--validationratio", default=0.0, type=float)
@click.option("--testratio", default=0.0, type=float)
def readmission_CNN_v0(input, output, vis, trainratio, validationratio, testratio):
    with mlflow.start_run() as mlrun:
        cmd = "CNN-trainvalidationtest.sh"  # modify
        lib_param = {}
        lib_param["--input"] = input
        lib_param["--output"] = output
        lib_param["--vis"] = vis
        lib_param["--trainratio"] = trainratio
        lib_param["--validationratio"] = validationratio
        lib_param["--testratio"] = testratio

        for k, v in lib_param.items():
            cmd = cmd + " " + str(k) + " " + str(v)
        
        cmd = "cd readmission_CNN_v0/ &&  bash " + cmd + " && cd .."
        print ("executing cmd: \n", cmd)
        os.system(cmd)  # modify

if __name__ == '__main__':
    readmission_CNN_v0()