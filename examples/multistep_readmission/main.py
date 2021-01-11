"""
Downloads the MovieLens dataset, ETLs it into Parquet, trains an
ALS model, and uses the ALS model to train a Keras neural network.

See README.rst for more details.
"""

import click
import os


import mlflow
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint

from mlflow.tracking.fluent import _get_experiment_id


def _already_ran(entry_point_name, parameters, git_commit, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = mlflow.tracking.MlflowClient()
    all_run_infos = reversed(client.list_run_infos(experiment_id))
    print ("all_run_infos: ", all_run_infos)
    for run_info in all_run_infos:
        print ("run_info: ", run_info)
        full_run = client.get_run(run_info.run_id)
        tags = full_run.data.tags
        print ("lv check entry_point_name1: ", tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None))
        print ("lv check entry_point_name2: ", entry_point_name)
        print ("tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name: ", (tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name))
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        match_failed = False
        for param_key, param_value in parameters.items():
            run_value = full_run.data.params.get(param_key)
            print ("lv check param1: ", run_value)
            print ("lv check type(run_value): ", type(run_value))
            print ("lv check param2: ", param_value)
            print ("lv check type(param_value): ", type(param_value))
            print ("lv check run_value != param_value: ", (run_value != param_value))
            ### lv check, int and str
            if type(param_value) is int or type(param_value) is float:
                param_value = str(param_value)
            if run_value != param_value:
                match_failed = True
                break
        print ("lv check after param_value match_failed: ", match_failed)
        if match_failed:
            continue
        print ("lv check pass entry point and param_value")
        if run_info.to_proto().status != RunStatus.FINISHED:
            eprint(
                ("Run matched, but is not FINISHED, so skipping " "(run_id=%s, status=%s)")
                % (run_info.run_id, run_info.status)
            )
            continue

        previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        if git_commit != previous_version:
            eprint(
                (
                    "Run matched, but has a different source version, so skipping "
                    "(found=%s, expected=%s)"
                )
                % (previous_version, git_commit)
            )
            continue
        return client.get_run(run_info.run_id)
    eprint("No matching run has been found.")
    return None


# TODO(aaron): This is not great because it doesn't account for:
# - changes in code
# - changes in dependant steps
def _get_or_run(entrypoint, parameters, git_commit, use_cache=True):
    existing_run = _already_ran(entrypoint, parameters, git_commit)
    if use_cache and existing_run:
        print("Found existing run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
        return existing_run
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters, use_conda=False)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


def workflow():
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run() as active_run:
        # os.environ["SPARK_CONF_DIR"] = os.path.abspath(".")
        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
        readmission_dice_v0_input = "/hdd2/feiyi/mlflow/multistep_readmission/dataset_nuh_read_q5/demographic.csv,/hdd2/feiyi/mlflow/multistep_readmission/dataset_nuh_read_q5/cases.csv"
        readmission_dice_v0_output = "./output/"
        readmission_dice_v0_vis = "./vis/"
        print ("lv check readmission_dice_v0 _get_or_run")
        readmission_dice_v0_run = _get_or_run(
            "readmission_dice_v0", 
            {"input": readmission_dice_v0_input, "output": readmission_dice_v0_output, "vis": readmission_dice_v0_vis}, 
            git_commit
        )

        readmission_extractor_v0_input = "../readmission_dice_v0/output/"
        readmission_extractor_v0_output = "./output/"
        readmission_extractor_v0_vis = "./vis/"
        print ("lv check readmission_extractor_v0 _get_or_run")
        readmission_extractor_v0_run = _get_or_run(
            "readmission_extractor_v0", 
            {"input": readmission_extractor_v0_input, "output": readmission_extractor_v0_output, "vis": readmission_extractor_v0_vis}, 
            git_commit
        )
        
        readmission_shared_data_preprocess_v0_input = "../readmission_extractor_v0/output/"
        readmission_shared_data_preprocess_v0_output = "./output/"
        readmission_shared_data_preprocess_v0_vis = "./vis/"
        # We specify a spark-defaults.conf to override the default driver memory. ALS requires
        # significant memory. The driver memory property cannot be set by the application itself.
        print ("lv check readmission_shared_data_preprocess_v0 _get_or_run")
        readmission_shared_data_preprocess_v0_run = _get_or_run(
            "readmission_shared_data_preprocess_v0", 
            {"input": readmission_shared_data_preprocess_v0_input, "output": readmission_shared_data_preprocess_v0_output, "vis": readmission_shared_data_preprocess_v0_vis}, 
            git_commit
        )

        readmission_CNN_v0_input = "../readmission_shared_data_preprocess_v0/output/"
        readmission_CNN_v0_output = "./output/"
        readmission_CNN_v0_vis = "./vis/"
        readmission_CNN_v0_trainratio = 1.0
        readmission_CNN_v0_validationratio = 0.0
        readmission_CNN_v0_testratio = 0.0

        readmission_CNN_v0_params = {
            "input": readmission_CNN_v0_input,
            "output": readmission_CNN_v0_output,
            "vis": readmission_CNN_v0_vis,
            "trainratio": readmission_CNN_v0_trainratio,
            "validationratio": readmission_CNN_v0_validationratio,
            "testratio": readmission_CNN_v0_testratio,
        }
        print ("lv check readmission_CNN_v0 _get_or_run")
        _get_or_run("readmission_CNN_v0", readmission_CNN_v0_params, git_commit)
        # _get_or_run("readmission_CNN_v0", readmission_CNN_v0_params, git_commit, use_cache=False)


if __name__ == "__main__":
    workflow()
