(1) als.py has errors because of mkdir /tmp/... (mlflow.spark.log_model(als_model, "als-model"))

change als.py and train_keras.py -->

mlflow.spark.log_model(als_model, "als-model", dfs_tmpdir="/hdd2/feiyi/mlflow")
als_model = mlflow.spark.load_model(als_model_uri, dfs_tmpdir="/hdd2/feiyi/mlflow").stages[0]
filepath = "/tmp/ALS_checkpoint_weights.hdf5"  --> filepath = "/hdd2/feiyi/mlflow/tmp/ALS_checkpoint_weights.hdf5" 

(2) adding print in main.py for reproduce

_already_ran

(3) etl can rerun

if type(param_value) is int:
    param_value = str(param_value)

(4) keras explicitly say not using cache ...

use_cache=False

even matches, still rerun ...

_get_or_run("train_keras", keras_params, git_commit, use_cache=False)
