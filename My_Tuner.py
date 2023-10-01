# set up Keras Tuners
import keras_tuner
from   build_so_model import build_model

tuner1 = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="mae",
    max_trials=1,
    executions_per_trial=1,
    overwrite=True,
    directory="my_dir",
    project_name="Regression",
)

tuner2 = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="mae",
    max_trials=1,
    executions_per_trial=1,
    overwrite=True,
    directory="my_dir",
    project_name="Regression",
)

tuner3 = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="mae",
    max_trials=1,
    executions_per_trial=1,
    overwrite=True,
    directory="my_dir",
    project_name="Regression",
)

tuner4 = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="mae",
    max_trials=1,
    executions_per_trial=1,
    overwrite=True,
    directory="my_dir",
    project_name="Regression",
)