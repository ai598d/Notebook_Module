import dependencies
from build_so_model import build_model
from build_so_model import train_model
import pandas as pd
import numpy as np



df = pd.read_csv('StaticMoveData2.csv')  # import data


# =============================================================================
# DATA PROCESSING
# =============================================================================

# Do random shuffle of rows/observations.
newdf=df.sample(frac=1)
newdf=newdf.reset_index(drop=True)

# seperate Targets and Inputs
Target = pd.DataFrame(newdf[['VM1X1', 'VM1X2','VM1X3', 'VM1X4', 'VM1X5', 'VM1X6', 'VM2X1', 'VM2X2', 'VM2X3', 'VM2X4','VM2X5', 'VM2X6']])
Input  = newdf.drop(columns=['VM1X1', 'VM1X2','VM1X3', 'VM1X4', 'VM1X5', 'VM1X6', 'VM2X1', 'VM2X2', 'VM2X3', 'VM2X4','VM2X5', 'VM2X6'])

# Drop zero values (we are only consedering state X1 and X2)
Input  = Input.drop(columns=['VIX3', 'VIX4','VIX5', 'VIX6','VGX3', 'VGX4','VGX5', 'VGX6' ])

# Convert to array
InputArray  = np.asarray(Input)
TargetArray = np.asarray(Target)

# Seperate individual targets

# x and y coordinates for Middle State 1
TargetM1x = TargetArray [:,0]
TargetM1y = TargetArray [:,1]

# x and y coordinates for Middle State 2
TargetM2x = TargetArray [:,6]
TargetM2y = TargetArray [:,7]

# just rename the array for readability
X_good = InputArray
Y_good1 = TargetM1x
Y_good2 = TargetM1y
Y_good3 = TargetM2x
Y_good4 = TargetM2y


# split total data into train:test by 50:50 ratio

ind1 = round(len(X_good)*.50)-1
ind2 = len(X_good)-1

train_data = X_good[0:ind1,:]
test_data = X_good[ind1+1:ind2 , :]


train_targets1 = Y_good1[0:ind1]
test_targets1 = Y_good1[ind1+1:ind2]

train_targets2 = Y_good2[0:ind1]
test_targets2 = Y_good2[ind1+1:ind2]

train_targets3 = Y_good3[0:ind1]
test_targets3 = Y_good3[ind1+1:ind2]

train_targets4 = Y_good4[0:ind1]
test_targets4 = Y_good4[ind1+1:ind2]

# =============================================================================
# SET KERAS TUNER INSTANCES
# =============================================================================
import keras_tuner

# set up Keras Tuners
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



# =============================================================================
# SEARCH TUNER & BUILD MODEL
# =============================================================================


# tune the tuner for best hyper-param (For Target 1)
tuner1.search(train_data, train_targets1, epochs=100, validation_data=(test_data, test_targets1))


# # tune the tuner for best hyper-param (For Target 2)
tuner2.search(train_data, train_targets2, epochs=100, validation_data=(test_data, test_targets2))

# # tune the tuner for best hyper-param (For Target 3)
tuner3.search(train_data, train_targets3, epochs=100, validation_data=(test_data, test_targets3))

# # tune the tuner for best hyper-param ((For Target 4)
tuner4.search(train_data, train_targets4, epochs=100, validation_data=(test_data, test_targets4))



# # Get the top 2 hyperparameters.
best_hps1 = tuner1.get_best_hyperparameters(5)
best_hps2 = tuner2.get_best_hyperparameters(5)
best_hps3 = tuner3.get_best_hyperparameters(5)
best_hps4 = tuner4.get_best_hyperparameters(5)


# # Build the model with the best hp.
model1 = build_model(best_hps1[0])
model2 = build_model(best_hps2[0])
model3 = build_model(best_hps3[0])
model4 = build_model(best_hps4[0])

# # =============================================================================
# # 
# # =============================================================================

all_mae_hist1 = train_model(train_data,train_targets1,model1,k=2,num_epochs = 100)
all_mae_hist2 = train_model(train_data,train_targets1,model2,k=2,num_epochs = 100)
all_mae_hist3 = train_model(train_data,train_targets1,model3,k=2,num_epochs = 100)
all_mae_hist4 = train_model(train_data,train_targets1,model4,k=2,num_epochs = 100)


# =============================================================================
# SAVE MODEL
# =============================================================================
model1.save('my_model1') # Save Model

model2.save('my_model2') # Save Model

model3.save('my_model3') # Save Model

model4.save('my_model4') # Save Model






