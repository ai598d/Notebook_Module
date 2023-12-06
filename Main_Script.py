

import os as os
#%%
# initialize working directory

path = "C:\\Users\\ai598\\Thesis\\Notebook_Modules"

os.chdir(path)

import dependencies
from build_so_model import build_model
from build_so_model import train_model
from build_so_model import train_model_FF
import pandas as pd
import numpy as np

from DataGen import Data_Array
from DataGen import Scale_Data
from DataGen import Filter_Good_Data
from DataGen import CSV_to_DataArray
from command import Cmove2
from compare import StaticCheckBad
#%%


myinput,mytarget = CSV_to_DataArray('TD_11252023.csv',array=True)

parts = 100
th = .1

good_input, good_target, count = Filter_Good_Data(myinput,mytarget,parts,th)


#%%

myinput = good_input
mytarget= good_target

inplength = len(good_input)
halflength = int(inplength/2)


Training_Input   = myinput[0:halflength]
Training_Target1 = mytarget[0:halflength,0]
Training_Target2 = mytarget[0:halflength,1]
Training_Target3 = mytarget[0:halflength,2]
Training_Target4 = mytarget[0:halflength,3]

Test_Input = myinput[halflength:inplength]
Test_Target1 = mytarget[halflength:inplength ,0]
Test_Target2 = mytarget[halflength:inplength ,1]
Test_Target3 = mytarget[halflength:inplength ,2]
Test_Target4 = mytarget[halflength:inplength ,3]

#%% ------------------------------------------------------------ Dont Run This Cell


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

#%%

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


#%%
# =============================================================================
# SEARCH TUNER & BUILD MODEL
# =============================================================================


# tune the tuner for best hyper-param (For Target 1)
tuner1.search(Training_Input, Training_Target1, epochs=100, validation_data=(Test_Input, Test_Target1))


# # tune the tuner for best hyper-param (For Target 2)
tuner2.search(Training_Input, Training_Target2, epochs=100, validation_data=(Test_Input, Test_Target2))

# # tune the tuner for best hyper-param (For Target 3)
tuner3.search(Training_Input, Training_Target3, epochs=100, validation_data=(Test_Input, Test_Target3))

# # tune the tuner for best hyper-param ((For Target 4)
tuner4.search(Training_Input, Training_Target4, epochs=100, validation_data=(Test_Input, Test_Target4))



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

#%%
model1,all_mae_hist1 = train_model_FF(Training_Input, Training_Target1,Test_Input,Test_Target1,num_epochs = 100, lr=.1)
model2,all_mae_hist2 = train_model_FF(Training_Input, Training_Target2,Test_Input,Test_Target2,num_epochs = 100, lr=.1)
model3,all_mae_hist3 = train_model_FF(Training_Input, Training_Target3,Test_Input,Test_Target3,num_epochs = 100, lr=.1)
model4,all_mae_hist4 = train_model_FF(Training_Input, Training_Target4,Test_Input,Test_Target4,num_epochs = 100, lr=.1)


# =============================================================================
# SAVE MODEL
# =============================================================================
model1.save('my_newFFmodel1') # Save Model

model2.save('my_newFFmodel2') # Save Model

model3.save('my_newFFmodel3') # Save Model

model4.save('my_newFFmodel4') # Save Model






