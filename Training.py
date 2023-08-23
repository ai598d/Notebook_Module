# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:01:42 2023

@author: ai598
"""

# %% ************************************************** Required Modules

import tensorflow as tf
from keras.datasets import imdb  # import imdb data 
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedKFold
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras import layers
from keras import models
import random as random
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import warnings
warnings.filterwarnings("ignore")
import keras_tuner
from tensorflow import keras
from keras import initializers 
from keras.layers import Activation, Dense


# ************************************************************************************

# Set Global Seed
tf.keras.utils.set_random_seed(1337) 

df = pd.read_csv('DataSet\MoveData1.csv') 


# drop unwanted columns and do random shuffle of rows. 
newdf=df.sample(frac=1)
newdf=newdf.reset_index(drop=True)
newdf = newdf.drop(columns=['OM1X1', 'OM1X2',
       'OM1X3', 'OM1X4', 'OM1X5', 'OM1X6', 'OM2X1', 'OM2X2', 'OM2X3', 'OM2X4',
       'OM2X5', 'OM2X6'])


#seperate label column from data
Label = pd.DataFrame(newdf['Label'])
newdf = newdf.drop(columns=['Label'])

# seperate targets
Target = pd.DataFrame(newdf[['VM1X1', 'VM1X2','VM1X3', 'VM1X4', 'VM1X5', 'VM1X6', 'VM2X1', 'VM2X2', 'VM2X3', 'VM2X4','VM2X5', 'VM2X6']])
Input  = newdf.drop(columns=['VM1X1', 'VM1X2','VM1X3', 'VM1X4', 'VM1X5', 'VM1X6', 'VM2X1', 'VM2X2', 'VM2X3', 'VM2X4','VM2X5', 'VM2X6'])

# Variable Reduction
Input  = Input.drop(columns = ['VIX3','VIX4','VIX5','VIX6'])
Input  = Input.drop(columns = ['VGX3','VGX4','VGX5','VGX6'])
Input  = Input.drop(columns = ['OIX3','OIX4','OIX5','OIX6'])
Input  = Input.drop(columns = ['OGX3','OGX4','OGX5','OGX6'])
Target = Target.drop(columns = ['VM1X3', 'VM1X4', 'VM1X5', 'VM1X6', 'VM2X3', 'VM2X4','VM2X5', 'VM2X6'])

#%%
# Convert dataframe to numpy array
InputArray  = np.asarray(Input)
TargetArray = np.asarray(Target)
LabelArray  = np.asarray(Label) 

#%%
# separate good bad
badcount = 26522
goodcount = InputArray.shape[0]-badcount

GoodInput  = np.zeros([goodcount,InputArray.shape[1]])
GoodTarget = np.zeros([goodcount,Target.shape[1]])

row = InputArray.shape[0]
i = 0
m = 0
while(i<row):
  if(LabelArray[i]==1):
    GoodInput[m] = InputArray[i]
    GoodTarget[m] = TargetArray[i]
    m=m+1
    i=i+1
  else:
    i=i+1

X_good = GoodInput
Y_good = GoodTarget




def build_model(hp):
  my_init = initializers.GlorotUniform()
  model = keras.Sequential()
  #model.add(layers.Dense(50, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
  #model.add(keras.layers.Flatten(input_shape=(InputArray.shape[1],1)))
  model.add(Dense(512, activation = 'relu', input_shape = (InputArray.shape[1],), 
   kernel_initializer = my_init))
  hp_units = hp.Int('units', min_value=32, max_value=512, step=8)
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(n_outputs, activation='softmax'))
 

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  #hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
  hp_learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-0, sampling="log")
  
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss='mse',
                metrics=['accuracy'])
  return model



n_inputs, n_outputs = X_good.shape[1], Y_good.shape[1]
tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    overwrite=True,
    objective='val_accuracy',
    max_trials=4,
    executions_per_trial=2,
    project_name="Navigation")




#stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(X_good, Y_good, epochs=50, validation_split=0.2) #callbacks=[stop_early]
best_hp = tuner.get_best_hyperparameters()[0]
best_model = tuner.hypermodel.build(best_hp)
#best_model = tuner.get_best_models(num_models=1)[0]


best_model.fit(X_good, Y_good,verbose=0, epochs=50)

# Save the model
filename = 'NewTuneMoveTrain24_4test.sav'
pickle.dump(best_model,open(filename,'wb'))
























#"C:\Users\ai598\Thesis\DataSet\MoveData1.csv"