# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:24:33 2023

@author: ai598
"""

import control as ct
import numpy as np
import NonLinSystem as NLS
import matplotlib.pyplot as plt


sys = NLS.RawSystem()
# Simulation parameter
T = np.linspace(0,500,1000)
x0 = np.array([0,0,0,0,0,0])  # initial condition
inp = np.ones(T.shape)
U = [inp,inp,inp]
# input response without controller
t, y = ct.input_output_response(sys, T, U, x0)


# Plot
plt.figure(1)
plt.figure(1,figsize=[12.8,9.6])
plt.subplot(231)
plt.plot(t,y[0])
plt.subplot(232)
plt.plot(t,y[1])
plt.subplot(233)
plt.plot(t,y[2])
plt.subplot(234)
plt.plot(t,y[3])
plt.subplot(235)
plt.plot(t,y[4])
plt.subplot(236)
plt.plot(t,y[5])
plt.show()

#%%

# Simulation parameter
syspos = NLS.PosControlSystem()
nT = 100;
T = np.linspace(0,3,nT)
x0 = np.array([0,0,0,0,0,0])  # initial condition
inp = np.ones(T.shape)
U = [inp,inp,inp]
# input response without controller
t, y = ct.input_output_response(syspos, T, U, x0)

# Plot
plt.figure(2)
plt.figure(2,figsize=[12.8,9.6])
plt.subplot(311)
plt.plot(t,y[0])
plt.subplot(312)
plt.plot(t,y[1])
plt.subplot(313)
plt.plot(t,y[2])

#%% 
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
import command as command
import compare as compare



filename ='NewTuneMoveTrain24_3.sav'
trained_model = pickle.load(open(filename,'rb'))

VI = np.array([.1,.2,0,0,0,0])

VG = np.array([.88,.55,0,0,0,0])

OI = np.array([.2,.3,0,0,0,0])

OG = np.array([.70,.8,0,0,0,0])

InputArray = np.concatenate((VI, VG, OI, OG), axis=None)

#%%

new_predict = trained_model.predict(InputArray)










