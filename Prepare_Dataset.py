# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 23:31:42 2023

@author: ai598
"""

import os as os
#%%
# initialize working directory

path = "C:\\Users\\ai598\\Thesis\\Notebook_Modules"

os.chdir(path)

#%%

import tensorflow as tf
from keras.datasets import imdb  # import imdb data
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedKFold
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
#from keras.utils.np_utils import to_categorical
from keras import layers
from keras import models
import random as random
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import warnings
warnings.filterwarnings("ignore")

from DataGen import Data_Array
from DataGen import Scale_Data
from DataGen import CSV_to_DataArray
from command import Cmove2
from compare import StaticCheckBad

#%%

# Generate Dataset

BCrangeArray = np.array([[1,190], [300,500], [501,700],[801,999],[210,290],[300,500],[501,700],[701,799]]) # range 
data = Data_Array(BCrangeArray,200000)

#%%

# Plot data distro
plt.figure()
n = np.linspace(0,500,500)
plt.plot(n,data[0:500,0],'*')


#%%
scale_data = Scale_Data(data)


#%%
# Plot data distro
plt.figure()
n = np.linspace(0,500,500)
plt.plot(n,scale_data[0:500,0],'*')

#%%

datacolumns = np.array(["VIX1","VIX2", "VM1X1","VM1X2", "VM2X1","VM2X2","VGX1","VGX2","OIX1","OIX2","OM1X1","OM1X2","OM2X1","OM2X2","OGX1","OGX2"])

dataframe= pd.DataFrame(scale_data ,columns=datacolumns)

#%%
dataframe.to_csv('TD_11252023.csv', index=False) # save Train data as excel |MoveData#| |MoveTestData# |

#%%

rawinp,rawtarget = CSV_to_DataArray('TD_11252023.csv',array=True)


#%%

def Gen_MoveC2(rawinp,rawtarget,i,parts=100):
    '''
    

    Parameters
    ----------
    rawinp : numpy array
        input array of 6 inputs: VIX1,VIX2,VGX1,VGX2,OIX1,OIX2
    rawtarget : numpy array
        VM1X1,VM2X2,VM2X1,VM2X2.
    i : int
        observation index.
    parts : int, optional
        number of points in a trjactory.There will be 3 additional points. The default is 100.

    Returns
    -------
    Trajectory array.

    '''
    i=100
    I  = np.array([rawinp[i,0],rawinp[i,1]])
    M1 = np.array([rawtarget[i,0],rawtarget[i,1]])
    M2 = np.array([rawtarget[i,2],rawtarget[i,3]])
    G  = np.array([rawinp[i,2],rawinp[i,3]])
    testmove = Cmove2(I,M1,M2,G,100)
    
    return testmove

def Generate_All_MoveC2(rawinp,rawtarget,parts=100):
    '''
    

    Parameters
    ----------
    rawinp : numpy array
        input array of 6 inputs: VIX1,VIX2,VGX1,VGX2,OIX1,OIX2
    rawtarget : numpy array
        VM1X1,VM2X2,VM2X1,VM2X2.
    parts : int, optional
        number of points in a trjactory.There will be 3 additional points. The default is 100.

    Returns
    -------
    All trajectory array.

    '''
    observation = len(rawinp)
    trajectory = []
    
    k=0
    
    while(k<observation):
        trajectory.append(Gen_MoveC2(rawinp,rawtarget,k,parts))
        k=k+1
        
    
    
    return np.asarray(trajectory)


testmove = Generate_All_MoveC2(rawinp,rawtarget,100)

#%%
# Good data filter

def Filter_Good_Data(rawinp,rawtarget,parts=100,th=.1):
    '''
    

    Parameters
    ----------
    rawinp : numpy array
        DESCRIPTION.
    rawtarget : numpy array
        DESCRIPTION.
    parts : int, optional
        DESCRIPTION. The default is 100.
    th : float, optional
        DESCRIPTION. The default is .1. must be 0<th<1

    Returns
    -------
    training_data_input : numpy array
        DESCRIPTION.
    taining_data_target : numpy array
        DESCRIPTION.
    count : int
        DESCRIPTION.

    '''
    

    Vmove = Generate_All_MoveC2(rawinp,rawtarget,parts)
    Opos= rawinp[:,4:6]
    count,index = StaticCheckBad(Vmove,Opos,th)
    training_data_input = np.delete(rawinp,index.astype(np.int64),0)
    taining_data_target = np.delete(rawtarget,index.astype(np.int64),0)

    

    return training_data_input  , taining_data_target , count


A,B,C = Filter_Good_Data(rawinp,rawtarget,100,.1)
    
#%%

Opos= rawinp[:,4:6]
count,index = StaticCheckBad(testmove,Opos,.1)
#%%
training_data_input = np.delete(rawinp,index.astype(np.int64),0)
taining_data_target = np.delete(rawtarget,index.astype(np.int64),0)

#%%
good_data = np.concatenate( (Good_data,Good_pos_data),axis=1)

#%%
plt.plot(testmove[0,:],testmove[1,:],'*')


