# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 23:30:37 2023

@author: ai598
"""

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

from command import Cmove2
from compare import StaticCheckBad


def Data_Array(BCrange,MaxIt):

  '''
  Input Arguments:
  BCrange:8x2 array type.
          contains boundary conditions in x direction.
          [initial boundary , Goal boundary ]
          [VI,VM1,VM2,VG,OI,OM1,OM2,OG1]

  MaxIt:  int type.
          number of observations.

  Returns:


  '''

  DataArray = np.zeros([MaxIt,16])

  i = 0
  j = 0
  k = 0

  while(i<MaxIt):
    while(j<16):

      if(j%2==0): # This is for x coordinates
        DataArray[i,j] = np.random.randint(BCrange[k,0],BCrange[k,1])


        k=k+1
        j=j+1



        #if(j==(BCrange.shape[0]-1)):
          #k=0
          #j=j+1
        #else:
          #k=k+1
          #j=j+1



      else:# This is for y coordinates
        DataArray[i,j] = np.random.randint(0,BCrange.max())

        j=j+1

        #if(j==BCrange.shape[0]-1):
          #k=0
          #j=j+1
        #else:

          #j=j+1


    j=0
    k=0
    i=i+1


  return DataArray


def Scale_Data(data):
  Scaled_Data = np.zeros([data.shape[0],data.shape[1]])

  i=0
  j=0
  max = np.max(data) # see figure
  min = np.min(data)

  MaxCol = len(data[0,:])
  MaxRow = len(data[:,0])
  while(j<MaxCol):
    while(i<MaxRow):
      if( data[0,j]+data[0,j]+data[0,j]+data[0,j] !=0 ): # sum of 4 consecutive number is not zero
        Scaled_Data[i,j] = (data[i,j]- min)/ (max-min)
      i=i+1

    i=0
    j=j+1
  
  return Scaled_Data


def CSV_to_DataArray(file,array=True):
    '''    

    Parameters
    ----------
    file : CSV
        pass filename.csv as a string. file location must be current work directory
    array : boolean, default returns numpy array
        DESCRIPTION. 

    Returns
    -------
    TYPE
        csv data to numpy array
    TYPE
        csv data to pandas dataframe

    '''
    df = pd.read_csv(file)
    
    # Do random shuffle of rows/observations.
    df=df.sample(frac=1)
    df=df.reset_index(drop=True)
    
    Static_Input  = df.drop(columns=['VM1X1', 'VM1X2', 'VM2X1', 'VM2X2', 'OM1X1','OM1X2','OM2X1','OM2X2','OGX1','OGX2'])
    Static_Target = pd.DataFrame(df[['VM1X1', 'VM1X2', 'VM2X1', 'VM2X2']])
    
    if(array):
        myinput  = Static_Input.to_numpy() #np.asarray(Static_Input)
        mytarget = Static_Target.to_numpy()#np.asarray(Static_Target)
        
        return myinput,mytarget
    else:
        return Static_Input, Static_Target
    
    
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