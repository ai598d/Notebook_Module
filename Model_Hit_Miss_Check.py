# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:42:00 2023

@author: ai598
"""

from command import Cmove2
from compare import StaticCheckBad

import compare as cp

import dependencies
from build_so_model import build_model
from build_so_model import train_model
from build_so_model import call_existing_code 
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import os as os 

import scipy as sc
from scipy import signal

#%%

path = 'C:\\Users\\ai598\\Thesis\\Notebook_Modules'
os.chdir(path)

# =============================================================================
# DATA PRE PROCESSING
# =============================================================================
df = pd.read_csv('StaticMoveTestData1.csv')  # import data

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



# =============================================================================
# CREATE RANDOM MODEL / UNTRAINED MODEL
# =============================================================================


rand_model1 = call_existing_code (10,'relu',.1, 1e-1)
rand_model2 = call_existing_code (10,'relu',.1, 1e-1)
rand_model3 = call_existing_code (10,'relu',.1, 1e-1)
rand_model4 = call_existing_code (10,'relu',.1, 1e-1)

# =============================================================================
# LOAD LEARNED MODEL
# =============================================================================

learned_model1 = keras.models.load_model('my_newFFmodel1')  # LOAD MODEL
learned_model2 = keras.models.load_model('my_newFFmodel2')  # LOAD MODEL
learned_model3 = keras.models.load_model('my_newFFmodel3')  # LOAD MODEL
learned_model4 = keras.models.load_model('my_newFFmodel4')  # LOAD MODEL



# =============================================================================
# PREDICT
# =============================================================================
items =8000
learned_Pred_M1x = learned_model1.predict(InputArray[:items])
learned_Pred_M1y = learned_model2.predict(InputArray[:items])
learned_Pred_M2x = learned_model3.predict(InputArray[:items])
learned_Pred_M2y = learned_model4.predict(InputArray[:items])


rand_pred_M1x = abs(rand_model1.predict(InputArray[:items]))
rand_pred_M1y = abs(rand_model2.predict(InputArray[:items]))
rand_pred_M2x = abs(rand_model3.predict(InputArray[:items]))
rand_pred_M2y = abs(rand_model4.predict(InputArray[:items]))


factor = 100/items

#%%


# =============================================================================
# Create Trajectory
# =============================================================================

i = 0

MaxIt = InputArray[:items].shape[0]
Learned_Move = []
Rand_Move = []

while(i<MaxIt):
    Is = np.array(InputArray[i,0:2],dtype='float64')
    Gs = np.array(InputArray[i,2:4],dtype='float64')
    Ms1 = np.array([learned_Pred_M1x[i],learned_Pred_M1y[i] ],dtype='float64')
    Ms2 = np.array([learned_Pred_M2x[i],learned_Pred_M2y[i]],dtype='float64')
    Learned_traj = Cmove2(Is,Ms1,Ms2,Gs,100)
    Learned_Move.append(Learned_traj)
    
    rand_Ms1 = np.array([rand_pred_M1x[i],rand_pred_M1y[i] ],dtype='float64')
    rand_Ms2 = np.array([rand_pred_M2x[i],rand_pred_M2y[i] ],dtype='float64')
    rand_traj = Cmove2(Is,rand_Ms1 ,rand_Ms2,Gs,100)
    Rand_Move.append(rand_traj)
    i=i+1


Learned_Trajectory = np.asarray(Learned_Move)
Rand_Trajectory = np.asarray(Rand_Move)


Learned_Trajectory = Learned_Trajectory[:,:,:,0]
Rand_Trajectory = Rand_Trajectory[:,:,:,0]

obstacle = InputArray[:,4:6]


#%%

# plot Trajectory

xL = sc.signal.decimate(Learned_Trajectory[800,0,:],4)
yL = sc.signal.decimate(Learned_Trajectory[800,1,:],4)

xR = sc.signal.decimate(Rand_Trajectory[800,0,:] ,4) 

yR = sc.signal.decimate(Rand_Trajectory[800,1,:] ,4) 

plt.plot(xL,yL ,'*')

plt.plot(xR ,yR, '*' )

#%%
# =============================================================================
# Check Hit Miss
# =============================================================================

Lcount, Lindex = StaticCheckBad(Learned_Trajectory,obstacle,.2)

Rcount, Rindex = StaticCheckBad(Rand_Trajectory,obstacle,.2)


#%%
fig, ax = plt.subplots()

models = ['Learned Model', 'Random Model']
counts = [Lcount, Rcount]
bar_labels = ['red', 'blue']
bar_colors = ['tab:red', 'tab:blue']

ax.bar(models, counts, label=bar_labels, color=bar_colors)

ax.set_ylabel('Number of hits')
ax.set_title('Hit-Miss comparison between models')
#ax.legend([['Learned model', 'Random Model'] ])

plt.show()

#%%


Lc = []

Rc = []

th = np.linspace(0,1,50)

i = 0

maxit = len(th)

while(i<maxit):
    
    Lcount, Lindex = StaticCheckBad(Learned_Trajectory,obstacle,th[i])

    Rcount, Rindex = StaticCheckBad(Rand_Trajectory,obstacle,th[i])
    
    Lc.append(Lcount)
    
    Rc.append(Rcount)
    
    i=i+1
    
#%%
    
Lc = np.asarray(Lc)*factor
Rc = np.asarray(Rc)*factor

# plot Trajectory

plt.figure()
plt.plot(th,Lc)
plt.plot(th,Rc)
plt.xlabel('collision threshold')
plt.ylabel('bad count %')
plt.legend(['learned','Random'])
plt.show()

