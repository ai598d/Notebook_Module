# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 00:11:26 2023

@author: ai598
"""

import os as os
#%%
# initialize working directory

path = "C:\\Users\\ai598\\Thesis\\Notebook_Modules"

os.chdir(path)

import dependencies
from build_so_model import build_model
from build_so_model import train_model
from build_so_model import train_model_FF
from build_so_model import train_model_WidenDeep
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

#%%
import matplotlib.pyplot as plt

plt.plot(myinput[:50,0],myinput[:50,1],'*')

#%%

import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt 

## generate the data and plot it for an ideal normal curve

## x-axis for the plot
x_data = myinput[:,0]

## y-axis as the gaussian
y_data = stats.norm.pdf(x_data, 0, 1)

## plot data
plt.plot(x_data, y_data)

plt.show()









