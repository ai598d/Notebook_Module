# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 23:01:15 2023

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






