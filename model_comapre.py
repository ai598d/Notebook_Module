import dependencies
from build_so_model import build_model
from build_so_model import train_model
from build_so_model import call_existing_code 
import pandas as pd
import numpy as np
import keras

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

learned_model1 = keras.models.load_model('my_model1')  # LOAD MODEL
learned_model2 = keras.models.load_model('my_model2')  # LOAD MODEL
learned_model3 = keras.models.load_model('my_model3')  # LOAD MODEL
learned_model4 = keras.models.load_model('my_model4')  # LOAD MODEL



# =============================================================================
# PREDICT
# =============================================================================

learned_Pred_M1x = learned_model1.predict(InputArray[:2000])
learned_Pred_M1y = learned_model2.predict(InputArray[:2000])
learned_Pred_M2x = learned_model3.predict(InputArray[:2000])
learned_Pred_M2y = learned_model4.predict(InputArray[:2000])


rand_pred_M1x = rand_model1.predict(InputArray[:2000])
rand_pred_M1y = rand_model2.predict(InputArray[:2000])
rand_pred_M2x = rand_model3.predict(InputArray[:2000])
rand_pred_M2y = rand_model4.predict(InputArray[:2000])


# =============================================================================
# MAE COMPARE
# =============================================================================


def my_mae(pred,real):
    
    return np.mean(np.abs(pred-real))



learned_mae_M1x = my_mae(learned_Pred_M1x,TargetM1x )
learned_mae_M1y = my_mae(learned_Pred_M1y,TargetM1y )
learned_mae_M2x = my_mae(learned_Pred_M2x,TargetM2x )
learned_mae_M2y = my_mae(learned_Pred_M2y,TargetM2y )


rand_mae_M1x = my_mae(rand_pred_M1x,TargetM1x )
rand_mae_M1y = my_mae(rand_pred_M1y,TargetM1y )
rand_mae_M2x = my_mae(rand_pred_M2x,TargetM2x )
rand_mae_M2y = my_mae(rand_pred_M2y,TargetM2y )



print("Learned mae:")
print(learned_mae_M1x )
print(learned_mae_M1y )
print(learned_mae_M2x )
print(learned_mae_M2y )

print("rand mae:")
print(rand_mae_M1x)
print(rand_mae_M1y)
print(rand_mae_M2x)
print(rand_mae_M2y)

