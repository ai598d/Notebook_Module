import keras
import numpy as np
import tensorflow as tf
import os as os
from keras.preprocessing import sequence
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense
from keras import models
from keras import layers
import keras_tuner
import matplotlib.pyplot as plt
import pandas as pd

def call_existing_code(units, activation, dropout, lr):

    """
    Return a neural network model. 

    This skeleton model is used as a framwork for actual model. Hyper parameters: unit, activation func, learning rate are passed as an argument. 

    :param kind: layer_units, activation function type, dropout, learning rate.
    :raise: If the kind is invalid.
    :return: Bad move counts, Array of indices for bad trajectories.
    :rtype: keras model

    """
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(units=units, activation=activation))
    if dropout:
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(12)) # extra layer 1
    model.add(layers.Dense(24)) # extra layer 2
    model.add(layers.Dense(48)) # extra layer 3
    model.add(layers.Dense(36)) # extra layer 3
    model.add(layers.Dense(1)) # extra layer 4
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse', metrics=['mae'],
    )
    return model


def build_model(hp):

    """
    Returns a complete neural network model

    :param kind: layer_units, activation function type, dropout, learning rate.
    :raise: If the kind is invalid.
    :return: Bad move counts, Array of indices for bad trajectories.
    :rtype: keras model

    """
        
    units = hp.Int("units", min_value=3, max_value=300, step=3)
    activation = hp.Choice("activation", ["relu", "tanh"])
    dropout = hp.Boolean("dropout")
    lr = hp.Float("lr", min_value=1e-4, max_value=1e-1, sampling="log")
    # call existing model-building code with the hyperparameter values.
    model = call_existing_code(
        units=units, activation=activation, dropout=dropout, lr=lr
    )
    return model


def train_model(train_data,train_target,model,k=2,num_epochs = 100):

  num_val_samples = len(train_data) // k
  all_scores = []
  all_mae_histories = []

  my_callbacks = [
  tf.keras.callbacks.EarlyStopping(patience=4),
  tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
  tf.keras.callbacks.TensorBoard(log_dir='./logs')
  ]

  for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_target = train_target[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
                        [train_data[:i * num_val_samples],
                        train_data[(i + 1) * num_val_samples:]],
                        axis=0)
    partial_train_target = np.concatenate(
                        [train_target[:i * num_val_samples],
                        train_target[(i + 1) * num_val_samples:]],
                        axis=0)
    #model = build_model()
    history = model.fit(partial_train_data, partial_train_target,
    validation_data=(val_data, val_target),
    epochs=num_epochs, batch_size=1, verbose=0,callbacks=my_callbacks)
    mae_history = history.history['mae']
    all_mae_histories.append(mae_history)

    return all_mae_histories

