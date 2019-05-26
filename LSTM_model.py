from alpha_model import AlphaModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import InputLayer, Input, Masking
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class LSTMModel(AlphaModel):
    def __init__(self):
        self.model = Sequential()

    def train(self, X, Y, look_back):
        assert len(X) == len(Y)
        self.model.add(LSTM(30, input_shape=(X.shape[1], X.shape[2]), return_sequences = True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(units=30, return_sequences = True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(units=30))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units = 1))
        self.model.summary()

        adam = optimizers.Adam(lr = 0.001, clipvalue = 0.25)
        self.model.compile(loss = 'mse', optimizer = adam)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
        self.model.fit(X, Y, epochs = 2, batch_size = 32, callbacks = [es])

    def predict(self, X):
        return self.model.predict(X)
