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
        self.model.add(LSTM(units=128, input_shape=(X.shape[1], X.shape[2]), return_sequences = True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(units=128, return_sequences = True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(units=128))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units = 1))
        self.model.add(Activation('linear'))
        self.model.summary()

        adam = optimizers.Adam(lr = 0.001, clipvalue = 0.5)
        self.model.compile(loss = 'mean_squared_error', optimizer = 'adagrad')
        es = EarlyStopping(monitor='loss', mode='min', verbose=1)
        self.model.fit(X, Y, epochs = 10, batch_size = 64, callbacks = [es])

    def predict(self, X):
        return self.model.predict(X)
