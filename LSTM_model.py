from alpha_model import AlphaModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import InputLayer, Input
from keras.layers import Dense, LSTM
from keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class LSTMModel(AlphaModel):
    def __init__(self):
        self.model = Sequential()

    def train(self, X, Y, look_back):
        assert len(X)==len(Y)
        self.model = Sequential()
        self.model.add(LSTM(units=30, return_sequences= True, input_shape=(X.shape[1],2)))
        self.model.add(LSTM(units=30, return_sequences=True))
        self.model.add(LSTM(units=30))
        self.model.add(Dense(units=1))
        self.model.summary()
        self.model.compile(loss='mse', optimizer='adam')
        self.model.fit(X, Y, epochs=200, batch_size=32)

    def predict(self, X):
        return self.model.predict(X)
