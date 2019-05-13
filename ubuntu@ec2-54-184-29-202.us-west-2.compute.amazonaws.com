from alpha_model import AlphaModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import InputLayer, Input
from keras.layers import Dense, LSTM

class LSTMModel(AlphaModel):
    def __init__(self):
        self.model = 0
        model = Sequential()
        model.add(LSTM(units=30, return_sequences= True, input_shape=(X.shape[1],2)))
        model.add(LSTM(units=30, return_sequences=True))
        model.add(LSTM(units=30))
        model.add(Dense(units=1))

    def train(self, X, Y):
        assert len(X)==len(Y)

    def predict(self, X):
        return 0
