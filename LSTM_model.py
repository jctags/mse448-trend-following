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
        self.model = Sequential()
   
    def train(self, X, Y):
        assert len(X)==len(Y)
        Xtrain = X.values 
        Ytrain = Y.values
        lookback = 50
     
        x = []
        y = []
        for i in range(len(Xtrain)-lookback-1):
            t = []
            for j in range(0,lookback):
                t.append(Xtrain[[(i+j)], :])
            x.append(t)
            y.append(Ytrain[i+ lookback])
         x, y = np.array(x), np.array(y)
         x = x.reshape(x.shape[0],lookback, 2)
  
        model.add(LSTM(units=30, return_sequences= True, input_shape=(x.shape[1],2)))
        model.add(LSTM(units=30, return_sequences=True))
        model.add(LSTM(units=30))
        model.add(Dense(units=1))
        model.summary()
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x, y, epochs=200, batch_size=32)
                
    def predict(self, X):
        Xtest = X.values
        lookback = 50
        
        x_test = []
        
        for i in range(len(Xtest)-lookback-1):
            t = []
            for j in range(0,lookback):
                t.append(Xtest[[(i+j)], :])
            x_test.append(t)
        
        x_test = x_test.reshape(x_test.shape[0],lookback, 2) 
        return model.predict(x_test)
        
