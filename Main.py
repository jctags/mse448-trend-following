import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from regression_model import RegressionModel
import os
import glob
from LSTM_model import LSTMModel
import re
import preprocessing
from sklearn.preprocessing import MinMaxScaler

def get_dataframe(filename):
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df["Unnamed: 0"], dayfirst=True)
    df.drop(["Unnamed: 0"],axis='columns', inplace=True)
    return df

def get_data_by_years(df, years, features, label):
    ydf = df[df.Date.map(lambda x: x.year in years)]
    return ydf[features].values, ydf[label].values, ydf['Date'].values

def get_data_not_by_years(df, years, features, label):
    ydf = df[df.Date.map(lambda x: x.year not in years)]
    return ydf[features].values, ydf[label].values, ydf['Date'].values

def get_data(df, valid_start, test_start, data_end, features, label):
    data = {}
    Date = {}
    train_years = list(range(valid_start, test_start))
    valid_years = list(range(valid_start, test_start))
    test_years  = list(range(test_start,  data_end + 1))
    data['Xtrain'], data['Ytrain'], Date['train'] = get_data_not_by_years(df, train_years, features, label)
    data['Xvalid'], data['Yvalid'], Date['valid'] = get_data_by_years(df, valid_years, features, label)
    data['Xtest'],  data['Ytest'], Date['test']  = get_data_by_years(df, test_years, features, label)
    return data, Date


look_back = 5
def create_dataset(Xtrain, Ytrain, look_back):
    dataX, dataY = [], []
    for i in range(len(Xtrain)-look_back-1):
        a = Xtrain[i:(i+look_back)]
        dataX.append(a)
        dataY.append(Ytrain[i + look_back])
    return np.array(dataX), np.array(dataY)

def main():
    features = [
        'EMA10',
        'EMA100',
        'EMA12',
        'EMA20',
        'EMA26',
        'EMA50',
        'SMA10',
        'SMA100',
        'SMA15',
        'SMA20',
        'SMA5',
        'SMA50',
        # 'Settle_Price'
    ]
    label = 'Settle_Price'
    features_directory = 'data'

    valid_start = 2017
    test_start = 2017
    data_end = 2019

    scale = MinMaxScaler(feature_range=(0, 1))
    for i, filename in enumerate(glob.glob("data/*.csv")):
        df = get_dataframe(filename)
        data, date = get_data(df, valid_start, test_start, data_end, features, label)
        data['Xtrain'] = scale.fit_transform(data['Xtrain'])
        data['Ytrain'] = scale.fit_transform(data['Ytrain'].reshape(-1,1))
        data['Xtest'] = scale.fit_transform(data['Xtest'])
        data['Ytest'] = scale.fit_transform(data['Ytest'].reshape(-1,1))
        trainX, trainY = create_dataset(data['Xtrain'], data['Ytrain'], look_back)
        testX, testY = create_dataset(data['Xtest'], data['Ytest'], look_back)
        print(trainX.shape, trainY.shape, testX.shape, testY.shape)
        model = LSTMModel()
        model.train(trainX, trainY, look_back)
        predictedY = model.predict(testX)
        predictedY = np.array(scale.inverse_transform(predictedY).ravel()).astype(str)
        date['test'] = np.array(date['test'][look_back + 1:]).astype(str)
        pred = np.vstack((predictedY, date['test']))
        pred = np.transpose(pred)
        pred_df = pd.DataFrame(pred)
        pred_df.to_csv('LSTM_output/predicted_price_' + filename[5:])

if __name__ == "__main__":
    main()
