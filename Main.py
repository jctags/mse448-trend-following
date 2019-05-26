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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def get_dataframe(filename):
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df["Unnamed: 0"], dayfirst=True)
    df.drop(["Unnamed: 0"],axis='columns', inplace=True)
    columns = list(df.columns)
    numerics = [x for x in columns[2:-3] if 'Cross' not in x and x not in ['MACD','Settle_Price']]
    for col in numerics:
        df[col+'norm'] = df[col] / df['Settle_Price'] - 1
    return df

def get_data_by_years(df, years, features, label):
    ydf = df[df.Date.map(lambda x: x.year in years)]
    return ydf[features].values, ydf[label].values

def get_data_not_by_years(df, years, features, label):
    ydf = df[df.Date.map(lambda x: x.year not in years)]
    return ydf[features].values, ydf[label].values

def get_data(df, valid_start, test_start, data_end, features, label):
    data = {}
    train_years = list(range(valid_start, test_start))
    valid_years = list(range(valid_start, test_start))
    test_years  = list(range(test_start,  data_end))
    data['Xtrain'], data['Ytrain'] = get_data_not_by_years(df, train_years, features, label)
    data['Xvalid'], data['Yvalid'] = get_data_by_years(df, valid_years, features, label)
    data['Xtest'],  data['Ytest']  = get_data_by_years(df, test_years, features, label)
    return data


look_back = 2
def create_dataset(Xtrain, Ytrain, look_back):
    dataX, dataY = [], []
    for i in range(len(Xtrain)-look_back-1):
        a = Xtrain[i:(i+look_back)]
        dataX.append(a)
        dataY.append(Ytrain[i + look_back])
    return np.array(dataX), np.array(dataY)

def main():
    features = [
        'EMA10norm',
        'EMA100norm',
        'EMA12norm',
        'EMA20norm',
        'EMA26norm',
        'EMA50norm',
        'SMA10norm',
        'SMA100norm',
        'SMA15norm',
        'SMA20norm',
        'SMA5norm',
        'SMA50norm'
    ]
    label = 'Settle_Price'
    features_directory = 'data'

    valid_start = 2017
    test_start = 2017
    data_end = 2019

    for i, filename in enumerate(glob.glob("data/*.csv")):
        df = get_dataframe(filename)
        data = get_data(df, valid_start, test_start, data_end, features, label)
        trainX, trainY = create_dataset(data['Xtrain'], data['Ytrain'], look_back)
        testX, testY = create_dataset(data['Xtest'], data['Ytest'], look_back)
        print(trainX.shape, trainY.shape, testX.shape, testY.shape)
        model = LSTMModel()
        model.train(trainX, trainY, look_back)
        predictedY = model.predict(testX)
        print(predictedY)
        return_df = pd.DataFrame(predictedY)
        return_df.to_csv('LSTM_output/predicted_price_' + filename[5:])

if __name__ == "__main__":
    main()
