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
    ]
    label = 'Daily_Return'
    data_directory = 'data'
    valid_start = 2017
    test_start = 2017
    data_end = 2019
    predicted_returns = dict()
    actual_returns = dict()
    train_returns = dict()

    scale = MinMaxScaler(feature_range=(0, 1))

    print("Loading Dataframes")
    dfs = []
    for i, filename in enumerate(os.listdir(data_directory)):
        if '.csv' not in filename:
            continue
        df = get_dataframe(data_directory + '/' + filename)
        dfs.append(df)

    print("Standardizing Date")
    minimal_dates = None
    for df in dfs:
        dates = set(df['Date'].map(lambda x: x.timestamp()))
        if minimal_dates is None:
            minimal_dates = dates
        else:
            minimal_dates = minimal_dates & dates

    new_dfs = []

    for df in dfs:
        new_df = df[df['Date'].map(lambda x: x.timestamp() in minimal_dates)]
        new_dfs.append(new_df)

    dfs = new_dfs

    for i, df in enumerate(dfs):
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
        y_pred = model.predict(testX)
        y_pred = scale.inverse_transform(y_pred).ravel()
        predicted_returns[str(i)] = y_pred
        actual_returns[str(i)] = data['Ytest']
        train_returns[str(i)] = data['Ytrain']
        date['test'] = np.array(date['test'][look_back + 1:]).astype(str)
        # pred = np.vstack((predictedY, date['test']))
        # pred = np.transpose(pred)
        # pred_df = pd.DataFrame(pred)
        # pred_df.to_csv('LSTM_output/predicted_price_' + filename[5:])

    pred_df = pd.DataFrame(predicted_returns)
    pred_df.to_csv('LSTM_output/predicted_price')
    actual_df = pd.DataFrame(actual_returns)

if __name__ == "__main__":
    main()
