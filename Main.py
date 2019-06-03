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
from preprocessing import create_return

def get_dataframe(filename):
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df["Unnamed: 0"], dayfirst=True)
    df.drop(["Unnamed: 0"],axis='columns', inplace=True)
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
    test_years  = list(range(test_start,  data_end + 1))
    data['Xtrain'], data['Ytrain'] = get_data_not_by_years(df, train_years, features, label)
    data['Xvalid'], data['Yvalid'] = get_data_by_years(df, valid_years, features, label)
    data['Xtest'],  data['Ytest']  = get_data_by_years(df, test_years, features, label)
    return data

look_back = 10
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
    label = 'Settle_Price'
    data_directory = 'data'
    valid_start = 2016
    test_start = 2017
    data_end = 2019
    predicted_price = dict()
    actual_price = dict()
    train_price = dict()
    valid_price = dict()

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

    forward_window = 1
    for i, df in enumerate(dfs):
        Next_Price = dfs[i]['Settle_Price'][forward_window:].values
        dfs[i] = dfs[i][:-forward_window]
        dfs[i] = dfs[i].assign(Next_Price = Next_Price)

    df_columns = []
    daily_test_returns = dict()

    for i, df in enumerate(dfs):
        data = get_data(df, valid_start, test_start, data_end, features, label)
        actual_price[str(i)] = data['Ytest'].ravel()
        train_price[str(i)] = data['Ytrain'].ravel()
        valid_price[str(i)] = data['Yvalid'].ravel()
        daily_test_returns[str(i)] = get_data(df, valid_start, test_start, data_end, features, 'Daily_Return')['Ytest'].ravel()
        df_columns.append(str(i))
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
        predicted_price[str(i)] = y_pred

    predicted_returns = dict()
    actual_returns = dict()
    train_returns = dict()
    valid_returns = dict()

    for w in df_columns:
        actual_returns[w] =  create_return(actual_price[w], forward_window)
        train_returns[w] = create_return(train_price[w], forward_window)
        predicted_returns[w] = create_return(predicted_price[w], forward_window)
        valid_returns[w] = create_return(valid_price[w], forward_window)

    daily_df = pd.DataFrame(daily_test_returns)
    daily_df = daily_df[df_columns]
    daily_df.to_csv('LSTM_output/daily_returns.csv')
    valid_df = pd.DataFrame(valid_returns)
    valid_df = valid_df[df_columns]
    valid_df.to_csv('LSTM_output/valid_returns.csv')
    pred_df = pd.DataFrame(predicted_returns)
    pred_df.to_csv('LSTM_output/predicted_returns.csv')
    actual_df = pd.DataFrame(actual_returns)
    train_df = pd.DataFrame(train_returns)
    train_df.to_csv('LSTM_output/train_returns.csv')
    actual_df.to_csv('LSTM_output/actual_returns.csv')

if __name__ == "__main__":
    main()
