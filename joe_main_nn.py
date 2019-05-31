import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from regression_model import RegressionModel
from nn_alpha import NeuralNetwork
import os
#from LSTM_model import LSTMModel
from simple_portfolio import SimplePortfolio

def get_dataframe(filename):
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df["Unnamed: 0"], dayfirst=True)
    df.drop(["Unnamed: 0"],axis='columns', inplace=True)
    columns = list(df.columns)
    numerics = [x for x in columns if 'MA' in x[:3] and x != 'MACD' and 'Cross' not in x]
    for col in numerics:
        df[col+'norm'] = df[col] / df['Settle_Price'] - 1
    return df

def get_data_by_years(df, years, features, label):
    ydf = df[df.Date.map(lambda x: x.year in years)]
    return ydf[features].values, ydf[label].values

def get_data(df, data_start, valid_start, test_start, data_end, features, label):
    data = {}
    train_years = list(range(data_start,  valid_start))
    valid_years = list(range(valid_start, test_start))
    test_years  = list(range(test_start,  data_end))
    data['Xtrain'], data['Ytrain'] = get_data_by_years(df, train_years, features, label)
    data['Xvalid'], data['Yvalid'] = get_data_by_years(df, valid_years, features, label)
    data['Xtest'],  data['Ytest']  = get_data_by_years(df, test_years,  features, label)
    return data

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
    'SMA50norm',
]

label = 'Daily_Return'
data_directory = 'data'
data_start = 1990
valid_start = 2015
test_start = 2017
data_end = 2019

predicted_returns = dict()
actual_returns = dict()
train_returns = dict()

filenames = os.listdir(data_directory)
filenames = ['Gold_3.csv', 'Gold_4.csv']

dfs = []

print("Loading Dataframes")

for i, filename in enumerate(filenames):
    if '.csv' not in filename:
        continue
    df = get_dataframe(data_directory + '/' + filename)
    dfs.append(df)

print("Standardizing Dates")

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

print("Generating Models")

for i, df in enumerate(dfs):
    data = get_data(df, data_start, valid_start, test_start, data_end, features, label)
    model = NeuralNetwork()
    model.train(data['Xtrain'], data['Ytrain'], data['Xvalid'], data['Yvalid'])
    y_pred = model.predict(data['Xtest'])
    predicted_returns[str(i)] = y_pred
    actual_returns[str(i)] = data['Ytest']
    train_returns[str(i)] = data['Ytrain']

cov = np.cov(pd.DataFrame(train_returns).values.T)

n = len(train_returns.keys())

opt = SimplePortfolio(n)
desired_variance = 1e-4

pred_df = pd.DataFrame(predicted_returns)
actual_df = pd.DataFrame(actual_returns)

print("Running Portfolio Simulation")

portfolio_value = 1.0
portfolio_over_time = [portfolio_value]
portfolio_returns = []
naive_value = 1.0
naive_over_time = [portfolio_value]
naive_returns = []
naive_allocation = np.ones(n)
naive_allocation = naive_allocation/np.sum(naive_allocation)
allocations = []
for i in range(len(pred_df)):
    naive_return = np.dot(naive_allocation, actual_df.iloc[i,:].values)
    naive_value *= (1+naive_return)
    naive_over_time.append(naive_value)
    naive_returns.append(naive_return)

    w = opt.optimize(pred_df.iloc[i, :].values, cov, desired_variance)
    allocations.append(w)
    p_return = np.dot(w, actual_df.iloc[i, :].values)
    portfolio_returns.append(p_return)
    portfolio_value *= (1+p_return)
    portfolio_over_time.append(portfolio_value)

print portfolio_value
print naive_value

plt.plot(portfolio_over_time)
plt.plot(naive_over_time)
plt.show()

import pdb; pdb.set_trace()
