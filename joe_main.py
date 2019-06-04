import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from regression_model import RegressionModel
#from nn_alpha import NeuralNetwork
import os
#from LSTM_model import LSTMModel
from simple_portfolio import SimplePortfolio
from portfolio_sim import PortfolioSimulator
from sklearn.covariance import LedoitWolf
from model_results import get_results

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

#label = '5-day Return'
label = 'Daily_Return'
data_directory = 'data'
data_start = 1990
valid_start = 2016
test_start = 2017
data_end = 2019

#desired variance is in the same
# use 4e-5 with daily returns to get variance close to naive
# use 2e-4 with 5-day returns to get variance close to naive
desired_variance = 4e-5
#desired_variance = 2e-4

transaction_costs = 0.0

# negative return allowed before position is closed
# None  to disable stoploss
stoploss_value = None
#stoploss_value = 0.2

model_type = 3 #see regression_model.py

predicted_returns = dict()
actual_returns = dict()
train_returns = dict()
valid_returns = dict()

filenames = os.listdir(data_directory)

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

daily_test_returns = dict()

print("Generating Models")

df_columns = []

for i, df in enumerate(dfs):
    data = get_data(df, data_start, valid_start, test_start, data_end, features, label)
    model = RegressionModel(model_type)
    model.train(data['Xtrain'], data['Ytrain'])
    y_pred = model.predict(data['Xtest'])
    predicted_returns[str(i)] = y_pred
    actual_returns[str(i)] = data['Ytest']
    train_returns[str(i)] = data['Ytrain']
    valid_returns[str(i)] = data['Yvalid']
    df_columns.append(str(i))
    daily_test_returns[str(i)] = get_data(df, data_start, valid_start, test_start, data_end, features, 'Daily_Return')['Ytest']


valid_df = pd.DataFrame(valid_returns)
valid_df = valid_df[df_columns]
X = valid_df.values
cov = LedoitWolf().fit(X)
cov = cov.covariance_
n = len(train_returns.keys())

daily_df = pd.DataFrame(daily_test_returns)
daily_df = daily_df[df_columns]

opt = SimplePortfolio(n)

pred_df = pd.DataFrame(predicted_returns)
pred_df = pred_df[df_columns]
pred_df.to_csv('pred_df_linear.csv')
actual_df = pd.DataFrame(actual_returns)
actual_df = actual_df[df_columns]
actual_df.to_csv('actual_df_linear.csv')

get_results(pred_df, actual_df)

print("Running Portfolio Simulation")

sim = PortfolioSimulator(opt, n)
sim.simulate(pred_df, daily_df, cov, desired_variance, transaction_costs, stoploss_value)

sim.plot_over_time()

import pdb; pdb.set_trace()
