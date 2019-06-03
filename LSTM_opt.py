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
from simple_portfolio import SimplePortfolio
from portfolio_sim import PortfolioSimulator
from sklearn.covariance import LedoitWolf
from model_results import get_results


print("Loading Dataframes")

data_directory = 'LSTM_output'
pred_df = pd.read_csv(data_directory + '/' + 'predicted_returns.csv')
pred_df.drop(["Unnamed: 0"],axis='columns', inplace=True)
actual_df = pd.read_csv(data_directory + '/' + 'actual_returns.csv')
actual_df.drop(["Unnamed: 0"],axis='columns', inplace=True)
train_df = pd.read_csv(data_directory + '/' + 'train_returns.csv')
train_df.drop(["Unnamed: 0"],axis='columns', inplace=True)
valid_df = pd.read_csv(data_directory + '/' + 'valid_returns.csv')
valid_df.drop(["Unnamed: 0"],axis='columns', inplace=True)
daily_df = pd.read_csv(data_directory + '/' + 'daily_returns.csv')
daily_df.drop(["Unnamed: 0"],axis='columns', inplace=True)

X = valid_df.values
cov = LedoitWolf().fit(X)
cov = cov.covariance_
n = len(train_df.keys())

opt = SimplePortfolio(n)
desired_variance = 2e-4
transaction_costs = 2e-4

get_results(pred_df, actual_df)

print("Running Portfolio Simulation")

sim = PortfolioSimulator(opt, n)
sim.simulate(pred_df, daily_df, cov, desired_variance, transaction_costs)
sim.plot_over_time()

# portfolio_value = 1.0
# portfolio_over_time = [portfolio_value]
# portfolio_returns = []
# naive_value = 1.0
# naive_over_time = [portfolio_value]
# naive_returns = []
# naive_allocation = np.ones(n)
# naive_allocation = naive_allocation/np.sum(naive_allocation)
# allocations = []
#
# for i in range(len(pred_df)):
#     naive_return = np.dot(naive_allocation, actual_df.iloc[i,:].values)
#     naive_value *= (1+naive_return)
#     naive_over_time.append(naive_value)
#     naive_returns.append(naive_return)
#     w = opt.optimize(pred_df.iloc[i, :].values, cov, desired_variance)
#     allocations.append(w)
#     p_return = np.dot(w, actual_df.iloc[i, :].values)
#     portfolio_returns.append(p_return)
#     portfolio_value *= (1+p_return)
#     portfolio_over_time.append(portfolio_value)
#
# print(portfolio_value)
# print(naive_value)
#
# plt.plot(portfolio_over_time)
#plt.plot(naive_over_time)
plt.show()
