import matplotlib.pyplot as plt
import pandas as pd
<<<<<<< HEAD
%matplotlib inline
from sklearn.linear_model import LinearRegression
import numpy as np
from regression_model import RegressionModel
import os

=======
from sklearn.linear_model import LinearRegression
import numpy as np
#from regression_model import RegressionModel
from LSTM_model import LSTMModel
import os


print("1")
>>>>>>> 991bd4a2c3911e8760d4847a5c36e68f10cd0bba
def get_dataframe(filename):
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df["Unnamed: 0"], dayfirst=True)
    df.drop(["Unnamed: 0"],axis='columns', inplace=True)
    columns = list(df.columns)
    numerics = [x for x in columns[2:-3] if 'Cross' not in x and x not in ['MACD','Settle_Price']]
    for col in numerics:
        df[col+'norm'] = df[col] / df['Settle_Price'] - 1
    return df

<<<<<<< HEAD
=======
print("2")
>>>>>>> 991bd4a2c3911e8760d4847a5c36e68f10cd0bba
def get_data_by_years(df, years, features, label):
    ydf = df[df.Date.map(lambda x: x.year in years)]
    return ydf[features].values, ydf[label].values

<<<<<<< HEAD
=======
print("3")
>>>>>>> 991bd4a2c3911e8760d4847a5c36e68f10cd0bba
def get_data(df, data_start, valid_start, test_start, data_end, features, label):
    data = {}
    train_years = list(range(data_start,  valid_start))
    valid_years = list(range(valid_start, test_start))
    test_years  = list(range(test_start,  data_end))
    data['Xtrain'], data['Ytrain'] = get_data_by_years(df, train_years, features, label)
    data['Xvalid'], data['Yvalid'] = get_data_by_years(df, valid_years, features, label)
    data['Xtest'],  data['Ytest']  = get_data_by_years(df, test_years,  features, label)
    return data

<<<<<<< HEAD
=======
print("4")
>>>>>>> 991bd4a2c3911e8760d4847a5c36e68f10cd0bba
features = ['MACD', 'Volume']
label = 'Daily_Return'
features_directory = 'data'
data_start = 1990
valid_start = 2010
test_start = 2014
data_end = 2018
<<<<<<< HEAD
model_class = RegressionModel

=======
model_class = LSTMModel

print("5")
>>>>>>> 991bd4a2c3911e8760d4847a5c36e68f10cd0bba
for i, filename in enumerate(os.listdir(features_directory)):
    df = get_dataframe(features_directory + '/' + filename)
    data = get_data(df, data_start, valid_start, test_start, data_end, features, label)
    model = model_class()
    model.train(data['Xtrain'], data['Ytrain'])
    output = {}
    output['predicted_'+str(i)] = model.predict(data['Xtest'])
    output['true_'+str(i)] = data['Ytest']
returns_df = pd.DataFrame(output)

<<<<<<< HEAD
=======
print("6")
>>>>>>> 991bd4a2c3911e8760d4847a5c36e68f10cd0bba
from simple_portfolio import SimplePortfolio
returns = np.array([0.01])
optimizer = SimplePortfolio(1)
#print(optimizer.optimize(returns, 1e-5))
vals = returns_df[['predicted_0']].values[0:5,:]
np.apply_along_axis(lambda x: optimizer.optimize(x, 1e-6), 1, vals)
