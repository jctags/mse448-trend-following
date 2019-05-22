import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from regression_model import RegressionModel
import os
from LSTM_model import LSTMModel
import re
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

def get_data_by_percentile(df, start, end, features, label):
    s = int(np.floor(len(df)*start))
    e = int(np.floor(len(df)*end)-1)
    ydf = df[s:e]
    return ydf[features].values, ydf[label].values

def get_data(df, data_start, valid_start, test_start, data_end, features, label):
    data = {}
    data['Xtrain'], data['Ytrain'] = get_data_by_percentile(df, data_start, valid_start,features, label)
    data['Xvalid'], data['Yvalid'] = get_data_by_percentile(df, valid_start, test_start,features, label)
    data['Xtest'],  data['Ytest']  = get_data_by_percentile(df, test_start, data_end, features, label)
    return data

features = ['MACD', 'Volume']
label = 'Daily_Return'
features_directory = 'data'
data_start = 0.00
valid_start = 0.70
test_start = 0.85
data_end = 1.00

df = get_dataframe(features_directory + '/' + 'Copper_2.csv')
data = get_data(df, data_start, valid_start, test_start, data_end, features, label)


def create_dataset(Xtrain, Ytrain, look_back = 1):
    dataX, dataY = [], []
    for i in range(len(Xtrain)-look_back-1):
        t = []
        for j in range(0, look_back):
            t.append(Xtrain[[(i+j)], :])
        dataX.append(t)
        dataY.append(Ytrain[i + look_back])
    return np.array(dataX), np.array(dataY)

look_back = 50

for i, filename in enumerate(os.listdir(features_directory)):
    if i == 1 and not re.match(filename, ".DS_Store"):
        df = get_dataframe(features_directory + '/' + filename)
        data = get_data(df, data_start, valid_start, test_start, data_end, features, label)
        trainX, trainY = create_dataset(data['Xtrain'], data['Ytrain'], look_back = 50)
        testX, testY = create_dataset(data['Xtest'], data['Ytest'], look_back = 50)
        trainX = trainX.reshape(trainX.shape[0], look_back, len(features))
        testX = testX.reshape(testX.shape[0], look_back, len(features))
        print(trainX.shape,testX.shape,trainY.shape,testY.shape)
        model = LSTMModel()
        model.train(trainX, trainY, look_back)
        output = {}
        output['predicted_'+str(i)] = model.predict(testX)
        output['true_'+str(i)] = data['Ytest']
        print(output)
returns_df = pd.DataFrame(output)

returns_df.to_csv('predicted_result')

