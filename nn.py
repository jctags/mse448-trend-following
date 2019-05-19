import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

df = pd.read_csv('Gold.csv')
df.head()
columns = list(df.columns)
numerics = [x for x in columns[2:-3] if 'Cross' not in x and x != 'MACD']
for col in numerics:
    df[col+'norm'] = df[col] / df['Settle_Price'] - 1
columns = list(df.columns)
features = [col for col in columns if 'norm' in col]
X = df[features]
Y = df['Daily_Return']

trainsplit = int(len(Y) * 0.9)
validsplit = trainsplit #+ int(len(Y) * 0.1)
Xtrain = X[:trainsplit]
Ytrain = Y[:trainsplit]
Xvalid = X[trainsplit:validsplit]
Yvalid = Y[trainsplit:validsplit]
Xtest = X[validsplit:]
Ytest = Y[validsplit:]


X = torch.tensor((Xtrain.values), dtype=torch.float)
y = torch.tensor(([Ytrain.values]), dtype=torch.float)
Xtest = torch.tensor((Xtest.values), dtype=torch.float)
ytest = torch.tensor((Ytest.values), dtype=torch.float)
y = torch.t(y)

#print(X.size())
#print(y.size())
#print(Xtest.size())
#print(ytest.size())

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden2, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))      # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        x = self.predict(x)             # linear output
        return x

net = Net(n_feature=12, n_hidden1= 10, n_hidden2= 5, n_output=1)     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

#plt.ion()   # something about plotting

for t in range(2000):
    prediction = net(X)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients


predY = net(Xtest)

a = predY.detach().numpy()
print(a)
b = ytest.detach().numpy()
print(b)


plt.plot(b,a, 'k.')
plt.title("Prediction accuracy")
plt.xlabel("True returns")
plt.ylabel("Predicted returns")

