from alpha_model import AlphaModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(AlphaModel):
    def __init__(self):
        self.model = Sequential()

    def train(self, X, Y):
        Xtrain = torch.tensor((X.values), dtype=torch.float)
        Ytrain = torch.tensor(([Y.values]), dtype=torch.float)
        N, D_in, H1, H2, H3, D_out = 64, 12, 100, 50, 20, 1

        self.model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H1),
            torch.nn.ReLU(),
            torch.nn.Linear(H1, H2),
            torch.nn.ReLU(),
            torch.nn.Linear(H2, H3),
            torch.nn.ReLU(),
            torch.nn.Linear(H3, D_out),
            )

        loss_fn = torch.nn.MSELoss(reduction = 'sum')


        learning_rate =  5 * 1e-6

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for t in range(10000):
    # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(Xtrain)

    # Compute and print loss.
            loss = loss_fn(y_pred, Ytrain)
            print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
            loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
            optimizer.step() 

    #    torch.save(self.model.state_dict(), PATH.pth)   

    def predict(self, X):
        Xtest = torch.tensor((X.values), dtype=torch.float)
        Y_pred = self.model(Xtest)
        return Y_pred
        


