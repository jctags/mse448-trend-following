from alpha_model import AlphaModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(AlphaModel):
    def __init__(self, learning_rate =  1e-3, convergence_error = 1e-6):
        self.model = torch.nn.Sequential()
        self.lr = learning_rate
        self.conv_err = convergence_error

    def train(self, X, Y, Xvalid=None, Yvalid=None):
        if Xvalid is not None and Yvalid is not None:
            use_validation = True
            print("USING VALIDATION SET FOR STOPPING")
        else:
            use_validation = False

        if use_validation:
            Xvalid = torch.tensor(Xvalid, dtype=torch.float)
            Yvalid = torch.tensor(Yvalid, dtype=torch.float)

        Xtrain = torch.tensor(X, dtype=torch.float)
        Ytrain = torch.tensor(Y, dtype=torch.float)

        D_in = Xtrain.shape[1]

        N, H1, H2, H3, D_out = 64, 100, 50, 20, 1

        self.model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H1),
            torch.nn.ReLU(),
            torch.nn.Linear(H1, H2),
            #torch.nn.ReLU(),
            #torch.nn.Linear(H2, H3),
            #torch.nn.ReLU(),
            torch.nn.Linear(H2, D_out),
        )

        self.model = torch.nn.Sequential(torch.nn.Linear(D_in, D_out))

        loss_fn = torch.nn.MSELoss(reduction = 'mean')

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if use_validation:
            prev_valid_loss = float('inf')
        else:
            prev_loss = float('inf')

        t=0
        while True:
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = self.model(Xtrain)

            # Compute and print loss.
            loss = loss_fn(y_pred, Ytrain)

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

            if use_validation:
                y_pred_valid = self.model(Xvalid)
                valid_loss = loss_fn(y_pred_valid, Yvalid)
                if prev_valid_loss - valid_loss.item() < self.conv_err:
                    break
                prev_valid_loss = valid_loss.item()
                if t % 10 == 0:
                    print(t, prev_valid_loss)
            else:
                if prev_loss - loss.item() < self.conv_err:
                    break
                prev_loss = loss.item()
                if t % 20 == 0:
                    print(t, prev_loss)
            t += 1

    #    torch.save(self.model.state_dict(), PATH.pth)

    def predict(self, X):
        Xtest = torch.tensor((X), dtype=torch.float)
        Y_pred = self.model(Xtest)
        return np.squeeze(Y_pred.detach().numpy())
