from alpha_model import AlphaModel
from sklearn.linear_model import LinearRegression
import numpy as np

class RegressionModel(AlphaModel):
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, Y):
        assert len(X)==len(Y)
        self.model.fit(X,Y)

    def predict(self, X):
        return np.array(self.model.predict(X))
